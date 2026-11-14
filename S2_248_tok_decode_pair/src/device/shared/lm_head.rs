
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Rw8, W};
use crate::device::layout::Slice;

pub(crate) type Cluster = m![W / 8192 % 2];
pub(crate) type VocabRows = m![W / 32 % 256];

pub(crate) type LogitSlices = m![W / 512 % 16, W / 16384];
pub(crate) type LogitsPerSlice = m![W % 512];

/// Logits per slice in the contraction's own layout: slice `v = W / 32 % 256`, elements `[band W / 16384, row W % 32]`.
pub(crate) type VocabTile = m![W / 16384, W % 32];

/// The LM-head table (the checkpoint's `[W, H]` embedding table) as it sits in HBM: STREAM ORDER. Inside a band of 16384 rows the 256 B chunks
/// (128 columns) are stored `[W % 32, H / 128, cluster, slice, H % 128]`, so one `(row % 32, chunk)` step is one contiguous 128 KB over both clusters and
/// all 256 slices and every DMA engine walks 16 KB blocks (`host::load::lm_table_bytes` writes this order; the bytes are a pure permutation of `[W, H]`).
pub(crate) type LmTableMap = m![W / 16384, W % 32, H / 128, W / 8192 % 2, W / 32 % 256, H % 128];
pub(crate) type LmTable = HbmTensor<bf16, Chip, LmTableMap>;

pub(crate) fn logits(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight: &LmTable,
) -> DmTensor<bf16, Chip, Cluster, LogitSlices, LogitsPerSlice> {
    let logits = logits_banded(device, x, weight);
    device
        .main
        .begin(logits.view())
        .fetch::<m![W / 16384], m![W % 32]>()
        .switch::<LogitSlices, m![W / 32 % 16]>(SwitchConfig::InterTranspose {
            slice1: 16,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![W / 16 % 32], m![W % 16]>()
        .commit_trim::<m![W % 16]>()
        .commit()
}

/// Band `i` (16384 vocabulary rows) of the stream-order table in the contraction's slice layout: slice `v = W / 32 % 256` of cluster `W / 8192 % 2` gets its 32 rows `[W % 32, H]`.
pub(crate) fn band_to_dm(device: &mut Device, weight: &LmTable, i: usize) -> DmTensor<bf16, Chip, Cluster, VocabRows, m![W % 32, H]> {
    weight
        .view()
        .tile::<m![W / 16384], 1, m![1 # 16, W % 32, H / 128, W / 8192 % 2, W / 32 % 256, H % 128]>(i)
        .to_dm(&mut device.tdma)
}

/// The LM-head contraction without the final cross-slice redistribution.
pub(crate) fn logits_banded(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight: &LmTable,
) -> DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile> {
    let x: DmTensor<bf16, Chip, Cluster, VocabRows, m![H]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .switch::<VocabRows, m![H / 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let x_trf: TrfTensor<bf16, Chip, Cluster, VocabRows, m![1], m![H]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();
    contract_bands(device, &x_trf, weight)
}

/// The band loop of the LM head for one hidden row in the TRF: 16 bands of 32 vocabulary rows per slice.
pub(crate) fn contract_bands(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, VocabRows, m![1], m![H]>,
    weight: &LmTable,
) -> DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile> {
    let mut logits: DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile> = DmTensor::new();
    const BANDS: usize = W::SIZE / 16384;

    for i in 0..BANDS {
        let weight_dm = band_to_dm(device, weight, i);

        device
            .main
            .begin(weight_dm.view())
            .fetch::<m![W % 32, H / 16], m![H % 16]>()
            .collect::<m![W % 32, H / 16], m![H % 16]>()
            .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![W % 32]>()
            .contract_lane::<m![W % 32], m![1 # 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![W / 4 % 8], m![W % 4 # 16]>()
            .commit_trim::<m![W % 4]>()
            .commit_view(logits.view_mut().tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32]>(i));
    }
    logits
}

/// tanh soft-cap of the bf16 logits, `30 * tanh(x / 30)`, rounded to bf16, over a per-slice tile flattened to one axis `F` (two passes: the
/// division and the tanh share the FpFpu ALU, so they cannot be fused). Elementwise identical to the passes of `final_norm_and_logits`.
pub(crate) fn softcap_flat<F: AxisName, C: M, S: M>(
    device: &mut Device,
    logits: &DmTensor<bf16, Chip, C, S, m![F]>,
) -> DmTensor<bf16, Chip, C, S, m![F]> {
    let scaled: DmTensor<f32, Chip, C, S, m![F]> = device
        .main
        .begin(logits.view())
        .fetch::<m![F / 16], m![F % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![F / 8], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4], m![F % 4]>()
        .vector_fp_div(crate::LOGIT_SOFTCAP)
        .vector_widen_concat::<m![F / 8], m![F % 8]>()
        .vector_final()
        .commit_trim::<m![F % 8]>()
        .commit();

    device
        .main
        .begin(scaled.view())
        .fetch::<m![F / 8], m![F % 8]>()
        .collect::<m![F / 8], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4], m![F % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), crate::LOGIT_SOFTCAP)
        .vector_widen_concat::<m![F / 8], m![F % 8]>()
        .vector_final()
        .cast::<bf16, m![F % 8 # 16]>()
        .commit_trim::<m![F % 8]>()
        .commit()
}

/// Per-slice tile of 8 rows: `[band W / 16384, row W % 32, lane Rw8]`.
pub(crate) type VocabTile8 = m![W / 16384, W % 32, Rw8];

/// Batched LM-head contraction: 8 hidden rows (one per TRF lane) share one pass over the table.
pub(crate) fn contract_bands_rows8(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, Cluster, VocabRows, m![Rw8], m![H]>,
    weight: &LmTable,
) -> DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile8> {
    let mut logits: DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile8> = DmTensor::new();
    const BANDS: usize = W::SIZE / 16384;

    for i in 0..BANDS {
        let weight_dm = band_to_dm(device, weight, i);

        device
            .main
            .begin(weight_dm.view())
            .fetch::<m![W % 32, H / 16], m![H % 16]>()
            .collect::<m![W % 32, H / 16], m![H % 16]>()
            .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![W % 32]>()
            .contract_lane::<m![W % 32], m![Rw8]>(LaneMode::Interleaved)
            .cast::<bf16, m![Rw8 # 16]>()
            .commit_trim::<m![Rw8]>()
            .commit_view(logits.view_mut().tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32, Rw8]>(i));
    }
    logits
}

/// Batched LM-head contraction for SEVERAL groups of 8 hidden rows in ONE pass over the table: every weight band is DMA'd once, and each group's rows are loaded into the TRF in turn
/// (the TRF holds 8 rows only). The compute of the extra groups (one TRF load + one contraction each per band) hides under the weight DMA. `x*` = the gathered rows of group `*`.
/// (Unrolled by a macro: device code cannot iterate over slices / Vecs.)
macro_rules! contract_groups_fn {
    ($name:ident, $ret:ty, $($x:ident => $l:ident),+) => {
        pub(crate) fn $name(
            device: &mut Device,
            $($x: &crate::device::lmhead::norm::Gathered<m![Rw8], Cluster, VocabRows>,)+
            weight: &LmTable,
        ) -> $ret {
            $(let mut $l: DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile8> = DmTensor::new();)+
            const BANDS: usize = W::SIZE / 16384;
            for i in 0..BANDS {
                let weight_dm = band_to_dm(device, weight, i);
                $(
                    {
                        let x_trf = crate::device::lmhead::norm::load_trf::<m![Rw8], Cluster, VocabRows>(device, $x);
                        device
                            .main
                            .begin(weight_dm.view())
                            .fetch::<m![W % 32, H / 16], m![H % 16]>()
                            .collect::<m![W % 32, H / 16], m![H % 16]>()
                            .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(&x_trf)
                            .contract_packet::<m![1]>()
                            .contract_time::<m![W % 32]>()
                            .contract_lane::<m![W % 32], m![Rw8]>(LaneMode::Interleaved)
                            .cast::<bf16, m![Rw8 # 16]>()
                            .commit_trim::<m![Rw8]>()
                            .commit_view($l.view_mut().tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32, Rw8]>(i));
                    }
                )+
            }
            ($($l),+)
        }
    };
}
type Logits8 = DmTensor<bf16, Chip, Cluster, VocabRows, VocabTile8>;
contract_groups_fn!(contract_bands_groups2, (Logits8, Logits8), xa => la, xb => lb);
