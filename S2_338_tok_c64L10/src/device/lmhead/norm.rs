//! Final RMSNorm of `Rows` hidden rows laid out for the LM head (see `RESULTS.md`).
//!
//! The rows are normalised where the DMA puts them, one row per group of 8 slices (`[Rows # 32, H / 480]`, 480 elements per slice), with the same
//! arithmetic as `shared::rmsnorm::normalize` (so the bf16 result is bit-identical), but the result is NOT gathered into one slice: it is
//! all-gathered straight from the 8 x `Rows` source slices into every slice of the LM head (`CustomBroadcast`, cost ~ source flits x ring span =
//! 30 flits instead of the 240 flits a single-slice source needs: 62k static cycles -> ~8k, and independent of `Rows` up to 32 rows per 8-slice groups).

use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// Normalised rows, bf16, `[H % 480]` per slice of `[Rows # 32, H / 480]`.
pub(crate) fn normalize_rows<Rows: M, Cluster: M>(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![{ Rows }, H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> {
    let x: DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> = x.to_dm(&mut device.tdma);
    normalize_rows_dm::<Rows, Cluster>(device, x, rms_weight)
}

/// `normalize_rows` for rows that are NOT contiguous in HBM: the first `H` columns of rows of the zero padded residual stream (`[Rows, H #{!} 4096]`, the layout of the hybrid pass's
/// residual `[T, Kc, Ke]`), read straight from the pass's residual tensor (no host round trip).
pub(crate) fn normalize_rows_padded<Rows: M, Cluster: M>(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![{ Rows }, H #{!} 4096]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> {
    let x: DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> = x.to_dm(&mut device.tdma);
    normalize_rows_dm::<Rows, Cluster>(device, x, rms_weight)
}

/// The arithmetic of `normalize_rows` on rows already in the slice layout.
fn normalize_rows_dm<Rows: M, Cluster: M>(
    device: &mut Device,
    x: DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> {

    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![{ Rows } # 32, Dummy8], m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![{ Rows } # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![{ Rows } # 32, Dummy8], m![1 # 8]> = device
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, m![{ Rows } # 32, H / 480], m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, m![{ Rows } # 32, H / 480], m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // Same normalisation pass as `shared::rmsnorm::normalize`; the bf16 rounding (Cast engine, RNE) happens here instead of after the gather.
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// All-gathers the normalised rows into every slice of the LM head: per slice `[H % 480 / 16, Rows, H / 480, H % 16]` bf16 (the switch puts the
/// source-slice axes innermost in time, so the flit index is outermost; the TRF load below reads it back in `[Rows, H]` order).
pub(crate) fn gather_rows<Rows: M, Cluster: M, VocabRows: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]>,
) -> DmTensor<bf16, Chip, Cluster, VocabRows, m![H % 480 / 16, { Rows }, H / 480, H % 16]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![H % 480 / 16], m![H % 16]>()
        .switch::<VocabRows, m![H % 480 / 16, { Rows }, H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H % 480 / 16, { Rows }, H / 480], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

/// The gathered rows of `gather_rows` (per slice `[H % 480 / 16, Rows, H / 480, H % 16]`).
pub(crate) type Gathered<Rows, Cluster, VocabRows> = DmTensor<bf16, Chip, Cluster, VocabRows, m![H % 480 / 16, { Rows }, H / 480, H % 16]>;

/// The TRF load of gathered rows (one row per lane).
pub(crate) fn load_trf<Rows: M, Cluster: M, VocabRows: M>(
    device: &mut Device,
    xb: &Gathered<Rows, Cluster, VocabRows>,
) -> TrfTensor<bf16, Chip, Cluster, VocabRows, Rows, m![H]> {
    device
        .sub
        .begin(xb.view())
        .fetch::<m![{ Rows }, H / 480, H % 480 / 16], m![H % 16]>()
        .collect::<m![{ Rows }, H / 480, H % 480 / 16], m![H % 16]>()
        .to_trf()
}

/// `gather_rows` followed by the TRF load (one row per lane).
pub(crate) fn rows_to_trf<Rows: M, Cluster: M, VocabRows: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, m![{ Rows } # 32, H / 480], m![H % 480]>,
) -> TrfTensor<bf16, Chip, Cluster, VocabRows, Rows, m![H]> {
    let xb = gather_rows::<Rows, Cluster, VocabRows>(device, x);
    load_trf::<Rows, Cluster, VocabRows>(device, &xb)
}
