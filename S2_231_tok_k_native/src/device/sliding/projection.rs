
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Dummy2, Dummy256, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// Both clusters: cluster c holds Q rows `2048c..` (heads `4c..4c+4`) and K/V rows `1024c..`
/// (the same heads). `Qs / 2048 == Ps / 1024 == Ns / 4`; the aliases below name the same
/// placement for tensors whose logical axes differ.
pub(crate) type QkvCluster = m![Qs / 2048];
/// The same two clusters for tensors whose row axis is `Ps`. (Naming the cluster axis after an
/// axis the tensor does not have makes the DMA *replicate* instead of split: QKV-003b loaded
/// K/V rows 0..1024 onto both clusters and the second half of k/v came out as garbage.)
type KvCluster = m![Ps / 1024];
/// 8 consecutive Q rows per slice over the full K = H: one contiguous 30 KiB run of FP8 in HBM
/// per slice (the previous `[64 row groups x 4 K-chunks]` layout fetched 32 x 960 B pieces per
/// slice and needed an inter-slice add of the four partials).
type QRows = m![Qs / 8 % 256];
/// 4 consecutive K or V rows per slice (15 KiB contiguous).
type KvRows = m![Ps / 4 % 256];
/// K-split pair layout for the activation load: slice `2g + d` gets K-half `d`.
type PairRows = m![Qs / 16 % 128, H / 1920];

pub(crate) type Hidden = DmTensor<bf16, Chip, QkvCluster, QRows, m![H]>;
/// Projection outputs gathered onto one slice per cluster.
type QueryOut = DmTensor<bf16, Chip, QkvCluster, Slice, m![Qs % 2048]>;
type KvOut = DmTensor<bf16, Chip, KvCluster, Slice, m![Ps % 1024]>;

/// The normalized hidden state on every slice of both clusters: a K-split DMA (each slice of a
/// pair gets one half, 3.4k cycles) and a 2-slice Switch ring so both slices hold all of x
/// (a pure-broadcast DMA to 256 slices took 18.4k cycles, a 256-slice Switch ring 62k).
pub(crate) fn load_hidden(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> Hidden {
    let x_half: DmTensor<bf16, Chip, QkvCluster, PairRows, m![H % 1920]> = x.to_dm(&mut device.tdma);
    let x_full: DmTensor<bf16, Chip, QkvCluster, m![Qs / 16 % 128, Dummy2], m![H]> = device
        .main
        .begin(x_half.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        // The source-slice axis moved into time must be the innermost time axis (verifier rule).
        .switch::<m![Qs / 16 % 128, Dummy2], m![H / 16 % 120, H / 1920]>(SwitchConfig::CustomBroadcast { ring_size: 2 })
        .collect::<m![H / 16 % 120, H / 1920], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    // Slice (g, d) of the pair layout is slice 2g + d of the row layouts; both hold all of x.
    unsafe { x_full.reshape() }
}

/// Rows `4t..4t+4` of every slice's 8 Q rows (15 KiB contiguous per slice, 7.9 MiB in all).
type QueryTile = DmTensor<f8e4m3, Chip, QkvCluster, QRows, m![Qs % 8 = 4, H]>;

fn load_query_tile(device: &mut Device, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>, t: usize) -> QueryTile {
    weight
        .view()
        .tile::<m![Qs % 8], 4, m![Qs / 8, Qs % 8 = 4 # 8, H]>(4 * t)
        .to_dm(&mut device.tdma)
}

/// One Q tile: FP8 -> BF16 lookup fused into the contraction, 4 output rows per slice (8 B of
/// bf16, no inter-slice reduce) into rows `4t..4t+4` of `q`.
fn project_query_tile(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, QkvCluster, QRows, m![1], m![H]>,
    tile: QueryTile,
    q: &mut DmTensor<bf16, Chip, QkvCluster, QRows, m![Qs % 8]>,
    t: usize,
) {
    device.main
        .begin(tile.view())
        .fetch::<m![Qs % 8 = 4, H / 16 % 240], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8 = 4, H / 16 % 240], m![H % 16]>()
        .contract_outer::<m![Qs % 8 = 4, H / 32 % 120], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8 = 4]>()
        .contract_lane::<m![Qs % 8 = 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Qs % 8 = 4 # 16]>()
        .commit_trim::<m![Qs % 8 = 4]>()
        .commit_view(q.view_mut().tile::<m![Qs % 8], 4, m![Qs % 8 = 4 #{!} 8]>(4 * t));
}

/// Q projection from its two weight tiles, gathered onto one slice per cluster.
fn project_query(device: &mut Device, x: &Hidden, w0: QueryTile, w1: QueryTile) -> QueryOut {
    let x_trf: TrfTensor<bf16, Chip, QkvCluster, QRows, m![1], m![H]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();
    let mut q: DmTensor<bf16, Chip, QkvCluster, QRows, m![Qs % 8]> = DmTensor::new();
    project_query_tile(device, &x_trf, w0, &mut q, 0);
    project_query_tile(device, &x_trf, w1, &mut q, 1);
    q.to_dm(&mut device.tdma)
}

/// K or V projection, 4 output rows per slice (8 B of bf16), gathered onto one slice per cluster.
fn project_key_or_value(device: &mut Device, x: &Hidden, weight: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]>) -> KvOut {
    // Same bytes as the Q placement: only the axis names differ.
    let x: DmTensorView<'_, bf16, Chip, KvCluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvCluster, KvRows, m![1], m![H]> = device
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();
    let kv: DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps % 4]> = device
        .main
        .begin(weight.view())
        .fetch::<m![Ps % 4, H / 16 % 240], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 4, H / 16 % 240], m![H % 16]>()
        .contract_outer::<m![Ps % 4, H / 32 % 120], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    kv.to_dm(&mut device.tdma)
}

// The channel scale and the hand-over to the single-slice per-head pipeline (RMSNorm, RoPE,
// stores) are the proven QKV-002b path: the two per-cluster halves meet in HBM (DM-to-DM DMA
// cannot change the cluster mapping), the scale is applied on 256 slices x 16 (8) channels,
// then one Switch broadcast puts the vector on one slice. A first attempt at running the
// per-head work on one slice per cluster and storing from both clusters returned zeros for one
// half of every output (Arena job 35403).

/// `q * channel_scale` (bf16), all 4096 channels on one slice, as `[Ns, Gs, Ds]`.
fn finish_query(device: &mut Device, q: QueryOut, weight_scale: &HbmTensor<bf16, Chip, m![Qs]>) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryRows = m![Qs / 16];
    let q: HbmTensor<bf16, Chip, m![Qs]> = q.to_hbm(&mut device.tdma);
    let q: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = q.to_dm(&mut device.tdma);
    let weight_scale: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QueryRows, m![Qs % 16]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();
    let scaled: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = device
        .main
        .begin(q.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 16]>()
        .switch::<Slice, m![Qs / 16]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();
    unsafe { output.reshape() }
}

/// `kv * channel_scale` (bf16), all 2048 channels on one slice, as `[Ns, Ds]`.
fn finish_key_or_value(device: &mut Device, kv: KvOut, weight_scale: &HbmTensor<bf16, Chip, m![Ps]>) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> {
    type Rows = m![Ps / 8];
    let kv: HbmTensor<bf16, Chip, m![Ps]> = kv.to_hbm(&mut device.tdma);
    let kv: DmTensor<bf16, Chip, Cluster, Rows, m![Ps % 8]> = kv.to_dm(&mut device.tdma);
    let weight_scale: DmTensor<bf16, Chip, Cluster, Rows, m![Ps % 8]> = weight_scale.to_dm(&mut device.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, Rows, m![Ps % 8]> = device
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();
    let scaled: DmTensor<bf16, Chip, Cluster, Rows, m![Ps % 8]> = device
        .main
        .begin(kv.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();
    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = device
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 8 # 16]>()
        .switch::<Slice, m![Ps / 8]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 8], m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();
    unsafe { output.reshape() }
}

/// Q, K and V projections with the channel scales applied, each on one slice, ready for the
/// per-head RMSNorm and RoPE. All three weight loads are issued first so they sit back to back
/// on the serial DMA queue (a load issued after the previous matrix's consumers reuses its
/// address and waits for them: QKV-003 lost 2.2k cycles that way).
pub(crate) fn project_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x = load_hidden(device, x);
    let q_w0 = load_query_tile(device, q_weight, 0);
    let q_w1 = load_query_tile(device, q_weight, 1);
    let k_w: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]> = k_weight.to_dm(&mut device.tdma);
    let v_w: DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps % 4, H]> = v_weight.to_dm(&mut device.tdma);
    let q = project_query(device, &x, q_w0, q_w1);
    let k = project_key_or_value(device, &x, k_w);
    let q = finish_query(device, q, q_weight_scale);
    let v = project_key_or_value(device, &x, v_w);
    let k = finish_key_or_value(device, k, k_weight_scale);
    let v = finish_key_or_value(device, v, v_weight_scale);
    (q, k, v)
}

pub(crate) type OutputCluster = m![H / 1920];
/// K split 16 (step 403): slice (g, k) of cluster c holds rows `1920 c + 120 g + [0, 120)` over K chunk
/// `k = Qs / 256` -- 120 pieces of 256 B per slice (step-82 probe: 612 B/cycle, vs 578 for the 1 KiB
/// pieces of `[H / 30 % 64, Qs / 1024]`). A row group's 120 rows are one epilogue slice's channels.
axes![AoQ = 32, AoR = 120];
pub(crate) type OutputRowsColumns = m![H / 120 % 16, Qs / 256];
type OutputRows = m![H / 120 % 16, 1 # 16];

pub(crate) fn load_output_input(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]> {
    // One 512 B chunk per K shard on the g = 0 slices (32 pieces), replicated over the 16 row groups
    // by the Switch; the padded source-slice axis becomes a padded element axis (innermost in time),
    // which the next pass (one link of the steering chain) drops.
    let seed: DmTensor<bf16, Chip, OutputCluster, m![1 # 16, Qs / 256], m![Qs % 256]> = x.to_dm(&mut device.tdma);
    // pinA probe: q_wfirst pin of the x switch (WAW on a Sub memset of its output).
    let mut gathered: DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs / 16 % 16, 1 # 16, Qs % 16]> = DmTensor::new();
    gathered.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    device
        .main
        .begin(seed.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .switch::<OutputRowsColumns, m![Qs / 16 % 16, 1 # 16]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 })
        .collect::<m![Qs / 16 % 16, 1 # 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit_view(gathered.view_mut());
    device
        .main
        .begin(gathered.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit()
}

/// The attention output as two FP8 vectors `[hi, lo]` with `hi = f8(kx)`, `lo = f8(kx - hi)`
/// (k = 128; 1/k is folded into the channel-scale pass). The FP8 weights contract against them
/// directly, without the FP8 -> BF16 table lookup that bounded the contraction pass (~12 B of
/// decoded output per cycle): the weight stream is read twice (the size-2 axis innermost in time,
/// borrowed from the skeleton's `Dummy2` -- `src/axes.rs` is not part of the submission) and the
/// time reducer sums the two products. |kx| must stay under 448 (|x| <= 3.5).
type OutputActivation = TrfTensor<f8e4m3, Chip, OutputCluster, OutputRowsColumns, m![1], m![Dummy2, Qs % 256]>;
const OUTPUT_ACTIVATION_SCALE: f32 = 128.0;
const OUTPUT_ACTIVATION_INV_SCALE: f32 = 1.0 / 128.0;

fn split_output_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]>,
) -> DmTensor<f8e4m3, Chip, OutputCluster, OutputRowsColumns, m![Dummy2, Qs % 256]> {
    type Rows = OutputRowsColumns;
    let mut out: DmTensor<f8e4m3, Chip, OutputCluster, Rows, m![Dummy2, Qs % 256]> = DmTensor::new();
    device.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), OUTPUT_ACTIVATION_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(out.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(0));
    let hi_neg: DmTensor<f8e4m3, Chip, OutputCluster, Rows, m![Qs % 256]> = device
        .main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -OUTPUT_ACTIVATION_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, OutputCluster, Rows, m![Qs % 256]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .to_vrf();
    device.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), OUTPUT_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(out.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Qs % 256]>(1));
    out
}

/// Rows `r0..r0+ROWS` of every slice's 120 output rows of the O weight (`ROWS` x 256 B pieces per
/// slice), and the contraction of one such tile (K-chunk partials summed over the 16 slices of a row
/// group) into rows `r0..r0+ROWS` of `result`.
macro_rules! output_tile_fns {
    ($load:ident, $tile:ident, $rows:literal, $r0:literal) => {
        fn $load(
            device: &mut Device,
            weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
        ) -> DmTensor<f8e4m3, Chip, OutputCluster, OutputRowsColumns, m![H % 120 = $rows, Qs % 256]> {
            let v: HbmTensorView<'_, f8e4m3, Chip, m![AoQ, AoR, Qs]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f8e4m3, Chip, m![AoQ / 16], m![AoQ % 16, Qs / 256], m![AoR = $rows, Qs % 256]> =
                v.tile::<m![AoR], $rows, m![AoQ, AoR = $rows # 120, Qs]>($r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }

        fn $tile(
            device: &mut Device,
            x_trf: &OutputActivation,
            weight: DmTensor<f8e4m3, Chip, OutputCluster, OutputRowsColumns, m![H % 120 = $rows, Qs % 256]>,
            result: &mut DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]>,
        ) {
            device.main
                .begin(weight.view())
                .fetch::<m![H % 120 = $rows, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
                .collect::<m![H % 120 = $rows, Qs / 64 % 4, Qs / 32 % 2], m![Qs % 32]>()
                .contract_outer::<m![H % 120 = $rows, Qs / 64 % 4, Dummy2], m![Qs % 64], _, _, _>(x_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![H % 120 = $rows]>()
                .contract_lane::<m![H % 120 = $rows], m![1 # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<OutputRows, m![H % 120 = $rows]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![H % 120 = $rows / 2], m![H % 120 = $rows % 2 # 8]>()
                .commit_trim::<m![H % 120 = $rows % 2]>()
                .commit_view(result.view_mut().tile::<m![H % 120], $rows, m![H % 120 = $rows #{!} 120]>($r0));
        }
    };
}
output_tile_fns!(load_output_tile_a, project_output_tile_a, 96, 0);
output_tile_fns!(load_output_tile_b, project_output_tile_b, 24, 96);

/// Identity copy of the activation (a Main fetch/commit pass).
fn copy_output_input(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]>,
) -> DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit()
}

/// A chain of 16 identity copies in front of the activation split (ver16's steering): it changes no
/// value; it only makes x_trf ready after the first weight tile has landed in the *static* schedule, so
/// the scheduler lists the second tile's DMA before the first contraction.
fn delay_output_input(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]>,
) -> DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]> {
    let x = copy_output_input(device, x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    let x = copy_output_input(device, &x);
    x
}

/// Attention output epilogue layout: 32 slices x 120 channels of cluster 0 (the per-slice
/// element count sets the cost of the big passes: 8 x 480 was ~0.3-0.5k slower than 16 x 240, and
/// 32 x 120 saves another ~0.3k; the all-gather Broadcast1{32,1} was validated with the lowering's
/// config_switch on the host and runs on hardware).
axes![AoEpi32 = 32];
pub(crate) type AoEpilogueSlices = m![1 # 8, H / 120];
type AoOperand = DmTensor<bf16, Chip, Cluster, AoEpilogueSlices, m![H % 120]>;

fn ao_operand_vrf(device: &mut Device, t: &AoOperand) -> VrfTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> {
    device
        .sub
        .begin(t.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf()
}

/// Attention output projection with its epilogue fused: `o = (W x) * channel_scale`, post-attention
/// RMSNorm, residual add, all on 32 slices x 120 channels (`AoEpilogueSlices`), returned ready to
/// store. The 64 row groups of both clusters are gathered straight into cluster 0's 32 epilogue
/// slices by one DM-to-DM DMA (ClusterSync; no per-cluster gather to one slice plus a move), and
/// the sqrt pass writes the rms VRF directly from Main (no DM commit + Sub StoVrf).
pub(crate) fn project_output_fused(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputCluster, OutputRowsColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, AoEpilogueSlices, m![H % 120]> {
    let x = delay_output_input(device, x);
    let x2 = split_output_activation(device, &x);
    let x_trf: OutputActivation = device
        .sub
        .begin(x2.view())
        .fetch::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Dummy2, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();
    // The epilogue's operands, on the DMA queue behind the weight.
    let weight_scale: AoOperand = weight_scale.to_dm(&mut device.tdma);
    let norm_weight: AoOperand = norm_weight.to_dm(&mut device.tdma);
    let residual: AoOperand = residual.to_dm(&mut device.tdma);
    // Weight in two row tiles (96 + 24 rows of the slice's 120; even counts so each tile's f32
    // output packs into whole 8 B row pairs) so the first contraction overlaps the second load.
    let w0 = load_output_tile_a(device, weight);
    let w1 = load_output_tile_b(device, weight);
    let mut result: DmTensor<f32, Chip, OutputCluster, OutputRows, m![H % 120]> = DmTensor::new();
    project_output_tile_a(device, &x_trf, w0, &mut result);
    project_output_tile_b(device, &x_trf, w1, &mut result);
    // One DM-to-DM gather: row group g of cluster c is epilogue slice 16 c + g (32 x 480 B pieces).
    // pinA probe aoY: memset the gather destination on Sub first (WAW for the cross-cluster gather).
    let mut y: DmTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = DmTensor::new();
    y.view_mut().memset(0.0f32, &mut device.sub);
    result.view().to_dm_view(&mut device.tdma, y.view_mut());

    // Operands of the two big passes, prepared on Sub: the channel scale, the norm weight and the
    // residual. The 1/128 activation scale is not folded into the channel scale: the mean square
    // is taken of `y * s` (128^2 times larger, exact: a power of two), EPS is scaled to match, and
    // `y * s / rms` is then the correctly scaled value.
    let weight_scale_vrf = ao_operand_vrf(device, &weight_scale);
    let residual_vrf = ao_operand_vrf(device, &residual);

    // (1) mean of (y * s)^2 per slice; (2) all-gather of the 32 partials (named Switch config, no
    // config DMA; an inter-slice reduce over padded live slices costs ~2.3k static on 0.8.1) summed
    // by the intra-slice reducer; (3) sqrt(ms + eps) -> VRF; (4) y * s / rms * w + r -> bf16.
    // (1)+(2) in one pass (step 411, port of step 201): per-slice mean of (y * s)^2, then the
    // 32-slice sum by the inter-slice reducer over the UNPADDED view (256 slices as 8 groups of 32;
    // group 0 = the epilogue slices, the others reduce their own garbage). (3) sqrt -> VRF. (4) final.
    let weight_scale_vrf_g: VrfTensor<f32, Chip, Cluster, m![Dummy256 / 32, H / 120], m![H % 120]> = unsafe { weight_scale_vrf.reshape() };
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![Dummy256 / 32, AoEpi32], m![1 # 8]> = device
        .main
        .begin(unsafe { y.view().reshape::<Chip, Cluster, m![Dummy256 / 32, H / 120], m![H % 120]>() })
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf_g)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Dummy256 / 32, AoEpi32], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = unsafe { weight_scale_vrf_g.reshape() };
    let mean_square: DmTensor<f32, Chip, Cluster, AoEpilogueSlices, m![1 # 8]> = unsafe { reduced_mean_square.reshape() };
    // Step 414: the sqrt pass folded into the final pass (see tools/p_414_v16_ao_sqrt_in_final.py).
    // inv_sw = 1 / (s * w): independent of y, runs while Main waits for the post-gather sync.
    let inv_sw_vrf: VrfTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = device
        .main
        .begin(norm_weight.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_fp_binary_with_mode(FpBinaryOp::DivF, BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    // y itself as a VRF operand (Sub, in parallel with the mean-square pass).
    let y_vrf: VrfTensor<f32, Chip, Cluster, AoEpilogueSlices, m![H % 120]> = device
        .sub
        .begin(y.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    // Final pass: the mean square broadcast over the channels in time (one live lane per step),
    // rms' = sqrt(ms + eps) / (s w), out = y / rms' + r, packed 4 bf16 per packet by the Transpose engine.
    device.main
        .begin(mean_square.view())
        .fetch::<m![H % 120], m![1 # 8]>()
        .collect::<m![H % 120], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS * OUTPUT_ACTIVATION_SCALE * OUTPUT_ACTIVATION_SCALE)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_sw_vrf)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, &y_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}
