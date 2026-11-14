use furiosa_opt_std::prelude::*;

use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Dummy8, Gs, H, Ps, Qs};

/// Q/K/V share one activation layout: 256 slices per cluster, every slice holding the whole
/// normalized hidden vector, and a slice's weights being a contiguous run of full rows.
///
/// The previous layout split K four ways (`m![Qs / 32 % 64, H / 960]`, 32 rows x 960 channels per
/// slice) so a slice's weights were 32 disjoint 960 B pieces strided through 3,840 B HBM rows:
/// the DMA ran at util 0.70-0.73 and the four K partials then needed an inter-slice reduce. Full
/// rows make a slice's weights one contiguous run (Q 8 x 3,840 B = 30 KiB, K/V 4 x 3,840 B =
/// 15 KiB) and delete the reduce. This is the layout `mlp.rs` already uses for the FFN matrices.
type QCluster = m![Qs / 2048];

/// Head-grouped strided rows: head h (512 rows) lives on slice group h; inside a head, adjacent
/// rows sit on adjacent slices (row r -> slice 64 * (r / 512 % 4) + r % 64, element r / 64 % 8),
/// so concurrently reading slices touch adjacent HBM rows: 602 B/cycle vs 505 for 8 consecutive
/// rows per slice (experiments/qkv-lab/RESULT.md, jobs 73542/73583/73595).
type QRows = m![Qs / 512 % 4, Qs % 64];

type QActivation = TrfTensor<f8e4m3, Chip, QCluster, QRows, m![1], m![Gs, H]>;

// FP8 hi is finite for abs(normalized activation) <= 28 at scale 16.
// Both components are reconstructed exactly on the supplied PRNG fixture,
// but arbitrary BF16 activations are not guaranteed to be represented exactly.
const QKV_ACTIVATION_SCALE: f32 = 16.0;

type QContraction = DmTensor<bf16, Chip, QCluster, QRows, m![Qs / 64 % 8]>;

type KvCluster = m![Ps / 1024];

type KvRows = m![Ps / 256 % 4, Ps % 64];

type KvContraction = DmTensor<bf16, Chip, KvCluster, KvRows, m![Ps / 64 % 4]>;

/// One Q weight tile (rows `r0..r0+ROWS` of every slice's 8-row group, `ROWS x 3,840 B`
/// contiguous per slice) as its own DM tensor, and its contraction into those rows.
///
/// Tiles are separate live tensors with their own consumer chains, which is what makes the load
/// queue stream the next tile while the previous one is on Main; a single whole-matrix `to_dm` is
/// only issued after the preceding Main pass finishes (see the same note in `mlp.rs`). Tiles are
/// a multiple of 4 rows because the Transpose Engine's `in_rows` is capped at 4 for 16-bit
/// elements (`computing-tensors/transpose-engine.html`).
fn load_q_tile8(device: &mut Device, weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>) -> QTile8 {
    weight.to_dm(&mut device.tdma)
}

fn q_tile8(device: &mut Device, x: &QActivation, weight: QTile8) -> QContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8]>()
        .contract_lane::<m![Qs / 64 % 8], m![1 # 8]>(LaneMode::Interleaved)
        // Restore activation scale before the original BF16 rounding boundary.
        // Per-channel weight scaling still happens after gather as before.
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 8 % 4]>()
        .commit()
}

pub(crate) type QTile8 = DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs / 64 % 8, H]>;

pub(crate) type KvWeight = DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]>;

/// All three weight matrices, with independent contraction consumers. Source ordering alone
/// does not force DMA overlap: the compiler schedules their dataflow and SRAM lifetimes.
pub(crate) type Weights = (QTile8, KvWeight, KvWeight);

pub(crate) fn load_weights(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> Weights {
    let q0 = load_q_tile8(device, q_weight);
    let k = load_kv(device, k_weight);
    let v = load_kv(device, v_weight);
    (q0, k, v)
}

/// One whole K or V matrix: with only four rows per slice there is nothing left to tile, but the
/// two matrices are still issued as separate live tensors ahead of either contraction.
fn load_kv(
    device: &mut Device,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> DmTensor<f8e4m3, Chip, KvCluster, KvRows, m![Ps / 64 % 4, H]> {
    weight.to_dm(&mut device.tdma)
}

/// K: gathered heads plus the per-channel scale (applied inside `normalize_key_scaled`).
fn finish_one_kv_matrix(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    gather_value(device, contraction, weight_scale)
}

/// K: ring-64 all-gather (`Broadcast1{64, 1}`) of each head's 64 contraction slices onto the first
/// slice of its PE (`rope_h4::KvHeadsAcrossSlices = [Ns % 4, 1 # 64]`), then the channel transpose.
/// The v7 gather (`TransposedBroadcast1{4, 64}` onto slices 0..3) was a ring-256 pass: 2.43k HW vs 0.79k.
pub(crate) fn gather_value(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered_p: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 64, Ps / 64 % 4]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps / 64 % 4 # 16]>()
            .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64]>(
                SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 },
            )
            .collect::<m![Ps % 64], m![Ps / 64 % 4 # 16]>()
            .commit_trim::<m![Ps / 64 % 4]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// K/V weights relabelled onto the Q axes (same cluster/slice/element extents), so one TRF load of
/// the activation serves all three contractions.
type KvWeightQ = DmTensor<f8e4m3, Chip, QCluster, QRows, m![Qs / 64 % 4, H]>;

type KvContractionQ = DmTensor<bf16, Chip, QCluster, QRows, m![Qs / 64 % 4]>;

pub(crate) fn contract_one_kv_matrix_q(device: &mut Device, x: &QActivation, weight: KvWeight) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// V gathered straight into natural channel order: the contraction output is fetched one channel per
/// packet with its 4 rows in time, the ring-64 all-gather appends the 64 slices innermost
/// (time = [row group j, slice s] = channel 64 j + s), and the Transpose Engine packs 4 steps per
/// packet. One pass instead of gather + transpose.
pub(crate) fn gather_value_f(
    device: &mut Device,
    contraction: KvContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Ps / 64 % 4], m![1 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps / 64 % 4, Ps % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

// x and the input RMS weight replicated 8x instead of 32x: one valid eight-slice shard group per 32
// slices (`1 # 4` between the group and the shard axis; the inter-slice reduce needs its shard axis
// innermost), so each replicated load writes a quarter of the bytes (2.75k -> ~1.5k HW each). A
// `Broadcast01{4, 8, 30}` then gives every slice all eight shards; the three padding slices' packets
// land in the `1 # 4` element padding of the result, which the TRF load skips.
axes![QkvQuad = 4];

type XRows8 = m![Qs / 256 % 8, 1 # 4, H / 480];

pub(crate) type BroadcastInput8 = DmTensor<bf16, Chip, QCluster, XRows8, m![H % 480]>;

type XHilo8 = DmTensor<f8e4m3, Chip, QCluster, QRows, m![1 # 4, Gs, H / 32 % 15, H / 480, H % 32]>;

fn split_qkv_activation8(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, QCluster, XRows8, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, QCluster, XRows8, m![Gs, H % 480]> {
    type Rows = XRows8;
    let mut out: DmTensor<f8e4m3, Chip, QCluster, Rows, m![Gs, H % 480]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(0));
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, QCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, QCluster, Rows, m![H % 480]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(1));
    out
}

/// Input RMSNorm on the 8x-replicated layout. The rms weight is loaded first and the mean square adds
/// `sum(0 * w)` (exactly 0 for finite weights): that data dependency makes the scheduler list the K
/// weight load right after x instead of after the whole Sub RMS chain (K streams from x landing).
pub(crate) fn normalize_input_x8w(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> BroadcastInput8 {
    type ReducingSlices = XRows8;

    // The rms weight is loaded first and the mean-square adds `sum(0 * w)` (exactly 0 for finite w): the
    // scheduler then lists the K weight load right after x instead of after the whole RMS chain.
    let weight_dm: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let zero: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(zero.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 256 % 8, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, crate::EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = device
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
        .commit();

    normalized
}

fn broadcast_hilo8(device: &mut Device, x: &BroadcastInput8) -> XHilo8 {
    let x = split_qkv_activation8(device, x);
    let full: DmTensor<f8e4m3, Chip, QCluster, m![Qs / 256 % 8, QkvQuad, Dummy8], m![1 # 4, Gs, H / 32 % 15, H / 480, H % 32]> =
        device.main
            .begin(x.view())
            .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
            .switch::<m![Qs / 256 % 8, QkvQuad, Dummy8], m![1 # 4, Gs, H / 32 % 15, H / 480]>(
                SwitchConfig::Broadcast01 { slice1: 4, slice0: 8, time0: 30 },
            )
            .collect::<m![1 # 4, Gs, H / 32 % 15, H / 480], m![H % 32]>()
            .commit_trim::<m![H % 32]>()
            .commit();
    unsafe { full.reshape() }
}

/// Q gathered straight into natural channel order (see `gather_value_f`): 8 rows in time, one
/// channel per packet, ring-64 all-gather appends the 64 slices innermost.
pub(crate) fn finish_query_f(
    device: &mut Device,
    contraction: QContraction,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Qs / 64 % 8], m![1 # 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 64], m![Qs / 64 % 8, Qs % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Qs / 64 % 8, Qs % 64], m![1 # 16]>()
        .transpose::<m![Qs % 512 / 4], m![Qs % 512 % 4 # 16]>()
        .commit_trim::<m![Qs % 512 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// Q, K, V projections from one TRF load (K/V weights relabelled onto the Q axes), heads gathered
/// onto one slice per PE, Q and V gathered straight into channel order, V normalized with its scale
/// folded in and the rms written to VRF by the sqrt pass.
pub(crate) fn project_x8g(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

/// A1: 64 B stream packets (PackSize 2) with hi/lo (`Gs`) as a stream-adapter time broadcast: each
/// weight flit is fetched once and multiplied against both activation halves, and every MAC cycle
/// uses the full 64 B packet (the 32 B form fetched every flit twice and idled half the multipliers).
fn q_tile8_p64(device: &mut Device, x: &QActivation, weight: QTile8) -> QContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8]>()
        .contract_lane::<m![Qs / 64 % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 8 % 4]>()
        .commit()
}

/// A1 form of `contract_one_kv_matrix_q` (see `q_tile8_p64`).
pub(crate) fn contract_one_kv_matrix_q_p64(device: &mut Device, x: &QActivation, weight: KvWeight) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// `project_x8g` with the A1 contractions at all three sites.
pub(crate) fn project_x8g_a1(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8_p64(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_p64(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

/// A1 on the V contraction only.
pub(crate) fn project_x8g_a1v(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

/// A1 on the K and V contractions.
pub(crate) fn project_x8g_a1kv(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_p64(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

/// A1 on the Q and V contractions.
pub(crate) fn project_x8g_a1qv(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8_p64(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

// ---------------------------------------------------------------------------------------------
// qd: queue-ahead dependencies. On hardware a DMA listed after a contraction is issued only when
// that contraction is dispatched (~0.29k after its weight lands), into an empty DMA queue, where a
// small load costs ~1.3k instead of the ~0.9k it costs queued behind another DMA. Making the
// contraction consume (through an exact 1/16 VRF in its epilogue) the first small load that follows
// its weight moves that load in front of the contraction in the static list, so hardware issues it
// early and it queues behind the weight; the next small load then ties with the contraction start
// and is listed first as well.
// ---------------------------------------------------------------------------------------------

type Inv16Vrf = VrfTensor<f32, Chip, QCluster, QRows, m![1 # 8]>;

/// Exact 1/16 on every contraction slice from a 4-byte i32 held by the head slices only: the other
/// slices read whatever is in DM there, which the bitwise AND clears before any float use.
fn inv16_from_i32(device: &mut Device, t: &DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>) -> Inv16Vrf {
    let t: DmTensorView<'_, i32, Chip, QCluster, QRows, m![1 # 8]> = unsafe { t.view().reshape() };
    device.main
        .begin(t)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0i32)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// Exact 1/16 on every contraction slice from a bf16 [Ds] tensor held by the head slices only.
fn inv16_from_ds(device: &mut Device, t: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>) -> Inv16Vrf {
    let t: DmTensorView<'_, bf16, Chip, QCluster, QRows, m![Ds]> = unsafe { t.view().reshape() };
    device.main
        .begin(t)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_final()
        .to_vrf(&mut device.sub)
}

fn q_tile8_dep(device: &mut Device, x: &QActivation, weight: QTile8, inv16: &Inv16Vrf) -> QContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8]>()
        .contract_lane::<m![Qs / 64 % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inv16)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 8 % 4]>()
        .commit()
}

fn contract_one_kv_matrix_q_dep(device: &mut Device, x: &QActivation, weight: KvWeight, inv16: &Inv16Vrf) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 32 % 120, Gs], m![H % 32], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inv16)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

pub(crate) type RopeOffsetDm = DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;
pub(crate) type HeadNormWeightDm = DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>;

/// `project_x8g` where the K contraction consumes `rope_offset` and the Q contraction consumes
/// `k_rms_weight` (both through an exact 1/16 epilogue VRF), so those loads are listed ahead of the
/// contractions and queue behind the K / Q weights on hardware. The loaded tensors are returned for
/// their real consumers (RoPE rows, K normalize).
pub(crate) fn project_x8g_qd(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    RopeOffsetDm,
    HeadNormWeightDm,
) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let off: RopeOffsetDm = rope_offset.to_dm(&mut device.tdma);
    let inv16_k = inv16_from_i32(device, &off);
    let k_rms: HeadNormWeightDm = k_rms_weight.to_dm(&mut device.tdma);
    let inv16_q = inv16_from_ds(device, &k_rms);
    let q = q_tile8_dep(device, &x_trf, q0, &inv16_q);
    let k = contract_one_kv_matrix_q_dep(device, &x_trf, k_weight, &inv16_k);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v, off, k_rms)
}

// ---------------------------------------------------------------------------------------------
// g: static-list steering. The static scheduler lets a TU command win a tie against a DMA, and a
// contraction starts (statically) the moment its weight lands, so the small loads that follow a
// weight in the DMA lane are listed after its contraction: on hardware they are issued ~0.29k after
// the contraction's dispatch into an idle lane (+~0.36k each). Making the contraction's epilogue
// multiply by an exact 1/16 derived from the LAST small load of that lane segment lists that load
// (and the ones before it, and the next weight) ahead of the contraction.
// ---------------------------------------------------------------------------------------------

/// A K/V row scale loaded straight into the head layout (the load `gather_value` does).
fn load_kv_scale_heads(device: &mut Device, s: &HbmTensor<bf16, Chip, m![Ps]>) -> HeadNormWeightDm {
    let t: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = s.to_dm(&mut device.tdma);
    unsafe { t.reshape() }
}

/// `gather_value` without its scale load.
fn gather_value_ns(device: &mut Device, contraction: KvContraction) -> HeadNormWeightDm {
    let gathered_p: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 64, Ps / 64 % 4]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps / 64 % 4 # 16]>()
            .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64]>(
                SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 },
            )
            .collect::<m![Ps % 64], m![Ps / 64 % 4 # 16]>()
            .commit_trim::<m![Ps / 64 % 4]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .collect::<m![Ps / 64 % 4, Ps % 64], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    unsafe { gathered.reshape() }
}

type ProjOut = (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
);

/// g1: the K contraction's epilogue 1/16 comes from `k_weight_scale` (loaded first, head layout).
pub(crate) fn project_x8g_g1(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> ProjOut {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let k_scale = load_kv_scale_heads(device, k_weight_scale);
    let inv16_k = inv16_from_ds(device, &k_scale);
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_dep(device, &x_trf, k_weight, &inv16_k);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let k = gather_value_ns(device, k);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    (q, q_scale, k, k_scale, v)
}

/// g2: the K contraction's epilogue 1/16 comes from `k_rms_weight` (so it is loaded before Q and the
/// K normalize can run before the Q contraction).
pub(crate) fn project_x8g_g2(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (ProjOut, HeadNormWeightDm) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let k_rms: HeadNormWeightDm = k_rms_weight.to_dm(&mut device.tdma);
    let inv16_k = inv16_from_ds(device, &k_rms);
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_dep(device, &x_trf, k_weight, &inv16_k);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), k_rms)
}

/// g12: K contraction depends on both k_scale and k_rms (1/16 from k_scale, times exact 1 from k_rms).
pub(crate) fn project_x8g_g12(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (ProjOut, HeadNormWeightDm) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let k_scale = load_kv_scale_heads(device, k_weight_scale);
    let k_rms: HeadNormWeightDm = k_rms_weight.to_dm(&mut device.tdma);
    let inv16_a = inv16_from_ds(device, &k_scale);
    let inv16_k = inv16_times_zero_ds(device, &k_rms, &inv16_a);
    let q = q_tile8(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_dep(device, &x_trf, k_weight, &inv16_k);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let k = gather_value_ns(device, k);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), k_rms)
}

/// `prev + sum(t AND 0)`: the same exact value as `prev`, with a data dependency on `t`.
fn inv16_times_zero_ds(device: &mut Device, t: &HeadNormWeightDm, prev: &Inv16Vrf) -> Inv16Vrf {
    let t: DmTensorView<'_, bf16, Chip, QCluster, QRows, m![Ds]> = unsafe { t.view().reshape() };
    device.main
        .begin(t)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, prev)
        .vector_final()
        .to_vrf(&mut device.sub)
}

// ---------------------------------------------------------------------------------------------
// k-pin: the K normalize waits for the Q contraction (an exact 0 from the Q contraction output is
// added to the K mean square), which keeps `k_rms_weight` behind the Q weight in the static list
// whatever the tail looks like (A1 on the V contraction otherwise moves it ahead of Q).
// ---------------------------------------------------------------------------------------------

pub(crate) type HeadZeroVrf = VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;

/// Exact 0.0 on the head slices, computed from the Q contraction output (bits ANDed with 0).
fn zero_from_q_contraction(device: &mut Device, q: &QContraction) -> HeadZeroVrf {
    let t: DmTensorView<'_, bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Qs / 64 % 8]> = unsafe { q.view().reshape() };
    device.main
        .begin(t)
        .fetch::<m![1], m![Qs / 64 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs / 64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `project_x8g` (optionally with A1 on V) that also returns the exact-zero VRF from the Q contraction.
pub(crate) fn project_x8g_kpin(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let zero = zero_from_q_contraction(device, &q);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}

pub(crate) fn project_x8g_a1v_kpin(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8(device, &x_trf, q0);
    let zero = zero_from_q_contraction(device, &q);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}

// ---------------------------------------------------------------------------------------------
// lay probes (timing only): weight DMAs in permuted mappings, relabelled to the base types.
// ---------------------------------------------------------------------------------------------
macro_rules! lay_weights {
    ($name:ident, $kc:ty, $ks:ty, $ke:ty, $qc:ty, $qs:ty, $qe:ty) => {
        pub(crate) fn $name(
            device: &mut Device,
            q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
            k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
            v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
        ) -> Weights {
            let q: DmTensor<f8e4m3, Chip, $qc, $qs, $qe> = q_weight.to_dm(&mut device.tdma);
            let k: DmTensor<f8e4m3, Chip, $kc, $ks, $ke> = k_weight.to_dm(&mut device.tdma);
            let v: DmTensor<f8e4m3, Chip, $kc, $ks, $ke> = v_weight.to_dm(&mut device.tdma);
            (unsafe { q.reshape() }, unsafe { k.reshape() }, unsafe { v.reshape() })
        }
    };
}
// P1: cluster = head parity (heads stay whole inside a PE; implementable).
lay_weights!(load_weights_p1,
    m![Ps / 256 % 2], m![Ps / 512 % 4, Ps % 64], m![Ps / 64 % 4, H],
    m![Qs / 512 % 2], m![Qs / 1024 % 4, Qs % 64], m![Qs / 64 % 8, H]);
// P2: cluster = row parity.
lay_weights!(load_weights_p2,
    m![Ps % 2], m![Ps / 512 % 4, Ps / 2 % 64], m![Ps / 128 % 4, H],
    m![Qs % 2], m![Qs / 1024 % 4, Qs / 2 % 64], m![Qs / 128 % 8, H]);
// P3: rows interleaved over the four PE engines of a cluster.
lay_weights!(load_weights_p3,
    m![Ps / 1024], m![Ps % 4, Ps / 4 % 64], m![Ps / 256 % 4, H],
    m![Qs / 2048], m![Qs % 4, Qs / 4 % 64], m![Qs / 256 % 8, H]);
// P4: both (cluster = row % 2, PE = row / 2 % 4).
lay_weights!(load_weights_p4,
    m![Ps % 2], m![Ps / 2 % 4, Ps / 8 % 64], m![Ps / 512 % 4, H],
    m![Qs % 2], m![Qs / 2 % 4, Qs / 8 % 64], m![Qs / 512 % 8, H]);

// ---------------------------------------------------------------------------------------------
// e: de-eagered VRF builds (Main vector pass -> Sub VRF instead of an eager Sub to_vrf).
// ---------------------------------------------------------------------------------------------

/// `normalize_input_x8w` whose rms-weight VRF is written by a Main pass (x is then queued behind rms_w).
pub(crate) fn normalize_input_x8w_e(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> BroadcastInput8 {
    type ReducingSlices = XRows8;
    let weight_dm: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let zero: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(zero.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 256 % 8, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, crate::EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };
    // The weight VRF from a Main pass (exact: x * 1.0), so it is placed next to the normalize instead of at
    // the moment rms_w lands (where a Sub to_vrf is listed ahead of the x load and gates its issue).
    let weight_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![H % 480]> = device
        .main
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let rms_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = device
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
        .commit();
    normalized
}

pub(crate) fn project_x8g_a1qv_kpin(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8_p64(device, &x_trf, q0);
    let zero = zero_from_q_contraction(device, &q);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}

/// Exact 0.0 on the head slices from the gathered q (bits ANDed with 0, then reduced).
pub(crate) fn zero_from_q_heads(device: &mut Device, q: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>) -> HeadZeroVrf {
    // [Gs, Ds] relabelled as one 512-element axis (same wire order) so a single reduce leaves one value.
    let t: DmTensorView<'_, bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Qs % 512]> = unsafe { q.view().reshape() };
    device.main
        .begin(t)
        .fetch::<m![Qs % 512 / 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs % 512 / 8], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Qs % 512 / 4], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// A1 on Q and V; the K normalize's final pass is pinned behind the Q GATHER (exact 0 from the gathered q),
/// which the static list places after the V weight load (V then queues behind the last small load).
pub(crate) fn project_x8g_a1qv_kpin3(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8_p64(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_p64(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let zero = zero_from_q_heads(device, &q);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}

// ---------------------------------------------------------------------------------------------
// lg: hi/lo (`Gs`) on two TRF lanes. The weight packet (64 B) is broadcast to both lanes and multiplied
// against x_hi and x_lo in the same step (half the MAC steps of A1); the two lane partials are summed in
// the epilogue.
// ---------------------------------------------------------------------------------------------
type QActivationLg = TrfTensor<f8e4m3, Chip, QCluster, QRows, m![Gs], m![H]>;

fn q_tile8_lg(device: &mut Device, x: &QActivationLg, weight: QTile8) -> QContraction {
    device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 8, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 8, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 8, H / 64 % 60], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 8]>()
        .contract_lane::<m![Qs / 64 % 8], m![Gs # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Gs # 4]>()
        .vector_intra_slice_reduce::<Gs, m![Qs / 64 % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 8 / 4], m![Qs / 64 % 8 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 8 % 4]>()
        .commit()
}

fn contract_one_kv_matrix_q_lg(device: &mut Device, x: &QActivationLg, weight: KvWeight) -> KvContraction {
    let weight: KvWeightQ = unsafe { weight.reshape() };
    let out: KvContractionQ = device.main
        .begin(weight.view())
        .fetch::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .collect::<m![Qs / 64 % 4, H / 64 % 60, H / 32 % 2], m![H % 32]>()
        .contract_outer::<m![Qs / 64 % 4, H / 64 % 60], m![H % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs / 64 % 4]>()
        .contract_lane::<m![Qs / 64 % 4], m![Gs # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Gs # 4]>()
        .vector_intra_slice_reduce::<Gs, m![Qs / 64 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 64 % 4 / 4], m![Qs / 64 % 4 # 16]>()
        .commit_trim::<m![Qs / 64 % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// a1qvk3 with all three contractions in lane form (hi/lo on TRF lanes).
pub(crate) fn project_x8g_lg_kpin3(
    device: &mut Device,
    x: &BroadcastInput8,
    w: Weights,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivationLg = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q = q_tile8_lg(device, &x_trf, q0);
    let k = contract_one_kv_matrix_q_lg(device, &x_trf, k_weight);
    let v = contract_one_kv_matrix_q_lg(device, &x_trf, v_weight);
    let (q, q_scale) = finish_query_f(device, q, q_weight_scale);
    let zero = zero_from_q_heads(device, &q);
    let (k, k_scale) = finish_one_kv_matrix(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}

// ---------------------------------------------------------------------------------------------
// h2: only x before the K weight. The x normalize adds an exact 0 taken from the landed K weight, so the
// static list places the rms-weight load behind K (the head, where neither cluster streams, shrinks by the
// rms_w load); the Sub mean-square chain still runs on x as soon as it lands.
// ---------------------------------------------------------------------------------------------
pub(crate) type XZeroVrf = VrfTensor<f32, Chip, QCluster, XRows8, m![1 # 8]>;

/// Exact 0.0 on every slice from one 32-byte tile of the K weight (bits ANDed with 0).
pub(crate) fn zero_from_kweight(device: &mut Device, k: &KvWeight) -> XZeroVrf {
    let z: DmTensor<f32, Chip, KvCluster, KvRows, m![1 # 8]> = device.main
        .begin(k.view().tile::<m![Ps / 64 % 4], 1, m![Ps / 64 % 4 = 1 # 4, H]>(0))
        .fetch::<m![H / 32], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let z: DmTensor<f32, Chip, QCluster, XRows8, m![1 # 8]> = unsafe { z.reshape() };
    device.sub
        .begin(z.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// Input RMSNorm with x loaded first (no dependency on the rms weight) and the normalize gated on `kz`.
pub(crate) fn normalize_input_x8h(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    kz: &XZeroVrf,
) -> BroadcastInput8 {
    type ReducingSlices = XRows8;
    let x: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let reduced_mean_square: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
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
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 256 % 8, 1 # 4, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, m![Qs / 256 % 8, 1 # 4, Dummy8], m![1 # 8]> = device
        .sub
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, crate::EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let weight_dm: DmTensor<bf16, Chip, QCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, QCluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, kz)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

// ---- v10 (qkv10 E301) far2 weight order, numerically real. Q/K/V weights land with the DMN bit of the slice index = row bit 8 (Q)
// / 7 (K, V) and the slice-in-DMN = row bits {6,7}+{0..2} (Q) / {5,6}+{0..2} (K, V); element = row bits 3-5 (Q) /
// 3-4 (K, V). Contractions run on the relabelled base types (row-independent), the gathers restore channel
// order (fused Q/V gathers: the commit reorders time; K pass 2 fetches in channel order). ----
type QRowsF2 = m![Qs / 512 % 4, Qs / 256 % 2, Qs / 64 % 4, Qs % 8];
pub(crate) type QTileF2 = DmTensor<f8e4m3, Chip, QCluster, QRowsF2, m![Qs / 8 % 8, H]>;
type QContractionF2 = DmTensor<bf16, Chip, QCluster, QRowsF2, m![Qs / 8 % 8]>;
type KvRowsF2 = m![Ps / 256 % 4, Ps / 128 % 2, Ps / 32 % 4, Ps % 8];
pub(crate) type KvWeightF2 = DmTensor<f8e4m3, Chip, KvCluster, KvRowsF2, m![Ps / 8 % 4, H]>;
type KvContractionF2 = DmTensor<bf16, Chip, KvCluster, KvRowsF2, m![Ps / 8 % 4]>;
pub(crate) type WeightsF2 = (QTileF2, KvWeightF2, KvWeightF2);

pub(crate) fn load_weights_f2(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> WeightsF2 {
    let q: QTileF2 = q_weight.to_dm(&mut device.tdma);
    let k: KvWeightF2 = k_weight.to_dm(&mut device.tdma);
    let v: KvWeightF2 = v_weight.to_dm(&mut device.tdma);
    (q, k, v)
}

fn finish_query_f2(
    device: &mut Device,
    contraction: QContractionF2,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Qs / 8 % 8], m![1 # 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 64], m![Qs / 8 % 8, Qs / 256 % 2, Qs / 64 % 4, Qs % 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Qs / 8 % 8, Qs / 256 % 2, Qs / 64 % 4, Qs % 8], m![1 # 16]>()
        .transpose::<m![Qs / 8 % 8, Qs / 256 % 2, Qs / 64 % 4, Qs % 8 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

fn gather_key_f2(
    device: &mut Device,
    contraction: KvContractionF2,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered_p: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps / 128 % 2, Ps / 32 % 4, Ps % 8, Ps / 8 % 4]> =
        device.main
            .begin(contraction.view())
            .fetch::<m![1], m![Ps / 8 % 4 # 16]>()
            .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps / 128 % 2, Ps / 32 % 4, Ps % 8]>(
                SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 },
            )
            .collect::<m![Ps / 128 % 2, Ps / 32 % 4, Ps % 8], m![Ps / 8 % 4 # 16]>()
            .commit_trim::<m![Ps / 8 % 4]>()
            .commit();
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(gathered_p.view())
        .fetch::<m![Ps / 128 % 2, Ps / 32 % 4, Ps / 8 % 4, Ps % 8], m![1 # 16]>()
        .collect::<m![Ps / 128 % 2, Ps / 32 % 4, Ps / 8 % 4, Ps % 8], m![1 # 16]>()
        .transpose::<m![Ps % 256 / 4], m![Ps % 256 % 4 # 16]>()
        .commit_trim::<m![Ps % 256 % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

fn gather_value_f2(
    device: &mut Device,
    contraction: KvContractionF2,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(contraction.view())
        .fetch::<m![Ps / 8 % 4], m![1 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps / 8 % 4, Ps / 128 % 2, Ps / 32 % 4, Ps % 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ps / 8 % 4, Ps / 128 % 2, Ps / 32 % 4, Ps % 8], m![1 # 16]>()
        .transpose::<m![Ps / 8 % 4, Ps / 128 % 2, Ps / 32 % 4, Ps % 8 / 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

pub(crate) fn project_x8g_a1qv_kpin3_f2(
    device: &mut Device,
    x: &BroadcastInput8,
    w: WeightsF2,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (ProjOut, HeadZeroVrf) {
    let (q0, k_weight, v_weight) = w;
    let xq = broadcast_hilo8(device, x);
    let x_trf: QActivation = device
        .sub
        .begin(xq.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let q: QContractionF2 = unsafe { q_tile8_p64(device, &x_trf, q0.reshape()).reshape() };
    let k: KvContractionF2 = unsafe { contract_one_kv_matrix_q(device, &x_trf, k_weight.reshape()).reshape() };
    let v: KvContractionF2 =
        unsafe { contract_one_kv_matrix_q_p64(device, &x_trf, v_weight.reshape()).reshape() };
    let (q, q_scale) = finish_query_f2(device, q, q_weight_scale);
    let zero = zero_from_q_heads(device, &q);
    let (k, k_scale) = gather_key_f2(device, k, k_weight_scale);
    let (v, v_scale) = gather_value_f2(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    ((q, q_scale, k, k_scale, v), zero)
}
