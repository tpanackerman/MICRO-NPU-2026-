
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy2, Dummy8, Dummy256, Gs, H, L};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const INVSQRT2: f32 = 0.70710678118f32;
const H_F32: f32 = H::SIZE as f32;

/// Layout of the fused Down epilogue (global scale, post-FFN RMSNorm, residual, layer gate):
/// 8 slices of cluster 0 with 480 channels each, the same reducing layout `rmsnorm::normalize`
/// uses. The kernel stores the result to HBM straight from it.
pub(crate) type EpilogueSlices = m![1 # 16, H / 240];
pub(crate) type EpilogueElem = m![H % 240];
axes![Epi16 = 16];

type UpCluster = m![L / 7680];
/// 256 slices per cluster, each holding 30 consecutive Up/Gate rows over the full K = H.
///
/// A slice's packed weights are then one contiguous 57.6 KiB run in HBM (30 rows x 1920 B) and
/// its block scales one contiguous 7.2 KiB run (30 x 240 B). The previous
/// `[128 row groups x 2 K-halves]` layout fetched 60 x 960 B and 60 x 120 B pieces per slice
/// (DMA util 0.74 and 0.28) and then needed an inter-slice add of the two K-half partials plus
/// its own relayout. Per-slice work is unchanged (30 x 3840 = 60 x 1920 weights).
type UpRows = m![L / 30 % 256];
type UpActivation = TrfTensor<f8e4m3, Chip, UpCluster, UpRows, m![1], m![Gs, H]>;

// V10-E101: x / rms weight replicated 4x (one 8-slice shard group per 64 slices); a named
// Broadcast01{8, 8, 30} all-gathers the shards (no ring-256 CustomBroadcast, no snoop-bitmap config DMA).
axes![HeadPad = 8];
type UpPairRows = m![L / 1920 % 4, 1 # 8, H / 480];
pub(crate) type NormalizedInput = (
    DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>,
    DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>,
);
type NormalizedX = DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>;
// Finite FP8 hi requires abs(x) <= 28. This is checked on the local fixture in
// numeric_check.py; arbitrary out-of-range activations need a different scaling scheme.
const UP_ACTIVATION_SCALE: f32 = 16.0;
fn split_up_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpCluster, UpPairRows, m![H % 480]>,
) -> DmTensor<f8e4m3, Chip, UpCluster, UpPairRows, m![Gs, H % 480]> {
    type Rows = UpPairRows;
    let hi: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, UpCluster, Rows, m![H % 480]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let lo: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![H % 480]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let mut out: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![Gs, H % 480]> = DmTensor::new();
    device.main
        .begin(hi.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(0));
    device.main
        .begin(lo.view())
        .fetch::<m![H / 32 % 15], m![H % 32]>()
        .collect::<m![H / 32 % 15], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, H % 480]>(1));
    out
}



// Split after the BF16 GeGLU rounding, while each active slice holds 120 values.
// Scale 256 needs abs(GeGLU) <= 1.75 for finite FP8 hi; validate on device.
const DOWN_ACTIVATION_SCALE: f32 = 256.0;
/// F9D usfg (usf on F9B_hd, refold in the epilogue): the up global scale su (in (2^-14, 2^-11]) leaves GeGLU. h is computed
/// with the constant 2^-13 in its place (h'' = h * 2^-13 / su, |h''| < 2 |h|). Down and the epilogue are linear in y up to the
/// global-scale product y * gs, so su * 2^13 is folded into the down global-scale VRF (gs' = (su * 2^13) * gs, one f32
/// rounding) that the mean-square and final passes already read: no new VRF operand in any Main pass.
const UP_FOLD: f32 = 1.0 / 131072.0; // 2^-13 / UP_ACTIVATION_SCALE
const UP_REFOLD: f32 = 8192.0; // 2^13
fn split_down_activation(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, UpCluster, UpPairedRows, m![L % 120]>,
) -> DmTensor<f8e4m3, Chip, UpCluster, UpPairedRows, m![Gs, L % 120]> {
    type Rows = UpPairedRows;
    let mut out: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![Gs, L % 120]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, L % 120]>(0));
    // -hi (exact) as the VRF operand of the lo pass: `SubF` with a VRF operand evaluates
    // `vrf - stream` and a MulF may not follow it, so the lo pass adds the negated hi.
    let hi_neg: DmTensor<f8e4m3, Chip, UpCluster, Rows, m![L % 120]> = device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -DOWN_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, UpCluster, Rows, m![L % 120]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, L % 120]>(1));
    out
}



/// Block scales, each slice's own 30 rows (7,200 B contiguous in HBM), loaded directly in the
/// layout the scale passes read: no pair seed and no InterTranspose pass on Main.
type UpScale = DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![L % 30, H / 16]>;

fn load_up_gate_scale(device: &mut Device, scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>) -> UpScale {
    scale.to_dm(&mut device.tdma)
}

axes![Dq4 = 4];
/// LAB q4: 120-row quad seed (28.8 KB per leader slice; every other quad starts 256 B aligned,
/// the pair seed only every fourth pair) and a ring-4 InterTranspose.
type UpScale4 = DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![Dq4, L % 30, H / 16]>;
type UpScaleSeed4 = DmTensor<f8e4m3, Chip, UpCluster, m![L / 120 % 64, 1 # 4], m![L % 120, H / 16]>;

axes![D2 = 2];
/// V14 FFN-OCT-b: 240-row octet seed, one leader per 8 slices.
type UpScaleSeed8 = DmTensor<f8e4m3, Chip, UpCluster, m![L / 240 % 32, 1 # 8], m![L % 240, H / 16]>;
/// V14 FFN-OCT-b: the relaid scale keeps a 2-way padded time digit (tile D2 = 0 holds the data).
type UpScale2 = DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![D2, L % 30, H / 16]>;

fn load_up_gate_scale4(device: &mut Device, scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>) -> UpScale2 {
    let seed: UpScaleSeed8 = scale.to_dm(&mut device.tdma);
    relayout_seed8(device, seed)
}

/// ffn20 R3: exact 0.0 on the RMSNorm slices from the landed Up seed (bits AND 0, reduced): a pure ordering
/// dependency that lists the input RMSNorm chain after the seed DMA.
fn zero_from_up_seed(device: &mut Device, seed: &UpScaleSeed8) -> VrfTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]> {
    // F9B_hd probe: the zeros also go through the 8-shard inter-slice reduce (exact: a sum of +0.0), a statically
    // longer zero pass, so the head table DMA may be listed right after it and queue behind the rms-weight load
    // instead of waiting for the rms-weight copy.
    let t: DmTensorView<'_, f8e4m3, Chip, UpCluster, UpPairRows, m![L % 120, H % 480]> = unsafe { seed.view().reshape() };
    let z: DmTensor<f32, Chip, UpCluster, m![L / 1920 % 4, 1 # 8, Dummy8], m![1 # 8]> = device
        .main
        .begin(t.tile::<m![L % 120], 1, m![L % 120 = 1 # 120, H % 480]>(0))
        .fetch::<m![H / 240 % 2], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![L / 1920 % 4, 1 # 8, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let z: DmTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]> = unsafe { z.reshape() };
    device
        .sub
        .begin(z.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn zero_from_seed_acc(device: &mut Device, seed: &UpScaleSeed8, zin: &VrfTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]>) -> VrfTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]> {
    // F9B_hd probe: the zeros also go through the 8-shard inter-slice reduce (exact: a sum of +0.0), a statically
    // longer zero pass, so the head table DMA may be listed right after it and queue behind the rms-weight load
    // instead of waiting for the rms-weight copy.
    let t: DmTensorView<'_, f8e4m3, Chip, UpCluster, UpPairRows, m![L % 120, H % 480]> = unsafe { seed.view().reshape() };
    let z: DmTensor<f32, Chip, UpCluster, m![L / 1920 % 4, 1 # 8, Dummy8], m![1 # 8]> = device
        .main
        .begin(t.tile::<m![L % 120], 1, m![L % 120 = 1 # 120, H % 480]>(0))
        .fetch::<m![H / 240 % 2], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zin)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![L / 1920 % 4, 1 # 8, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let z: DmTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]> = unsafe { z.reshape() };
    device
        .sub
        .begin(z.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn relayout_seed8(device: &mut Device, seed: UpScaleSeed8) -> UpScale2 {
    // V14 FFN-OCT-b: 240-row octet seed (57.6 KB per leader on every 8th slice, each run a whole number of
    // 256 B units starting 256-aligned), relaid by two InterTransposes on Main that replace the q4 one:
    // A (ring-4, stride 1, 1,800 flits) swaps Dq4 <-> L % 120 / 30; B (ring-2, stride 4, on tile Dq4 = 0,
    // 450 flits) swaps D2 <-> L % 240 / 120. Pure data movement (bit-identical).
    let seed: DmTensor<f8e4m3, Chip, UpCluster, m![L / 240 % 32, D2, Dq4], m![L % 240 / 2, H / 8]> =
        unsafe { seed.reshape() };
    let stage_a: DmTensor<f8e4m3, Chip, UpCluster, m![L / 240 % 32, D2, L % 120 / 30], m![Dq4, L % 240 / 120, L % 30 / 2, H / 8]> = device.main
        .begin(seed.view())
        .fetch::<m![L % 240 / 120, L % 120 / 30, L % 30 / 2], m![H / 8]>()
        .switch::<m![L / 240 % 32, D2, L % 120 / 30], m![L % 240 / 120, L % 30 / 2, Dq4]>(
            SwitchConfig::InterTranspose { slice1: 4, slice0: 1, time0: 15 },
        )
        .collect::<m![L % 240 / 120, L % 30 / 2, Dq4, H / 8 / 32 % 15], m![H / 8 % 32]>()
        .commit_trim::<m![H / 8 % 32]>()
        .commit();
    let switched: DmTensor<f8e4m3, Chip, UpCluster, m![L / 120 % 64, L % 120 / 30], m![D2, L % 30 / 2, H / 8]> = device.main
        .begin(stage_a.view().tile::<m![Dq4], 1, m![Dq4 = 1 # 4, L % 240 / 120, L % 30 / 2, H / 8]>(0))
        .fetch::<m![L % 240 / 120, L % 30 / 2], m![H / 8]>()
        .switch::<m![L / 120 % 64, L % 120 / 30], m![L % 30 / 2, D2]>(
            SwitchConfig::InterTranspose { slice1: 2, slice0: 4, time0: 15 },
        )
        .collect::<m![L % 30 / 2, D2, H / 8 / 32 % 15], m![H / 8 % 32]>()
        .commit_trim::<m![H / 8 % 32]>()
        .commit();
    unsafe { switched.reshape() }
}

// X3 variant A: up_gate_scale_pass4 removed (block scales applied by the contraction engine).

/// Block-16 partial dots for the slice's 30 rows (30 x 240 x 4 B = 28.8 KiB).
type UpBlocks = DmTensor<bf16, Chip, UpCluster, UpRows, m![L % 30, H / 16 % 240]>;

/// f32 result: a 30 x bf16 = 60 B in-slice extent is rejected (SRAM access width is 8 B).
type UpResult = DmTensor<f32, Chip, UpCluster, UpRows, m![L % 30]>;

/// Decode FP4 once to FP8, then stream it twice against scaled activation hi/lo.
/// Keep each block-16 dot separate until its own block scale has been applied.
/// Each tile owns a separate block buffer to avoid mutable-view owner conflicts in
/// this compiler. All input weights retain the original external layout.
macro_rules! up_gate_tile_fns {
    ($load:ident, $tile:ident, $rows:literal) => {
        fn $load(
            device: &mut Device,
            weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
            r0: usize,
        ) -> DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![L % 30 = $rows, H]> {
            weight
                .view()
                .tile::<m![L % 30], $rows, m![L / 30, L % 30 = $rows # 30, H]>(r0)
                .to_dm(&mut device.tdma)
        }

        fn $tile(
            device: &mut Device,
            x: &UpActivation,
            packed: DmTensor<f4e2m1, Chip, UpCluster, UpRows, m![L % 30 = $rows, H]>,
            blocks: &mut UpBlocks,
            r0: usize,
        ) {
            // api081 P1: FP4 -> FP8 lookup fused into the contraction pass; Gs (hi/lo) is a
            // stream-adapter time broadcast, so each weight packet is fetched and decoded once.
            device.main
                .begin(packed.view())
                .fetch::<m![L % 30 = $rows], m![H]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![L % 30 = $rows, H / 64 % 60, H / 32 % 2], m![H % 32]>()
                .contract_outer::<m![L % 30 = $rows, H / 64 % 60, Gs], m![H % 64], _, _, _>(x)
                .contract_packet::<m![H / 16 % 4]>()
                .contract_time::<m![L % 30 = $rows, H / 64 % 60]>()
                .contract_lane::<m![L % 30 = $rows, H / 64 % 60], m![H / 16 % 4 # 8]>(LaneMode::Sequential)
                // X3 variant A: block sums leave the pass as bf16 (the scale contraction's operand type).
                .cast::<bf16, m![H / 16 % 4 # 16]>()
                .commit_trim::<m![H / 16 % 4]>()
                .commit_view(blocks.view_mut().tile::<m![L % 30], $rows, m![L % 30 = $rows #{!} 30, H / 16 % 240]>(r0));
        }
    };
}
up_gate_tile_fns!(load_up_gate_tile6, up_gate_tile6, 6);
up_gate_tile_fns!(load_up_gate_tile12, up_gate_tile12, 12);
up_gate_tile_fns!(load_up_gate_tile18, up_gate_tile18, 18);
up_gate_tile_fns!(load_up_gate_tile24, up_gate_tile24, 24);
up_gate_tile_fns!(load_up_gate_tile30, up_gate_tile30, 30);

/// Block scales applied and summed over K for rows `r0..r0+6`. The VRF holds at most 8 KiB =
/// 2,048 f32 block scales (8.5 rows x 240), so tiles are finished in 6-row passes; each pass
/// writes whole 8 B row pairs.
macro_rules! up_gate_scale_pass_fn {
    ($name:ident, $g:literal) => {
fn $name(device: &mut Device, blocks: &UpBlocks, scale: &UpScale, result: &mut UpResult, r0: usize) {
    // Written from Main through the vector engine: a Sub `to_vrf` that reads the directly loaded
    // scale would be scheduled eagerly after the scale DMA and hold the next weight DMA's issue.
    // 1/UP_ACTIVATION_SCALE (a power of two, exact) is folded into the operand.
    let scale_vrf: VrfTensor<f32, Chip, UpCluster, UpRows, m![L % 30 = $g, H / 16]> = device
        .main
        .begin(scale.view().tile::<m![L % 30], $g, m![L % 30 = $g # 30, H / 16]>(r0))
        .fetch::<m![L % 30 = $g], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30 = $g, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32 / UP_ACTIVATION_SCALE)
        .vector_widen_concat::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(blocks.view().tile::<m![L % 30], $g, m![L % 30 = $g # 30, H / 16 % 240]>(r0))
        .fetch::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .collect::<m![L % 30 = $g, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 30 = $g, H / 16 / 4 % 60], m![H / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = $g], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 = $g / 2], m![L % 30 = $g % 2 # 8]>()
        .commit_trim::<m![L % 30 = $g % 2]>()
        .commit_view(result.view_mut().tile::<m![L % 30], $g, m![L % 30 = $g #{!} 30]>(r0));
}
    };
}
// X3 variant A: the VE scale passes are no longer instantiated.

/// X3 variant A: block scales applied by the contraction engine. The f8 scales become bf16
/// (exact) on Sub through the fetch adapter and the cast engine, sit in the TRF, and one Main
/// contraction per matrix multiplies each row's 240 block sums by its 240 scales and sums them in
/// the time reducer. Replaces the Up scale passes (5 Main + 5 Sub) and the Gate scale passes (8 Main).
/// 1/UP_ACTIVATION_SCALE moves into the GeGLU global-scale operands.
type UpScaleBf16 = DmTensor<bf16, Chip, UpCluster, UpRows, m![L % 30, H / 16 % 240]>;
type UpScaleTrf = TrfTensor<bf16, Chip, UpCluster, UpRows, m![1], m![L % 30, H / 16 % 240]>;

fn gate_scale_to_bf16(device: &mut Device, scale: &UpScale) -> UpScaleBf16 {
    // V15-B mce: on MAIN (32 B/cycle fetch). The Sub version waits for the Vector Engine the Gate contraction holds and
    // then runs at the Sub rate (3.9k HW, cc1ct) on the Gate chain; the Main pass is the up_scale4_to_bf16 form (2.1k).
    device
        .main
        .begin(scale.view())
        .fetch::<m![L % 30], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>()
        .commit_trim::<m![H / 16 % 8]>()
        .commit()
}

fn up_scale4_to_bf16(device: &mut Device, scale: &UpScale2) -> UpScaleBf16 {
    // One Main pass over tile D2 = 0 (a Sub pass on this source hits a lowering ICE on 0.8 toolchain).
    device
        .main
        .begin(scale.view().tile::<m![D2], 1, m![D2 = 1 # 2, L % 30, H / 16]>(0))
        .fetch::<m![L % 30], m![H / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30, H / 16 / 8 % 30], m![H / 16 % 8]>()
        .cast::<bf16, m![H / 16 % 8 # 16]>()
        .commit_trim::<m![H / 16 % 8]>()
        .commit()
}

fn apply_block_scales_ce(device: &mut Device, blocks: &UpBlocks, scale: &UpScaleBf16) -> UpResult {
    // V15-B mce: the block-scale TRF staging on MAIN (32 B/cycle; the Sub StoTrf took 3.9k HW on the Gate chain).
    let scale_trf: UpScaleTrf = device
        .main
        .begin(scale.view())
        .fetch::<m![L % 30, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .collect::<m![L % 30, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .to_trf();
    device
        .main
        .begin(blocks.view())
        .fetch::<m![L % 30, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .collect::<m![L % 30, H / 16 / 16 % 15], m![H / 16 % 16]>()
        .contract_outer::<m![L % 30, H / 16 / 16 % 15], m![H / 16 % 16], _, _, _>(&scale_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 30]>()
        .contract_lane::<m![L % 30], m![1 # 8]>(LaneMode::Sequential)
        .transpose::<m![L % 30 / 2], m![L % 30 % 2 # 8]>()
        .commit_trim::<m![L % 30 % 2]>()
        .commit()
}

fn project_up_and_gate(
    device: &mut Device,
    x: &NormalizedX,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_seed: UpScaleSeed8,
    gate_seed: UpScaleSeed8,
) -> (UpResult, UpResult) {
    // Normalize and split only eight 480-channel shards per cluster, then broadcast
    // [hi, lo] to every row slice while independent weight DMA runs.
    // Both FP8 components together occupy the same logical bytes as the BF16 input.
    let x_half = split_up_activation(device, x);
    let x_full: DmTensor<f8e4m3, Chip, UpCluster, m![L / 1920 % 4, HeadPad, Dummy8], m![1 # 8, Gs, H / 32 % 15, H / 480, H % 32]> = device
        .main
        .begin(x_half.view())
        .fetch::<m![Gs, H / 32 % 15], m![H % 32]>()
        .switch::<m![L / 1920 % 4, HeadPad, Dummy8], m![1 # 8, Gs, H / 32 % 15, H / 480]>(
            SwitchConfig::Broadcast01 { slice1: 8, slice0: 8, time0: 30 },
        )
        .collect::<m![1 # 8, Gs, H / 32 % 15, H / 480], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();
    // All 256 slices of each cluster now hold identical [hi, lo] full-H vectors (the padding slices'
    // packets land in the `1 # 8` element padding, which the TRF load skips).
    let x: DmTensor<f8e4m3, Chip, UpCluster, UpRows, m![1 # 8, Gs, H / 32 % 15, H / 480, H % 32]> = unsafe { x_full.reshape() };
    let x_trf: UpActivation = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, H / 32], m![H % 32]>()
        .collect::<m![Gs, H / 32], m![H % 32]>()
        .to_trf();
    let up = project_up_gate_small_first(device, &x_trf, up_weight_packed, up_seed);
    let gate = project_up_gate_small_last(device, &x_trf, gate_weight_packed, gate_seed);
    (up, gate)
}

/// Up: one full 30-row tile; decoded once, with scales consumed in six-row passes.
fn project_up_gate_small_first(
    device: &mut Device,
    x: &UpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    seed: UpScaleSeed8,
) -> UpResult {
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = relayout_seed8(device, seed);
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    let scale = up_scale4_to_bf16(device, &scale);
    apply_block_scales_ce(device, &blocks0, &scale)
}

/// Gate: one full 30-row tile, reducing LUT loads and DMA command startup.
fn project_up_gate_small_last(
    device: &mut Device,
    x: &UpActivation,
    weight: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    seed: UpScaleSeed8,
) -> UpResult {
    // fgs: the Gate seed DMA is issued at the head (feedforward); only its relayout stays here.
    let w0 = load_up_gate_tile30(device, weight, 0);
    let scale = relayout_seed8(device, seed);
    let mut blocks0: UpBlocks = DmTensor::new();
    up_gate_tile30(device, x, w0, &mut blocks0, 0);
    let scale = up_scale4_to_bf16(device, &scale);
    apply_block_scales_ce(device, &blocks0, &scale)
}


/// The 7,680 Up (or Gate) values of a cluster on 64 of its slices, 120 per slice: the same rows
/// the contraction produced, regrouped in fours. 120 elements is what makes every packet whole
/// -- 8 f32 is exactly one 32 B flit, and 120 bf16 is 240 B -- where the contraction's own 30
/// rows are neither (30 bf16 = 60 B is not a multiple of the 8 B SRAM access width, and 30 f32
/// is 3.75 flits). This is a DM-to-DM permutation inside the cluster, not a gather: no HBM.
// CC-1: the regroup's broadcast axis is REAL (Dq4): all 4 neighbouring slices hold the 120-value group, so the
// GeGLU chain runs on every slice (same per-slice time) and the ring-64 h gather below can keep Dq4 as its slice0.
type UpPairedRows = m![L / 120 % 64, Dq4];
type UpPaired = DmTensor<f32, Chip, UpCluster, UpPairedRows, m![L % 120]>;

/// GeGLU where Up and Gate were computed. Both results sit in the same distributed layout, so
/// the activation function is slice-local and only the finished 15 KiB per cluster leaves for
/// HBM (where the Down contraction reads it back replicated).
///
/// Replaces, per matrix, a gather to one slice + an HBM store + a reload into the single-cluster
/// 60-rows-per-slice layout + a cast pass, and then a pairing Switch and two global-scale passes:
/// 7.9k of DMA and eight Main passes, for 2.2k of DMA and four passes. The intermediate bf16
/// roundings of `up * scale` and `gate * scale` are gone (the values stay f32 until the GeGLU
/// output), which moves the result by far less than the block-scale error already present.
/// Four adjacent slices' 30 rows onto the first of them (120 values) with a ring-4 Switch on
/// Main, instead of a DM-to-DM DMA that queues behind the Down weight load already streaming.
fn gather_up_gate(device: &mut Device, x: UpResult) -> UpPaired {
    device.main
        .begin(x.view())
        .fetch::<m![L / 2 % 15], m![L % 2 # 8]>()
        .switch::<UpPairedRows, m![L / 2 % 15, L / 30 % 4]>(
            SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 },
        )
        .collect::<m![L / 2 % 15, L / 30 % 4], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

/// ffnB13 F-4: the global scale lands on 16 slices per cluster (one per 16-slice group); a named Broadcast1{4, 4} fans it out to
/// the 64 paired-row slices on Main (no ring config DMA), instead of a 64-slice replicated DMA per cluster.
/// The fan-out's padded time slots land in the element padding, which the VRF pass's fetch skips.
type GsSrcRows = m![L / 480 % 16, 1 # 16];
// CC-1: ring-16 fan-out to ALL 16 slices of each group (the GeGLU chain now runs on all 256 slices).
axes![Xg64 = 64, Xg16 = 16];
type GsFanRows = m![L / 480 % 16, Xg16];
fn global_scale_fanout(
    device: &mut Device,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<f32, Chip, UpCluster, UpPairedRows, m![1 # 16, 1 # 8]> {
    let src: DmTensor<f32, Chip, UpCluster, GsSrcRows, m![1 # 8]> = global_scale.to_dm(&mut device.tdma);
    let fan: DmTensor<f32, Chip, UpCluster, GsFanRows, m![1 # 16, 1 # 8]> = device
        .main
        .begin(src.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<GsFanRows, m![1 # 16]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![1 # 16], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    unsafe { fan.reshape() }
}

fn geglu_local(
    device: &mut Device,
    up: UpResult,
    gate: UpResult,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![Gs, L % 7680]> {
    let up = gather_up_gate(device, up);
    let gate = gather_up_gate(device, gate);

    let _ = up_global_scale; // F9D usfg: folded into the down global-scale VRF in the epilogue
    let gate_scale = global_scale_fanout(device, gate_global_scale);
    let gate_scale_vrf: VrfTensor<f32, Chip, UpCluster, UpPairedRows, m![1 # 8]> = device
        .main
        .begin(gate_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1f32 / UP_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);

    let gate_scaled: UpPaired = device
        .main
        .begin(gate.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    // g * (1 + erf(g / sqrt(2))); GeGLU's factor 1/2 is folded into the product below.
    let gelu: UpPaired = device
        .sub
        .begin(gate_scaled.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    let gelu_vrf: VrfTensor<f32, Chip, UpCluster, UpPairedRows, m![L % 120]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();
    let out: DmTensor<bf16, Chip, UpCluster, UpPairedRows, m![L % 120]> = device
        .main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_FOLD)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();
    // Split GeGLU locally into [hi, lo] (every slice holds its 120-value group).
    let out = split_down_activation(device, &out);
    // CC-1: h never leaves the cluster. Ring of 64 at stride 4 (Broadcast1{64, 4}, 24 B packets): every slice of
    // the cluster receives all 64 groups; the commit puts them in natural L order. No DMA, no HBM, no sync.
    device
        .main
        .begin(out.view())
        .fetch::<m![Gs, L % 120 / 24], m![L % 24]>()
        .switch::<DownRows, m![Gs, L % 120 / 24, L / 120 % 64]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![Gs, L % 120 / 24, L / 120 % 64], m![L % 24 # 32]>()
        .commit_trim::<m![L % 24]>()
        .commit()
}

pub(crate) fn feedforward(
    device: &mut Device,
    x: NormalizedInput,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    // ffn20 R3: the Up seed DMA first; the input RMSNorm chain is gated on an exact 0 from the landed seed.
    let (x_raw, weight_dm) = x;
    let up_seed: UpScaleSeed8 = up_weight_scale.to_dm(&mut device.tdma);
    let zero_vrf = zero_from_up_seed(device, &up_seed);
    // fgs p68: the Main copy of the rms weight adds zG (AND-0 of the Gate seed).
    let gate_seed: UpScaleSeed8 = gate_weight_scale.to_dm(&mut device.tdma);
    let zg = zero_from_up_seed(device, &gate_seed);
    let weight_dm = pin_weight_copy(device, &weight_dm, &zg);
    let x = normalize_finish(device, &x_raw, &weight_dm, &zero_vrf);
    let (up, gate) = project_up_and_gate(
        device,
        &x,
        up_weight_packed,
        gate_weight_packed,
        up_seed,
        gate_seed,
    );
    let x = geglu_local(device, up, gate, up_global_scale, gate_global_scale);
    // The epilogue's small operands, issued here so they sit on the DMA queue long before the
    // tail needs them.
    let residual: DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> = residual.to_dm(&mut device.tdma);
    let norm_weight: DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> = post_ff_rms_weight.to_dm(&mut device.tdma);
    let global_scale: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = down_global_scale.to_dm(&mut device.tdma);
    let layer_scalar: DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![1 # 8]> = layer_scalar.to_dm(&mut device.tdma);
    // F9D usfg2: su lands on BOTH clusters (UpCluster broadcast, like the old GeGLU gs source), so cluster 1's padding
    // epilogue multiplies its garbage by a finite su * 2^13 instead of stale DM (the usf trace 831_90 showed c1's ms pass
    // 1.48-1.58k vs 0.97k with a stale su). Cluster 0's data and every value it computes are unchanged vs usfg.
    let up_gs: DmTensor<f32, Chip, UpCluster, EpilogueSlices, m![1 # 8]> = up_global_scale.to_dm(&mut device.tdma);
    let up_gs: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = unsafe { up_gs.reshape() };
    let y = project_down_i16(device, &x, down_weight_packed, down_weight_scale);
    down_epilogue(device, y, &global_scale, &norm_weight, &residual, &layer_scalar, &up_gs)
}

// CC-1: Down is split across the clusters by L (the Up/Gate ownership), so the GeGLU output never leaves its
// cluster. Each cluster contracts its own 7,680 columns for all 3,840 rows; only f32 partial sums cross clusters.
type DownCluster = m![L / 7680];
// Down row h = 60 g + 4 r + q  (g = H / 60, r = H % 60 / 4 in 0..15, q = H % 4).
axes![KhG = 64, KhR = 15, KhQ = 4];
/// 256 slices per cluster: 64 groups of 60 rows x 4 interleaved rows (`H % 4`), 15 rows per slice.
///
/// A slice's packed weights are still 15 pieces of 3,840 B (the cluster's half of a 7,680 B Down row),
/// whole 256 B units on 256 B boundaries; the two clusters now read the two halves of the same rows.
/// The compute side names the slice axes as the broadcast axes of the h gather (`Xg64`, `Dq4`), so the
/// gathered activation's TRF and the weight tiles share one slice mapping; the result is relabelled after.
type DownRows = m![Xg64, Dq4];
type DownRowsH = m![H / 60 % 64, H % 4];
type DownActivation = TrfTensor<f8e4m3, Chip, DownCluster, DownRows, m![1], m![Gs, L % 7680]>;
type DownResult = DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4, 1 # 2]>;
/// Partial sums regrouped in the cluster: 240 rows on the first slice of each 16-slice ring.
type DownPartialRows = m![H / 240 % 16, 1 # 16];

macro_rules! down_tile_fns {
    ($load:ident, $tile:ident, $rows:literal) => {
        fn $load(device: &mut Device, weight: &HbmTensor<f4e2m1, Chip, m![H, L]>, r0: usize)
        -> DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]> {
            // Relabel H as (KhG, KhR, KhQ) so the row window sits on an axis the view names directly
            // (0.8 toolchain rejects a non-divisor window on an inner digit of H).
            let v: HbmTensorView<'_, f4e2m1, Chip, m![KhG, KhR, KhQ, L]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f4e2m1, Chip, m![L / 7680], m![KhG, KhQ], m![KhR = $rows, L % 7680]> =
                v.tile::<m![KhR], $rows, m![KhG, KhR = $rows # 15, KhQ, L]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }
        fn $tile(device: &mut Device,
            packed: DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]>)
            -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]> {
            device.main.begin(packed.view())
                .fetch::<m![H % 60 / 4 = $rows, L / 3840 % 2], m![L % 3840]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 60 / 4 = $rows, L / 32 % 240], m![L % 32]>()
                .commit_trim::<m![L % 32]>().commit()
        }
    };
}
down_tile_fns!(load_down_tile4, decode_down_tile4_unused, 4);
down_tile_fns!(load_down_tile11, decode_down_tile11, 11);

macro_rules! down_scale_fn {
    ($name:ident, $rows:literal) => {
        fn $name(
            device: &mut Device,
            scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
            r0: usize,
        ) -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L / 16 % 480]> {
            let v: HbmTensorView<'_, f8e4m3, Chip, m![KhG, KhR, KhQ, L / 16]> = unsafe { scale.view().reshape() };
            let d: DmTensor<f8e4m3, Chip, m![L / 7680], m![KhG, KhQ], m![KhR = $rows, L / 16 % 480]> =
                v.tile::<m![KhR], $rows, m![KhG, KhR = $rows # 15, KhQ, L / 16]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }
    };
}
down_scale_fn!(load_down_scale4, 4);
down_scale_fn!(load_down_scale3, 3);
down_scale_fn!(load_down_scale8, 8);
down_scale_fn!(load_down_scale15, 15);

macro_rules! down_stream_group_fn {
    ($name:ident, $tile_rows:literal, $group:literal) => {
        fn $name(
            device: &mut Device,
            x: &DownActivation,
            fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $tile_rows, L % 7680]>,
            tile_offset: usize,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]>,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view())
                .fetch::<m![H % 60 / 4 = $group, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $group, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(fp8.view().tile::<m![H % 60 / 4 = $tile_rows], $group, m![H % 60 / 4 = $group # $tile_rows, L % 7680]>(tile_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .collect::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 / 4 = $group, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 / 4 = $group, L / 64 % 120]>()
                .contract_lane::<m![H % 60 / 4 = $group, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $group], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                // One row per 8 B SRAM write unit: [value, pad].
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $group, m![H % 60 / 4 = $group #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
macro_rules! down_stream_group_s7_fn {
    ($name:ident, $tile_rows:literal, $group:literal) => {
        fn $name(
            device: &mut Device,
            x: &DownActivation,
            fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $tile_rows, L % 7680]>,
            tile_offset: usize,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 8, L / 16 % 480]>,
            scale_offset: usize,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view().tile::<m![H % 60 / 4 = 8], $group, m![H % 60 / 4 = $group # 8, L / 16 % 480]>(scale_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $group, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(fp8.view().tile::<m![H % 60 / 4 = $tile_rows], $group, m![H % 60 / 4 = $group # $tile_rows, L % 7680]>(tile_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .collect::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 / 4 = $group, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 / 4 = $group, L / 64 % 120]>()
                .contract_lane::<m![H % 60 / 4 = $group, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $group], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                // One row per 8 B SRAM write unit: [value, pad].
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $group, m![H % 60 / 4 = $group #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
down_stream_group_s7_fn!(down_stream_group3_t11_s7, 11, 3);
down_stream_group_fn!(down_stream_group3_t11, 11, 3);

/// Group of 4 rows whose block scales are a 4-row tile of an 8-row scale load (rows 0..7 in one DMA; fp8 tile of the 11-row decode).
fn down_stream_group4_s8_t11(
    device: &mut Device,
    x: &DownActivation,
    fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 11, L % 7680]>,
    tile_offset: usize,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 8, L / 16 % 480]>,
    scale_offset: usize,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 8], 4, m![H % 60 / 4 = 4 # 8, L / 16 % 480]>(scale_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(fp8.view().tile::<m![H % 60 / 4 = 11], 4, m![H % 60 / 4 = 4 # 11, L % 7680]>(tile_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

/// Group of 4 rows whose block scales are a 4-row tile of an 8-row scale load (rows 0..7 in one DMA; fp8 tile of the 11-row decode).
fn down_stream_group4_s15_t11(
    device: &mut Device,
    x: &DownActivation,
    fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 11, L % 7680]>,
    tile_offset: usize,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
    scale_offset: usize,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 15], 4, m![H % 60 / 4 = 4 # 15, L / 16 % 480]>(scale_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(fp8.view().tile::<m![H % 60 / 4 = 11], 4, m![H % 60 / 4 = 4 # 11, L % 7680]>(tile_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

/// V15-B k11: the last 4-row tile (rows 11..14): FP4 -> FP8 lookup fused into the group's contraction (no decode pass, no FP8
macro_rules! down_stream_group_s15_fn {
    ($name:ident, $tile_rows:literal, $group:literal) => {
        fn $name(
            device: &mut Device,
            x: &DownActivation,
            fp8: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $tile_rows, L % 7680]>,
            tile_offset: usize,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
            scale_offset: usize,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $group, L / 16 % 480]> = device
                .sub
                .begin(scale.view().tile::<m![H % 60 / 4 = 15], $group, m![H % 60 / 4 = $group # 15, L / 16 % 480]>(scale_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $group, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(fp8.view().tile::<m![H % 60 / 4 = $tile_rows], $group, m![H % 60 / 4 = $group # $tile_rows, L % 7680]>(tile_offset))
                .fetch::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .collect::<m![H % 60 / 4 = $group, L / 64 % 120, L / 32 % 2], m![L % 32]>()
                .contract_outer::<m![H % 60 / 4 = $group, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
                .contract_packet::<m![L / 16 % 4]>()
                .contract_time::<m![H % 60 / 4 = $group, L / 64 % 120]>()
                .contract_lane::<m![H % 60 / 4 = $group, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $group], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                // One row per 8 B SRAM write unit: [value, pad].
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $group, m![H % 60 / 4 = $group #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
down_stream_group_s15_fn!(down_stream_group3_t11_s15, 11, 3);

/// V15-B k11: the last 4-row tile (rows 11..14): FP4 -> FP8 lookup fused into the group's contraction (no decode pass, no FP8
/// buffer); x hi/lo is a stream-adapter time broadcast as in Up/Gate.
fn down_fused_group4(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]>,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 4], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

fn down_fused_group4_s15(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]>,
    scale_offset: usize,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 15], 4, m![H % 60 / 4 = 4 # 15, L / 16 % 480]>(scale_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 4], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

fn down_fused_group4_s7(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L % 7680]>,
    scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 8, L / 16 % 480]>,
    scale_offset: usize,
    result: &mut DownResult,
    result_offset: usize,
) {
    let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]> = device
        .sub
        .begin(scale.view().tile::<m![H % 60 / 4 = 8], 4, m![H % 60 / 4 = 4 # 8, L / 16 % 480]>(scale_offset))
        .fetch::<m![H % 60 / 4 = 4, L / 16 / 240 % 2], m![L / 16 % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 / 4 = 4, L / 16 / 8 % 60], m![L / 16 % 8]>()
        .to_vrf();
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 4], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<L, m![H % 60 / 4 = 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_view(result.view_mut().tile::<m![H % 60 / 4], 4, m![H % 60 / 4 = 4 #{!} 15, 1 # 2]>(result_offset));
}

// V18 ND: piece A = fused FP4 lookup contraction, f32 block-16 sums (no VE); scale passes on the VE afterwards.
type DownBlocksA11 = DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 11, L / 16 % 480]>;
type DownBlocksB4 = DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L / 16 % 480]>;

fn down_blocks_a11(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 11, L % 7680]>,
) -> DownBlocksA11 {
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 11], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 11, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 11, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 11, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 11, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![L / 16 % 4]>()
        .commit()
}

fn down_blocks_b4(
    device: &mut Device,
    x: &DownActivation,
    packed: &DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = 4, L % 7680]>,
) -> DownBlocksB4 {
    device.main
        .begin(packed.view())
        .fetch::<m![H % 60 / 4 = 4], m![L % 7680]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 / 4 = 4, L / 64 % 120, L / 32 % 2], m![L % 32]>()
        .contract_outer::<m![H % 60 / 4 = 4, L / 64 % 120, Gs], m![L % 64], _, _, _>(x)
        .contract_packet::<m![L / 16 % 4]>()
        .contract_time::<m![H % 60 / 4 = 4, L / 64 % 120]>()
        .contract_lane::<m![H % 60 / 4 = 4, L / 64 % 120], m![L / 16 % 4 # 8]>(LaneMode::Sequential)
        .commit_trim::<m![L / 16 % 4]>()
        .commit()
}

/// VE scale pass: g rows of f32 block sums (tile of a T-row block-sum tensor) times their f32 block scales (tile of an
/// S-row scale load, via the VRF), reduced over L, / DOWN_ACTIVATION_SCALE, into DownResult rows.
macro_rules! down_scale_pass_fn {
    ($name:ident, $t:literal, $sr:literal, $g:literal) => {
        fn $name(
            device: &mut Device,
            blocks: &DmTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $t, L / 16 % 480]>,
            tile_offset: usize,
            scale: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = $sr, L / 16 % 480]>,
            scale_offset: usize,
            result: &mut DownResult,
            result_offset: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, DownCluster, DownRows, m![H % 60 / 4 = $g, L / 16 % 480]> = device
                .sub
                .begin(scale.view().tile::<m![H % 60 / 4 = $sr], $g, m![H % 60 / 4 = $g # $sr, L / 16 % 480]>(scale_offset))
                .fetch::<m![H % 60 / 4 = $g, L / 16 / 240 % 2], m![L / 16 % 240]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 / 4 = $g, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .to_vrf();
            device.main
                .begin(blocks.view().tile::<m![H % 60 / 4 = $t], $g, m![H % 60 / 4 = $g # $t, L / 16 % 480]>(tile_offset))
                .fetch::<m![H % 60 / 4 = $g, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .collect::<m![H % 60 / 4 = $g, L / 16 / 8 % 60], m![L / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 60 / 4 = $g, L / 16 / 4 % 120], m![L / 16 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_intra_slice_reduce::<L, m![H % 60 / 4 = $g], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DOWN_ACTIVATION_SCALE)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 2]>()
                .commit_view(result.view_mut().tile::<m![H % 60 / 4], $g, m![H % 60 / 4 = $g #{!} 15, 1 # 2]>(result_offset));
        }
    };
}
down_scale_pass_fn!(down_scale_pass4_t11_s15, 11, 15, 4);
down_scale_pass_fn!(down_scale_pass3_t11_s15, 11, 15, 3);
down_scale_pass_fn!(down_scale_pass4_t11_s11, 11, 11, 4);
down_scale_pass_fn!(down_scale_pass3_t11_s11, 11, 11, 3);
down_scale_pass_fn!(down_scale_pass4_t4_s4, 4, 4, 4);
down_scale_fn!(load_down_scale11, 11);

// F72 fsi16: Down row h = 240 G + 16 r + Q: slice s = 16 G + Q (relabelled as the compute's Xg64 x Dq4, same wire order)
// holds rows 240 G + Q + 16 r, r = 0..14. At every row step the 16 adjacent slices of a ring read 16 ADJACENT rows
// (the base reads 4), so the HBM footprint of the Down weight (3,840 B pieces) and block-scale (480 B pieces) loads is
// 16 consecutive rows wide; piece shapes, per-slice work and every compute pass are unchanged. The ring of 16 that
// regroups the partials is exactly the 16 slices of epilogue slice G, in natural H % 240 = 16 r + Q order.
axes![KiG = 16, KiR = 15, KiQ = 16];
/// fsi16: the true H digits of the Down slice axis (s = 16 G + Q); regroups the compute's (Xg64, Dq4).
type DownRowsI16 = m![H / 240 % 16, H % 16];
macro_rules! down_tile_i16_fns {
    ($load:ident, $rows:literal) => {
        fn $load(device: &mut Device, weight: &HbmTensor<f4e2m1, Chip, m![H, L]>, r0: usize)
        -> DmTensor<f4e2m1, Chip, DownCluster, DownRows, m![H % 60 / 4 = $rows, L % 7680]> {
            let v: HbmTensorView<'_, f4e2m1, Chip, m![KiG, KiR, KiQ, L]> = unsafe { weight.view().reshape() };
            let d: DmTensor<f4e2m1, Chip, m![L / 7680], m![KiG, KiQ], m![KiR = $rows, L % 7680]> =
                v.tile::<m![KiR], $rows, m![KiG, KiR = $rows # 15, KiQ, L]>(r0).to_dm(&mut device.tdma);
            unsafe { d.reshape() }
        }
    };
}
down_tile_i16_fns!(load_down_i16_tile11, 11);
down_tile_i16_fns!(load_down_i16_tile4, 4);

fn load_down_i16_scale15(
    device: &mut Device,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![H % 60 / 4 = 15, L / 16 % 480]> {
    let v: HbmTensorView<'_, f8e4m3, Chip, m![KiG, KiR, KiQ, L / 16]> = unsafe { scale.view().reshape() };
    let d: DmTensor<f8e4m3, Chip, m![L / 7680], m![KiG, KiQ], m![KiR = 15, L / 16 % 480]> =
        v.tile::<m![KiR], 15, m![KiG, KiR = 15 # 15, KiQ, L / 16]>(0).to_dm(&mut device.tdma);
    unsafe { d.reshape() }
}

fn project_down_i16(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![Gs, L % 7680]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    let w0 = load_down_i16_tile11(device, down_weight_packed, 0);
    let x: DownActivation = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, L / 32 % 240], m![L % 32]>()
        .collect::<m![Gs, L / 32 % 240], m![L % 32]>()
        .to_trf();
    let scale15 = load_down_i16_scale15(device, down_weight_scale);
    let w1 = load_down_i16_tile4(device, down_weight_packed, 11);
    let blocks_a = down_blocks_a11(device, &x, &w0);
    let mut result: DownResult = DmTensor::new();
    down_scale_pass4_t11_s15(device, &blocks_a, 0, &scale15, 0, &mut result, 0);
    down_scale_pass4_t11_s15(device, &blocks_a, 4, &scale15, 4, &mut result, 4);
    down_scale_pass3_t11_s15(device, &blocks_a, 8, &scale15, 8, &mut result, 8);
    down_fused_group4_s15(device, &x, &w1, &scale15, 11, &mut result, 11);
    // True digits: slice (G, Q) = (H / 240 % 16, H % 16), local row r = H % 240 / 16.
    let result: DmTensor<f32, Chip, DownCluster, DownRowsI16, m![H % 240 / 16, 1 # 2]> = unsafe { result.reshape() };
    // One pass per cluster: the 16 slices of epilogue slice G meet on a ring of 16; stream time (r, Q) is natural
    // H % 240 order and H % 2 = Q % 2, so row pairs are committed as whole 8 B units (as the base).
    let part: DmTensor<f32, Chip, DownCluster, DownPartialRows, m![H % 240]> = device
        .main
        .begin(result.view())
        .fetch::<m![H % 240 / 16], m![1 # 2]>()
        .switch::<DownPartialRows, m![H % 240 / 16, H % 16]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H % 240 / 16, H % 16], m![1 # 8]>()
        .transpose::<m![H % 240 / 16, H % 16 / 2], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();
    let parts: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![L / 7680, H % 240]> = part.to_dm(&mut device.tdma);
    device
        .main
        .begin(parts.view())
        .fetch::<m![H % 240 / 4, L / 7680], m![H % 4]>()
        .collect::<m![H % 240 / 4, L / 7680], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        .vector_intra_slice_reduce::<L, m![H % 240 / 4], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![H % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![H % 4]>()
        .commit()
}

/// CC-1 Down projection: each cluster contracts its own L half (the gathered h, already in DM on every slice)
/// for all 3,840 rows; the two clusters' f32 partials meet on cluster 0's epilogue slices and are added there.
fn project_down(
    device: &mut Device,
    x: &DmTensor<f8e4m3, Chip, DownCluster, DownRows, m![Gs, L % 7680]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    // Weight tiles do not depend on GeGLU, so the queue streams them behind Up/Gate.
    let w0 = load_down_tile11(device, down_weight_packed, 0);
    // CC-1c: the gathered activation into the TRF on Sub through a ring-1 CustomBroadcast (identity routing). Lab 94091 trace: a plain
    // fetch/collect/to_trf is a StoTrf command that holds the in-order issue until it completes (it delayed decode12
    // and the Down scale DMAs by 2.7-3.0k); a Sub pass with a switch runs as a TuExec beside Main's decode12.
    // V15-B mtrf: the gathered activation into the TRF on MAIN (32 B/cycle fetch; no CustomBroadcast config DMA after
    // W12, and Sub stays free for the s0 VRF load and the fused3 table StoTab during decode12).
    let x: DownActivation = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, L / 32 % 240], m![L % 32]>()
        .collect::<m![Gs, L / 32 % 240], m![L % 32]>()
        .to_trf();
    // V15-B k11: Down rows split 11 + 4 (was 12 + 3). The first weight piece lands ~3.3k earlier (one row of lane moves
    // behind it), decode11 is 1.35k shorter, the fused lookup group takes 4 rows (VRF 4 x 480 x 4 B = 7.7 KB fits).
    // V15-B k15: every Down block scale in ONE 15-row load (1.84 MB): one fixed command less than k11mmce8, no re-read,
    // and the fused table (always listed after the last scale load) lands ~1.5k earlier.
    // V18 nd11: no decode pass. Piece A: one fused lookup contraction (f32 block sums, no VE) + three VE scale passes.
    let scale15 = load_down_scale15(device, down_weight_scale, 0);
    let w1 = load_down_tile4(device, down_weight_packed, 11);
    let blocks_a = down_blocks_a11(device, &x, &w0);
    let mut result: DownResult = DmTensor::new();
    down_scale_pass4_t11_s15(device, &blocks_a, 0, &scale15, 0, &mut result, 0);
    down_scale_pass4_t11_s15(device, &blocks_a, 4, &scale15, 4, &mut result, 4);
    down_scale_pass3_t11_s15(device, &blocks_a, 8, &scale15, 8, &mut result, 8);
    down_fused_group4_s15(device, &x, &w1, &scale15, 11, &mut result, 11);
    let result: DmTensor<f32, Chip, DownCluster, DownRowsH, m![H % 60 / 4, 1 # 2]> = unsafe { result.reshape() };
    // One pass per cluster: the 16 row slices of a 240-row block meet on a ring of 16 (rows 60 g4 + 4 r + q),
    // and row pairs are committed as whole 8 B units in natural H order.
    let part: DmTensor<f32, Chip, DownCluster, DownPartialRows, m![H % 240]> = device
        .main
        .begin(result.view())
        .fetch::<m![H % 60 / 4], m![1 # 2]>()
        .switch::<DownPartialRows, m![H % 60 / 4, H / 60 % 4, H % 4]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H % 60 / 4, H / 60 % 4, H % 4], m![1 # 8]>()
        .transpose::<m![H % 60 / 4, H / 60 % 4, H / 2 % 2], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();
    // Both clusters' partials onto cluster 0's 16 epilogue slices (DM-to-DM, cluster mapping changes; the
    // L half becomes an element axis), then one add pass: the intra-slice reducer sums over L / 7680.
    let parts: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![L / 7680, H % 240]> = part.to_dm(&mut device.tdma);
    device
        .main
        .begin(parts.view())
        .fetch::<m![H % 240 / 4, L / 7680], m![H % 4]>()
        .collect::<m![H % 240 / 4, L / 7680], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        .vector_intra_slice_reduce::<L, m![H % 240 / 4], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![H % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![H % 4]>()
        .commit()
}


/// Down global scale, post-FFN RMSNorm, residual add and layer gate on 8 slices x 480 channels.
///
/// Replaces: gather to one slice, HBM round trip, f32 -> bf16 cast pass, global-scale pass,
/// `rmsnorm::normalize` (its own 8-slice relayout and 1-slice broadcast back), a 2-tile residual
/// add and a layer-gate pass -- 14k static cycles of serial small operations after the last
/// contraction. Rounding differs from that chain in two places, each by at most one bf16 ulp:
/// `bf16(y * gs)` instead of `bf16(bf16(y) * gs)`, and `bf16((n + r) * gate)` instead of
/// `bf16(bf16(n + r) * gate)`. The normalized value is still rounded to bf16 before the add.
fn down_epilogue(
    device: &mut Device,
    y: DmTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]>,
    global_scale: &DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
    norm_weight: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]>,
    residual: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]>,
    layer_scalar: &DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
    up_gs: &DmTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, EpilogueSlices, m![H % 240]> {
    let down_gs_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // F9D usfg: gs' = (su * 2^13) * gs on Sub (su * 2^13 is exact; one f32 rounding), used wherever gs was.
    let global_scale_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(up_gs.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_REFOLD)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_gs_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();
    let layer_scalar_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = device
        .sub
        .begin(layer_scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // Operands the final pass needs, prepared on Sub while Main is still contracting:
    // the residual pre-multiplied by the layer gate (exact: bf16 x bf16 fits f32).
    // C2: the gated operand goes straight to the VRF (Sub vector -> to_vrf, HW-proven by ffn_scale E1).
    let residual_gated_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> = device
        .sub
        .begin(residual.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &layer_scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .to_vrf();
    // ... and the norm weight pre-multiplied by the layer gate (also exact).
    let weight_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![H % 240]> = device
        .sub
        .begin(norm_weight.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &layer_scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .to_vrf();

    // Three passes on the serial tail. (1) mean of (y * gs)^2 straight from the f32 sums, summed
    // over the 16 epilogue slices; (2) sqrt(ms + eps) into the VRF; (3) y * gs / rms * (w * gate) +
    // r * gate -> bf16. Against the unfused chain this drops the bf16 rounding of `y * gs` before
    // both the square and the normalization, and of the normalized value before the residual add:
    // each at most one bf16 ulp of the output, the same order as the roundings that chain skipped.
    // Pass (1) does the per-slice mean square and the 16-slice sum in one pass (ver15 AO lesson):
    // the inter-slice reducer runs over the UNPADDED view (256 slices as 16 groups of 16; group 0 is
    // the epilogue, the other groups reduce their own garbage). An inter-slice reduce on the padded
    // slice axis (`1 # 16`) is priced ~5.8x by furiosa-opt 0.8 toolchain; on the regrouped view it is cheap.
    // On hardware this pass takes 0.97k where the old partial pass + Broadcast1{16,1} gather pass took
    // 1.44k (post-ClusterSync chain 2.92k vs 3.40k, lab job 80046).
    let global_scale_vrf_g: VrfTensor<f32, Chip, Cluster, m![Dummy256 / 16, H / 240], m![1 # 8]> = unsafe { global_scale_vrf.reshape() };
    let reduced_mean_square: VrfTensor<f32, Chip, Cluster, m![Dummy256 / 16, Epi16], m![1 # 8]> = device
        .main
        .begin(unsafe { y.view().reshape::<Chip, Cluster, m![Dummy256 / 16, H / 240], m![H % 240]>() })
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf_g)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        // SQF: EPS split over the 16 summed partials; the all-reduce result (ms + eps) goes straight to the VRF.
        .vector_clip(ClipBinaryOpF32::Add, EPS / 16.0)
        .vector_inter_slice_reduce::<m![Dummy256 / 16, Epi16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .to_vrf(&mut device.sub);
    let global_scale_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = unsafe { global_scale_vrf_g.reshape() };
    let ms_vrf: VrfTensor<f32, Chip, Cluster, EpilogueSlices, m![1 # 8]> = unsafe { reduced_mean_square.reshape() };

    device.main
        .begin(y.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_stash()
        // SQF: stream * 0 + (ms + eps), sqrt, then stash / rms on the dedicated FpDiv stage.
        .vector_fp_ternary_with_mode(FpTernaryOp::FmaF, TernaryArgMode::Mode021, (&ms_vrf, 0.0f32))
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_gated_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Normalize independently in each cluster's eight-slice seed group. The RMS reduction,
/// division, weight multiplication and BF16 rounding match shared::rmsnorm::normalize.
/// Returning the shards directly avoids gathering to a slice and staging through HBM.
pub(crate) fn normalize_input(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> NormalizedInput {
    type ReducingSlices = UpPairRows;

    let x: DmTensor<bf16, Chip, UpCluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut device.tdma);
    let weight_dm: DmTensor<bf16, Chip, UpCluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut device.tdma);
    // A Main copy of the weight (Main is busy with the reduce when it lands) so the Sub `to_vrf` below
    // is listed after the Up weight DMA: on hardware a Sub command listed before that DMA waits behind
    // the head lookup-table StoTab (5.7k cold) and holds the Up weight issue.
    (x, weight_dm)
}

/// fgs p68: identity copy of the rms weight (+0.0 from the Gate seed) on Main.
fn pin_weight_copy(device: &mut Device, w: &NormalizedX, zg: &VrfTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]>) -> NormalizedX {
    device
        .main
        .begin(w.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zg)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// ffn20 R3: the input RMSNorm proper (same passes as before), gated on `zero_vrf` (exact 0 from the Up seed).
fn normalize_finish(
    device: &mut Device,
    x: &NormalizedX,
    weight_dm: &NormalizedX,
    zero_vrf: &VrfTensor<f32, Chip, UpCluster, UpPairRows, m![1 # 8]>,
) -> NormalizedX {
    type ReducingSlices = UpPairRows;

    let reduced_mean_square: DmTensor<f32, Chip, UpCluster, m![L / 1920 % 4, 1 # 8, Dummy8], m![1 # 8]> = device
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
        .vector_fp_binary(FpBinaryOp::AddF, zero_vrf)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![L / 1920 % 4, 1 # 8, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, UpCluster, m![L / 1920 % 4, 1 # 8, Dummy8], m![1 # 8]> = device
        .main
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
    let rms: DmTensor<f32, Chip, UpCluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_vrf: VrfTensor<f32, Chip, UpCluster, ReducingSlices, m![H % 480]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, UpCluster, ReducingSlices, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, UpCluster, ReducingSlices, m![H % 480]> = device
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
