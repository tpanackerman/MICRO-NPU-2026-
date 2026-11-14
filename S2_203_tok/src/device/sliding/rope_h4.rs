use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{C, Ds, Dummy2, Dummy8, E, Gf, Gs, Ns};
/// Four KV heads (and their two Q groups) remain in each projection cluster.
pub(crate) type HeadCluster = m![Ns / 4];
type Cluster = HeadCluster;

pub(crate) type KvHeadsAcrossSlices = m![Ns % 4, 1 # 64];

/// The selected cos and sin rows, `[cos, sin]`, on the four head slices of both clusters.
pub(crate) type RopeRows = DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Dummy2, Ds]>;

/// RoPE rows computed on chip (step 40). The `cos`/`sin` inputs are the model's standard
/// theta = 10,000 tables, so the row a position selects is a pure function of that position:
/// `cos_row[i] = cos(POS * f(i))`, `f(i) = 10000^(-(i mod 128) / 128)`, and
/// `sin_row[i] = -/+ sin(POS * f(i))` with the low half negated (the `rotate_half` convention the
/// reference bakes into its sin table). Computing them here replaces two `dma_gather`s, an HBM
/// staging store/load pair and the random write->read sync they carried (ban 02 hardware trace:
/// 6.6k cycles of DMA lane plus 1-20k of sync, then 5k of RoPE waiting on the tail) with one 4-byte
/// load of `rope_offset` and ~22 tiny Main passes on the four head slices, which run while Main is
/// otherwise idle under the weight loads.
///
/// The ramp `f(i)` is built without any index primitive (the toolchain does not lower
/// `TagMode::AxisToggle`): eight doublings of a geometric sequence, each writing `[v, v * r^n]` into
/// the two tiles of a length-doubled tensor, in a one-value-per-flit `[N, 1 # 8]` layout, then one
/// Transpose pass packs the 256 flits into a dense `[Ds]` vector. The last doubling uses factor 1
/// because `f` repeats over the two halves. Precision: <= 8 f32 multiplies per value (~5e-7
/// relative), and the results are rounded to bf16 exactly as the tables are.
type RopeF32<E> = DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, E>;
const ROPE_R1: f32 = 0.930_572_032_9;
const ROPE_R2: f32 = 0.865_964_353_1;
const ROPE_R4: f32 = 0.749_894_201_8;
const ROPE_R8: f32 = 0.562_341_332_4;
const ROPE_R16: f32 = 0.316_227_763_9;
const ROPE_R32: f32 = 0.1;
const ROPE_R64: f32 = 0.01;
/// `rope_offset` is a byte offset (`POS * 256 * 2`); read as a Q24.7 fixed point by
/// `vector_fxp_to_fp(24)` (value = raw * 2^(24 - 31)) it is `4 * POS`.
const ROPE_OFFSET_TO_POS: f32 = 0.25;
/// Range reduction constants in the same Q24.7 fixed point (`int_width` 24, 7 fraction bits):
/// `turns = round(angle / 2pi * 128)` is stored as an i32 whose value read back is `n` (exact),
/// scaled back by `-2pi * 128`... i.e. `angle * (1 / 2pi) / 128` goes in, `raw / 128 * (-2pi * 128)`
/// comes out.
const ROPE_INV_TWO_PI_Q24: f32 = 0.001_243_398_0;
const ROPE_NEG_TWO_PI_Q24: f32 = -804.247_719_3;
const ROPE_TURNS_INT_WIDTH: u32 = 24;

macro_rules! rope_ramp_double {
    ($name:ident, $in_time:ty, $in_elem:ty, $mid_elem:ty, $mid_tile:ty, $out_elem:ty) => {
        /// `[v, v * factor]`: the first tile copies `v`, the second scales it, and the doubled
        /// tensor is relabelled to its canonical length axis (a pure regrouping of the wire order).
        fn $name(device: &mut Device, v: &RopeF32<$in_elem>, factor: f32) -> RopeF32<$out_elem> {
            let mut mid: RopeF32<$mid_elem> = DmTensor::new();
            device.main
                .begin(v.view())
                .fetch::<$in_time, m![1 # 8]>()
                .collect::<$in_time, m![1 # 8]>()
                .commit_trim::<m![1 # 8]>()
                .commit_view(mid.view_mut().tile::<m![Dummy2], 1, $mid_tile>(0));
            device.main
                .begin(v.view())
                .fetch::<$in_time, m![1 # 8]>()
                .collect::<$in_time, m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit_view(mid.view_mut().tile::<m![Dummy2], 1, $mid_tile>(1));
            unsafe { mid.reshape() }
        }
    };
}
rope_ramp_double!(rope_ramp_2, m![1], m![1 # 8], m![Dummy2, 1 # 8], m![Dummy2 = 1 #{!} 2, 1 # 8], m![Gs, 1 # 8]);
rope_ramp_double!(rope_ramp_4, m![Gs], m![Gs, 1 # 8], m![Dummy2, Gs, 1 # 8], m![Dummy2 = 1 #{!} 2, Gs, 1 # 8], m![Dummy8 / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_8, m![Dummy8 / 2], m![Dummy8 / 2, 1 # 8], m![Dummy2, Dummy8 / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, Dummy8 / 2, 1 # 8], m![Dummy8, 1 # 8]);
rope_ramp_double!(rope_ramp_16, m![Dummy8], m![Dummy8, 1 # 8], m![Dummy2, Dummy8, 1 # 8], m![Dummy2 = 1 #{!} 2, Dummy8, 1 # 8], m![Gf, 1 # 8]);
rope_ramp_double!(rope_ramp_32, m![Gf], m![Gf, 1 # 8], m![Dummy2, Gf, 1 # 8], m![Dummy2 = 1 #{!} 2, Gf, 1 # 8], m![C / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_64, m![C / 2], m![C / 2, 1 # 8], m![Dummy2, C / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, C / 2, 1 # 8], m![C, 1 # 8]);
rope_ramp_double!(rope_ramp_128, m![C], m![C, 1 # 8], m![Dummy2, C, 1 # 8], m![Dummy2 = 1 #{!} 2, C, 1 # 8], m![Ds / 2, 1 # 8]);
rope_ramp_double!(rope_ramp_256, m![Ds / 2], m![Ds / 2, 1 # 8], m![Dummy2, Ds / 2, 1 # 8], m![Dummy2 = 1 #{!} 2, Ds / 2, 1 # 8], m![Ds, 1 # 8]);

/// 1/Ds: summed over the Ds channels of a head it is exactly 1.0.
const ROPE_ONE_OVER_DS: f32 = 0.00390625;

pub(crate) fn stage_rope_rows_r17(
    device: &mut Device,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    after: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) -> RopeRows {
    // `0.0 * sum(after)`: a data dependency and nothing else. Hardware issues commands in the
    // static schedule's order, one context at a time, so the tiny passes below -- which the
    // scheduler would otherwise place at the very head of the list, where they run 3-12x slower
    // than modelled -- held back the issue of the K / Q / V weight loads behind them.
    // `sum(0 * k + 1/256) = 1.0` on every head slice: the data dependency on the finished K that
    // keeps the chain off the head of the list (it would hold back the weight issue there), now
    // entering at `one` so that `pos` and the 4-byte rope_offset load stay at the head instead of
    // sitting between the K and Q weight loads.
    let one_k: RopeF32<m![1 # 8]> = device
        .main
        .begin(after.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, ROPE_ONE_OVER_DS)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let one_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(one_k.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // POS on every head slice: the one small DMA of this path, consumed at the head.
    let off: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = rope_offset.to_dm(&mut device.tdma);
    let pos: RopeF32<m![1 # 8]> = device
        .main
        .begin(off.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_OFFSET_TO_POS)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let pos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // f(0) = 1, then the geometric ramp by doubling: [1] -> [1, r] -> [1, r, r^2, r^3] -> ...
    let one: RopeF32<m![1 # 8]> = device
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, &one_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let v = rope_ramp_2(device, &one, ROPE_R1);
    let v = rope_ramp_4(device, &v, ROPE_R2);
    let v = rope_ramp_8(device, &v, ROPE_R4);
    let v = rope_ramp_16(device, &v, ROPE_R8);
    let v = rope_ramp_32(device, &v, ROPE_R16);
    let v = rope_ramp_64(device, &v, ROPE_R32);
    let v = rope_ramp_128(device, &v, ROPE_R64);
    let v = rope_ramp_256(device, &v, 1.0f32);
    // One value per flit -> dense [Ds]: the Transpose Engine packs two f32 rows per flit.
    let f: RopeF32<m![Ds]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds], m![1 # 8]>()
        .collect::<m![Ds], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ds / 2], m![Ds % 2 # 8]>()
        .commit_trim::<m![Ds % 2]>()
        .commit();
    // angle = POS * f as f32 on every head slice (one pass), then reduced to [-pi, pi]:
    // turns = round(angle / 2pi) through the Vector Engine's f32 -> i32 conversion (the host model
    // `float_to_fixedpoint` rounds to nearest; int_width 24 = the same fixed point the POS read uses,
    // so `turns` holds 128 * n), r = angle - 2pi * n. cos(angle) = cos(r), sin(angle) = sin(r), and
    // the hardware Sin/Cos never see an argument beyond pi (ban 40_01 fed them up to POS = 137 rad
    // and q/k came back non-finite).
    let angle: RopeF32<m![Ds]> = device
        .main
        .begin(f.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &pos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let angle_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let turns: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .main
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_INV_TWO_PI_Q24)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_fp_to_fxp(ROPE_TURNS_INT_WIDTH)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    // cos(r) over all 256 channels, -sin(r) over the low half, +sin(r) over the high half.
    let mut rows: RopeRows = DmTensor::new();
    device.main
        .begin(turns.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
        .vector_fp_binary(FpBinaryOp::AddF, &angle_vrf)
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    // The two sin halves as a macro: a vector-engine immediate must be a literal (an `if`-selected
    // sign does not const-fold). They land in their own [Ds] tensor through single-level tiles (a
    // nested Dummy2 x Ds tile write fails verification) and are then copied into `rows`.
    let mut sin_full: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    macro_rules! sin_half {
        ($half:literal, $sign:literal) => {
            let angle_half: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
                .sub
                .begin(angle.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            device.main
                .begin(turns.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
                .vector_fp_binary(FpBinaryOp::AddF, &angle_half)
                .vector_fp_unary(FpUnaryOp::Sin)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), $sign)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(sin_full.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128 * $half));
        };
    }
    sin_half!(0, -1.0f32);
    sin_half!(1, 1.0f32);
    device.main
        .begin(sin_full.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    rows
}

/// `stage_rope_rows_r17` with `rope_offset` already loaded (qd: loaded ahead of the K contraction).
pub(crate) fn stage_rope_rows_r17_dm(
    device: &mut Device,
    off: &DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>,
    after: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) -> RopeRows {
    // `0.0 * sum(after)`: a data dependency and nothing else. Hardware issues commands in the
    // static schedule's order, one context at a time, so the tiny passes below -- which the
    // scheduler would otherwise place at the very head of the list, where they run 3-12x slower
    // than modelled -- held back the issue of the K / Q / V weight loads behind them.
    // `sum(0 * k + 1/256) = 1.0` on every head slice: the data dependency on the finished K that
    // keeps the chain off the head of the list (it would hold back the weight issue there), now
    // entering at `one` so that `pos` and the 4-byte rope_offset load stay at the head instead of
    // sitting between the K and Q weight loads.
    let one_k: RopeF32<m![1 # 8]> = device
        .main
        .begin(after.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, ROPE_ONE_OVER_DS)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let one_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(one_k.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // POS on every head slice: the one small DMA of this path, consumed at the head.
    let pos: RopeF32<m![1 # 8]> = device
        .main
        .begin(off.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_OFFSET_TO_POS)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let pos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // f(0) = 1, then the geometric ramp by doubling: [1] -> [1, r] -> [1, r, r^2, r^3] -> ...
    let one: RopeF32<m![1 # 8]> = device
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, &one_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let v = rope_ramp_2(device, &one, ROPE_R1);
    let v = rope_ramp_4(device, &v, ROPE_R2);
    let v = rope_ramp_8(device, &v, ROPE_R4);
    let v = rope_ramp_16(device, &v, ROPE_R8);
    let v = rope_ramp_32(device, &v, ROPE_R16);
    let v = rope_ramp_64(device, &v, ROPE_R32);
    let v = rope_ramp_128(device, &v, ROPE_R64);
    let v = rope_ramp_256(device, &v, 1.0f32);
    // One value per flit -> dense [Ds]: the Transpose Engine packs two f32 rows per flit.
    let f: RopeF32<m![Ds]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds], m![1 # 8]>()
        .collect::<m![Ds], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ds / 2], m![Ds % 2 # 8]>()
        .commit_trim::<m![Ds % 2]>()
        .commit();
    // angle = POS * f as f32 on every head slice (one pass), then reduced to [-pi, pi]:
    // turns = round(angle / 2pi) through the Vector Engine's f32 -> i32 conversion (the host model
    // `float_to_fixedpoint` rounds to nearest; int_width 24 = the same fixed point the POS read uses,
    // so `turns` holds 128 * n), r = angle - 2pi * n. cos(angle) = cos(r), sin(angle) = sin(r), and
    // the hardware Sin/Cos never see an argument beyond pi (ban 40_01 fed them up to POS = 137 rad
    // and q/k came back non-finite).
    let angle: RopeF32<m![Ds]> = device
        .main
        .begin(f.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &pos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let angle_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let turns: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .main
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_INV_TWO_PI_Q24)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_fp_to_fxp(ROPE_TURNS_INT_WIDTH)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    // cos(r) over all 256 channels, -sin(r) over the low half, +sin(r) over the high half.
    let mut rows: RopeRows = DmTensor::new();
    device.main
        .begin(turns.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
        .vector_fp_binary(FpBinaryOp::AddF, &angle_vrf)
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    // The two sin halves as a macro: a vector-engine immediate must be a literal (an `if`-selected
    // sign does not const-fold). They land in their own [Ds] tensor through single-level tiles (a
    // nested Dummy2 x Ds tile write fails verification) and are then copied into `rows`.
    let mut sin_full: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    macro_rules! sin_half {
        ($half:literal, $sign:literal) => {
            let angle_half: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
                .sub
                .begin(angle.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            device.main
                .begin(turns.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
                .vector_fp_binary(FpBinaryOp::AddF, &angle_half)
                .vector_fp_unary(FpUnaryOp::Sin)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), $sign)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(sin_full.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128 * $half));
        };
    }
    sin_half!(0, -1.0f32);
    sin_half!(1, 1.0f32);
    device.main
        .begin(sin_full.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    rows
}

/// `apply_rope` with the sin products computed straight from the two half tiles of q / k (the low
/// half of the output is the high half of the input times the low half of sin, and vice versa):
/// two passes per tensor instead of two copy passes plus one product pass.
pub(crate) fn apply_rope_r21(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    rows: &RopeRows,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let sin = rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1);
    let sin_lo: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let sin_hi: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let mut q_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = DmTensor::new();
    // q_sin[.., 0..128] = q[.., 128..256] * sin[0..128]
    device.main
        .begin(q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128))
        .fetch::<m![Gs], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0));
    // q_sin[.., 128..256] = q[.., 0..128] * sin[128..256]
    device.main
        .begin(q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0))
        .fetch::<m![Gs], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi)
        .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(q_sin.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128));
    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = device
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let result_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = device
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let mut k_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    device.main
        .begin(k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_lo)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));
    device.main
        .begin(k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_hi)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(k_sin.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));
    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let result_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    (result_q, result_k)
}

/// V12 KSq (qkvD2): `stage_rope_rows_r17` that also returns its exact-1.0 VRF (`one_vrf`, from the gathered k).
pub(crate) fn stage_rope_rows_r17o(
    device: &mut Device,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    after: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) -> (RopeRows, VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>) {
    // `0.0 * sum(after)`: a data dependency and nothing else. Hardware issues commands in the
    // static schedule's order, one context at a time, so the tiny passes below -- which the
    // scheduler would otherwise place at the very head of the list, where they run 3-12x slower
    // than modelled -- held back the issue of the K / Q / V weight loads behind them.
    // `sum(0 * k + 1/256) = 1.0` on every head slice: the data dependency on the finished K that
    // keeps the chain off the head of the list (it would hold back the weight issue there), now
    // entering at `one` so that `pos` and the 4-byte rope_offset load stay at the head instead of
    // sitting between the K and Q weight loads.
    let one_k: RopeF32<m![1 # 8]> = device
        .main
        .begin(after.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, ROPE_ONE_OVER_DS)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let one_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(one_k.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // POS on every head slice: the one small DMA of this path, consumed at the head.
    let off: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = rope_offset.to_dm(&mut device.tdma);
    let pos: RopeF32<m![1 # 8]> = device
        .main
        .begin(off.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_OFFSET_TO_POS)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let pos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = device
        .sub
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // f(0) = 1, then the geometric ramp by doubling: [1] -> [1, r] -> [1, r, r^2, r^3] -> ...
    let one: RopeF32<m![1 # 8]> = device
        .main
        .begin(pos.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0f32)
        .vector_fp_binary(FpBinaryOp::AddF, &one_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let v = rope_ramp_2(device, &one, ROPE_R1);
    let v = rope_ramp_4(device, &v, ROPE_R2);
    let v = rope_ramp_8(device, &v, ROPE_R4);
    let v = rope_ramp_16(device, &v, ROPE_R8);
    let v = rope_ramp_32(device, &v, ROPE_R16);
    let v = rope_ramp_64(device, &v, ROPE_R32);
    let v = rope_ramp_128(device, &v, ROPE_R64);
    let v = rope_ramp_256(device, &v, 1.0f32);
    // One value per flit -> dense [Ds]: the Transpose Engine packs two f32 rows per flit.
    let f: RopeF32<m![Ds]> = device
        .main
        .begin(v.view())
        .fetch::<m![Ds], m![1 # 8]>()
        .collect::<m![Ds], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ds / 2], m![Ds % 2 # 8]>()
        .commit_trim::<m![Ds % 2]>()
        .commit();
    // angle = POS * f as f32 on every head slice (one pass), then reduced to [-pi, pi]:
    // turns = round(angle / 2pi) through the Vector Engine's f32 -> i32 conversion (the host model
    // `float_to_fixedpoint` rounds to nearest; int_width 24 = the same fixed point the POS read uses,
    // so `turns` holds 128 * n), r = angle - 2pi * n. cos(angle) = cos(r), sin(angle) = sin(r), and
    // the hardware Sin/Cos never see an argument beyond pi (ban 40_01 fed them up to POS = 137 rad
    // and q/k came back non-finite).
    let angle: RopeF32<m![Ds]> = device
        .main
        .begin(f.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &pos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let angle_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let turns: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .main
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_INV_TWO_PI_Q24)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_fp_to_fxp(ROPE_TURNS_INT_WIDTH)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    // cos(r) over all 256 channels, -sin(r) over the low half, +sin(r) over the high half.
    let mut rows: RopeRows = DmTensor::new();
    device.main
        .begin(turns.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
        .vector_fp_binary(FpBinaryOp::AddF, &angle_vrf)
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    // The two sin halves as a macro: a vector-engine immediate must be a literal (an `if`-selected
    // sign does not const-fold). They land in their own [Ds] tensor through single-level tiles (a
    // nested Dummy2 x Ds tile write fails verification) and are then copied into `rows`.
    let mut sin_full: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    macro_rules! sin_half {
        ($half:literal, $sign:literal) => {
            let angle_half: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
                .sub
                .begin(angle.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            device.main
                .begin(turns.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
                .vector_fp_binary(FpBinaryOp::AddF, &angle_half)
                .vector_fp_unary(FpUnaryOp::Sin)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), $sign)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(sin_full.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128 * $half));
        };
    }
    sin_half!(0, -1.0f32);
    sin_half!(1, 1.0f32);
    device.main
        .begin(sin_full.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    (rows, one_vrf)
}


// ---- V15-B qkv15 bzr: RoPE as two two-group (unzip / zip) passes per tensor instead of two half products + a Sub
// VRF load + a final pass. out[..128] = x[..128] * cos[..128] + x[128..] * sin[..128] and out[128..] = x[128..] *
// cos[128..] + x[..128] * sin[128..], each ONE Main pass: the two halves enter as the two groups of an interleaved
// stream (`begin_interleaved`), each group is multiplied by its own VRF (fp stage), and `vector_fp_zip(AddF)` adds
// them. Same f32 products and the same single add as `apply_rope_r21` (q * cos + q_sin), one bf16 cast at the end:
// bit-identical (ARCH_QKV_B, external review R2). ----
axes![Iz = 2];

/// The cos row as two [Ds = 128] VRF halves (Sub loads from the rope rows).
pub(crate) fn rope_cos_halves(
    device: &mut Device,
    rows: &RopeRows,
) -> (
    VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
) {
    let cos = rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0);
    let lo: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(cos.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let hi: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(cos.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    (lo, hi)
}


/// The two sin half rows as f32 VRFs (as `rope_vrfs`, without the cos row).
pub(crate) fn rope_sin_halves(
    device: &mut Device,
    rows: &RopeRows,
) -> (
    VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
) {
    let sin = rows.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1);
    let lo: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let hi: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
        .sub
        .begin(sin.tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    (lo, hi)
}

/// q RoPE, two zip passes (see above).
pub(crate) fn apply_rope_q_zip(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    cos_lo: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    cos_hi: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    sin_lo: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    sin_hi: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
) -> DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> {
    let mut out: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>($own),
                    q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>($other),
                )
                .fetch::<m![Iz, Gs], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Gs, Ds = 128 / 8]>()
                .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half!(0, 128, cos_lo, sin_lo);
    zip_half!(128, 0, cos_hi, sin_hi);
    out
}

/// k RoPE, two zip passes.
pub(crate) fn apply_rope_k_zip(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    cos_lo: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    cos_hi: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    sin_lo: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
    sin_hi: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]>,
) -> DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> {
    let mut out: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>($own),
                    k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>($other),
                )
                .fetch::<m![Iz], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Ds = 128 / 8]>()
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half_k!(0, 128, cos_lo, sin_lo);
    zip_half_k!(128, 0, cos_hi, sin_hi);
    out
}

// ---- V17 QKV qf7 (engineer qkv18): the RoPE rows with POS gated on the chunk-mask seed, so the ramp, POS and the
// cos/sin rows run under the first (Q) weight stream. Values identical to `stage_rope_rows_r17o` (POS * 1.0 exact;
// same ramp, angle, range reduction and cos/sin passes). ----
/// The ramp `f(i) = 10000^(-(i mod 128) / 128)` as a dense `[Ds]` f32 row on the head slices.
pub(crate) type RopeRamp = DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>;

/// The exact 1.0 from the gathered k (VRF), as returned by `stage_rope_rows_r17o`.
pub(crate) type RopeOneVrf = VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;

/// The 8 doublings and the packing transpose of `stage_rope_rows_r17o`, from a seed whose live value is 1.0.
fn rope_ramp_from(device: &mut Device, one: &RopeF32<m![1 # 8]>) -> RopeRamp {
    let v = rope_ramp_2(device, one, ROPE_R1);
    let v = rope_ramp_4(device, &v, ROPE_R2);
    let v = rope_ramp_8(device, &v, ROPE_R4);
    let v = rope_ramp_16(device, &v, ROPE_R8);
    let v = rope_ramp_32(device, &v, ROPE_R16);
    let v = rope_ramp_64(device, &v, ROPE_R32);
    let v = rope_ramp_128(device, &v, ROPE_R64);
    let v = rope_ramp_256(device, &v, 1.0f32);
    device
        .main
        .begin(v.view())
        .fetch::<m![Ds], m![1 # 8]>()
        .collect::<m![Ds], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ds / 2], m![Ds % 2 # 8]>()
        .commit_trim::<m![Ds % 2]>()
        .commit()
}

/// h1K (qkv14): a DM seed whose live value is exactly 1.0, and the ramp from it.
pub(crate) type RopeOneDm = DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;

pub(crate) fn rope_ramp_seeded(device: &mut Device, one: &RopeOneDm) -> RopeRamp {
    rope_ramp_from(device, one)
}

/// POS on the head slices, `(4 POS) * 0.25 * one`: the exact 1.0 from the gathered k only gates the pass, so the
/// rope_offset load's first consumer runs after the K gather instead of between the small head loads.
fn rope_pos_gated(
    device: &mut Device,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    one_vrf: &RopeOneVrf,
) -> VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> {
    let off: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = rope_offset.to_dm(&mut device.tdma);
    let pos: RopeF32<m![1 # 8]> = device
        .main
        .begin(off.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(24)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_OFFSET_TO_POS)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), one_vrf)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device.sub.begin(pos.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// angle = POS * f, range reduction, cos / -sin / +sin rows (the tail of `stage_rope_rows_r17o`, unchanged).
fn rope_rows_from(
    device: &mut Device,
    f: &RopeRamp,
    pos_vrf: &VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>,
) -> RopeRows {
    let angle: RopeF32<m![Ds]> = device
        .main
        .begin(f.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), pos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let angle_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .sub
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let turns: DmTensor<i32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = device
        .main
        .begin(angle.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_INV_TWO_PI_Q24)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_fp_to_fxp(ROPE_TURNS_INT_WIDTH)
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let mut rows: RopeRows = DmTensor::new();
    device.main
        .begin(turns.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
        .vector_fp_binary(FpBinaryOp::AddF, &angle_vrf)
        .vector_fp_unary(FpUnaryOp::Cos)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(0));
    let mut sin_full: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();
    macro_rules! sin_half {
        ($half:literal, $sign:literal) => {
            let angle_half: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds = 128]> = device
                .sub
                .begin(angle.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            device.main
                .begin(turns.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128 * $half))
                .fetch::<m![1], m![Ds = 128]>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp_to_fp(ROPE_TURNS_INT_WIDTH)
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ROPE_NEG_TWO_PI_Q24)
                .vector_fp_binary(FpBinaryOp::AddF, &angle_half)
                .vector_fp_unary(FpUnaryOp::Sin)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), $sign)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(sin_full.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128 * $half));
        };
    }
    sin_half!(0, -1.0f32);
    sin_half!(1, 1.0f32);
    device.main
        .begin(sin_full.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![Dummy2], 1, m![Dummy2 = 1 #{!} 2, Ds]>(1));
    rows
}

pub(crate) fn stage_rope_rows_m18(
    device: &mut Device,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    f: &RopeRamp,
    seed: &RopeOneDm,
) -> (RopeRows, RopeOneVrf) {
    let one_vrf: RopeOneVrf = device.sub.begin(seed.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let pos_vrf = rope_pos_gated(device, rope_offset, &one_vrf);
    (rope_rows_from(device, f, &pos_vrf), one_vrf)
}
