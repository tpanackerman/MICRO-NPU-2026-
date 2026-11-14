//! QKV projection with a K-split-15 weight layout (V13 qkvB13, experiments/qkvB13_notes.md).
//!
//! Every weight row's 3,840 B are 15 aligned 256 B chunks; chunk k of the rows of row group g lands on slice (g, k)
//! of the cluster (16 slots per group, the 16th is padding the DMA never writes). A DMA engine's packet window then
//! walks whole rows (AO-like coverage): K+Q+V stream ~1.3k faster than the far2 layout (qkvRate Arena 90846 /
//! qkvB13 trace 91691).
//! - x and the input RMS weight are loaded replicated to the 16 row groups in the chunk layout; x's buffer is zeroed
//!   first through the all-live 16-slot type, so the padded slot's x is exactly +0.0.
//! - A -1 / 0 slot mask (from the zero-padded x) ANDs the padded slot's contraction partial to +0.0 (its weights are
//!   never written); the inter-slice reducer sums the 16 slots while moving the row's low 4 bits into the slot axis.
//! - Ordering as qkvD2 ksq2: K lands first, and its gather, rope-row staging and K-norm part 1 run during the Q
//!   weight DMA; the K scale load is pinned ahead of the Q weight by an exact zero added into the K contraction, and
//!   the Q contraction is pinned behind the K rms (exact zero) so the post-Q chain is q gather -> K-norm part 2 /
//!   q normalize -> rope. Same-job labs vs V12 QKV: Arena 91814 mean -362, 91889 mean -429 / E[max2] -513 (SE 203).
use furiosa_opt_std::prelude::*;

use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices;
use crate::Chip;
use crate::axes::{Ds, Gs, H, Ps, Qs};

axes![Hk = 15, Hp = 256, K16 = 16, D16 = 16];

const QKV_ACTIVATION_SCALE: f32 = 16.0;

type QCluster = m![Qs / 2048];
type KvCluster = m![Ps / 1024];

/// Contraction slices: row group x chunk slot (15 live chunks of 16 slots).
type QSl = m![Qs / 128 % 16, Hk # 16];
type KvSl = m![Ps / 64 % 16, Hk # 16];
/// All-reduced scalar slices (the slot axis replaced by a replicated axis).
type QSlAll = m![Qs / 128 % 16, D16];
/// The contraction slices as an all-live 16-slot type: the slot reduce-scatter (row % 16 into the slot axis) is
/// "Promotion with ValidCountGenerator is not supported yet" from the padded `Hk # 16` source (bisect probe
/// t_k15p_rs) but lowers from an all-live one (t_k15p_rsfull). Slot 15 is garbage the mask ANDs to +0.0.
type QFull = m![Qs / 128 % 16, K16];

pub(crate) type QW15 = DmTensor<f8e4m3, Chip, QCluster, QSl, m![Qs % 128, Hp]>;
pub(crate) type KvW15 = DmTensor<f8e4m3, Chip, KvCluster, KvSl, m![Ps % 64, Hp]>;
pub(crate) type Weights15 = (QW15, KvW15, KvW15);
/// Q weights on the all-live contraction slices; K/V weights relabelled onto them (the contraction is row-independent).
type QW15Full = DmTensor<f8e4m3, Chip, QCluster, QFull, m![Qs % 128, Hp]>;
type KvW15Q = DmTensor<f8e4m3, Chip, QCluster, QFull, m![Qs % 64, Hp]>;

pub(crate) type XTrf15 = TrfTensor<f8e4m3, Chip, QCluster, QFull, m![1], m![Gs, Hp]>;
/// -1 (all bits) on the 15 live chunk slots, 0 on the padded slot.
pub(crate) type ChunkMask = VrfTensor<i32, Chip, QCluster, QFull, m![1 # 8]>;

/// After the reduce: slice (g, row % 16), rows / 16 in the element.
type QPart15 = DmTensor<bf16, Chip, QCluster, m![Qs / 128 % 16, Qs % 16], m![Qs % 128 / 16]>;
type KvPart15Q = DmTensor<bf16, Chip, QCluster, m![Qs / 128 % 16, Qs % 16], m![Qs % 64 / 16]>;
type KvPart15 = DmTensor<bf16, Chip, KvCluster, m![Ps / 64 % 16, Ps % 16], m![Ps % 64 / 16]>;

/// Exact zero on the 256 contraction slices (the Q / K contraction pins).
type QZero15 = VrfTensor<f32, Chip, QCluster, QFull, m![1 # 8]>;

/// The K weight scale loaded ahead of the Q weight (see `qkv_k15_ksq2`).
type KvScaleDm = DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]>;

pub(crate) fn load_weights_k15(
    device: &mut Device,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> Weights15 {
    let qv: HbmTensorView<'_, f8e4m3, Chip, m![Qs, Hk, Hp]> = unsafe { q_weight.view().reshape() };
    let kv: HbmTensorView<'_, f8e4m3, Chip, m![Ps, Hk, Hp]> = unsafe { k_weight.view().reshape() };
    let vv: HbmTensorView<'_, f8e4m3, Chip, m![Ps, Hk, Hp]> = unsafe { v_weight.view().reshape() };
    let q: QW15 = qv.to_dm(&mut device.tdma);
    let k: KvW15 = kv.to_dm(&mut device.tdma);
    let v: KvW15 = vv.to_dm(&mut device.tdma);
    (q, k, v)
}

/// x in the chunk layout, replicated to the 16 row groups by the DMA (the row-group axis is absent from x). The
/// buffer is zeroed through the all-live 16-slot type first; the DMA through the padded `Hk # 16` type never writes
/// slot 15 (qkvRate descriptor decode), so the padded slot's x is exactly +0.0 whatever the DM held before.
fn load_x15(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> DmTensor<bf16, Chip, QCluster, QSl, m![Hp]> {
    let mut buf: DmTensor<bf16, Chip, QCluster, QFull, m![Hp]> = DmTensor::new();
    buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let mut buf: DmTensor<bf16, Chip, QCluster, QSl, m![Hp]> = unsafe { buf.reshape() };
    let xv: HbmTensorView<'_, bf16, Chip, m![Hk, Hp]> = unsafe { x.view().reshape() };
    xv.to_dm_view(&mut device.tdma, buf.view_mut());
    buf
}

/// The chunk-slot mask from the zero-padded x (no extra DMA; a memset / tile on the slice axis is not lowered, and
/// what a compute pass through the padded type does on slot 15 is not relied on: the pass runs on the all-live
/// type): -1 where the slot's x chunk has a nonzero value, 0 where it is all +-0 -- the padded slot, or a live chunk
/// whose true contribution is exactly 0 anyway. -max|bits| via BitAnd 0x7fff0000 (bf16 bits sit in the f32 high
/// half), MulInt -1 and a Min reduce; Clip Max -1 maps it to -1 / 0.
fn chunk_mask(device: &mut Device, x: &DmTensor<bf16, Chip, QCluster, QSl, m![Hp]>) -> ChunkMask {
    type Full = QFull;
    let xf: DmTensorView<'_, bf16, Chip, QCluster, Full, m![Hp]> = unsafe { x.view().reshape() };
    let m: DmTensor<i32, Chip, QCluster, Full, m![1 # 8]> = device
        .sub
        .begin(xf)
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_0000i32)
        .vector_fxp(FxpBinaryOp::MulInt, -1i32)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpI32::Max, -1i32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device.sub.begin(m.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// Input RMSNorm in the chunk layout (every row group holds the whole x), hi/lo split, TRF load.
/// Steering as `qkv_h4::normalize_input_x8w`: the rms weight is loaded first and the mean square adds an exact
/// zero taken from it (bitwise AND 0: exact for any bits, the padded slot's garbage included). Also returns the
/// chunk-slot mask (computed after the TRF load, off the head chain; first read by the Q contraction epilogue).
pub(crate) fn normalize_input_k15(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (XTrf15, ChunkMask) {
    type S = QSl;
    let wv: HbmTensorView<'_, bf16, Chip, m![Hk, Hp]> = unsafe { rms_weight.view().reshape() };
    let weight_dm: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = wv.to_dm(&mut device.tdma);
    let zero: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(zero.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let x = load_x15(device, x);
    // Mean square: chunk partial (the padded slot's x is +0), summed over the 16 slots and broadcast back to all.
    let mean_square: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QSlAll, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<bf16, m![Hp % 8 # 16]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hilo = split_hilo15(device, &normalized);
    let hilo: DmTensor<f8e4m3, Chip, QCluster, QFull, m![Gs, Hp]> = unsafe { hilo.reshape() };
    let trf: XTrf15 = device
        .sub
        .begin(hilo.view())
        .fetch::<m![Gs, Hp / 32], m![Hp % 32]>()
        .collect::<m![Gs, Hp / 32], m![Hp % 32]>()
        .to_trf();
    let mask = chunk_mask(device, &x);
    (trf, mask)
}

/// x = hi + lo with hi = f8(16 x), lo = f8(16 x - hi) (as `qkv_h4::split_qkv_activation8`), per chunk slice.
fn split_hilo15(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, QCluster, QSl, m![Hp]>,
) -> DmTensor<f8e4m3, Chip, QCluster, QSl, m![Gs, Hp]> {
    type S = QSl;
    let mut out: DmTensor<f8e4m3, Chip, QCluster, S, m![Gs, Hp]> = DmTensor::new();
    device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hp % 8 # 32]>()
        .commit_trim::<m![Hp % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Hp]>(0));
    let hi_neg: DmTensor<f8e4m3, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -QKV_ACTIVATION_SCALE)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hp % 8 # 32]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hi_neg_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(hi_neg.view())
        .fetch::<m![Hp / 32], m![Hp % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, &hi_neg_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hp % 8 # 32]>()
        .commit_trim::<m![Hp % 8]>()
        .commit_view(out.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Hp]>(1));
    out
}

/// K / V contraction (64 rows per group), same epilogue.
fn contract_kv15(device: &mut Device, x: &XTrf15, weight: KvW15, mask: &ChunkMask) -> KvPart15 {
    let weight: KvW15Q = unsafe { weight.reshape() };
    let out: KvPart15Q = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 64, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .collect::<m![Qs % 64, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .contract_outer::<m![Qs % 64, Hp / 64, Gs], m![Hp % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64]>()
        .contract_lane::<m![Qs % 64], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, mask)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 128 % 16, Qs % 16], m![Qs % 64 / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs % 64 / 16 / 4], m![Qs % 64 / 16 # 16]>()
        .commit_trim::<m![Qs % 64 / 16]>()
        .commit();
    unsafe { out.reshape() }
}

/// Q gathered into channel order on the head slices: ring-64 all-gather per head (time = [row / 16, group,
/// row % 16]; channel = 128 group + row), row % 16 innermost packed 4 per packet.
fn finish_query_k15(
    device: &mut Device,
    part: QPart15,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> = device
        .main
        .begin(part.view())
        .fetch::<m![Qs % 128 / 16], m![1 # 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 64], m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16], m![1 # 16]>()
        .transpose::<m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// K or V gathered into channel order (see `finish_query_k15`): time = [row / 16, group, row % 16].
fn gather_kv15(
    device: &mut Device,
    part: KvPart15,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(part.view())
        .fetch::<m![Ps % 64 / 16], m![1 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16], m![1 # 16]>()
        .transpose::<m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16 / 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    let weight_scale: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> =
        weight_scale.to_dm(&mut device.tdma);
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// Exact +0.0 on the 256 contraction slices from a head-slice f32 tensor (padding slices' bits ANDed away).
fn zero_q15_from_head(device: &mut Device, t: &DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>) -> QZero15 {
    let v: DmTensorView<'_, f32, Chip, QCluster, QFull, m![1 # 8]> = unsafe { t.view().reshape() };
    device
        .main
        .begin(v)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `contract_q15` whose epilogue adds the exact-zero pin `z`.
fn contract_q15z(device: &mut Device, x: &XTrf15, weight: QW15, mask: &ChunkMask, z: &QZero15) -> QPart15 {
    let weight: QW15Full = unsafe { weight.reshape() };
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 128, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .collect::<m![Qs % 128, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .contract_outer::<m![Qs % 128, Hp / 64, Gs], m![Hp % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 128]>()
        .contract_lane::<m![Qs % 128], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, mask)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, z)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 128 % 16, Qs % 16], m![Qs % 128 / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs % 128 / 64], m![Qs % 128 / 16 % 4 # 16]>()
        .commit_trim::<m![Qs % 128 / 16 % 4]>()
        .commit()
}

fn zero_q15_from_kscale(device: &mut Device, s: &KvScaleDm) -> QZero15 {
    let v: DmTensorView<'_, bf16, Chip, QCluster, QFull, m![Ps % 256]> = unsafe { s.view().reshape() };
    device
        .main
        .begin(v)
        .fetch::<m![Ps % 256 / 16], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps % 256 / 8], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ps % 256 / 4], m![Ps % 4]>()
        .vector_intra_slice_reduce::<Ps, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `contract_kv15` whose epilogue adds the exact-zero pin `z`.
fn contract_kv15z(device: &mut Device, x: &XTrf15, weight: KvW15, mask: &ChunkMask, z: &QZero15) -> KvPart15 {
    let weight: KvW15Q = unsafe { weight.reshape() };
    let out: KvPart15Q = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 64, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .collect::<m![Qs % 64, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .contract_outer::<m![Qs % 64, Hp / 64, Gs], m![Hp % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 64]>()
        .contract_lane::<m![Qs % 64], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, mask)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_fp_binary(FpBinaryOp::AddF, z)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 128 % 16, Qs % 16], m![Qs % 64 / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs % 64 / 16 / 4], m![Qs % 64 / 16 # 16]>()
        .commit_trim::<m![Qs % 64 / 16]>()
        .commit();
    unsafe { out.reshape() }
}

/// `gather_kv15` with the scale already loaded.
fn gather_kv15s(
    device: &mut Device,
    part: KvPart15,
    weight_scale: KvScaleDm,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let gathered: DmTensor<bf16, Chip, KvCluster, m![Ps / 256 % 4, 1 # 64], m![Ps % 256]> = device
        .main
        .begin(part.view())
        .fetch::<m![Ps % 64 / 16], m![1 # 16]>()
        .switch::<m![Ps / 256 % 4, 1 # 64], m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16], m![1 # 16]>()
        .transpose::<m![Ps % 64 / 16, Ps / 64 % 4, Ps % 16 / 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    (unsafe { gathered.reshape() }, unsafe { weight_scale.reshape() })
}

/// x15k2 QKV body (see above).
pub(crate) fn qkv_k15_ksq2(
    device: &mut Device,
    x_trf: &XTrf15,
    mask: &ChunkMask,
    w: Weights15,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let k_scale_dm: KvScaleDm = k_weight_scale.to_dm(&mut device.tdma);
    let kz = zero_q15_from_kscale(device, &k_scale_dm);
    let k = contract_kv15z(device, x_trf, k_weight, mask, &kz);
    let (k, k_scale) = gather_kv15s(device, k, k_scale_dm);
    let (rope_rows, one) = super::rope_h4::stage_rope_rows_r17o(device, rope_offset, &k);
    let (ks_vrf, krms, krms_vrf, kx_vrf) = super::qkv_h4_norm::normalize_key_p1(device, &k, &k_scale, &one);
    let qz = zero_q15_from_head(device, &krms);
    let q = contract_q15z(device, x_trf, q0, mask, &qz);
    let v = contract_kv15(device, x_trf, v_weight, mask);
    let (q, q_scale) = finish_query_k15(device, q, q_weight_scale);
    let zero = super::qkv_h4::zero_from_q_heads(device, &q);
    let (v, v_scale) = gather_kv15(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    let k = super::qkv_h4_norm::normalize_key_p2(device, k_rms_weight, &zero, &ks_vrf, &krms_vrf, &kx_vrf);
    let q = super::qkv_h4_norm::normalize_query_scaled(device, &q, &q_scale, q_rms_weight);
    let (q, k) = super::rope_h4::apply_rope_r21(device, &q, &k, &rope_rows);
    (q, k, v)
}

// ---- V16 qkv17 g6z (experiments/qkv17_progress.md): the input RMS weight loaded ONCE per PE (each 256-element chunk
// as four 64-element quarters over the PE's four row groups, 30,720 B per cluster instead of 122,880 B) and fanned out
// on chip by a ring-64 Broadcast1 on Sub; the rms-weight exact-zero pin reads the once-per-PE weight directly (all-live
// view), so the K scale and K weight go out right behind x. The shorter head makes the scheduler load the K scale ahead
// of K: the K contraction runs at K's landing, the Q contraction runs fast and the post-Q small loads go out at Q's
// landing. Body = x15k2 with the q / k RoPE as two-group zip passes (V15 bzr). Every value bit-identical to x15k2
// (per-output errors identical in every lab). ----
axes![G4 = 4];

/// One copy per PE: quarter q of chunk k on slice (p, q, k); slot 15 is padding the DMA never writes.
type X4S = m![Qs / 512 % 4, Hp / 64, Hk # 16];
/// The same slices as an all-live 16-slot type (the fan-out source).
type X4SFull = m![Qs / 512 % 4, Hp / 64, K16];
/// Fan-out destination: PE, row group in the PE (the new broadcast axis), slot.
type X16F = m![Qs / 512 % 4, G4, K16];
type X4Tensor = DmTensor<bf16, Chip, QCluster, X4S, m![Hp % 64]>;

/// The input rms weight once per PE (slot 15 garbage, as the replicated load leaves it).
fn load_w4s(device: &mut Device, w: &HbmTensor<bf16, Chip, m![H]>) -> X4Tensor {
    let wv: HbmTensorView<'_, bf16, Chip, m![Hk, Hp]> = unsafe { w.view().reshape() };
    wv.to_dm(&mut device.tdma)
}

/// Ring-64 Broadcast1 over the four row groups of each PE: slice (p, g, k) gets quarters q = 0..3 of slot k.
fn fanout_w4s(device: &mut Device, t: &X4Tensor) -> DmTensor<bf16, Chip, QCluster, QSl, m![Hp]> {
    let v: DmTensorView<'_, bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = unsafe { t.view().reshape() };
    let out: DmTensor<bf16, Chip, QCluster, X16F, m![Hp]> = device
        .sub
        .begin(v)
        .fetch::<m![1], m![Hp % 64]>()
        .switch::<X16F, m![Hp / 64]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![Hp / 64, Hp % 64 / 16], m![Hp % 16]>()
        .commit_trim::<m![Hp % 16]>()
        .commit();
    unsafe { out.reshape() }
}


pub(crate) fn normalize_input_k15g6(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (XTrf15, ChunkMask) {
    type S = QSl;
    let w4 = load_w4s(device, rms_weight);
    let w4f: DmTensorView<'_, bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = unsafe { w4.view().reshape() };
    let zero4: DmTensor<f32, Chip, QCluster, X4SFull, m![1 # 8]> = device
        .sub
        .begin(w4f)
        .fetch::<m![Hp % 64 / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp % 64 / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Hp % 64 / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { zero4.reshape() };
    let zero_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(zero.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_dm: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &w4);
    let x: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = load_x15(device, x);
    let mean_square: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QSlAll, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<bf16, m![Hp % 8 # 16]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hilo = split_hilo15(device, &normalized);
    let hilo: DmTensor<f8e4m3, Chip, QCluster, QFull, m![Gs, Hp]> = unsafe { hilo.reshape() };
    let trf: XTrf15 = device
        .sub
        .begin(hilo.view())
        .fetch::<m![Gs, Hp / 32], m![Hp % 32]>()
        .collect::<m![Gs, Hp / 32], m![Hp % 32]>()
        .to_trf();
    let mask = chunk_mask(device, &x);
    (trf, mask)
}

/// x15k2's QKV body (`qkv_k15_ksq2`) with the zip RoPE (`rope_h4::apply_rope_{q,k}_zip`, bit-identical products).
pub(crate) fn qkv_k15_ksq2_zip(
    device: &mut Device,
    x_trf: &XTrf15,
    mask: &ChunkMask,
    w: Weights15,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
    DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>,
) {
    let (q0, k_weight, v_weight) = w;
    let k_scale_dm: KvScaleDm = k_weight_scale.to_dm(&mut device.tdma);
    let kz = zero_q15_from_kscale(device, &k_scale_dm);
    let k = contract_kv15z(device, x_trf, k_weight, mask, &kz);
    let (k, k_scale) = gather_kv15s(device, k, k_scale_dm);
    let (rope_rows, one) = super::rope_h4::stage_rope_rows_r17o(device, rope_offset, &k);
    let (ks_vrf, krms, krms_vrf, kx_vrf) = super::qkv_h4_norm::normalize_key_p1(device, &k, &k_scale, &one);
    let qz = zero_q15_from_head(device, &krms);
    let q = contract_q15z(device, x_trf, q0, mask, &qz);
    let v = contract_kv15(device, x_trf, v_weight, mask);
    let (q, q_scale) = finish_query_k15(device, q, q_weight_scale);
    let zero = super::qkv_h4::zero_from_q_heads(device, &q);
    let (v, v_scale) = gather_kv15(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    let k = super::qkv_h4_norm::normalize_key_p2(device, k_rms_weight, &zero, &ks_vrf, &krms_vrf, &kx_vrf);
    let q = super::qkv_h4_norm::normalize_query_scaled(device, &q, &q_scale, q_rms_weight);
    let (cos_lo, cos_hi) = super::rope_h4::rope_cos_halves(device, &rope_rows);
    let (sin_lo, sin_hi) = super::rope_h4::rope_sin_halves(device, &rope_rows);
    let q = super::rope_h4::apply_rope_q_zip(device, &q, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k = super::rope_h4::apply_rope_k_zip(device, &k, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    (q, k, v)
}

// ---- V17 QKV qf7 (engineer qkv18: experiments/qkv18_progress.md, analysis/v17/GAP_QKV.md C1): Q-FIRST weight order
// on the V16 g6z head. The Q weight streams right behind x; the RoPE ramp, POS and the cos/sin rows are gated on the
// chunk-mask seed (end of the x chain) and run under the Q stream (rope_offset is listed first in the head); the Q
// contraction, q gather and q-norm part 1 run under the K stream; the K contraction is pinned on exact zeros from the
// q rms and from the K weight scale (so the q and k scales are listed ahead of K); the K epilogue, q/k-norm part 2,
// the zip RoPE and the V epilogue run under / after the V stream. DMA list: rope_off | rms_w W x | Q W q_scale |
// k_scale | K W k_rms | q_rms | V. Every value bit-identical to V16 (x * 1.0 and x + 0.0 exact; same f32 ops per
// output). Same-job INFO labs vs V16 g6z: Arena 100939 -636 (SE 126, 6/6), 100958 -413 (SE 151, 8/8). ----

/// V16's `normalize_input_k15g6` (every pass identical) that also returns the chunk mask's DM tensor (the ramp seed).
pub(crate) fn normalize_input_k15g6m18(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (XTrf15, ChunkMask, MaskDm) {
    type S = QSl;
    let w4 = load_w4s(device, rms_weight);
    let w4f: DmTensorView<'_, bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = unsafe { w4.view().reshape() };
    let zero4: DmTensor<f32, Chip, QCluster, X4SFull, m![1 # 8]> = device
        .sub
        .begin(w4f)
        .fetch::<m![Hp % 64 / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp % 64 / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Hp % 64 / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { zero4.reshape() };
    let zero_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(zero.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_dm: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &w4);
    let x: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = load_x15(device, x);
    let mean_square: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QSlAll, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<bf16, m![Hp % 8 # 16]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hilo = split_hilo15(device, &normalized);
    let hilo: DmTensor<f8e4m3, Chip, QCluster, QFull, m![Gs, Hp]> = unsafe { hilo.reshape() };
    let trf: XTrf15 = device
        .sub
        .begin(hilo.view())
        .fetch::<m![Gs, Hp / 32], m![Hp % 32]>()
        .collect::<m![Gs, Hp / 32], m![Hp % 32]>()
        .to_trf();
    let (mask, mdm) = chunk_mask_dm(device, &x);
    (trf, mask, mdm)
}

/// `chunk_mask` that also returns the mask's DM tensor.
fn chunk_mask_dm(device: &mut Device, x: &DmTensor<bf16, Chip, QCluster, QSl, m![Hp]>) -> (ChunkMask, MaskDm) {
    type Full = QFull;
    let xf: DmTensorView<'_, bf16, Chip, QCluster, Full, m![Hp]> = unsafe { x.view().reshape() };
    let m: DmTensor<i32, Chip, QCluster, Full, m![1 # 8]> = device
        .sub
        .begin(xf)
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_0000i32)
        .vector_fxp(FxpBinaryOp::MulInt, -1i32)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpI32::Max, -1i32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let v: ChunkMask = device.sub.begin(m.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    (v, m)
}

type MaskDm = DmTensor<i32, Chip, QCluster, QFull, m![1 # 8]>;

/// Ramp seed on the head slices: `(mask & 0) as f32 + 1.0` == 1.0 exactly (integer AND, so NaN-free by construction).
fn ramp_seed_from_mask(device: &mut Device, mdm: &MaskDm) -> super::rope_h4::RopeOneDm {
    let v: DmTensorView<'_, i32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]> = unsafe { mdm.view().reshape() };
    device
        .main
        .begin(v)
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Q contraction (64 B packets, hi/lo in the TRF), masked, scaled by 1/16 (exact) and reduced over the 16 chunk
/// slots with the row's low 4 bits moved into the slot axis.
fn contract_q15(device: &mut Device, x: &XTrf15, weight: QW15, mask: &ChunkMask) -> QPart15 {
    let weight: QW15Full = unsafe { weight.reshape() };
    device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 128, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .collect::<m![Qs % 128, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .contract_outer::<m![Qs % 128, Hp / 64, Gs], m![Hp % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 128]>()
        .contract_lane::<m![Qs % 128], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, mask)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 128 % 16, Qs % 16], m![Qs % 128 / 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs % 128 / 64], m![Qs % 128 / 16 % 4 # 16]>()
        .commit_trim::<m![Qs % 128 / 16 % 4]>()
        .commit()
}

/// The q all-gather of `finish_query_k15` without its scale load.
fn gather_q15(device: &mut Device, part: QPart15) -> HeadGsDs15 {
    let gathered: DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]> = device
        .main
        .begin(part.view())
        .fetch::<m![Qs % 128 / 16], m![1 # 16]>()
        .switch::<m![Qs / 512 % 4, 1 # 64], m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16], m![1 # 16]>()
        .transpose::<m![Qs % 128 / 16, Qs / 128 % 4, Qs % 16 / 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    unsafe { gathered.reshape() }
}

type QScaleDm15 = DmTensor<bf16, Chip, QCluster, m![Qs / 512 % 4, 1 # 64], m![Qs % 512]>;

type HeadDs15 = DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>;

type HeadGsDs15 = DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>;

pub(crate) type HeadZeroVrf = VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;

/// Exact zero from a 256-element bf16 head-slice tensor.
fn hz_256(device: &mut Device, t: DmTensorView<'_, bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>) -> HeadZeroVrf {
    device
        .main
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
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// Exact +0.0 on the contraction slices from a head-slice `[Gs, 1 # 8]` f32 tensor (the q rms), read as 16 f32 per
/// slice (the padding's bits ANDed away).
fn qf18_zero_from_grms(
    device: &mut Device,
    t: &DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, 1 # 8]>,
) -> QZero15 {
    let v: DmTensorView<'_, f32, Chip, QCluster, QFull, m![D16]> = unsafe { t.view().reshape() };
    device
        .main
        .begin(v)
        .fetch::<m![D16 / 8], m![D16 % 8]>()
        .collect::<m![D16 / 8], m![D16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![D16 / 4], m![D16 % 4]>()
        .vector_intra_slice_reduce::<D16, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `zero_q15_from_kscale` plus a previous exact zero (on the 256 contraction slices).
fn zero_q15_from_kscale_add(device: &mut Device, s: &KvScaleDm, prev: &QZero15) -> QZero15 {
    let v: DmTensorView<'_, bf16, Chip, QCluster, QFull, m![Ps % 256]> = unsafe { s.view().reshape() };
    device
        .main
        .begin(v)
        .fetch::<m![Ps % 256 / 16], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps % 256 / 8], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ps % 256 / 4], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, prev)
        .vector_intra_slice_reduce::<Ps, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// qf7 body (see the block comment above). Returns (q, k, v) like `qkv_k15_ksq2_zip`.
pub(crate) fn qkv18_qf7(
    device: &mut Device,
    x_trf: &XTrf15,
    mask: &ChunkMask,
    mdm: &MaskDm,
    w: Weights15,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (HeadGsDs15, HeadDs15, HeadDs15) {
    let (q0, k_weight, v_weight) = w;
    let seed = ramp_seed_from_mask(device, mdm);
    let f = super::rope_h4::rope_ramp_seeded(device, &seed);
    let q_scale_dm: QScaleDm15 = q_weight_scale.to_dm(&mut device.tdma);
    let q_scale: HeadGsDs15 = unsafe { q_scale_dm.reshape() };
    let (rope_rows, one) = super::rope_h4::stage_rope_rows_m18(device, rope_offset, &f, &seed);
    let q = contract_q15(device, x_trf, q0, mask);
    let q = gather_q15(device, q);
    let (qs_vrf, qrms, qrms_vrf, qx_vrf) = super::qkv_h4_norm::normalize_query_p1(device, &q, &q_scale, &one);
    let k_scale_dm: KvScaleDm = k_weight_scale.to_dm(&mut device.tdma);
    let kz0 = qf18_zero_from_grms(device, &qrms);
    let kz = zero_q15_from_kscale_add(device, &k_scale_dm, &kz0);
    let k = contract_kv15z(device, x_trf, k_weight, mask, &kz);
    let (k, k_scale) = gather_kv15s(device, k, k_scale_dm);
    let (ks_vrf, _krms, krms_vrf, kx_vrf) = super::qkv_h4_norm::normalize_key_p1(device, &k, &k_scale, &one);
    let v = contract_kv15(device, x_trf, v_weight, mask);
    let zero = hz_256(device, k.view());
    let (v, v_scale) = gather_kv15(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sv(device, &v, &v_scale);
    let k = super::qkv_h4_norm::normalize_key_p2(device, k_rms_weight, &zero, &ks_vrf, &krms_vrf, &kx_vrf);
    let q = super::qkv_h4_norm::normalize_query_p2(device, q_rms_weight, &zero, &qs_vrf, &qrms_vrf, &qx_vrf);
    let (cos_lo, cos_hi) = super::rope_h4::rope_cos_halves(device, &rope_rows);
    let (sin_lo, sin_hi) = super::rope_h4::rope_sin_halves(device, &rope_rows);
    let q = super::rope_h4::apply_rope_q_zip(device, &q, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k = super::rope_h4::apply_rope_k_zip(device, &k, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    (q, k, v)
}

// ---- V18 qkv19 q19n2 (experiments/qkv19_progress.md): qf7 with the V normalize in two passes
// (`normalize_value_scaled_sqf`, V10-E311 form: the mean-square pass writes ms + EPS straight to the VRF, the final
// pass takes the sqrt in-stream and divides the stashed s * v on the FpDiv stage), so the separate sqrt pass after the
// V contraction is gone. Every other statement is `qkv18_qf7`'s, in the same order. Values bit-identical (Arena labs
// 101974 / 102090 / 102103 / 102114: all outputs' max/mean |d| identical to qf7). ----
pub(crate) fn qkv19_qf7n2(
    device: &mut Device,
    x_trf: &XTrf15,
    mask: &ChunkMask,
    mdm: &MaskDm,
    w: Weights15,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (HeadGsDs15, HeadDs15, HeadDs15) {
    let (q0, k_weight, v_weight) = w;
    let seed = ramp_seed_from_mask(device, mdm);
    let f = super::rope_h4::rope_ramp_seeded(device, &seed);
    let q_scale_dm: QScaleDm15 = q_weight_scale.to_dm(&mut device.tdma);
    let q_scale: HeadGsDs15 = unsafe { q_scale_dm.reshape() };
    let (rope_rows, one) = super::rope_h4::stage_rope_rows_m18(device, rope_offset, &f, &seed);
    let q = contract_q15(device, x_trf, q0, mask);
    let q = gather_q15(device, q);
    let (qs_vrf, qrms, qrms_vrf, qx_vrf) = super::qkv_h4_norm::normalize_query_p1(device, &q, &q_scale, &one);
    let k_scale_dm: KvScaleDm = k_weight_scale.to_dm(&mut device.tdma);
    let kz0 = qf18_zero_from_grms(device, &qrms);
    let kz = zero_q15_from_kscale_add(device, &k_scale_dm, &kz0);
    let k = contract_kv15z(device, x_trf, k_weight, mask, &kz);
    let (k, k_scale) = gather_kv15s(device, k, k_scale_dm);
    let (ks_vrf, _krms, krms_vrf, kx_vrf) = super::qkv_h4_norm::normalize_key_p1(device, &k, &k_scale, &one);
    let v = contract_kv15(device, x_trf, v_weight, mask);
    let zero = hz_256(device, k.view());
    let (v, v_scale) = gather_kv15(device, v, v_weight_scale);
    let v = super::qkv_h4_norm::normalize_value_scaled_sqf(device, &v, &v_scale);
    let k = super::qkv_h4_norm::normalize_key_p2(device, k_rms_weight, &zero, &ks_vrf, &krms_vrf, &kx_vrf);
    let q = super::qkv_h4_norm::normalize_query_p2(device, q_rms_weight, &zero, &qs_vrf, &qrms_vrf, &qx_vrf);
    let (cos_lo, cos_hi) = super::rope_h4::rope_cos_halves(device, &rope_rows);
    let (sin_lo, sin_hi) = super::rope_h4::rope_sin_halves(device, &rope_rows);
    let q = super::rope_h4::apply_rope_q_zip(device, &q, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k = super::rope_h4::apply_rope_k_zip(device, &k, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    (q, k, v)
}

// ---- Step 641 x4: x loaded ONCE per PE (30,720 B per cluster instead of 122,880 B) into a zeroed buffer and fanned
// out on chip to the four row groups of each PE by the same ring-64 Broadcast1 as the rms weight (`fanout_w4s`). The
// x DMA is the last command ahead of the Q weight on the serial DMA queue (trace 600_90: 1.7k, vs 1.0k for the
// once-per-PE rms weight). Slot 15 of the zeroed buffer is never written by the DMA, so the fanned-out x keeps +0.0 on
// the padded slot and the chunk mask is unchanged. Every value bit-identical. ----
fn load_x4s(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> X4Tensor {
    let mut buf: DmTensor<bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = DmTensor::new();
    buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let mut buf: X4Tensor = unsafe { buf.reshape() };
    let xv: HbmTensorView<'_, bf16, Chip, m![Hk, Hp]> = unsafe { x.view().reshape() };
    xv.to_dm_view(&mut device.tdma, buf.view_mut());
    buf
}

/// `normalize_input_k15g6m18` with x loaded once per PE and fanned out on chip (see above).
pub(crate) fn normalize_input_k15x4(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (XTrf15, ChunkMask, MaskDm) {
    type S = QSl;
    let w4 = load_w4s(device, rms_weight);
    let w4f: DmTensorView<'_, bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = unsafe { w4.view().reshape() };
    let zero4: DmTensor<f32, Chip, QCluster, X4SFull, m![1 # 8]> = device
        .sub
        .begin(w4f)
        .fetch::<m![Hp % 64 / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp % 64 / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Hp % 64 / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { zero4.reshape() };
    let zero_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(zero.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_dm: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &w4);
    let x4 = load_x4s(device, x);
    let x: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &x4);
    let mean_square: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QSlAll, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<bf16, m![Hp % 8 # 16]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hilo = split_hilo15(device, &normalized);
    let hilo: DmTensor<f8e4m3, Chip, QCluster, QFull, m![Gs, Hp]> = unsafe { hilo.reshape() };
    let trf: XTrf15 = device
        .sub
        .begin(hilo.view())
        .fetch::<m![Gs, Hp / 32], m![Hp % 32]>()
        .collect::<m![Gs, Hp / 32], m![Hp % 32]>()
        .to_trf();
    let (mask, mdm) = chunk_mask_dm(device, &x);
    (trf, mask, mdm)
}

// ---- Q800 vtail: the V tail as two switch + VE passes over a 4-consecutive-channel V partial. ----
// V174 tail after the last V byte (HW trace 885_90, c0): V contraction 2.5k -> ring-64 gather 1.18k -> mean square
// 0.77k -> normalize 0.76k -> v scatter 1.66k. Here:
// - The V contraction walks its 64 rows in time order [row % 4, row / 4] and its reduce-scatter hands row / 4 (the
//   inner 16 time steps) to the slot axis, so contraction slice (g, j) holds the 4 CONSECUTIVE channels 64g + 4j + i
//   as one 8 B element (V174: channels 64g + 16t + s, strided). Same MACs, same slot sums.
// - One 16 B f32 packet per slice feeds the ring-64 Broadcast1 (64 flits per destination instead of V174's 256; HW fit
//   663 + 4 x bytes fed per slice: ~0.73k vs 1.18k), and the post-switch stream (time [Ds / 64, Ds % 64 / 4] x
//   packet [Ds % 4]) is in natural channel order, so the natural-order v-scale VRF (f32 [Ds], the Sub load of
//   normalize_value_scaled_sqf) is read by the 4-lane packet as 4 consecutive cells (a packed gather of V174's strided
//   partial could not index it: PacketNotIndexable, furiosa-opt-lower verify/engine/vector/operand.rs:122).
// - Pass 1 = gather + mean square of s * v (+ EPS) straight to the VRF; pass 2 = gather + s * v / sqrt(ms) (SQF form)
//   committed to the head layout. The stand-alone gather pass is gone (3 post-contraction passes -> 2).
// Every VE op, and the channel order the mean-square reduce walks, is that of normalize_value_scaled_sqf, so v is
// expected bit-identical to V174 (q and k are computed by the same passes).
// Static list: shortening the V tail exposes the v-scale path (q store + v_scale + Sub VRF, 1,333 static) that V174
// hides behind its gather; unpinned, the scheduler then hoists the v_scale load ahead of the Q weight and re-plans the
// head (rope_offset between K and V: the rc shape, +2.3..+2.5k on HW). The pin in `v_scale_after_qm` restores V174's
// first 10 DMAs (order and gates); see its comment for the probes that did not.
use crate::axes::Ns;

/// V partial after the c4 reduce-scatter: slice (g, row / 4), element row % 4 (Qs vocabulary of the contraction).
type KvPart4Q = DmTensor<bf16, Chip, QCluster, m![Qs / 128 % 16, Qs % 64 / 4], m![Qs % 4]>;
/// The same tensor in the head vocabulary: slice (head, Ds / 64, Ds % 64 / 4), element Ds % 4.
type KvPart4 = DmTensor<bf16, Chip, Cluster, m![Ns % 4, Ds / 64, Ds % 64 / 4], m![Ds % 4]>;
type HeadScaleVrf = VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]>;
type HeadMsVrf = VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8]>;

/// `contract_kv15` with the rows walked as [row % 4, row / 4] and row / 4 moved into the slot axis by the reduce.
fn contract_kv15c4(device: &mut Device, x: &XTrf15, weight: KvW15, mask: &ChunkMask) -> KvPart4 {
    let weight: KvW15Q = unsafe { weight.reshape() };
    let out: KvPart4Q = device
        .main
        .begin(weight.view())
        .fetch::<m![Qs % 4, Qs % 64 / 4, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .collect::<m![Qs % 4, Qs % 64 / 4, Hp / 64, Hp / 32 % 2], m![Hp % 32]>()
        .contract_outer::<m![Qs % 4, Qs % 64 / 4, Hp / 64, Gs], m![Hp % 64], _, _, _>(x)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 4, Qs % 64 / 4]>()
        .contract_lane::<m![Qs % 4, Qs % 64 / 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, mask)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32 / QKV_ACTIVATION_SCALE)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Qs / 128 % 16, Qs % 64 / 4], m![Qs % 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();
    unsafe { out.reshape() }
}

/// The v scale as the natural-order f32 VRF (the Sub load of normalize_value_scaled_sqf).
fn vscale_vrf_h(device: &mut Device, s: &HeadDs15) -> HeadScaleVrf {
    device
        .sub
        .begin(s.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// Pass 1: packed ring-64 gather + mean square of s * v, / 256, + EPS -> VRF (Main -> Sub VRF).
fn v_ms_gather4(device: &mut Device, part: &KvPart4, scale: &HeadScaleVrf) -> HeadMsVrf {
    device
        .main
        .begin(part.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<KvHeadsAcrossSlices, m![Ds / 64, Ds % 64 / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 64, Ds % 64 / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::EPS)
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// Pass 2: packed ring-64 gather + s * v / sqrt(ms) (SQF form) -> bf16 head layout.
fn v_norm_gather4(device: &mut Device, part: &KvPart4, scale: &HeadScaleVrf, ms: &HeadMsVrf) -> HeadDs15 {
    device
        .main
        .begin(part.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<KvHeadsAcrossSlices, m![Ds / 64, Ds % 64 / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 64, Ds % 64 / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale)
        .vector_stash()
        .vector_fp_ternary_with_mode(FpTernaryOp::FmaF, TernaryArgMode::Mode021, (ms, 0.0f32))
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

/// The v-scale DMA into a buffer first written by a Main copy of the q-norm weight tile Gs = 0 (a write-after-write
/// ordering pin; the copied values are overwritten by the DMA). It orders the load after the q_rms load and makes the
/// V tail depend on q_rms, which keeps V174's list: rope_off | rms | x | Q | q_scale, k_scale, K (gate: Q contraction)
/// | k_rms, q_rms (gate: K contraction) | V (gate: K gather) | v_scale, q store, k scatter, v scatter.
/// Static-list probes that did not (all compiled on 0.8.1 unless noted):
/// - unpinned: v_scale ahead of the Q weight, rope_offset between K and V (the rc shape);
/// - Sub copy of the gathered k: v_scale between K and V, the V weight gated by the Sub scale VRF, rope_offset moved;
/// - Sub copy of the roped k: rope_offset and q_rms between Q and K;
/// - Sub copy of the pre-RoPE normalized k: q_rms behind V, K gated by a Sub RoPE VRF load, q store last;
/// - Main pass reading the hz_256 zero: rope_offset between K and V again;
/// - Main pass with the K-norm part-1 rms VRF (with or without the q_rms tile, with or without a Sub spacer copy):
///   first 10 DMAs as V174, but the k scatter behind the v scatter (+~1.66k on HW);
/// - Sub copy of the q_rms tile: tail as here, but the q-norm scale pass ahead of the Q contraction (q_scale gated
///   by a RoPE pass, k_scale by the scale pass: the kp3 shape);
/// - Sub cast of a V-weight row (to make v_scale ready at V landing): ICE, npu-isa sfr_impl.rs:868 (cast on Sub);
/// - DMA into a buffer that earlier passes read (k_scale's, the gathered k's): compiler panic, npu-ir-common
///   abstract_buffer.rs:309 (owner mismatch).
fn v_scale_after_qm(
    device: &mut Device,
    q_rms: &DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvScaleDm {
    let mut buf: HeadDs15 = DmTensor::new();
    device
        .main
        .begin(q_rms.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(buf.view_mut());
    let mut buf: KvScaleDm = unsafe { buf.reshape() };
    let vsv: HbmTensorView<'_, bf16, Chip, m![Ps]> = v_weight_scale.view();
    vsv.to_dm_view(&mut device.tdma, buf.view_mut());
    buf
}

/// qkv19_qf7n2 with the V tail replaced (see the block comment above). The q-norm weight is loaded in the body (the
/// DMA normalize_query_p2 issues) and passed to normalize_query_p2_dm, so the v-scale pin can read it.
pub(crate) fn qkv63_vtail(
    device: &mut Device,
    x_trf: &XTrf15,
    mask: &ChunkMask,
    mdm: &MaskDm,
    w: Weights15,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> (HeadGsDs15, HeadDs15, HeadDs15) {
    let (q0, k_weight, v_weight) = w;
    let seed = ramp_seed_from_mask(device, mdm);
    let f = super::rope_h4::rope_ramp_seeded(device, &seed);
    let q_scale_dm: QScaleDm15 = q_weight_scale.to_dm(&mut device.tdma);
    let q_scale: HeadGsDs15 = unsafe { q_scale_dm.reshape() };
    let (rope_rows, one) = super::rope_h4::stage_rope_rows_m18(device, rope_offset, &f, &seed);
    let q = contract_q15(device, x_trf, q0, mask);
    let q = gather_q15(device, q);
    let (qs_vrf, qrms, qrms_vrf, qx_vrf) = super::qkv_h4_norm::normalize_query_p1(device, &q, &q_scale, &one);
    let k_scale_dm: KvScaleDm = k_weight_scale.to_dm(&mut device.tdma);
    let kz0 = qf18_zero_from_grms(device, &qrms);
    let kz = zero_q15_from_kscale_add(device, &k_scale_dm, &kz0);
    let k = contract_kv15z(device, x_trf, k_weight, mask, &kz);
    let (k, k_scale) = gather_kv15s(device, k, k_scale_dm);
    let (ks_vrf, _krms, krms_vrf, kx_vrf) = super::qkv_h4_norm::normalize_key_p1(device, &k, &k_scale, &one);
    let v = contract_kv15c4(device, x_trf, v_weight, mask);
    let zero = hz_256(device, k.view());
    let q_rms_dm: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = q_rms_weight.to_dm(&mut device.tdma);
    let v_scale_dm: KvScaleDm = v_scale_after_qm(device, &q_rms_dm, v_weight_scale);
    let v_scale: HeadDs15 = unsafe { v_scale_dm.reshape() };
    let v_scale = vscale_vrf_h(device, &v_scale);
    let v_ms = v_ms_gather4(device, &v, &v_scale);
    let v = v_norm_gather4(device, &v, &v_scale, &v_ms);
    let k = super::qkv_h4_norm::normalize_key_p2(device, k_rms_weight, &zero, &ks_vrf, &krms_vrf, &kx_vrf);
    let q = super::qkv_h4_norm::normalize_query_p2_dm(device, &q_rms_dm, &zero, &qs_vrf, &qrms_vrf, &qx_vrf);
    let (cos_lo, cos_hi) = super::rope_h4::rope_cos_halves(device, &rope_rows);
    let (sin_lo, sin_hi) = super::rope_h4::rope_sin_halves(device, &rope_rows);
    let q = super::rope_h4::apply_rope_q_zip(device, &q, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k = super::rope_h4::apply_rope_k_zip(device, &k, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    (q, k, v)
}

/// Side variant (NOT called by `sliding_project_qkv`; for a separate A/B): `normalize_input_k15x4` with the input mean
/// square read through the all-live QFull slot type, so its inter-slice reduce no longer runs over the padded Hk # 16
/// axis (0.8.1 prices a reduce over a padded axis ~5.8x on hardware). Bit-exact: slot 15's x is +0.0. Compiled with
/// the Q800 body (ops.rs calling it): static list identical to Q800's (all 14 DMAs, gates and times; makespan 38,008).
#[allow(dead_code)]
pub(crate) fn normalize_input_k15x4f(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (XTrf15, ChunkMask, MaskDm) {
    type S = QSl;
    let w4 = load_w4s(device, rms_weight);
    let w4f: DmTensorView<'_, bf16, Chip, QCluster, X4SFull, m![Hp % 64]> = unsafe { w4.view().reshape() };
    let zero4: DmTensor<f32, Chip, QCluster, X4SFull, m![1 # 8]> = device
        .sub
        .begin(w4f)
        .fetch::<m![Hp % 64 / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp % 64 / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Hp % 64 / 4], m![Hp % 4]>()
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let zero: DmTensor<f32, Chip, QCluster, QFull, m![1 # 8]> = unsafe { zero4.reshape() };
    let zero_vrf: VrfTensor<f32, Chip, QCluster, QFull, m![1 # 8]> =
        device.sub.begin(zero.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_dm: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &w4);
    let x4 = load_x4s(device, x);
    let x: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = fanout_w4s(device, &x4);
    // Relabelled onto the all-live 16-slot type: the reduce no longer runs over the padded Hk # 16 axis (slot 15's x
    // is exactly +0.0, so the sum is unchanged).
    let xf: DmTensorView<'_, bf16, Chip, QCluster, QFull, m![Hp]> = unsafe { x.view().reshape() };
    let mean_square: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(xf)
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, &zero_vrf)
        .vector_intra_slice_reduce::<Hp, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<QSlAll, m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, QCluster, QSlAll, m![1 # 8]> = device
        .sub
        .begin(mean_square.view())
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
    let rms: DmTensor<f32, Chip, QCluster, S, m![1 # 8]> = unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QCluster, S, m![Hp]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, QCluster, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let normalized: DmTensor<bf16, Chip, QCluster, S, m![Hp]> = device
        .main
        .begin(x.view())
        .fetch::<m![Hp / 16], m![Hp % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Hp / 8], m![Hp % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hp / 4], m![Hp % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Hp / 8], m![Hp % 8]>()
        .vector_final()
        .cast::<bf16, m![Hp % 8 # 16]>()
        .commit_trim::<m![Hp % 8]>()
        .commit();
    let hilo = split_hilo15(device, &normalized);
    let hilo: DmTensor<f8e4m3, Chip, QCluster, QFull, m![Gs, Hp]> = unsafe { hilo.reshape() };
    let trf: XTrf15 = device
        .sub
        .begin(hilo.view())
        .fetch::<m![Gs, Hp / 32], m![Hp % 32]>()
        .collect::<m![Gs, Hp / 32], m![Hp % 32]>()
        .to_trf();
    let (mask, mdm) = chunk_mask_dm(device, &x);
    (trf, mask, mdm)
}
