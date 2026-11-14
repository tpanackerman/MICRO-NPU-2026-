//! `S2_I8=A` body of the native T = 256 sliding layer (two 128-token halves, the `f_t128::layer_sliding_pair_v` structure of the fused K / K' launches):
//! the up / gate projections run int8 x int8 (weights int8 per (row x 128-chunk), activations int8 per (token x 128-chunk), both scales applied to the
//! per-slice i32 partial before the f32 inter-slice reduce); AO, GeGLU, the down projection and every norm are COPIES of the production code of
//! `device/ffnc/ffn.rs` (f_t128 instance), so the only numerical change is the up / gate GEMM.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy8, H, Rh};
use crate::device::ffnc::norm::EPS;
use crate::device::ffnc::normp;
use crate::device::ffnc::{C32, C64, Dummy32, E32, E256, Fb8, Fb16, Fc2, Fh2, Fi4, Fi5, G32, Gd8, Rt80, Rt96, Rt240, Rt256, Tk128};

pub(crate) type Cl = m![G32 / 8];
pub(crate) type Sl = m![G32 % 8, C32];
type Cl8 = m![Gd8 / 4];
type Sl8 = m![Gd8 % 4, C64];
type SlOut8 = m![Gd8 % 4, 1 # 64];
type SpSl = m![G32 % 8, Tk128 / 4];
type TCl = m![Tk128 / 64];
type Pre = m![1 # 2, Tk128 % 64];
type PartDm = DmTensor<bf16, Chip, TCl, m![{ Pre }, H / 1920], m![H % 1920]>;

const INVSQRT2: f32 = 0.70710678118f32;
const INV_H: f32 = 1.0 / 3840.0;
const INV127: f32 = 1.0 / 127.0;
const TINY: f32 = 1e-30;

// all8b: Hadamard-32 rotation of every 32-element block of the activation chunk before the int8 quantiser (i8free recipe had32; weights rotated on the host)
axes![Ho32 = 32];
pub(crate) type HadT1 = HbmTensor<bf16, Chip, m![Ho32, E32 % 32]>;
/// rotp: the Hadamard matrix pre-replicated once per Q1 slice (`[ClK, SlK]` outer, 2 KiB contiguous each) so the per-launch load is one contiguous
/// 1 MiB copy instead of 512 reads of the same 2 KiB (static util 0.15)
pub(crate) type HadT = HbmTensor<bf16, Chip, m![Tk128 / 64, Tk128 % 64 / 8, C32, Ho32, E32 % 32]>;
pub(crate) type HadHost = HostTensor<bf16, m![Tk128 / 64, Tk128 % 64 / 8, C32, Ho32, E32 % 32]>;

/// x' = H32 x on each 32-block (unnormalised +-1 H, exact products, f32 accumulation), committed in bf16 in the same [Tk128, E32] order.
fn rot32(
    device: &mut Device,
    x_n: &DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]>,
    had: &HadT1,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]> {
    let h_dm: DmTensor<bf16, Chip, Cl, Sl, m![Ho32, E32 % 32]> = had.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, Cl, Sl, m![Ho32 % 8], m![Ho32 / 8, E32 % 32]> = device
        .sub
        .begin(h_dm.view())
        .fetch::<m![Ho32 % 8, Ho32 / 8, E32 % 32 / 16], m![E32 % 16]>()
        .collect::<m![Ho32 % 8, Ho32 / 8, E32 % 32 / 16], m![E32 % 16]>()
        .to_trf();
    let r: DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32 / 32, Ho32]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .collect::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .contract_outer::<m![Tk128, E32 / 32, Ho32 / 8], m![E32 % 32], _, _, _>(&h_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128, E32 / 32, Ho32 / 8]>()
        .contract_lane::<m![Tk128, E32 / 32, Ho32 / 8], m![Ho32 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ho32 % 8 # 16]>()
        .commit_trim::<m![Ho32 % 8]>()
        .commit();
    unsafe { r.reshape() }
}

/// = `ffn_mod!::prenorm` (f_t128 instance): x [T, 128] (a K chunk of the residual) -> bf16(x / rms * w), rms from the whole row.
fn prenorm(
    device: &mut Device,
    x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]>,
    w: &HbmTensor<bf16, Chip, m![C32, E32]>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]> {
    let ss: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, E32 / 4], m![E32 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<E32, m![Tk128], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cl, m![G32 % 8, Dummy32], m![Tk128, 1 # 8]> = device
        .main
        .begin(ss.view())
        .fetch::<m![Tk128], m![1 # 8]>()
        .collect::<m![Tk128], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![G32 % 8, Dummy32], m![Tk128]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INV_H)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![Tk128], m![1 # 8]>().collect::<m![Tk128], m![1 # 8]>().to_vrf();
    let w_dm: DmTensor<bf16, Chip, Cl, Sl, m![E32]> = w.to_dm(&mut device.tdma);
    let w_vrf: VrfTensor<f32, Chip, Cl, Sl, m![E32]> = device
        .sub
        .begin(w_dm.view())
        .fetch::<m![E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![E32 / 8], m![E32 % 8]>()
        .to_vrf();
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_final()
        .cast::<bf16, m![E32 % 8 # 16]>()
        .commit_trim::<m![E32 % 8]>()
        .commit()
}

/// Pre-FF RMSNorm + int8 quantiser of one 128-token half in the K-split replicated layout (slice (g, c) = chunk c of every token):
/// x_n = production `prenorm` (bf16, bit-identical to the bf16 path), s = sqrt(max_j x_n_j^2 + 1e-30) / 127 per (token, chunk),
/// x_q = clip(round(x_n / s), +-127) (i8). Returns (x_q, s): s IS the dequantisation factor of this chunk's partial (x_n is already normalised).
/// Every VRF is used by exactly one pass (rms + gamma in the normalise pass, s in the quantise pass). Pad chunks / pad tokens: x_n = 0, s ~ 8e-18, q = 0.
pub(crate) fn prenorm_q8(
    device: &mut Device,
    x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]>,
    w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    had: &HadT1,
) -> (DmTensor<i8, Chip, Cl, Sl, m![Tk128, E32]>, DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]>) {
    let x_n = prenorm(device, x_dm, w);
    let x_n = rot32(device, &x_n, had);
    let am2: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, E32 / 4], m![E32 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<E32, m![Tk128], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // TINY is added BEFORE the square root (no sqrt(0) on the pad chunks / pad tokens)
    let s: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = device
        .main
        .begin(am2.view())
        .fetch::<m![Tk128], m![1 # 8]>()
        .collect::<m![Tk128], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, TINY)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INV127)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![Tk128], m![1 # 8]>().collect::<m![Tk128], m![1 # 8]>().to_vrf();
    let q: DmTensor<i8, Chip, Cl, Sl, m![Tk128, E32]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &s_vrf)
        .vector_widen_concat::<m![Tk128, E32 / 8], m![E32 % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Min, 127i32)
        .vector_clip(ClipBinaryOpI32::Max, -127i32)
        .vector_final()
        .cast::<i8, m![E32 % 8 # 32]>()
        .commit_trim::<m![E32 % 8]>()
        .commit();
    (q, s)
}

// ---------------------------------------------------------------------------------------------------- R15 layout Q1: the pre-FF quantiser ONCE, on the UN-replicated K-split layout
// Same slice structure as the replicated FFN layout (C32 = innermost slice axis, so the 32-way rms all-reduce and every per-(token, chunk) pass see exactly the operands of
// `prenorm_q8`), but the 8 outer slice positions hold 8 different token groups instead of 8 copies: cluster = token / 64, slice = (token % 64 / 8, chunk), 8 tokens per slice.
// Every pass handles 8 x 128 elements per slice instead of 128 x 128 (16x less VE / TU work), the int8 rows and f32 chunk scales are stored chunk-major
// ([C32, Rh, T, E32] / [C32, Rh, T, 1 # 8]: 1 KiB / 256 B contiguous per slice), and the up / gate operands are loaded from there replicated 16x (16 KiB i8 + 4 KiB f32
// contiguous per slice instead of 128 pieces of 256 B bf16). BIT class by construction: the per-token sums use the same intra-slice reduce and the same 32-slice inter-slice
// reduce; every other pass is per element / per (token, chunk).
type ClK = m![Tk128 / 64];
type SlK = m![Tk128 % 64 / 8, C32];
type SlKr = m![Tk128 % 64 / 8, Dummy32];
pub(crate) type XqHbm = HbmTensor<i8, Chip, m![C32, Rh, Tk128, E32]>;
/// (the scale rows keep their 8-lane f32 pad as real `Dummy8` lanes in HBM: a padded `1 # 8` element is stored as 4-byte pieces, 32 B runs are not)
pub(crate) type SqHbm = HbmTensor<f32, Chip, m![C32, Rh, Tk128, Dummy8]>;

/// = `rot32` on the Q1 layout
fn rot32_k(device: &mut Device, x_n: &DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32]>, had: &HadT) -> DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32]> {
    let h_dm: DmTensor<bf16, Chip, ClK, SlK, m![Ho32, E32 % 32]> = had.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, ClK, SlK, m![Ho32 % 8], m![Ho32 / 8, E32 % 32]> = device
        .sub
        .begin(h_dm.view())
        .fetch::<m![Ho32 % 8, Ho32 / 8, E32 % 32 / 16], m![E32 % 16]>()
        .collect::<m![Ho32 % 8, Ho32 / 8, E32 % 32 / 16], m![E32 % 16]>()
        .to_trf();
    let r: DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32 / 32, Ho32]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .collect::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .contract_outer::<m![Tk128 % 8, E32 / 32, Ho32 / 8], m![E32 % 32], _, _, _>(&h_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128 % 8, E32 / 32, Ho32 / 8]>()
        .contract_lane::<m![Tk128 % 8, E32 / 32, Ho32 / 8], m![Ho32 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ho32 % 8 # 16]>()
        .commit_trim::<m![Ho32 % 8]>()
        .commit();
    unsafe { r.reshape() }
}

/// = `prenorm` on the Q1 layout
fn prenorm_k(
    device: &mut Device,
    x_dm: &DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32]>,
    w: &HbmTensor<bf16, Chip, m![C32, E32]>,
) -> DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32]> {
    let ss: DmTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> = device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 8, E32 / 4], m![E32 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<E32, m![Tk128 % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, ClK, SlKr, m![Tk128 % 8, 1 # 8]> = device
        .main
        .begin(ss.view())
        .fetch::<m![Tk128 % 8], m![1 # 8]>()
        .collect::<m![Tk128 % 8], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<SlKr, m![Tk128 % 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INV_H)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![Tk128 % 8], m![1 # 8]>().collect::<m![Tk128 % 8], m![1 # 8]>().to_vrf();
    let w_dm: DmTensor<bf16, Chip, ClK, SlK, m![E32]> = w.to_dm(&mut device.tdma);
    let w_vrf: VrfTensor<f32, Chip, ClK, SlK, m![E32]> = device
        .sub
        .begin(w_dm.view())
        .fetch::<m![E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![E32 / 8], m![E32 % 8]>()
        .to_vrf();
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 8, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_final()
        .cast::<bf16, m![E32 % 8 # 16]>()
        .commit_trim::<m![E32 % 8]>()
        .commit()
}

/// Q1 quantiser of one half: mid rows (HBM `[T, C32, E32]`, zero pad chunks) -> int8 rows + f32 chunk scales stored into half `half` of `xq` / `sq`
/// (= `prenorm_q8` per element; chunks 30 / 31 give q = 0, s = sqrt(1e-30) / 127 exactly as before).
fn q1_half(
    device: &mut Device,
    x_hbm: &HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
    w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    had: &HadT,
    xq: &mut XqHbm,
    sq: &mut SqHbm,
    half: usize,
) {
    let x_dm: DmTensor<bf16, Chip, ClK, SlK, m![Tk128 % 8, E32]> = x_hbm.to_dm(&mut device.tdma);
    let x_n = prenorm_k(device, &x_dm, w);
    let x_n = rot32_k(device, &x_n, had);
    let am2: DmTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 8, E32 / 4], m![E32 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<E32, m![Tk128 % 8], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s: DmTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> = device
        .main
        .begin(am2.view())
        .fetch::<m![Tk128 % 8], m![1 # 8]>()
        .collect::<m![Tk128 % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, TINY)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INV127)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_vrf: VrfTensor<f32, Chip, ClK, SlK, m![Tk128 % 8, 1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![Tk128 % 8], m![1 # 8]>().collect::<m![Tk128 % 8], m![1 # 8]>().to_vrf();
    let q: DmTensor<i8, Chip, ClK, SlK, m![Tk128 % 8, E32]> = device
        .main
        .begin(x_n.view())
        .fetch::<m![Tk128 % 8, E32 / 16], m![E32 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 8, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &s_vrf)
        .vector_widen_concat::<m![Tk128 % 8, E32 / 8], m![E32 % 8]>()
        .vector_fp_to_fxp(31)
        .vector_clip(ClipBinaryOpI32::Min, 127i32)
        .vector_clip(ClipBinaryOpI32::Max, -127i32)
        .vector_final()
        .cast::<i8, m![E32 % 8 # 32]>()
        .commit_trim::<m![E32 % 8]>()
        .commit();
    q.view().to_hbm_view(&mut device.tdma, xq.view_mut().tile::<m![Rh], 1, m![C32, Rh = 1 #{!} 2, Tk128, E32]>(half));
    let sv: DmTensorView<'_, f32, Chip, ClK, SlK, m![Tk128 % 8, Dummy8]> = unsafe { s.view().reshape() };
    sv.to_hbm_view(&mut device.tdma, sq.view_mut().tile::<m![Rh], 1, m![C32, Rh = 1 #{!} 2, Tk128, Dummy8]>(half));
}

/// int8 weight tile (DM) -> TRF (Sub); 30 KiB per slice (two tiles fit the 64 KiB TRF).
pub(crate) fn ffu_trf_i8(device: &mut Device, w: &DmTensor<i8, Chip, Cl, Sl, m![Rt240, E32]>) -> TrfTensor<i8, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]> {
    device
        .sub
        .begin(w.view())
        .fetch::<m![Rt240 % 8, Rt240 / 8, E32 / 32], m![E32 % 32]>()
        .collect::<m![Rt240 % 8, Rt240 / 8, E32 / 32], m![E32 % 32]>()
        .to_trf()
}

/// Per-slice f32 VRF of the per-(row x chunk) weight scales (global NVFP4 scale folded in by the host).
/// The weight scales pass through one tiny Main pass `ws x one` (`one` = exactly 1.0, derived from the token scales of half 1) before their VRF load:
/// a VRF loaded straight from the HBM tensor has no data dependency on the activations, the scheduler hoists it ahead of the AO core and its
/// 960 B/slice then blocks the AO norm's 7,680 B VRF (lean compiles 1-3: "Cannot find evict target for Vrf"; diagnostic `ffn_a` / `ffn_b`, runs/lean5.out).
pub(crate) fn ws_vrf(
    device: &mut Device,
    ws: &DmTensor<f32, Chip, Cl, Sl, m![Rt240]>,
    one: &VrfTensor<f32, Chip, Cl, Sl, m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cl, Sl, m![Rt240]> {
    let ws1: DmTensor<f32, Chip, Cl, Sl, m![Rt240]> = device
        .main
        .begin(ws.view())
        .fetch::<m![Rt240 / 8], m![Rt240 % 8]>()
        .collect::<m![Rt240 / 8], m![Rt240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt240 / 4], m![Rt240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), one)
        .vector_widen_concat::<m![Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .commit_trim::<m![Rt240 % 8]>()
        .commit();
    device.sub.begin(ws1.view()).fetch::<m![Rt240 / 8], m![Rt240 % 8]>().collect::<m![Rt240 / 8], m![Rt240 % 8]>().to_vrf()
}

/// Exactly 1.0 per slice (`0 x s + 1`, s finite and > 0), computed from the token scales of one half: the activation dependency of `ws_vrf`.
pub(crate) fn one_vrf(device: &mut Device, s: &DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]>) -> VrfTensor<f32, Chip, Cl, Sl, m![1 # 8]> {
    let one: DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> = device
        .main
        .begin(s.view().tile::<m![Tk128], 1, m![Tk128 = 1 # 128, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device.sub.begin(one.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// The same 1.0 on the slices of the down layout (Cl8 / Sl8 = the same 512 slices under the other mapping; every value is 1.0).
fn one8_vrf(device: &mut Device, s: &DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]>) -> VrfTensor<f32, Chip, Cl8, Sl8, m![1 # 8]> {
    let one: DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> = device
        .main
        .begin(s.view().tile::<m![Tk128], 1, m![Tk128 = 1 # 128, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0f32)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let one: DmTensor<f32, Chip, Cl8, Sl8, m![1 # 8]> = unsafe { one.reshape() };
    device.sub.begin(one.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}



/// int8 x int8 contraction of one TRF-resident tile against one token half, both scales on the per-slice partial (Mul0 = weight scale per row,
/// Mul1 = token scale per (token, chunk)), f32 reduce over the 32 chunk slices STRAIGHT into the token-spread pair buffer (half `half`, sub-tile `h`),
/// the production `ffu_con_into` target. No intermediate out tile and no DMA re-shard: an out tile re-used by the next contraction's commit while the
/// re-shard DMA still read it was a write-after-DMA-read race (w8a8 `scripts/war4.py`, lab job 198925: first ~40 tokens of each half wrong).
#[allow(clippy::too_many_arguments)]
fn ffu_con_i8_sp(
    device: &mut Device,
    x_q: &DmTensor<i8, Chip, Cl, Sl, m![Tk128, E32]>,
    w_trf: &TrfTensor<i8, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]>,
    ws: &VrfTensor<f32, Chip, Cl, Sl, m![Rt240]>,
    tk: &VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]>,
    buf: &mut DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fh2, Tk128 % 4, Rt240]>,
    h: usize,
    half: usize,
) {
    device
        .main
        .begin(x_q.view())
        .fetch::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .collect::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .contract_outer::<m![Tk128, E32 / 64, Rt240 / 8], m![E32 % 64], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128, Rt240 / 8]>()
        .contract_lane::<m![Tk128, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_to_fp(31)
        .vector_narrow_split::<m![Tk128, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), ws)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), tk)
        .vector_widen_concat::<m![Tk128, Rt240 / 8], m![Rt240 % 8]>()
        .vector_inter_slice_reduce::<SpSl, m![Tk128 % 4, Rt240 / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Rt240 % 8 # 16]>()
        .commit_trim::<m![Rt240 % 8]>()
        .commit_view(
            buf.view_mut()
                .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Fh2, Tk128 % 4, Rt240]>(half)
                .tile::<m![Fh2], 1, m![Rh = 1 #{!} 2, Fh2 = 1 #{!} 2, Tk128 % 4, Rt240]>(h),
        );
}

// ---------------------------------------------------------------------------------------------------- production copies (f_t128 instance of ffn_mod!)

/// = `ffn_mod!::scalar_vrf8`
fn scalar_vrf8(device: &mut Device, gs: &HbmTensor<f32, Chip, impl M>) -> VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> {
    let gs_dm: DmTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> = gs.to_dm(&mut device.tdma);
    device.sub.begin(gs_dm.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// = `ffn_mod!::ffd_dec8` + the `one8` scheduling anchor (same values)
fn ffd_dec8(
    device: &mut Device,
    w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]>,
    scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]>,
    one8: &VrfTensor<f32, Chip, Cl8, Sl8, m![1 # 8]>,
) -> TrfTensor<bf16, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> {
    // i8 body only: the block scales pass through `x one8` (exactly 1.0, activation-dependent) before their VRF load, so the 6 KiB scale VRF of a down
    // tile is not hoisted into the int8 up / gate phase (whose epilogue holds ~5 KiB of VRF); the decoded bf16 values are unchanged
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_f: DmTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
        .main
        .begin(scale_view)
        .fetch::<m![Rt96], m![Rt256 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt96, Rt256 / 16 / 4], m![Rt256 / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), one8)
        .vector_widen_concat::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .vector_final()
        .commit_trim::<m![Rt256 / 16 % 8]>()
        .commit();
    let scale_vrf: VrfTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
        .sub
        .begin(scale_f.view())
        .fetch::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .to_vrf();
    let w_bf: DmTensor<bf16, Chip, Cl8, Sl8, m![Rt96, Rt256]> = device
        .main
        .begin(w4_dm.view())
        .fetch::<m![Rt96], m![Rt256]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt96, Rt256 / 4], m![Rt256 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt256 % 8 # 16]>()
        .commit_trim::<m![Rt256 % 8]>()
        .commit();
    device
        .sub
        .begin(w_bf.view())
        .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
        .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
        .to_trf()
}

/// Direct TRF load of once-decoded BF16 checkpoint weights. Reduction and global scale stay unchanged.
fn ffd_trf_bf16(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, Cl8, Sl8, m![Rt96, Rt256]>,
) -> TrfTensor<bf16, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> {
    device.sub.begin(w.view())
        .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
        .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
        .to_trf()
}

/// = `ffn_mod!::ffd_con8`
#[allow(clippy::too_many_arguments)]
fn ffd_con8(
    device: &mut Device,
    x_dm: &DmTensor<bf16, Chip, Cl8, Sl8, m![Tk128, Rt256]>,
    w_trf: &TrfTensor<bf16, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]>,
    buf: &mut DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]>,
    i: usize,
    half: usize,
) {
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, Rt256 / 16], m![Rt256 % 16]>()
        .collect::<m![Tk128, Rt256 / 16], m![Rt256 % 16]>()
        .contract_outer::<m![Tk128, Rt256 / 32, Rt96 / 8], m![Rt256 % 32], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128, Rt96 / 8]>()
        .contract_lane::<m![Tk128, Rt96 / 8], m![Rt96 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SlOut8, m![Tk128, Rt96 / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, Rt96 / 8, Rt96 / 4 % 2], m![Rt96 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
        .vector_widen_concat::<m![Tk128, Rt96 / 8], m![Rt96 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt96 % 8 # 16]>()
        .commit_trim::<m![Rt96 % 8]>()
        .commit_view(buf.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, Fi5, Rt96]>(half).tile::<m![Fi5], 1, m![Rh = 1 #{!} 2, Tk128, Fi5 = 1 #{!} 5, Rt96]>(i));
}

/// = `ffn_mod!::geglu_pair_sp` (R15 overlap: each half has its OWN h buffer, so the down hop of half 0 does not wait for the GeGLU of half 1;
/// same arithmetic as `geglu_pair_sp_rh`, only the destination handle differs)
fn geglu_pair_sp_h(
    device: &mut Device,
    up_p: DmTensor<bf16, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt240]>,
    gate_p: &DmTensor<bf16, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt240]>,
    h_buf: &mut DmTensor<bf16, Chip, Cl, SpSl, m![Fc2, Fh2, Tk128 % 4, Rt256]>,
    c2: usize,
) {
    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt240]> = device
        .sub
        .begin(gate_p.view())
        .fetch::<m![Fh2, Tk128 % 4, Rt240 / 16], m![Rt240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Fh2, Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Fh2, Tk128 % 4, Rt240 / 4], m![Rt240 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![Fh2, Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .commit_trim::<m![Rt240 % 8]>()
        .commit();
    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt256 = 240]> = unsafe { gelu.reshape() };
    let up_sp: DmTensor<bf16, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt256 = 240]> = unsafe { up_p.reshape() };
    let gelu_vrf: VrfTensor<f32, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt256 = 240]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![Fh2, Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .collect::<m![Fh2, Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .to_vrf();
    device
        .main
        .begin(up_sp.view())
        .fetch::<m![Fh2, Tk128 % 4, Rt256 = 240 / 16], m![Rt256 = 240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Fh2, Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Fh2, Tk128 % 4, Rt256 = 240 / 4], m![Rt256 = 240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![Fh2, Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt256 = 240 % 8 # 16]>()
        .commit_trim::<m![Rt256 = 240 % 8]>()
        .commit_view(
            h_buf
                .view_mut()
                .tile::<m![Fc2], 1, m![Fc2 = 1 #{!} 2, Fh2, Tk128 % 4, Rt256]>(c2)
                .tile::<m![Rt256], 240, m![Fc2 = 1 #{!} 2, Fh2, Tk128 % 4, Rt256 = 240 #{!} 256]>(0),
        );
}

/// = `ffn_mod!::down_tail8_half`
fn down_tail8_half(
    device: &mut Device,
    dbuf: &DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]>,
    half: usize,
    res: &PartDm,
    post_w: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
) -> PartDm {
    let mut yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, Gd8 / 4], m![Gd8 % 4, Fi5, Rt96]> = DmTensor::new();
    dbuf.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, Fi5, Rt96]>(half).to_dm_view(&mut device.tdma, yb.view_mut());
    let yb: PartDm = unsafe { yb.reshape() };
    let summed = normp::p1920::norm_residual::<TCl, Pre>(device, &yb, res, post_w);
    normp::p1920::scale_rows::<TCl, Pre>(device, &summed, layer_scalar)
}

// ---------------------------------------------------------------------------------------------------- the i8 FFN body and layer

/// (T = 256, `S2_I8=A`) `ffn_core_d8_pair` with int8 up / gate: every int8 weight tile is loaded into the TRF ONCE and contracted against both halves.
#[allow(clippy::too_many_arguments)]
fn ffn_core_i8_pair(
    device: &mut Device,
    x_hbm0: &HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
    x_hbm1: &HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
    res0: &PartDm,
    res1: &PartDm,
    pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    up_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    gate_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    down_w: &HbmTensor<bf16, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_w: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    had: &HadT,
) -> (PartDm, PartDm) {
    // R15 layout Q1: quantise once (un-replicated K-split layout), chunk-major int8 / scale stores, 1-byte replicated loads
    let mut xq_hbm: XqHbm = HbmTensor::new();
    let mut sq_hbm: SqHbm = HbmTensor::new();
    q1_half(device, x_hbm0, pre_w, had, &mut xq_hbm, &mut sq_hbm, 0);
    q1_half(device, x_hbm1, pre_w, had, &mut xq_hbm, &mut sq_hbm, 1);
    let xq0: DmTensor<i8, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(0).to_dm(&mut device.tdma);
    let tk0: DmTensor<f32, Chip, Cl, Sl, m![Tk128, Dummy8]> = sq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, Dummy8]>(0).to_dm(&mut device.tdma);
    let tk0: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = unsafe { tk0.reshape() };
    let xq1: DmTensor<i8, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(1).to_dm(&mut device.tdma);
    let tk1: DmTensor<f32, Chip, Cl, Sl, m![Tk128, Dummy8]> = sq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, Dummy8]>(1).to_dm(&mut device.tdma);
    let tk1: DmTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> = unsafe { tk1.reshape() };
    // from half 1: the AO core of half 1 (7,680 B norm VRF) is independent of half 0's quantiser, so a half-0 anchor still lets the weight scales in early
    let one = one_vrf(device, &tk1);

    // R15 overlap (ffnpipe h1d1 in the fused int8 body): ONE h buffer PER HALF. With one shared handle the hop of half 0 (h -> HBM -> 8x replicate)
    // waited for the last GeGLU of half 1 (an ordering through the handle, not a data dependency); with two handles it is issued right after the
    // GeGLU of half 0 and runs on the DMA queue under the GeGLU of half 1 and the first down decode. Values are unchanged.
    let mut h_buf0: DmTensor<bf16, Chip, Cl, SpSl, m![Fc2, Fh2, Tk128 % 4, Rt256]> = DmTensor::new();
    h_buf0.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let mut h_buf1: DmTensor<bf16, Chip, Cl, SpSl, m![Fc2, Fh2, Tk128 % 4, Rt256]> = DmTensor::new();
    h_buf1.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! xh {
        ($hb:ident) => {{
            let h: HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, Tk128, Rt256]> = $hb.to_hbm(&mut device.tdma);
            let hv: HbmTensorView<'_, bf16, Chip, m![C64, Tk128, Rt256]> = unsafe { h.view().reshape() };
            let x_h: DmTensor<bf16, Chip, Cl8, Sl8, m![Tk128, Rt256]> = hv.to_dm(&mut device.tdma);
            x_h
        }};
    }
    macro_rules! mcon {
        ($q:ident, $ws:ident, $i:expr, $buf:ident, $h:expr) => {{
            let w_dm: DmTensor<i8, Chip, Cl, Sl, m![Rt240, E32]> = $q.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i).to_dm(&mut device.tdma);
            let ws_dm: DmTensor<f32, Chip, Cl, Sl, m![Rt240]> = $ws.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240]>($i).to_dm(&mut device.tdma);
            let t = ffu_trf_i8(device, &w_dm);
            let wsv = ws_vrf(device, &ws_dm, &one);
            // the token-chunk scale VRF of one half is 4 KiB (half the VRF): loaded per half, right before its contraction
            {
                let tkv: VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> =
                    device.sub.begin(tk0.view()).fetch::<m![Tk128], m![1 # 8]>().collect::<m![Tk128], m![1 # 8]>().to_vrf();
                ffu_con_i8_sp(device, &xq0, &t, &wsv, &tkv, &mut $buf, $h, 0);
            }
            {
                let tkv: VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> =
                    device.sub.begin(tk1.view()).fetch::<m![Tk128], m![1 # 8]>().collect::<m![Tk128], m![1 # 8]>().to_vrf();
                ffu_con_i8_sp(device, &xq1, &t, &wsv, &tkv, &mut $buf, $h, 1);
            }
        }};
    }
    macro_rules! mload {
        ($q:ident, $ws:ident, $i:expr) => {{
            let w_dm: DmTensor<i8, Chip, Cl, Sl, m![Rt240, E32]> = $q.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i).to_dm(&mut device.tdma);
            let ws_dm: DmTensor<f32, Chip, Cl, Sl, m![Rt240]> = $ws.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240]>($i).to_dm(&mut device.tdma);
            let t = ffu_trf_i8(device, &w_dm);
            let wsv = ws_vrf(device, &ws_dm, &one);
            (t, wsv)
        }};
    }
    #[allow(unused_macros)]
    macro_rules! mc {
        ($t:ident, $wsv:ident, $buf:ident, $h:expr, $xq:ident, $tk:ident, $half:expr) => {{
            let tkv: VrfTensor<f32, Chip, Cl, Sl, m![Tk128, 1 # 8]> =
                device.sub.begin($tk.view()).fetch::<m![Tk128], m![1 # 8]>().collect::<m![Tk128], m![1 # 8]>().to_vrf();
            ffu_con_i8_sp(device, &$xq, &$t, &$wsv, &tkv, &mut $buf, $h, $half);
        }};
    }
    macro_rules! hcopy {
        ($buf:ident, $half:expr) => {{
            let mut t: DmTensor<bf16, Chip, Cl, SpSl, m![Fh2, Tk128 % 4, Rt240]> = DmTensor::new();
            $buf.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Fh2, Tk128 % 4, Rt240]>($half).to_dm_view(&mut device.tdma, t.view_mut());
            t
        }};
    }
    // ---- quad 0 (up / gate tiles 0, 1)
    let mut up_p: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fh2, Tk128 % 4, Rt240]> = DmTensor::new();
    let mut gate_p: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fh2, Tk128 % 4, Rt240]> = DmTensor::new();
    mcon!(up_q, up_ws, 0, up_p, 0);
    mcon!(gate_q, gate_ws, 0, gate_p, 0);
    mcon!(up_q, up_ws, 1, up_p, 1);
    mcon!(gate_q, gate_ws, 1, gate_p, 1);
    {
        let up0 = hcopy!(up_p, 0);
        let gate0 = hcopy!(gate_p, 0);
        geglu_pair_sp_h(device, up0, &gate0, &mut h_buf0, 0);
        let up1 = hcopy!(up_p, 1);
        let gate1 = hcopy!(gate_p, 1);
        geglu_pair_sp_h(device, up1, &gate1, &mut h_buf1, 0);
    }
    // ---- quad 1 (tiles 2, 3); the down hop of each half is issued right after that half's last GeGLU
    let mut up_p1: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fh2, Tk128 % 4, Rt240]> = DmTensor::new();
    let mut gate_p1: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fh2, Tk128 % 4, Rt240]> = DmTensor::new();
    mcon!(up_q, up_ws, 2, up_p1, 0);
    mcon!(gate_q, gate_ws, 2, gate_p1, 0);
    let x_h0;
    let x_h1;
    {
        // R15 overlap P8 (end skew): (up 3, gate 3) both TRF-resident, half 0 first -> GeGLU h0 -> the h0 down hop (DMA) runs under the half-1 contractions
        // P8b: the weight-scale VRF is re-derived per half (a VRF live across the GeGLU's 7.5 KiB VRF cannot be evicted); the TRF tiles stay shared
        let wsa_dm: DmTensor<f32, Chip, Cl, Sl, m![Rt240]> = up_ws.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240]>(3).to_dm(&mut device.tdma);
        let wsb_dm: DmTensor<f32, Chip, Cl, Sl, m![Rt240]> = gate_ws.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240]>(3).to_dm(&mut device.tdma);
        let wa_dm: DmTensor<i8, Chip, Cl, Sl, m![Rt240, E32]> = up_q.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>(3).to_dm(&mut device.tdma);
        let ta = ffu_trf_i8(device, &wa_dm);
        let wb_dm: DmTensor<i8, Chip, Cl, Sl, m![Rt240, E32]> = gate_q.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>(3).to_dm(&mut device.tdma);
        let tb = ffu_trf_i8(device, &wb_dm);
        {
            let wa = ws_vrf(device, &wsa_dm, &one);
            mc!(ta, wa, up_p1, 1, xq0, tk0, 0);
        }
        {
            let wb = ws_vrf(device, &wsb_dm, &one);
            mc!(tb, wb, gate_p1, 1, xq0, tk0, 0);
        }
        let up0 = hcopy!(up_p1, 0);
        let gate0 = hcopy!(gate_p1, 0);
        geglu_pair_sp_h(device, up0, &gate0, &mut h_buf0, 1);
        x_h0 = xh!(h_buf0);
        {
            let wa = ws_vrf(device, &wsa_dm, &one);
            mc!(ta, wa, up_p1, 1, xq1, tk1, 1);
        }
        {
            let wb = ws_vrf(device, &wsb_dm, &one);
            mc!(tb, wb, gate_p1, 1, xq1, tk1, 1);
        }
        let up1 = hcopy!(up_p1, 1);
        let gate1 = hcopy!(gate_p1, 1);
        geglu_pair_sp_h(device, up1, &gate1, &mut h_buf1, 1);
        x_h1 = xh!(h_buf1);
    }
    let d_gs = scalar_vrf8(device, dgs);
    let mut dbuf: DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]> = DmTensor::new();
    macro_rules! dtile {
        ($i:expr) => {{
            let w_dm: DmTensor<bf16, Chip, Cl8, Sl8, m![Rt96, Rt256]> = down_w.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>($i).to_dm(&mut device.tdma);
            let t = ffd_trf_bf16(device, &w_dm);
            ffd_con8(device, &x_h0, &t, &d_gs, &mut dbuf, $i, 0);
            ffd_con8(device, &x_h1, &t, &d_gs, &mut dbuf, $i, 1);
        }};
    }
    dtile!(0);
    dtile!(1);
    dtile!(2);
    dtile!(3);
    dtile!(4);
    let o0 = down_tail8_half(device, &dbuf, 0, res0, post_w, layer_scalar);
    let o1 = down_tail8_half(device, &dbuf, 1, res1, post_w, layer_scalar);
    (o0, o1)
}

/// (T = 256, `S2_I8=A`) = `f_t128::layer_sliding_pair_v` with the int8 up / gate body: AO of both halves from views of the attention output handles,
/// mid rows through an HBM scratch (30 real chunks + 2 zero chunks), FFN, residual rows stored to `res_out`.
/// R15 gapimpl F1: the attention rows arrive as the e4m3 AO8 input (`[C32, T, E]`, cast in the attention epilogue) -> `sliding_core_halfv8_t128`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_pair_v_i8(
    device: &mut Device,
    attn0: HbmTensorView<'_, f8e4m3, Chip, m![C32, Tk128, E32]>,
    attn1: HbmTensorView<'_, f8e4m3, Chip, m![C32, Tk128, E32]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    up_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    gate_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    down_w: &HbmTensor<bf16, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    had: &HadT,
) {
    // one zero pad for both halves, written BEFORE the AO cores: a memset allocated after an AO re-shard DMA reused that DMA's source memory
    // with no ordering (war4.py, Arm D lean: slack 0)
    let mut zero_pad: DmTensor<bf16, Chip, TCl, m![{ Pre }, 1 # 2], m![C32 = 2, E32]> = DmTensor::new();
    zero_pad.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! half {
        ($half:expr, $attn:expr) => {{
            let res: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let res: PartDm = unsafe { res.reshape() };
            let mid = crate::device::ffnc::ao::sliding_core_halfv8_t128(device, $attn, w0, w1, w2, s0, s1, s2, post_attn_norm, &res);
            // mid residual rows -> HBM scratch [T, C32, E32] (30 real chunks + 2 explicit zero chunks)
            let mut mid_hbm: HbmTensor<bf16, Chip, m![Tk128, C32, E32]> = HbmTensor::new();
            normp::p1920::store_rows::<Tk128, TCl, Pre>(device, &mid, &mut mid_hbm);
            zero_pad.view().to_hbm_view(&mut device.tdma, mid_hbm.view_mut().tile::<m![C32], 2, m![Tk128, C32 = 2 #{!} 32, E32]>(30));
            (mid, mid_hbm)
        }};
    }
    let (mid0, mid_hbm0) = half!(0, attn0);
    let (mid1, mid_hbm1) = half!(1, attn1);
    let (out0, out1) = ffn_core_i8_pair(
        device, &mid_hbm0, &mid_hbm1, &mid0, &mid1, pre_w, up_q, up_ws, gate_q, gate_ws, down_w, down_s, dgs, post_ff, layer_scalar, had,
    );
    macro_rules! store_half {
        ($out:expr, $half:expr) => {{
            let xv: DmTensorView<'_, bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { $out.view().reshape() };
            xv.to_hbm_view(
                &mut device.tdma,
                res_out
                    .view_mut()
                    .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, C32, E32]>($half)
                    .tile::<m![C32], 30, m![Rh = 1 #{!} 2, Tk128, C32 = 30 #{!} 32, E32]>(0),
            );
        }};
    }
    store_half!(out0, 0);
    store_half!(out1, 1);
}

/// all8b S3: the int8 FULL-attention layer (T = 256): full AO (E256, bf16 production core per half) + the int8 FFN body (identical to the sliding one:
/// the FFN shapes do not depend on the attention type).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_full_pair_i8(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E256]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    up_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    gate_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    down_w: &HbmTensor<bf16, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    had: &HadT,
) {
    let mut zero_pad: DmTensor<bf16, Chip, TCl, m![{ Pre }, 1 # 2], m![C32 = 2, E32]> = DmTensor::new();
    zero_pad.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! half {
        ($half:expr) => {{
            let res: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let res: PartDm = unsafe { res.reshape() };
            let mid = crate::device::ffnc::ao::full_core_half_t128(device, attn, $half, w0, w1, w2, s0, s1, s2, post_attn_norm, &res);
            let mut mid_hbm: HbmTensor<bf16, Chip, m![Tk128, C32, E32]> = HbmTensor::new();
            normp::p1920::store_rows::<Tk128, TCl, Pre>(device, &mid, &mut mid_hbm);
            zero_pad.view().to_hbm_view(&mut device.tdma, mid_hbm.view_mut().tile::<m![C32], 2, m![Tk128, C32 = 2 #{!} 32, E32]>(30));
            (mid, mid_hbm)
        }};
    }
    let (mid0, mid_hbm0) = half!(0);
    let (mid1, mid_hbm1) = half!(1);
    let (out0, out1) = ffn_core_i8_pair(
        device, &mid_hbm0, &mid_hbm1, &mid0, &mid1, pre_w, up_q, up_ws, gate_q, gate_ws, down_w, down_s, dgs, post_ff, layer_scalar, had,
    );
    macro_rules! store_half {
        ($out:expr, $half:expr) => {{
            let xv: DmTensorView<'_, bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { $out.view().reshape() };
            xv.to_hbm_view(
                &mut device.tdma,
                res_out
                    .view_mut()
                    .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, C32, E32]>($half)
                    .tile::<m![C32], 30, m![Rh = 1 #{!} 2, Tk128, C32 = 30 #{!} 32, E32]>(0),
            );
        }};
    }
    store_half!(out0, 0);
    store_half!(out1, 1);
}

/// all8b S2: the int8 SLIDING layer from a whole [Rh, T, C32, E32] attention tensor (native T = 256 launch): AO8 half cores + the int8 FFN body.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_pair_i8(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
    up_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    gate_q: &HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_ws: &HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    down_w: &HbmTensor<bf16, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    had: &HadT,
) {
    let mut zero_pad: DmTensor<bf16, Chip, TCl, m![{ Pre }, 1 # 2], m![C32 = 2, E32]> = DmTensor::new();
    zero_pad.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! half {
        ($half:expr) => {{
            let res: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let res: PartDm = unsafe { res.reshape() };
            let mid = crate::device::ffnc::ao::sliding_core_half8_t128(device, attn, $half, w0, w1, w2, s0, s1, s2, post_attn_norm, &res);
            let mut mid_hbm: HbmTensor<bf16, Chip, m![Tk128, C32, E32]> = HbmTensor::new();
            normp::p1920::store_rows::<Tk128, TCl, Pre>(device, &mid, &mut mid_hbm);
            zero_pad.view().to_hbm_view(&mut device.tdma, mid_hbm.view_mut().tile::<m![C32], 2, m![Tk128, C32 = 2 #{!} 32, E32]>(30));
            (mid, mid_hbm)
        }};
    }
    let (mid0, mid_hbm0) = half!(0);
    let (mid1, mid_hbm1) = half!(1);
    let (out0, out1) = ffn_core_i8_pair(
        device, &mid_hbm0, &mid_hbm1, &mid0, &mid1, pre_w, up_q, up_ws, gate_q, gate_ws, down_w, down_s, dgs, post_ff, layer_scalar, had,
    );
    macro_rules! store_half {
        ($out:expr, $half:expr) => {{
            let xv: DmTensorView<'_, bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { $out.view().reshape() };
            xv.to_hbm_view(
                &mut device.tdma,
                res_out
                    .view_mut()
                    .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, C32, E32]>($half)
                    .tile::<m![C32], 30, m![Rh = 1 #{!} 2, Tk128, C32 = 30 #{!} 32, E32]>(0),
            );
        }};
    }
    store_half!(out0, 0);
    store_half!(out1, 1);
}
