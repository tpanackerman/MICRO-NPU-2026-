//! LOSSY FP8 WEIGHT FOLD of the FFN (`S2_FP8=fold`, DEFAULT ON in the fold1 build (S2_FP8=off disables); needs the organisers' ruling), T = 128 hybrid layer, on the fin6 d8 FFN-down layout.
//! The NVFP4 up / gate / down weights are decoded on chip to e4m3 with the block scale folded in (lookup + f32 widening + multiply by `block_scale / 8` + e4m3 cast in ONE Main pass; `/ 8` is a
//! power-of-two change, exact in e4m3 for every scale of the checkpoint), and contracted e4m3 x e4m3 against dynamically quantised activations (per-token amax scale, applied to the raw GEMM
//! results before the post norms). The global scale passed in the `*gs` arguments is `8 x` the reciprocal NVFP4 global scale (host). The weights are re-encoded lossily (about 2.2 % rms of the weight).
//!
//! Layout of the chain (everything per-token is in the token-sharded 2-part layout):
//!   AO core (bf16, the production one) -> mid residual (part layout, DM) -> pre-FF RMSNorm + quantiser (`tok`) -> xq e4m3 HBM hop (chunk-major, + zero pad chunks) and the token scale (HBM)
//!   -> up / gate tiles (e4m3 TRF, K split 32) -> GeGLU in the token-spread layout, the token scale applied through VRF operands -> h (bf16, HBM hop) -> quantiser on the part layout
//!   -> hq e4m3 HBM hop -> down tiles in the 8-row-group layout of fin6 (K split over 64 slices, chunk 256, 96-row tiles, 4 row groups per cluster) -> y (raw) x token scale
//!   -> post-FF RMSNorm + residual + layer scalar.
use furiosa_opt_std::prelude::*;

use super::tok::{self, Pre, TCl, YDm};
use super::{Xa1920, Xa8192};
use crate::Chip;
use crate::axes::{Dummy8, H, Rh};
use crate::device::ffnc::normp;
use crate::device::ffnc::{C32, C64, E32, E256, Fb8, Fb16, Fc2, Fh2, Fi4, Fi5, G32, Gd8, Pd2, Rt80, Rt96, Rt240, Rt256, Tk128};

type Cl = m![G32 / 8];
type Sl = m![G32 % 8, C32];
type SlOut = m![G32 % 8, 1 # 32];
type SpSl = m![G32 % 8, Tk128 / 4];

const INVSQRT2: f32 = 0.70710678118f32;

/// bigfa (PAPERS_FP8 probe P2): the fold weight is e4m3(ALPHA x f4 x bs / 8) instead of e4m3(f4 x bs / 8); the host divides the fold global scales by ALPHA.
/// ALPHA = 4/3 makes the NVFP4 codes 6 / 3 / 1.5 (mantissa 1.5, the block maximum) exact in e4m3: fold weight output error 1.9 -> 1.0 % on held-out rows.
pub(crate) const FOLD_ALPHA: f32 = 4.0 / 3.0;

fn scalar_vrf(device: &mut Device, gs: &HbmTensor<f32, Chip, impl M>) -> VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]> {
    let gs_dm: DmTensor<f32, Chip, Cl, SlOut, m![1 # 8]> = gs.to_dm(&mut device.tdma);
    device.sub.begin(gs_dm.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// One 240-row tile of up or gate: f4 -> e4m3 lookup, f32 widening, block scale (/8 folded), e4m3 cast in one Main pass, TRF (Sub), e4m3 x e4m3 contraction, K reduce, global scale (x8);
/// result: the raw tile `[T, Rt240]` (bf16, WITHOUT the token scale) on the live slices.
fn ffu_tile8(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]>,
    w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]>,
    scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cl, SlOut, m![Tk128, Rt240]> {
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt240, E32 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt240, E32 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt240], m![E32 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240], m![E32 / 16]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, E32]> = device
        .main
        .begin(w4_dm.view())
        .fetch::<m![Rt240], m![E32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt240, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![E32 % 8 # 32]>()
        .commit_trim::<m![E32 % 8]>()
        .commit();
    let w_trf = ffu_trf8(device, &w8s);
    ffu_con8(device, x_dm, &w_trf, gs_vrf)
}

/// Decode of one 240-row up / gate tile into the TRF (the first half of `ffu_tile8`).
fn ffu_dec8(
    device: &mut Device,
    w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]>,
    scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]>,
) -> TrfTensor<f8e4m3, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]> {
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt240, E32 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt240, E32 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt240], m![E32 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240], m![E32 / 16]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, E32]> = device
        .main
        .begin(w4_dm.view())
        .fetch::<m![Rt240], m![E32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt240, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![E32 % 8 # 32]>()
        .commit_trim::<m![E32 % 8]>()
        .commit();
    ffu_trf8(device, &w8s)
}


/// `ffu_con8` over the 256 tokens (Rh, Tk128) of both halves.
fn ffu_con8_t256(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl, Sl, m![Rh, Tk128, E32]>,
    w_trf: &TrfTensor<f8e4m3, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cl, SlOut, m![Rh, Tk128, Rt240]> {
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Rh, Tk128, E32 / 32], m![E32 % 32]>()
        .collect::<m![Rh, Tk128, E32 / 32], m![E32 % 32]>()
        .contract_outer::<m![Rh, Tk128, E32 / 64, Rt240 / 8], m![E32 % 64], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rh, Tk128, Rt240 / 8]>()
        .contract_lane::<m![Rh, Tk128, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SlOut, m![Rh, Tk128, Rt240 / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rh, Tk128, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
        .vector_widen_concat::<m![Rh, Tk128, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt240 % 8 # 16]>()
        .commit_trim::<m![Rt240 % 8]>()
        .commit()
}


/// e4m3 up / gate tile (DM, `[Rt240, E32]`) -> TRF (Sub).
fn ffu_trf8(device: &mut Device, w8s: &DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, E32]>) -> TrfTensor<f8e4m3, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]> {
    device
        .sub
        .begin(w8s.view())
        .fetch::<m![Rt240 % 8, Rt240 / 8, E32 / 32], m![E32 % 32]>()
        .collect::<m![Rt240 % 8, Rt240 / 8, E32 / 32], m![E32 % 32]>()
        .to_trf()
}

/// e4m3 x e4m3 contraction of one 240-row tile (TRF resident), K reduce, global scale (x8): the raw tile `[T, Rt240]` (bf16, without the token scale).
fn ffu_con8(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]>,
    w_trf: &TrfTensor<f8e4m3, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cl, SlOut, m![Tk128, Rt240]> {
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .collect::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .contract_outer::<m![Tk128, E32 / 64, Rt240 / 8], m![E32 % 64], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128, Rt240 / 8]>()
        .contract_lane::<m![Tk128, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SlOut, m![Tk128, Rt240 / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
        .vector_widen_concat::<m![Tk128, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt240 % 8 # 16]>()
        .commit_trim::<m![Rt240 % 8]>()
        .commit()
}

/// keytrick L2 (lab probe): `ffu_con8` with a REDUCE-SCATTER epilogue straight into the GeGLU token-spread layout (`SpSl` = 4 tokens per slice, as the int8 body's
/// `ffu_con_i8_sp` does), so the DM -> DM re-shard `t.to_dm()` of every (tile, half) disappears from the DMA queue. Same 32-way K reduce, same global-scale multiply after it.
fn ffu_con8_sp(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]>,
    w_trf: &TrfTensor<f8e4m3, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl, SpSl, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> {
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .collect::<m![Tk128, E32 / 32], m![E32 % 32]>()
        .contract_outer::<m![Tk128, E32 / 64, Rt240 / 8], m![E32 % 64], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Tk128, Rt240 / 8]>()
        .contract_lane::<m![Tk128, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SpSl, m![Tk128 % 4, Rt240 / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 4, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
        .vector_widen_concat::<m![Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt240 % 8 # 16]>()
        .commit_trim::<m![Rt240 % 8]>()
        .commit()
}

fn scalar_vrf_sp(device: &mut Device, gs: &HbmTensor<f32, Chip, impl M>) -> VrfTensor<f32, Chip, Cl, SpSl, m![1 # 8]> {
    let gs_dm: DmTensor<f32, Chip, Cl, SpSl, m![1 # 8]> = gs.to_dm(&mut device.tdma);
    device.sub.begin(gs_dm.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// Per-token scale VRFs of the GeGLU token-spread layout (4 tokens per slice): `s / sqrt2` (the erf argument of the scaled gate) and `s^2` (up x gelu both carry one factor s; the GeGLU `/ 2` is folded in: `s^2 / 2`).
struct SpreadScales {
    s_a: VrfTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]>,
    s_b: VrfTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]>,
}

fn spread_scales(device: &mut Device, s_hbm: &HbmTensor<f32, Chip, m![Tk128, Pd2, 1 # 8]>) -> SpreadScales {
    let s_sp: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Pd2, 1 # 8]> = s_hbm.to_dm(&mut device.tdma);
    let s_a_dm: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]> = device
        .main
        .begin(s_sp.view().tile::<m![Pd2], 1, m![Tk128 % 4, Pd2 = 1 # 2, 1 # 8]>(0))
        .fetch::<m![Tk128 % 4], m![1 # 8]>()
        .collect::<m![Tk128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_b_dm: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]> = device
        .main
        .begin(s_sp.view().tile::<m![Pd2], 1, m![Tk128 % 4, Pd2 = 1 # 2, 1 # 8]>(0))
        .fetch::<m![Tk128 % 4], m![1 # 8]>()
        .collect::<m![Tk128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_a = device.sub.begin(s_a_dm.view()).fetch::<m![Tk128 % 4], m![1 # 8]>().collect::<m![Tk128 % 4], m![1 # 8]>().to_vrf();
    let s_b = device.sub.begin(s_b_dm.view()).fetch::<m![Tk128 % 4], m![1 # 8]>().collect::<m![Tk128 % 4], m![1 # 8]>().to_vrf();
    SpreadScales { s_a, s_b }
}

fn spread_scales_from(device: &mut Device, s_sp: &DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Pd2, 1 # 8]>) -> SpreadScales {
    let s_a_dm: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]> = device
        .main
        .begin(s_sp.view().tile::<m![Pd2], 1, m![Tk128 % 4, Pd2 = 1 # 2, 1 # 8]>(0))
        .fetch::<m![Tk128 % 4], m![1 # 8]>()
        .collect::<m![Tk128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_b_dm: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, 1 # 8]> = device
        .main
        .begin(s_sp.view().tile::<m![Pd2], 1, m![Tk128 % 4, Pd2 = 1 # 2, 1 # 8]>(0))
        .fetch::<m![Tk128 % 4], m![1 # 8]>()
        .collect::<m![Tk128 % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_a = device.sub.begin(s_a_dm.view()).fetch::<m![Tk128 % 4], m![1 # 8]>().collect::<m![Tk128 % 4], m![1 # 8]>().to_vrf();
    let s_b = device.sub.begin(s_b_dm.view()).fetch::<m![Tk128 % 4], m![1 # 8]>().collect::<m![Tk128 % 4], m![1 # 8]>().to_vrf();
    SpreadScales { s_a, s_b }
}


/// Re-shard of ONE 128-token half of an up / gate tile of the 256-token contraction into the token-spread layout (4 tokens per slice).
fn reshard_half(device: &mut Device, t: &DmTensor<bf16, Chip, Cl, SlOut, m![Rh, Tk128, Rt240]>, half: usize) -> DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> {
    let mut sp: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = DmTensor::new();
    t.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, Rt240]>(half).to_dm_view(&mut device.tdma, sp.view_mut());
    sp
}

/// GeGLU of one tile pair on the RAW (token-scale-free) up / gate tiles, spread over the slices (4 tokens per slice): with `a` = raw gate, `u` = raw up, scale `s`,
/// `h = (u s) * gelu2(a s) / 2` where `gelu2(g) = g (1 + erf(g / sqrt2))`  =>  `h = s^2 * u * [a (1 + erf(a s / sqrt2))] / 2`: the gate pass multiplies by `s / sqrt2` (VRF) before the erf and the
/// product pass by `s^2` (VRF).  Result -> tile `i` = (c2, h) of `h_buf` (`[Fc2, Fh2, T % 4, Rt256]`, zero pad columns).
fn geglu_tile8(
    device: &mut Device,
    up_t: &DmTensor<bf16, Chip, Cl, SlOut, m![Tk128, Rt240]>,
    gate_t: &DmTensor<bf16, Chip, Cl, SlOut, m![Tk128, Rt240]>,
    sc: &SpreadScales,
    h_buf: &mut DmTensor<bf16, Chip, Cl, SpSl, m![Fc2, Fh2, Tk128 % 4, Rt256]>,
    i: usize,
) {
    let up_sp: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = up_t.to_dm(&mut device.tdma);
    let gate_sp: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = gate_t.to_dm(&mut device.tdma);

    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = device
        .sub
        .begin(gate_sp.view())
        .fetch::<m![Tk128 % 4, Rt240 / 16], m![Rt240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 4, Rt240 / 4], m![Rt240 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sc.s_a)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .commit_trim::<m![Rt240 % 8]>()
        .commit();
    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = unsafe { gelu.reshape() };
    let up_sp: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = unsafe { up_sp.reshape() };
    let gelu_vrf: VrfTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .collect::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .to_vrf();

    device
        .main
        .begin(up_sp.view())
        .fetch::<m![Tk128 % 4, Rt256 = 240 / 16], m![Rt256 = 240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 4, Rt256 = 240 / 4], m![Rt256 = 240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sc.s_b)
        .vector_widen_concat::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt256 = 240 % 8 # 16]>()
        .commit_trim::<m![Rt256 = 240 % 8]>()
        .commit_view(
            h_buf
                .view_mut()
                .tile::<m![Fc2], 1, m![Fc2 = 1 #{!} 2, Fh2, Tk128 % 4, Rt256]>(i / 2)
                .tile::<m![Fh2], 1, m![Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, Tk128 % 4, Rt256]>(i % 2)
                .tile::<m![Rt256], 240, m![Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, Tk128 % 4, Rt256 = 240 #{!} 256]>(0),
        );
}

/// `geglu_tile8` on tiles that are already in the token-spread layout (`reshard_half` of the contraction output).
fn geglu_sp8(
    device: &mut Device,
    up_sp: &DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]>,
    gate_sp: &DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]>,
    sc: &SpreadScales,
    h_buf: &mut DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fc2, Fh2, Tk128 % 4, Rt256]>,
    i: usize,
    half: usize,
) {

    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = device
        .sub
        .begin(gate_sp.view())
        .fetch::<m![Tk128 % 4, Rt240 / 16], m![Rt240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 4, Rt240 / 4], m![Rt240 % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sc.s_a)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![Tk128 % 4, Rt240 / 8], m![Rt240 % 8]>()
        .vector_final()
        .commit_trim::<m![Rt240 % 8]>()
        .commit();
    let gelu: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = unsafe { gelu.reshape() };
    let up_sp: DmTensorView<'_, bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = unsafe { up_sp.view().reshape() };
    let gelu_vrf: VrfTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Rt256 = 240]> = device
        .sub
        .begin(gelu.view())
        .fetch::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .collect::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .to_vrf();

    device
        .main
        .begin(up_sp)
        .fetch::<m![Tk128 % 4, Rt256 = 240 / 16], m![Rt256 = 240 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Tk128 % 4, Rt256 = 240 / 4], m![Rt256 = 240 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sc.s_b)
        .vector_widen_concat::<m![Tk128 % 4, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt256 = 240 % 8 # 16]>()
        .commit_trim::<m![Rt256 = 240 % 8]>()
        .commit_view(
            h_buf
                .view_mut()
                .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Fc2, Fh2, Tk128 % 4, Rt256]>(half)
                .tile::<m![Fc2], 1, m![Rh = 1 #{!} 2, Fc2 = 1 #{!} 2, Fh2, Tk128 % 4, Rt256]>(i / 2)
                .tile::<m![Fh2], 1, m![Rh = 1 #{!} 2, Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, Tk128 % 4, Rt256]>(i % 2)
                .tile::<m![Rt256], 240, m![Rh = 1 #{!} 2, Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, Tk128 % 4, Rt256 = 240 #{!} 256]>(0),
        );
}


// ---- down projection in the fin6 8-row-group layout (K split over 64 slices, chunk 256, 4 row groups per cluster, 96-row tiles) ----
type Cl8 = m![Gd8 / 4];
type Sl8 = m![Gd8 % 4, C64];
type SlOut8 = m![Gd8 % 4, 1 # 64];

/// the down-projection global scale (8 x reciprocal) on the live slices of the 8-group down layout
fn scalar_vrf8(device: &mut Device, gs: &HbmTensor<f32, Chip, impl M>) -> VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> {
    let gs_dm: DmTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> = gs.to_dm(&mut device.tdma);
    device.sub.begin(gs_dm.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf()
}

/// One 96-row tile of the down projection on the e4m3 h (chunk = [Rt256]: 240 real + 16 zero columns): fused decode (lookup + widen + scale/8 + e4m3 cast), e4m3 x e4m3 contraction;
/// raw result (no token scale, global scale applied) -> tile `i` of `buf` ([T, Fi5, Rt96]) on the 4 live slices of each cluster.
fn ffd_tile8(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Tk128, Rt256]>,
    w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]>,
    scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]>,
    buf: &mut DmTensor<bf16, Chip, Cl8, SlOut8, m![Tk128, Fi5, Rt96]>,
    i: usize,
) {
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt96], m![Rt256 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256]> = device
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Rt256 % 8 # 32]>()
        .commit_trim::<m![Rt256 % 8]>()
        .commit();
    let w_trf: TrfTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> = device
        .sub
        .begin(w8s.view())
        .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
        .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
        .to_trf();
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Tk128, Rt256 / 32], m![Rt256 % 32]>()
        .collect::<m![Tk128, Rt256 / 32], m![Rt256 % 32]>()
        .contract_outer::<m![Tk128, Rt256 / 64, Rt96 / 8], m![Rt256 % 64], _, _, _>(&w_trf)
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
        .commit_view(buf.view_mut().tile::<m![Fi5], 1, m![Tk128, Fi5 = 1 #{!} 5, Rt96]>(i));
}

/// Decode of one 96-row down tile into the TRF (first half of `ffd_tile8`).
fn ffd_dec8(
    device: &mut Device,
    w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]>,
    scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]>,
) -> TrfTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> {
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt96], m![Rt256 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256]> = device
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Rt256 % 8 # 32]>()
        .commit_trim::<m![Rt256 % 8]>()
        .commit();
    device
        .sub
        .begin(w8s.view())
        .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
        .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
        .to_trf()
}

/// Contraction of one down tile (TRF resident) against the e4m3 h of the 256 tokens (Rh, Tk128): raw result (global scale applied) -> tile `i` of `buf`.
fn ffd_con8_t256(
    device: &mut Device,
    x_dm: &DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rh, Tk128, Rt256]>,
    w_trf: &TrfTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]>,
    gs_vrf: &VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]>,
    buf: &mut DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]>,
    i: usize,
) {
    device
        .main
        .begin(x_dm.view())
        .fetch::<m![Rh, Tk128, Rt256 / 32], m![Rt256 % 32]>()
        .collect::<m![Rh, Tk128, Rt256 / 32], m![Rt256 % 32]>()
        .contract_outer::<m![Rh, Tk128, Rt256 / 64, Rt96 / 8], m![Rt256 % 64], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rh, Tk128, Rt96 / 8]>()
        .contract_lane::<m![Rh, Tk128, Rt96 / 8], m![Rt96 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SlOut8, m![Rh, Tk128, Rt96 / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rh, Tk128, Rt96 / 8, Rt96 / 4 % 2], m![Rt96 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
        .vector_widen_concat::<m![Rh, Tk128, Rt96 / 8], m![Rt96 % 8]>()
        .vector_final()
        .cast::<bf16, m![Rt96 % 8 # 16]>()
        .commit_trim::<m![Rt96 % 8]>()
        .commit_view(buf.view_mut().tile::<m![Fi5], 1, m![Rh, Tk128, Fi5 = 1 #{!} 5, Rt96]>(i));
}


/// The FFN of the T = 128 chunk from the mid residual rows (part layout, DM): returns the new residual rows (part layout).
#[allow(clippy::too_many_arguments)]
fn ffn_core8(
    device: &mut Device,
    mid: &YDm<bf16>,
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_w: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
) -> YDm<bf16> {
    // ---- pre-FF RMSNorm + dynamic quantiser on the part layout; xq (e4m3, pad chunks zero) and the token scale hop through HBM ----
    let xn = tok::norm_only(device, mid, pre_w);
    let xn1: DmTensor<bf16, Chip, TCl, m![{ Pre }, H / 1920], m![Xa1920]> = unsafe { xn.reshape() };
    let (xq, s_ff) = tok::quant_xn(device, &xn1);
    let xq: YDm<f8e4m3> = unsafe { xq.reshape() };
    let mut xq_hbm: HbmTensor<f8e4m3, Chip, m![C32, Tk128, E32]> = HbmTensor::new();
    tok::store_rows_q_cm(device, xq, &mut xq_hbm);
    tok::zero_pad_q_cm(device, &mut xq_hbm);
    let mut s_hbm: HbmTensor<f32, Chip, m![Tk128, Pd2, 1 # 8]> = HbmTensor::new();
    let s_pd: DmTensorView<'_, f32, Chip, TCl, m![{ Pre }, Pd2], m![1 # 8]> = unsafe { s_ff.view().reshape() };
    s_pd.to_hbm_view(&mut device.tdma, s_hbm.view_mut());

    // ---- up / gate ----
    let x_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.to_dm(&mut device.tdma);
    let up_gs = scalar_vrf(device, ugs);
    let gate_gs = scalar_vrf(device, ggs);
    let sc = spread_scales(device, &s_hbm);
    let mut h_buf: DmTensor<bf16, Chip, Cl, SpSl, m![Fc2, Fh2, Tk128 % 4, Rt256]> = DmTensor::new();
    h_buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! mtile {
        ($w:ident, $s:ident, $gs:ident, $i:expr) => {{
            let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]> = $w.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i).to_dm(&mut device.tdma);
            let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]> = $s.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, Fb8]>($i).to_dm(&mut device.tdma);
            ffu_tile8(device, &x_dm, w4_dm, scale_dm, &$gs)
        }};
    }
    macro_rules! mpair {
        ($i:expr) => {{
            let up_t = mtile!(up_w, up_s, up_gs, $i);
            let gate_t = mtile!(gate_w, gate_s, gate_gs, $i);
            geglu_tile8(device, &up_t, &gate_t, &sc, &mut h_buf, $i);
        }};
    }
    mpair!(0);
    mpair!(1);
    mpair!(2);
    mpair!(3);
    let h_buf = h_buf;

    // ---- h -> HBM -> quantiser on the part layout -> hq (e4m3) HBM -> replicated 8x for the down projection ----
    let h: HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, Tk128, Rt256]> = h_buf.to_hbm(&mut device.tdma);
    let h_part: DmTensor<bf16, Chip, TCl, m![{ Pre }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> = h.to_dm(&mut device.tdma);
    let h_part: DmTensor<bf16, Chip, TCl, m![{ Pre }, G32 / 8], m![Xa8192]> = unsafe { h_part.reshape() };
    let (hq, s_h) = tok::quant_h(device, &h_part);
    let hq: DmTensorView<'_, f8e4m3, Chip, TCl, m![{ Pre }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> = unsafe { hq.view().reshape() };
    let mut hq_hbm: HbmTensor<f8e4m3, Chip, m![G32, Fc2, Fh2, Tk128, Rt256]> = HbmTensor::new();
    hq.to_hbm_view(&mut device.tdma, hq_hbm.view_mut());
    let hv: HbmTensorView<'_, f8e4m3, Chip, m![C64, Tk128, Rt256]> = unsafe { hq_hbm.view().reshape() };
    let x_h: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Tk128, Rt256]> = hv.to_dm(&mut device.tdma);

    // ---- down (8-row-group layout) ----
    let d_gs = scalar_vrf8(device, dgs);
    let mut dbuf: DmTensor<bf16, Chip, Cl8, SlOut8, m![Tk128, Fi5, Rt96]> = DmTensor::new();
    macro_rules! dtile {
        ($i:expr) => {{
            let w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]> =
                down_w.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>($i).to_dm(&mut device.tdma);
            let scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]> =
                down_s.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Fb16]>($i).to_dm(&mut device.tdma);
            ffd_tile8(device, &x_h, w4_dm, scale_dm, &d_gs, &mut dbuf, $i);
        }};
    }
    dtile!(0);
    dtile!(1);
    dtile!(2);
    dtile!(3);
    dtile!(4);

    // ---- epilogue: raw y x token scale (token-sharded) -> post-FF norm + residual -> layer scalar ----
    let yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, Gd8 / 4], m![Gd8 % 4, Fi5, Rt96]> = dbuf.to_dm(&mut device.tdma);
    let yb: YDm<bf16> = unsafe { yb.reshape() };
    let ys = tok::scale_tok(device, &yb, &s_h);
    let summed = normp::p1920::norm_residual::<TCl, Pre>(device, &ys, mid, post_w);
    normp::p1920::scale_rows::<TCl, Pre>(device, &summed, layer_scalar)
}

macro_rules! layer_fold {
    ($fname:ident, $E:ident, $core:path) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $fname(
            device: &mut Device,
            attn: &HbmTensor<bf16, Chip, m![Tk128, C32, $E]>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
            res_in: &HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, impl M>,
            up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ugs: &HbmTensor<f32, Chip, impl M>,
            gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ggs: &HbmTensor<f32, Chip, impl M>,
            down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
            down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
            dgs: &HbmTensor<f32, Chip, impl M>,
            post_ff: &HbmTensor<bf16, Chip, impl M>,
            layer_scalar: &HbmTensor<bf16, Chip, impl M>,
            res_out: &mut HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
        ) {
            let t0: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w0.to_dm(&mut device.tdma);
            let t1: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w1.to_dm(&mut device.tdma);
            let t2: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w2.to_dm(&mut device.tdma);
            let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
            let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
            let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
            let res = normp::p1920::load_rows::<Tk128, TCl, Pre>(device, res_in);
            let mid = $core(device, attn, t0, t1, t2, sc0, sc1, sc2, post_attn_norm, &res);
            let out = ffn_core8(device, &mid, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff, layer_scalar);
            normp::p1920::store_rows::<Tk128, TCl, Pre>(device, &out, res_out);
        }
    };
}
// fold FFN behind the production bf16 AO core (QKV and AO stay bf16)
layer_fold!(layer_sliding_f8fb, E32, crate::device::ffnc::ao::sliding_core_t128);
layer_fold!(layer_full_f8fb, E256, crate::device::ffnc::ao::full_core_t128);

/// AO core of ONE 128-token half of a T = 256 layer: `ao::sliding_core_t128` with the activation rows read through a view of the `[Rh, Tk128, ..]` tensor.
#[allow(clippy::too_many_arguments)]
fn ao_core_half(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    half: usize,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_norm: &HbmTensor<bf16, Chip, impl M>,
    res: &YDm<bf16>,
) -> YDm<bf16> {
    use crate::device::ffnc::Hi3;
    use crate::device::ffnc::ao::{load_t128, proc_t128};
    let t0 = load_t128(device, w0);
    let t1 = load_t128(device, w1);
    let t2 = load_t128(device, w2);
    let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
    let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
    let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tk128, E32]> = x.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>(half).to_dm(&mut device.tdma);
    let mut ybuf: DmTensor<bf16, Chip, Cl, SlOut, m![Tk128, Hi3, Rt80]> = DmTensor::new();
    proc_t128(device, &x_dm, t0, sc0, &mut ybuf, 0);
    proc_t128(device, &x_dm, t1, sc1, &mut ybuf, 1);
    proc_t128(device, &x_dm, t2, sc2, &mut ybuf, 2);
    let yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, G32 / 8], m![G32 % 8, Hi3, Rt80]> = ybuf.to_dm(&mut device.tdma);
    let yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
    normp::p1920::norm_residual::<TCl, Pre>(device, &yb, res, post_norm)
}

/// The FFN of the T = 256 layer from the mid residual rows of the two 128-token halves (T = 128 part layout): every weight tile is decoded and loaded into the TRF ONCE and contracted against
/// all 256 tokens in one contraction.
#[allow(clippy::too_many_arguments)]
fn ffn_core8_pair(
    device: &mut Device,
    mid0: &YDm<bf16>,
    mid1: &YDm<bf16>,
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_w: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
) -> (YDm<bf16>, YDm<bf16>) {
    // ---- per half: pre-FF RMSNorm + quantiser (T = 128 layout); xq (e4m3, both halves in ONE chunk-major tensor) and the token scale hop through HBM ----
    let mut xq_hbm: HbmTensor<f8e4m3, Chip, m![C32, Rh, Tk128, E32]> = HbmTensor::new();
    let mut s_hbm: SHbm = HbmTensor::new();
    macro_rules! quant_half {
        ($m:expr, $half:expr) => {{
            let xn = tok::norm_only(device, $m, pre_w);
            let xn1: DmTensor<bf16, Chip, TCl, m![{ Pre }, H / 1920], m![Xa1920]> = unsafe { xn.reshape() };
            let (xq, s_ff) = tok::quant_xn(device, &xn1);
            let xq: YDm<f8e4m3> = unsafe { xq.reshape() };
            tok::store_rows_q_cm_half(device, xq, &mut xq_hbm, $half);
            tok::zero_pad_q_cm_half(device, &mut xq_hbm, $half);
            store_scales_half(device, &s_ff, &mut s_hbm, $half);
        }};
    }
    quant_half!(mid0, 0);
    quant_half!(mid1, 1);

    // ---- up / gate: each weight tile decoded ONCE, contracted against the 256 tokens, re-sharded per half, GeGLU per half ----
    let x_dm0: DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(0).to_dm(&mut device.tdma);
    let x_dm1: DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(1).to_dm(&mut device.tdma);
    let up_gs = scalar_vrf(device, ugs);
    let gate_gs = scalar_vrf(device, ggs);
    // ONE h_buf for both halves: every GeGLU store goes through tiles of the same handle, so the compiler keeps the (tile, half) order of the program (one weight tile in the TRF at a time)
    let mut h_buf: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fc2, Fh2, Tk128 % 4, Rt256]> = DmTensor::new();
    h_buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! mtile {
        ($w:ident, $s:ident, $gs:ident, $i:expr) => {{
            let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]> = $w.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i).to_dm(&mut device.tdma);
            let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]> = $s.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, Fb8]>($i).to_dm(&mut device.tdma);
            let w_trf = ffu_dec8(device, w4_dm, scale_dm);
            // both halves contract against the TRF-resident tile; each result is re-sharded right away (the 256-token output would not fit the DM next to the prefetched weights)
            let t0 = ffu_con8(device, &x_dm0, &w_trf, &$gs);
            let sp0: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = t0.to_dm(&mut device.tdma);
            let t1 = ffu_con8(device, &x_dm1, &w_trf, &$gs);
            let sp1: DmTensor<bf16, Chip, Cl, SpSl, m![Tk128 % 4, Rt240]> = t1.to_dm(&mut device.tdma);
            (sp0, sp1)
        }};
    }
    macro_rules! mpair {
        ($i:expr) => {{
            let (up0, up1) = mtile!(up_w, up_s, up_gs, $i);
            let (gate0, gate1) = mtile!(gate_w, gate_s, gate_gs, $i);
            // the token-scale VRFs live only inside their GeGLU (the norm weight operand takes 7680 of the 8192 B of VRF per slice, so two halves of scales cannot stay resident)
            let sc0 = spread_scales_half(device, &s_hbm, 0);
            geglu_sp8(device, &up0, &gate0, &sc0, &mut h_buf, $i, 0);
            let sc1 = spread_scales_half(device, &s_hbm, 1);
            geglu_sp8(device, &up1, &gate1, &sc1, &mut h_buf, $i, 1);
        }};
    }
    mpair!(0);
    mpair!(1);
    mpair!(2);
    mpair!(3);
    let h_buf = h_buf;

    // ---- per half: h -> HBM -> quantiser on the part layout; hq (e4m3) of both halves in ONE HBM tensor -> replicated 8x for the down projection ----
    let mut hq_hbm: HbmTensor<f8e4m3, Chip, m![G32, Fc2, Fh2, Rh, Tk128, Rt256]> = HbmTensor::new();
    let h: HbmTensor<bf16, Chip, m![G32, Rh, Fc2, Fh2, Tk128, Rt256]> = h_buf.to_hbm(&mut device.tdma);
    macro_rules! hq_half {
        ($half:expr) => {{
            let h_part: DmTensor<bf16, Chip, TCl, m![{ Pre }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> =
                h.view().tile::<m![Rh], 1, m![G32, Rh = 1 # 2, Fc2, Fh2, Tk128, Rt256]>($half).to_dm(&mut device.tdma);
            let h_part: DmTensor<bf16, Chip, TCl, m![{ Pre }, G32 / 8], m![Xa8192]> = unsafe { h_part.reshape() };
            let (hq, s_h) = tok::quant_h(device, &h_part);
            let hq: DmTensorView<'_, f8e4m3, Chip, TCl, m![{ Pre }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> = unsafe { hq.view().reshape() };
            hq.to_hbm_view(&mut device.tdma, hq_hbm.view_mut().tile::<m![Rh], 1, m![G32, Fc2, Fh2, Rh = 1 #{!} 2, Tk128, Rt256]>($half));
            s_h
        }};
    }
    let s_h0 = hq_half!(0);
    let s_h1 = hq_half!(1);
    let hv: HbmTensorView<'_, f8e4m3, Chip, m![C64, Rh, Tk128, Rt256]> = unsafe { hq_hbm.view().reshape() };
    let x_h: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rh, Tk128, Rt256]> = hv.to_dm(&mut device.tdma);

    // ---- down (8-row-group layout): each weight tile decoded once, contracted against the 256 tokens ----
    let d_gs = scalar_vrf8(device, dgs);
    let mut dbuf: DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]> = DmTensor::new();
    macro_rules! dtile {
        ($i:expr) => {{
            let w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]> =
                down_w.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>($i).to_dm(&mut device.tdma);
            let scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]> =
                down_s.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Fb16]>($i).to_dm(&mut device.tdma);
            let w_trf = ffd_dec8(device, w4_dm, scale_dm);
            ffd_con8_t256(device, &x_h, &w_trf, &d_gs, &mut dbuf, $i);
        }};
    }
    dtile!(0);
    dtile!(1);
    dtile!(2);
    dtile!(3);
    dtile!(4);

    // ---- per half epilogue: raw y x token scale -> post-FF norm + residual -> layer scalar ----
    macro_rules! epi_half {
        ($s_h:expr, $m:expr, $half:expr) => {{
            let mut yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, Gd8 / 4], m![Gd8 % 4, Fi5, Rt96]> = DmTensor::new();
            dbuf.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, Fi5, Rt96]>($half).to_dm_view(&mut device.tdma, yb.view_mut());
            let yb: YDm<bf16> = unsafe { yb.reshape() };
            let ys = tok::scale_tok(device, &yb, &$s_h);
            let summed = normp::p1920::norm_residual::<TCl, Pre>(device, &ys, $m, post_w);
            normp::p1920::scale_rows::<TCl, Pre>(device, &summed, layer_scalar)
        }};
    }
    let out0 = epi_half!(s_h0, mid0, 0);
    let out1 = epi_half!(s_h1, mid1, 1);
    (out0, out1)
}

/// R14 fixedcost PREBAKE: `ffn_core8_pair` with the up / gate / down weights already decoded to e4m3 in HBM (the exact output of `ffu_dec8` / `ffd_dec8`'s commit, produced once
/// at load by `prebake_up8` / `prebake_down8`): every tile is DMA'd straight to DM and loaded into the TRF (Sub); no table lookup, no scale VRF, no Main decode pass.
/// Same TRF contents bit for bit, same contractions in the same order => same results as `ffn_core8_pair`.
#[allow(clippy::too_many_arguments)]
fn ffn_core8_pair_pb(
    device: &mut Device,
    mid0: &YDm<bf16>,
    mid1: &YDm<bf16>,
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w8: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_w: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
) -> (YDm<bf16>, YDm<bf16>) {
    let mut xq_hbm: HbmTensor<f8e4m3, Chip, m![C32, Rh, Tk128, E32]> = HbmTensor::new();
    let mut s_hbm: SHbm = HbmTensor::new();
    // cpk: pre-FF norm weight DMA'd ONCE for both halves
    let pre_dm: YDm<bf16> = pre_w.to_dm(&mut device.tdma);
    macro_rules! quant_half {
        ($m:expr, $half:expr) => {{
            let xn = tok::norm_only_dm(device, $m, &pre_dm);
            let xn1: DmTensor<bf16, Chip, TCl, m![{ Pre }, H / 1920], m![Xa1920]> = unsafe { xn.reshape() };
            let (xq, s_ff) = tok::quant_xn(device, &xn1);
            let xq: YDm<f8e4m3> = unsafe { xq.reshape() };
            tok::store_rows_q_cm_half(device, xq, &mut xq_hbm, $half);
            tok::zero_pad_q_cm_half(device, &mut xq_hbm, $half);
            store_scales_half(device, &s_ff, &mut s_hbm, $half);
        }};
    }
    quant_half!(mid0, 0);
    quant_half!(mid1, 1);

    let x_dm0: DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(0).to_dm(&mut device.tdma);
    let x_dm1: DmTensor<f8e4m3, Chip, Cl, Sl, m![Tk128, E32]> = xq_hbm.view().tile::<m![Rh], 1, m![C32, Rh = 1 # 2, Tk128, E32]>(1).to_dm(&mut device.tdma);
    let up_gs = scalar_vrf_sp(device, ugs);
    let gate_gs = scalar_vrf_sp(device, ggs);
    let mut h_buf: DmTensor<bf16, Chip, Cl, SpSl, m![Rh, Fc2, Fh2, Tk128 % 4, Rt256]> = DmTensor::new();
    h_buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    macro_rules! mtile {
        ($w:ident, $gs:ident, $i:expr) => {{
            let w8s: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, E32]> = $w.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i).to_dm(&mut device.tdma);
            let w_trf = ffu_trf8(device, &w8s);
            // keytrick L2: reduce-scatter epilogue straight into the GeGLU layout (no DM -> DM re-shard)
            let sp0 = ffu_con8_sp(device, &x_dm0, &w_trf, &$gs);
            let sp1 = ffu_con8_sp(device, &x_dm1, &w_trf, &$gs);
            (sp0, sp1)
        }};
    }
    macro_rules! mpair {
        ($i:expr) => {{
            let (up0, up1) = mtile!(up_w8, up_gs, $i);
            let (gate0, gate1) = mtile!(gate_w8, gate_gs, $i);
            let sc0 = spread_scales_half(device, &s_hbm, 0);
            geglu_sp8(device, &up0, &gate0, &sc0, &mut h_buf, $i, 0);
            let sc1 = spread_scales_half(device, &s_hbm, 1);
            geglu_sp8(device, &up1, &gate1, &sc1, &mut h_buf, $i, 1);
        }};
    }
    mpair!(0);
    mpair!(1);
    mpair!(2);
    mpair!(3);
    let h_buf = h_buf;

    let mut hq_hbm: HbmTensor<f8e4m3, Chip, m![G32, Fc2, Fh2, Rh, Tk128, Rt256]> = HbmTensor::new();
    let h: HbmTensor<bf16, Chip, m![G32, Rh, Fc2, Fh2, Tk128, Rt256]> = h_buf.to_hbm(&mut device.tdma);
    macro_rules! hq_half {
        ($half:expr) => {{
            // act3 hsp: h rows / hq rows / the token amax on `Pre2` (all 4 PEs of a cluster): the h load runs 7,950 -> 4,191 static, the hq store
            // 3,896 -> 2,150; the per-token scale (32 B per slice, lane 0 = value) is re-laid onto the epilogue's `Pre` slices with its pad lanes as data
            let h_part: DmTensor<bf16, Chip, TCl, m![{ tok::Pre2 }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> =
                h.view().tile::<m![Rh], 1, m![G32, Rh = 1 # 2, Fc2, Fh2, Tk128, Rt256]>($half).to_dm(&mut device.tdma);
            let h_part: DmTensor<bf16, Chip, TCl, m![{ tok::Pre2 }, G32 / 8], m![Xa8192]> = unsafe { h_part.reshape() };
            let (hq, s_h2) = tok::quant_h_sp(device, &h_part);
            let hq: DmTensorView<'_, f8e4m3, Chip, TCl, m![{ tok::Pre2 }, G32 / 8], m![G32 % 8, Fc2, Fh2, Rt256]> = unsafe { hq.view().reshape() };
            hq.to_hbm_view(&mut device.tdma, hq_hbm.view_mut().tile::<m![Rh], 1, m![G32, Fc2, Fh2, Rh = 1 #{!} 2, Tk128, Rt256]>($half));
            let s_h2: DmTensorView<'_, f32, Chip, TCl, m![{ tok::Pre2 }, H / 1920], m![Dummy8]> = unsafe { s_h2.view().reshape() };
            let mut s_h: DmTensor<f32, Chip, TCl, m![{ Pre }, H / 1920], m![Dummy8]> = DmTensor::new();
            s_h2.to_dm_view(&mut device.tdma, s_h.view_mut());
            let s_h: tok::ScaleDm = unsafe { s_h.reshape() };
            s_h
        }};
    }
    let s_h0 = hq_half!(0);
    let s_h1 = hq_half!(1);
    let hv: HbmTensorView<'_, f8e4m3, Chip, m![C64, Rh, Tk128, Rt256]> = unsafe { hq_hbm.view().reshape() };
    let x_h: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rh, Tk128, Rt256]> = hv.to_dm(&mut device.tdma);

    let d_gs = scalar_vrf8(device, dgs);
    let mut dbuf: DmTensor<bf16, Chip, Cl8, SlOut8, m![Rh, Tk128, Fi5, Rt96]> = DmTensor::new();
    macro_rules! dtile {
        ($i:expr) => {{
            let w8s: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256]> =
                down_w8.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>($i).to_dm(&mut device.tdma);
            let w_trf: TrfTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> = device
                .sub
                .begin(w8s.view())
                .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
                .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 32], m![Rt256 % 32]>()
                .to_trf();
            ffd_con8_t256(device, &x_h, &w_trf, &d_gs, &mut dbuf, $i);
        }};
    }
    dtile!(0);
    dtile!(1);
    dtile!(2);
    dtile!(3);
    dtile!(4);

    // cpk: post-FF norm weight + layer scalar DMA'd ONCE for both halves
    let post_dm = normp::p1920::load_weight_dm::<TCl, Pre>(device, post_w);
    let scalar_dm = normp::p1920::load_scalar_dm::<TCl, Pre>(device, layer_scalar);
    macro_rules! epi_half {
        ($s_h:expr, $m:expr, $half:expr) => {{
            let mut yb: DmTensor<bf16, Chip, TCl, m![{ Pre }, Gd8 / 4], m![Gd8 % 4, Fi5, Rt96]> = DmTensor::new();
            dbuf.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, Fi5, Rt96]>($half).to_dm_view(&mut device.tdma, yb.view_mut());
            let yb: YDm<bf16> = unsafe { yb.reshape() };
            let ys = tok::scale_tok(device, &yb, &$s_h);
            let summed = normp::p1920::norm_residual_dm::<TCl, Pre>(device, &ys, $m, &post_dm);
            normp::p1920::scale_rows_dm::<TCl, Pre>(device, &summed, &scalar_dm)
        }};
    }
    let out0 = epi_half!(s_h0, mid0, 0);
    let out1 = epi_half!(s_h1, mid1, 1);
    (out0, out1)
}

/// R14 fixedcost PREBAKE: `layer_sliding_f8fb_pair` on pre-decoded e4m3 FFN weights (`ffn_core8_pair_pb`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_f8pb_pair(
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
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w8: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    macro_rules! res_half {
        ($half:expr) => {{
            let x: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let x: YDm<bf16> = unsafe { x.reshape() };
            x
        }};
    }
    let res0 = res_half!(0);
    let res1 = res_half!(1);
    let mid0 = ao_core_half(device, attn, 0, w0, w1, w2, s0, s1, s2, post_attn_norm, &res0);
    let mid1 = ao_core_half(device, attn, 1, w0, w1, w2, s0, s1, s2, post_attn_norm, &res1);
    let (out0, out1) = ffn_core8_pair_pb(device, &mid0, &mid1, pre_w, up_w8, ugs, gate_w8, ggs, down_w8, dgs, post_ff, layer_scalar);
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

/// R14 fixedcost PREBAKE (load time, once per folded layer and matrix): the up / gate weight decode of `ffu_dec8` (table lookup, x block scale, e4m3 cast) for ONE `Fi4` tile,
/// committed to DM and stored to the matching tile of the pre-decoded HBM tensor. Instruction-for-instruction the decode pass of `ffu_dec8` => bit-identical e4m3 values.
pub(crate) fn prebake_up8_tile(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    out: &mut HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    i: usize,
) {
    let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]> = w.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>(i).to_dm(&mut device.tdma);
    let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]> = s.view().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, Fb8]>(i).to_dm(&mut device.tdma);
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt240, E32 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt240, E32 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt240], m![E32 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240], m![E32 / 16]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, E32]> = device
        .main
        .begin(w4_dm.view())
        .fetch::<m![Rt240], m![E32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rt240, E32 / 4], m![E32 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt240, E32 / 8], m![E32 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![E32 % 8 # 32]>()
        .commit_trim::<m![E32 % 8]>()
        .commit();
    w8s.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 #{!} 4, Rt240, E32]>(i));
}

/// R14 fixedcost PREBAKE: the decode pass of `ffd_dec8` for ONE `Fi5` down tile -> pre-decoded HBM tensor (bit-identical e4m3 values).
pub(crate) fn prebake_down8_tile(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    out: &mut HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    i: usize,
) {
    let w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]> = w.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>(i).to_dm(&mut device.tdma);
    let scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]> = s.view().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Fb16]>(i).to_dm(&mut device.tdma);
    let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = unsafe { scale_dm.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
        .sub
        .begin(scale_view)
        .fetch::<m![Rt96], m![Rt256 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
        .to_vrf();
    let w8s: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256]> = device
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
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), FOLD_ALPHA)
        .vector_widen_concat::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Rt256 % 8 # 32]>()
        .commit_trim::<m![Rt256 % 8]>()
        .commit();
    w8s.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 #{!} 5, Rt96, Rt256]>(i));
}

/// Token scales of ONE half (`spread_scales` on a view of the `[Rh, Tk128, ..]` scale tensor).
fn spread_scales_half(device: &mut Device, s_hbm: &SHbm, half: usize) -> SpreadScales {
    let mut s_sp: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Pd2, Dummy8]> = DmTensor::new();
    s_hbm.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, Pd2, Dummy8]>(half).to_dm_view(&mut device.tdma, s_sp.view_mut());
    // lane 0 of every 8-lane group is the scale; lanes 1..8 are the producer's pad lanes, never read as values (`spread_scales_from` trims to lane 0's group)
    let s_sp: DmTensor<f32, Chip, Cl, SpSl, m![Tk128 % 4, Pd2, 1 # 8]> = unsafe { s_sp.reshape() };
    spread_scales_from(device, &s_sp)
}

/// act3 sg3: the per-token scale of the T = 256 pair as a DENSE HBM tensor (the 7 pad lanes of each `[1 # 8]` slot travel as data, as `int8::ffn`'s `sq` /
/// Dummy8): a padded HBM tensor makes the store 256 single 4 B writes (6,130 static); dense + gathered into one slice per cluster it is one 4 KiB run per
/// cluster (DM -> DM 716 + store 461). Lane 0 (the value) is unchanged.
type SHbm = HbmTensor<f32, Chip, m![Rh, Tk128, Pd2, Dummy8]>;

fn store_scales_half(device: &mut Device, s_ff: &tok::ScaleDm, s_hbm: &mut SHbm, half: usize) {
    let s_pd: DmTensorView<'_, f32, Chip, TCl, m![{ Pre }, Pd2], m![Dummy8]> = unsafe { s_ff.view().reshape() };
    let mut s_g: DmTensor<f32, Chip, TCl, m![1 # 256], m![Tk128 % 64, Pd2, Dummy8]> = DmTensor::new();
    s_pd.to_dm_view(&mut device.tdma, s_g.view_mut());
    s_g.view().to_hbm_view(&mut device.tdma, s_hbm.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, Pd2, Dummy8]>(half));
}

/// R14 fp8pass: `layer_sliding_f8fb_pair` with the attention rows of each half read through a VIEW of the attention kernel's own output handle (the fused K / K' launch of
/// `ops_fl`: ordering through the same handle). AO = `ffnc::ao::sliding_core_halfv_t128` (the same instructions as this file's `ao_core_half`: load_t128 / proc_t128 /
/// norm_residual), FFN = `ffn_core8_pair`: per-token arithmetic identical to `layer_sliding_f8fb_pair`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_f8fb_pair_v(
    device: &mut Device,
    attn0: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>,
    attn1: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    macro_rules! res_half {
        ($half:expr) => {{
            let x: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let x: YDm<bf16> = unsafe { x.reshape() };
            x
        }};
    }
    let res0 = res_half!(0);
    let res1 = res_half!(1);
    let mid0 = crate::device::ffnc::ao::sliding_core_halfv_t128(device, attn0, w0, w1, w2, s0, s1, s2, post_attn_norm, &res0);
    let mid1 = crate::device::ffnc::ao::sliding_core_halfv_t128(device, attn1, w0, w1, w2, s0, s1, s2, post_attn_norm, &res1);
    let (out0, out1) = ffn_core8_pair(device, &mid0, &mid1, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff, layer_scalar);
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

/// R14 fixedcost PREBAKE: `layer_sliding_f8fb_pair_v` on pre-decoded e4m3 FFN weights (`ffn_core8_pair_pb`); bit-identical results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_f8pb_pair_v(
    device: &mut Device,
    attn0: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>,
    attn1: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, impl M>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w8: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    macro_rules! res_half {
        ($half:expr) => {{
            let x: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let x: YDm<bf16> = unsafe { x.reshape() };
            x
        }};
    }
    let res0 = res_half!(0);
    let res1 = res_half!(1);
    // R15 overlap P3: o_proj tiles / row scales DMA'd ONCE for both halves (each half TRF-loads them); -3 tile DMAs in the DMA-bound head
    use crate::device::ffnc::ao as aom;
    let at0 = aom::load_t128(device, w0);
    let at1 = aom::load_t128(device, w1);
    let at2 = aom::load_t128(device, w2);
    let asc0: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
    let asc1: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
    let asc2: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
    // cpk: the post-attention norm weight (2 MB replicated) DMA'd ONCE for both halves
    let pan_dm = normp::p1920::load_weight_dm::<m![Tk128 / 64], m![1 # 2, Tk128 % 64]>(device, post_attn_norm);
    let mid0 = aom::sliding_core_halfvw_t128(device, attn0, &at0, &at1, &at2, &asc0, &asc1, &asc2, &pan_dm, &res0);
    let mid1 = aom::sliding_core_halfvw_t128(device, attn1, &at0, &at1, &at2, &asc0, &asc1, &asc2, &pan_dm, &res1);
    let (out0, out1) = ffn_core8_pair_pb(device, &mid0, &mid1, pre_w, up_w8, ugs, gate_w8, ggs, down_w8, dgs, post_ff, layer_scalar);
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

/// R15 layout F1f: `layer_sliding_f8pb_pair_v` with the e4m3 AO8 attention input (gapimpl F1 on the fold K).
/// R14 fixedcost PREBAKE: `layer_sliding_f8fb_pair_v` on pre-decoded e4m3 FFN weights (`ffn_core8_pair_pb`); bit-identical results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_f8pb_pair_v8(
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
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w8: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    macro_rules! res_half {
        ($half:expr) => {{
            let x: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let x: YDm<bf16> = unsafe { x.reshape() };
            x
        }};
    }
    let res0 = res_half!(0);
    let res1 = res_half!(1);
    // R15 overlap P3 (on the F1f AO8 path): o_proj tiles / row scales DMA'd ONCE for both halves (each half TRF-loads them)
    use crate::device::ffnc::ao as aom;
    let at0 = aom::load_t128(device, w0);
    let at1 = aom::load_t128(device, w1);
    let at2 = aom::load_t128(device, w2);
    let asc0: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
    let asc1: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
    let asc2: DmTensor<bf16, Chip, aom::Cl, aom::SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
    // cpk: the post-attention norm weight (2 MB replicated) DMA'd ONCE for both halves
    let pan_dm = normp::p1920::load_weight_dm::<m![Tk128 / 64], m![1 # 2, Tk128 % 64]>(device, post_attn_norm);
    let mid0 = aom::sliding_core_halfv8w_t128(device, attn0, &at0, &at1, &at2, &asc0, &asc1, &asc2, &pan_dm, &res0);
    let mid1 = aom::sliding_core_halfv8w_t128(device, attn1, &at0, &at1, &at2, &asc0, &asc1, &asc2, &pan_dm, &res1);
    let (out0, out1) = ffn_core8_pair_pb(device, &mid0, &mid1, pre_w, up_w8, ugs, gate_w8, ggs, down_w8, dgs, post_ff, layer_scalar);
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

/// NATIVE T = 256 form of `layer_sliding_f8fb`: the activation tensors are `[Rh, Tk128, ..]` (half-major, the byte layout of 256 consecutive rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_sliding_f8fb_pair(
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
    pre_w: &HbmTensor<bf16, Chip, impl M>,
    up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ugs: &HbmTensor<f32, Chip, impl M>,
    gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    ggs: &HbmTensor<f32, Chip, impl M>,
    down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    dgs: &HbmTensor<f32, Chip, impl M>,
    post_ff: &HbmTensor<bf16, Chip, impl M>,
    layer_scalar: &HbmTensor<bf16, Chip, impl M>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    macro_rules! res_half {
        ($half:expr) => {{
            let x: DmTensor<bf16, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res_in
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tk128, C32, E32]>($half)
                .tile::<m![C32], 30, m![Rh = 1 # 2, Tk128, C32 = 30 # 32, E32]>(0)
                .to_dm(&mut device.tdma);
            let x: YDm<bf16> = unsafe { x.reshape() };
            x
        }};
    }
    let res0 = res_half!(0);
    let res1 = res_half!(1);
    let mid0 = ao_core_half(device, attn, 0, w0, w1, w2, s0, s1, s2, post_attn_norm, &res0);
    let mid1 = ao_core_half(device, attn, 1, w0, w1, w2, s0, s1, s2, post_attn_norm, &res1);
    let (out0, out1) = ffn_core8_pair(device, &mid0, &mid1, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff, layer_scalar);
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
