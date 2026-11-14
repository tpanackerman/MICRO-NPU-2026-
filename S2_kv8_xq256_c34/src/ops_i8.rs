//! R14 w8a8 (`S2_I8_LAYERS`, LOSSY, default OFF): `#[device]` entry points. The fused K launch of `ops_fl` with the int8 up / gate FFN body
//! (`device/int8/ffn.rs`). Attention (N3), AO, GeGLU, the down
//! projection and the next layer's quad QKV are the production kernels, called exactly as in `ops_fl`.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::ffnc::{C32, C64, E32, E256, Fb16, Fh2, Fi4, Fi5, G32, Gd8, Rt80, Rt96, Rt240, Rt256, Tk128};
use crate::device::{dattn, int8, rpa};
use crate::ops_fl::{N3In16, QkvNext};

/// AO + FFN weights of one sliding layer with the int8 up / gate tensors (`S2_I8_LAYERS`).
#[derive(DeviceSend)]
pub struct LayerW8<'a> {
    pub w0: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    pub w1: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    pub w2: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    pub s0: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub s1: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub s2: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub post_attn_norm: &'a HbmTensor<bf16, Chip, m![H]>,
    pub pre_w: &'a HbmTensor<bf16, Chip, m![C32, E32]>,
    pub up_q: &'a HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub up_ws: &'a HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    pub gate_q: &'a HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub gate_ws: &'a HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    pub down_w: &'a HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    pub down_s: &'a HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    pub dgs: &'a HbmTensor<f32, Chip, m![1]>,
    pub post_ff: &'a HbmTensor<bf16, Chip, m![H]>,
    /// all8b: the +-1 Hadamard-32 matrix of the activation rotation (host `i8w::had32_bytes`)
    pub had: &'a int8::ffn::HadT,
}

type Res = HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>;


#[allow(clippy::too_many_arguments)]
pub(crate) fn n3_layer_i8(
    device: &mut Device,
    n: &N3In16<'_>,
    w: &LayerW8<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    res_in: &Res,
    out_p0: &mut HbmTensor<f8e4m3, Chip, rpa::prefill_kvd256::O8P0>,
    out_p1: &mut HbmTensor<f8e4m3, Chip, rpa::prefill_kvd256::O8P1>,
    res_out: &mut Res,
) {
    // R15 gapimpl F1: the attention epilogues write the e4m3 AO8 input (slice-contiguous `[C32, Tk128, E32]`), the AO reads it without a cast
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_o8(device, n.q_p0, n.k_cache, n.v_cache, n.win0, n.mask_p0, n.pen_p0, out_p0);
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_r16_o8(device, n.q_p1, n.k_cache, n.v_cache, n.win1, n.mask_p1, n.pen_p1, out_p1);
    dattn::sliding_blk::sliding_decode_attend16_blk_view_o8(device, n.q_d, n.k_blk, n.v_blk, n.index_d, n.mask_d, out_p1);
    let a0: HbmTensorView<'_, f8e4m3, Chip, m![C32, Tk128, E32]> = unsafe { out_p0.view().reshape() };
    let a1: HbmTensorView<'_, f8e4m3, Chip, m![C32, Tk128, E32]> = unsafe { out_p1.view().reshape() };
    int8::ffn::layer_sliding_pair_v_i8(
        device, a0, a1, w.w0, w.w1, w.w2, w.s0, w.s1, w.s2, w.post_attn_norm, res_in, w.pre_w, w.up_q, w.up_ws, w.gate_q, w.gate_ws, w.down_w, w.down_s, w.dgs,
        w.post_ff, layer_scalar, res_out, w.had,
    );
}

// big8: no int8 K' entry (an unreferenced device entry still costs image / build time; the K'-slot layer runs big's bf16 path).

/// K (16 decode rows), int8 up / gate: N3(l) + AO + FFN(l) + QKV(l+1) (big's production quadp QKV with kidx4, reading `res_out` through a view of the same handle).
#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn sliding_n3_layer_qkv_b16_i8(
    device: &mut Device,
    n: &N3In16<'_>,
    w: &LayerW8<'_>,
    qn: &QkvNext<'_>,
    kidx4: &HbmTensor<i32, Chip, m![Ub, Tc]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    res_in: &Res,
    out_p0: &mut HbmTensor<f8e4m3, Chip, rpa::prefill_kvd256::O8P0>,
    out_p1: &mut HbmTensor<f8e4m3, Chip, rpa::prefill_kvd256::O8P1>,
    res_out: &mut Res,
    xn: &mut crate::qkv8::XnScratch,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]>,
    k_next: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_next: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    n3_layer_i8(device, n, w, layer_scalar, res_in, out_p0, out_p1, res_out);
    let x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]> = unsafe { res_out.view().reshape() };
    // compile-time cut (10-11): big's production quad QKV (`quadp`, kidx4), as in the bf16 / fold K, instead of the anchored 4-kidx copy
    // (`int8::qkvq`: +280 schedule instructions); the int8 FFN epilogue no longer needs the anchors (lean OK, static 931,217 vs 998,694)
    // all8b: the same quad selector as the bf16 / fold K (e4m3 QKV when qkv8 is ON)
    crate::qkv8::quad_sel(
        device, x, qn.input_norm, xn, qn.wq, qn.sq, qn.wk, qn.sk, qn.wv, qn.sv, qn.q_norm, qn.k_norm, qn.cos, qn.sin, q_out, kidx4, k_next, v_next,
    );
}


/// all8b S2: the int8 sliding layer as a NATIVE T = 256 launch (AO + int8 FFN), for the K'-slot layer (its QKV already done by K_{l-1}) and for every
/// unfused pass of an int8 layer, so an int8 layer computes int8 in every hybrid pass shape. Same body as the fused K (`layer_sliding_pair_v_i8`).
#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn layer_sliding_t256_i8(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    w: &LayerW8<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    res_in: &Res,
    res_out: &mut Res,
) {
    int8::ffn::layer_sliding_pair_i8(
        device, attn, w.w0, w.w1, w.w2, w.s0, w.s1, w.s2, w.post_attn_norm, res_in, w.pre_w, w.up_q, w.up_ws, w.gate_q, w.gate_ws, w.down_w, w.down_s, w.dgs,
        w.post_ff, layer_scalar, res_out, w.had,
    );
}

/// all8b S3: AO + FFN weights of one FULL layer with the int8 up / gate tensors.
#[derive(DeviceSend)]
pub struct LayerW8F<'a> {
    pub w0: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    pub w1: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    pub w2: &'a HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E256]>,
    pub s0: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub s1: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub s2: &'a HbmTensor<bf16, Chip, m![G32, Rt80]>,
    pub post_attn_norm: &'a HbmTensor<bf16, Chip, m![H]>,
    pub pre_w: &'a HbmTensor<bf16, Chip, m![C32, E32]>,
    pub up_q: &'a HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub up_ws: &'a HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    pub gate_q: &'a HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub gate_ws: &'a HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    pub down_w: &'a HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    pub down_s: &'a HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    pub dgs: &'a HbmTensor<f32, Chip, m![1]>,
    pub post_ff: &'a HbmTensor<bf16, Chip, m![H]>,
    pub had: &'a int8::ffn::HadT,
}

/// all8b S3: int8 full-attention layer, native T = 256 (replaces `ops_fl::layer_full_t256r2` for int8 full layers).
#[device(chip = 1)]
pub fn layer_full_t256_i8(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E256]>,
    w: &LayerW8F<'_>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    res_in: &Res,
    res_out: &mut Res,
) {
    int8::ffn::layer_full_pair_i8(
        device, attn, w.w0, w.w1, w.w2, w.s0, w.s1, w.s2, w.post_attn_norm, res_in, w.pre_w, w.up_q, w.up_ws, w.gate_q, w.gate_ws, w.down_w, w.down_s, w.dgs,
        w.post_ff, layer_scalar, res_out, w.had,
    );
}
