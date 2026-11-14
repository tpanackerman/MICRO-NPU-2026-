//! R14 qkvre QH ("quad halves, half-outer"): the sliding quad QKV of `quad::sliding_qkv_quad_direct` with the SAME per-token arithmetic (bit-identical) and a different schedule shape.
//! b11 quad: joint norm -> xn (HBM) -> ONE 32 MiB replicated load (36.4k static with Main idle; attntrace: 52k HW TU-idle cycles per quad) -> 8 tiles x 256 tokens -> post of the 4 blocks.
//! QH: joint norm -> xn -> TWO 16 MiB replicated loads (half A = blocks 0,1; half B = blocks 2,3). Every weight tile is DMA'd and decoded ONCE and stays in DM; all 8 tiles are contracted
//! against half A while half B is still loading, then against half B (a second TRF load of the decoded tile, Sub is ~18 % busy in the tail). The post of blocks 0,1 (reshards, norms,
//! RoPE, scatters, q stores) runs while half B is contracted; only the post of blocks 2,3 is left as the exposed tail. Per-half output buffers (separate handles) so the post of half A
//! carries no dependency on the half-B commits.
use furiosa_opt_std::prelude::*;

use super::quad::{input_norm_quad, load_rope_block, rope_k_s, rope_q_s};
use super::{gemm, post};
use crate::Chip;
use crate::axes::*;

/// Half `j` (blocks 2j, 2j+1) of the normalised xn, replicated to the GEMM layout: [Rh, Tc, Ke] per slice (16 MiB, one DMA).
fn load_x_half(device: &mut Device, xn: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>, j: usize) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]> {
    let x: DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Ub = 2, Tc, Ke]> = xn.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc, Kc, Ke]>(2 * j).to_dm(&mut device.tdma);
    unsafe { x.reshape() }
}

/// f8 -> bf16 decode of one weight tile (Main), the same pass as `quad::tile_quad_fn`; the decoded tile is kept in DM and loaded into the TRF once per half.
fn dec_tile(device: &mut Device, w_f8: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Rt, Ke]>) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rt, Ke]> {
    device
        .main
        .begin(w_f8.view())
        .fetch::<m![Rt, Ke / 32], m![Ke % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Rt, Ke / 32, Ke % 32 / 8], m![Ke % 8]>()
        .cast::<bf16, m![Ke % 8 # 16]>()
        .commit_trim::<m![Ke % 8]>()
        .commit()
}

/// One decoded row tile contracted against ONE 128-token half (the `quad::tile_quad_fn` pipeline with [Rh, Tc] tokens): TRF load (Sub), contraction, inter-slice reduce, row scale, bf16.
macro_rules! con_half_fn {
    ($name:ident, $Rq:ident, $nrq:literal) => {
        fn $name(
            device: &mut Device,
            xh: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
            w_bf: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rt, Ke]>,
            scale_dm: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![$Rq, Rt]>,
            y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, $Rq, Rt]>,
            i: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![Rt]> = device
                .sub
                .begin(scale_dm.view().tile::<m![$Rq], 1, m![$Rq = 1 # $nrq, Rt]>(i))
                .fetch::<m![Rt / 8], m![Rt % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rt / 8], m![Rt % 8]>()
                .to_vrf();
            let w_trf: TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rt % 8], m![Rt / 8, Ke]> = device
                .sub
                .begin(w_bf.view())
                .fetch::<m![Rt % 8, Rt / 8, Ke / 16], m![Ke % 16]>()
                .collect::<m![Rt % 8, Rt / 8, Ke / 16], m![Ke % 16]>()
                .to_trf();
            device
                .main
                .begin(xh.view())
                .fetch::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .collect::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .contract_outer::<m![Rh, Tc, Ke / 32, Rt / 8], m![Ke % 32], _, _, _>(&w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Rh, Tc, Rt / 8]>()
                .contract_lane::<m![Rh, Tc, Rt / 8], m![Rt % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<gemm::SlOut, m![Rh, Tc, Rt / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rh, Tc, Rt / 8, Rt / 4 % 2], m![Rt % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_widen_concat::<m![Rh, Tc, Rt / 8], m![Rt % 8]>()
                .vector_final()
                .cast::<bf16, m![Rt % 8 # 16]>()
                .commit_trim::<m![Rt % 8]>()
                .commit_view(y.view_mut().tile::<m![$Rq], 1, m![Rh, Tc, $Rq = 1 #{!} $nrq, Rt]>(i));
        }
    };
}
con_half_fn!(con_q_half, Rq4, 4);
con_half_fn!(con_kv_half, Rq2, 2);

/// KVD post stage of block `hh` of a half buffer (global block `h` = 2 j + hh): the passes of `quad::sliding_post_quad`, reading the per-half buffers.
#[allow(clippy::too_many_arguments)]
fn sliding_post_half(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]>,
    yk: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    yv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    h: usize,
    hh: usize,
) {
    let (cos_lo, sin_lo) = load_rope_block(device, cos, sin, h);

    let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yk.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(hh).to_dm_view(&mut device.tdma, k_tok.view_mut());
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = rope_k_s(device, &k_n, &cos_lo, &sin_lo);
    let k_r = crate::device::kv8::kv8_0(device, &k_r);
    k_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, k_r);

    let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yv.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(hh).to_dm_view(&mut device.tdma, v_tok.view_mut());
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    let v_n = crate::device::kv8::kv8_0(device, &v_n);
    v_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, v_n);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Ds]>(hh).to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = rope_q_s(device, &q_n, &cos_lo, &sin_lo);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Ns, Gs, Ds]>(h));
}

/// Per-half output buffers -> the post layout (the reshape of `quad::sliding_qkv_quad_direct` with Ub -> Rh).
macro_rules! post_half {
    ($device:ident, $yq:ident, $yk:ident, $yv:ident, $q_w:ident, $k_w:ident, $cos:ident, $sin:ident, $q_out:ident, $ia:ident, $ib:ident, $kc:ident, $vc:ident, $h0:expr) => {{
        let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]> = unsafe { $yq.reshape() };
        let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { $yk.reshape() };
        let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { $yv.reshape() };
        sliding_post_half($device, &yq, &yk, &yv, &$q_w, &$k_w, $cos, $sin, $q_out, $ia, $kc, $vc, $h0, 0);
        sliding_post_half($device, &yq, &yk, &yv, &$q_w, &$k_w, $cos, $sin, $q_out, $ib, $kc, $vc, $h0 + 1, 1);
    }};
}

/// The eight contractions of one half (K and V first: their post feeds the ring scatters; Q last).
macro_rules! half_tiles {
    ($device:ident, $x:ident, $k0:ident, $k1:ident, $v0:ident, $v1:ident, $q0:ident, $q1:ident, $q2:ident, $q3:ident, $sq:ident, $sk:ident, $sv:ident, $yq:ident, $yk:ident, $yv:ident) => {{
        con_kv_half($device, &$x, &$k0, &$sk, &mut $yk, 0);
        con_kv_half($device, &$x, &$k1, &$sk, &mut $yk, 1);
        con_kv_half($device, &$x, &$v0, &$sv, &mut $yv, 0);
        con_kv_half($device, &$x, &$v1, &$sv, &mut $yv, 1);
        con_q_half($device, &$x, &$q0, &$sq, &mut $yq, 0);
        con_q_half($device, &$x, &$q1, &$sq, &mut $yq, 1);
        con_q_half($device, &$x, &$q2, &$sq, &mut $yq, 2);
        con_q_half($device, &$x, &$q3, &$sq, &mut $yq, 3);
    }};
}

/// QH body (fused K: `x` = a view of the layer output `res_out`; prologue: a view of the residual input). Same arguments as `quad::sliding_qkv_quad_direct_v`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_quad_h(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    index2: &HbmTensor<i32, Chip, m![Tc]>,
    index3: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    input_norm_quad(device, x, input_rms_weight, xn);
    let xa = load_x_half(device, xn, 0);
    let xb = load_x_half(device, xn, 1);

    let sq = gemm::load_s_q(device, sq);
    let sk = gemm::load_s_kv(device, sk);
    let sv = gemm::load_s_kv(device, sv);
    let w_k0 = gemm::load_w_kv(device, wk, 0);
    let k0 = dec_tile(device, w_k0);
    let w_k1 = gemm::load_w_kv(device, wk, 1);
    let k1 = dec_tile(device, w_k1);
    let w_v0 = gemm::load_w_kv(device, wv, 0);
    let v0 = dec_tile(device, w_v0);
    let w_v1 = gemm::load_w_kv(device, wv, 1);
    let v1 = dec_tile(device, w_v1);
    let w_q0 = gemm::load_w_q(device, wq, 0);
    let q0 = dec_tile(device, w_q0);
    let w_q1 = gemm::load_w_q(device, wq, 1);
    let q1 = dec_tile(device, w_q1);
    let w_q2 = gemm::load_w_q(device, wq, 2);
    let q2 = dec_tile(device, w_q2);
    let w_q3 = gemm::load_w_q(device, wq, 3);
    let q3 = dec_tile(device, w_q3);

    let mut yqa: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq4, Rt]> = DmTensor::new();
    let mut yka: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    let mut yva: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    half_tiles!(device, xa, k0, k1, v0, v1, q0, q1, q2, q3, sq, sk, sv, yqa, yka, yva);

    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    post_half!(device, yqa, yka, yva, q_w, k_w, cos, sin, q_out, index0, index1, k_cache, v_cache, 0);

    let mut yqb: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq4, Rt]> = DmTensor::new();
    let mut ykb: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    let mut yvb: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    half_tiles!(device, xb, k0, k1, v0, v1, q0, q1, q2, q3, sq, sk, sv, yqb, ykb, yvb);
    post_half!(device, yqb, ykb, yvb, q_w, k_w, cos, sin, q_out, index2, index3, k_cache, v_cache, 2);
}
