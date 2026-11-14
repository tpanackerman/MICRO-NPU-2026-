//! PAIR chunk QKV for the hybrid pass (two 64-row token blocks = the 128 rows of a pass, ONE launch per sliding layer): the weight tiles are streamed, decoded and loaded into the TRF ONCE
//! (`gemm::proc_tile_*_pair`), both blocks are contracted against them; the input norm and the post stage (per-head norms, RoPE) run per block with the unchanged 64-row code.
use furiosa_opt_std::prelude::*;

use super::{gemm, post, xnorm};
use crate::Chip;
use crate::axes::*;

/// Post stage of ONE 64-token block of a sliding layer (the second half of `sliding::sliding_qkv_tail`): re-shard, per-head norms, RoPE, stores.
#[allow(clippy::too_many_arguments)]
pub(super) fn sliding_post(
    device: &mut Device,
    yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]>,
    yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]>,
    yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    let (cos_lo, cos_hi, sin_lo, sin_hi) = post::load_rope(device, cos, sin);

    // q: head = group (Gg % 8 = (Ns % 4, Gs)); re-shard to slice (token, Ns % 4), element [Gs, Ds]
    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Tc, Ds]> = unsafe { yq.reshape() };
    let q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = yq.to_dm(&mut device.tdma);
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    // k: group = (head, half); element [Dh, Dq] = [Ds]
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Tc, Dq]> = unsafe { yk.reshape() };
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = yk.to_dm(&mut device.tdma);
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = post::rope_k(device, &k_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k_r: DmTensor<bf16, Chip, post::PCl, post::StageSl, m![Ds]> = unsafe { k_r.reshape() };
    k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut());

    // v
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Tc, Dq]> = unsafe { yv.reshape() };
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = yv.to_dm(&mut device.tdma);
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    let v_n: DmTensor<bf16, Chip, post::PCl, post::StageSl, m![Ds]> = unsafe { v_n.reshape() };
    v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

/// Sliding-layer QKV of the two halves of a 128-row residual `[Rh = 2, Tc, Kc, Ke]` in one launch. `xn0` / `xn1` are the zero-initialised scratches of the two blocks (two handles, so
/// the write of block 1 can never race the read of block 0). Outputs are the 64-row stage pieces of each block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_pair(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn0: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    xnorm::input_norm_h0(device, x, input_rms_weight, xn0);
    xnorm::input_norm_h1(device, x, input_rms_weight, xn1);
    let x0 = gemm::load_x(device, xn0);
    let x1 = gemm::load_x(device, xn1);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    let mut yq1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    gemm::proc_tile_q_pair(device, &x0, &x1, q0, &sq, &mut yq0, &mut yq1, 0);
    gemm::proc_tile_q_pair(device, &x0, &x1, q1, &sq, &mut yq0, &mut yq1, 1);
    gemm::proc_tile_q_pair(device, &x0, &x1, q2, &sq, &mut yq0, &mut yq1, 2);
    gemm::proc_tile_q_pair(device, &x0, &x1, q3, &sq, &mut yq0, &mut yq1, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    let mut yk1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv_pair(device, &x0, &x1, k0, &sk, &mut yk0, &mut yk1, 0);
    gemm::proc_tile_kv_pair(device, &x0, &x1, k1, &sk, &mut yk0, &mut yk1, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    let mut yv1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv_pair(device, &x0, &x1, v0, &sv, &mut yv0, &mut yv1, 0);
    gemm::proc_tile_kv_pair(device, &x0, &x1, v1, &sv, &mut yv0, &mut yv1, 1);

    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post(device, yq0, yk0, yv0, &q_w, &k_w, cos0, sin0, q_out0, k_out0, v_out0);
    sliding_post(device, yq1, yk1, yv1, &q_w, &k_w, cos1, sin1, q_out1, k_out1, v_out1);
}

/// KVD post stage of ONE 64-token block: like `sliding_post`, but k and v are SCATTERED straight into the flat ring caches (`index[t]` = byte offset of the cache row of token t; padding
/// rows name the scratch row) instead of being stored to a stage tensor.
#[allow(clippy::too_many_arguments)]
fn sliding_post_direct(
    device: &mut Device,
    yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]>,
    yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]>,
    yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    let (cos_lo, cos_hi, sin_lo, sin_hi) = post::load_rope(device, cos, sin);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Tc, Ds]> = unsafe { yq.reshape() };
    let q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = yq.to_dm(&mut device.tdma);
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Tc, Dq]> = unsafe { yk.reshape() };
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = yk.to_dm(&mut device.tdma);
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = post::rope_k(device, &k_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    k_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, k_r);

    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Tc, Dq]> = unsafe { yv.reshape() };
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = yv.to_dm(&mut device.tdma);
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    v_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, v_n);
}

/// KVD pair QKV: `sliding_qkv_pair` with the K / V stores replaced by scatters into the ring caches (`index0` / `index1` = per-block cache row byte offsets).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_pair_direct(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn0: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    xnorm::input_norm_h0(device, x, input_rms_weight, xn0);
    xnorm::input_norm_h1(device, x, input_rms_weight, xn1);
    let x0 = gemm::load_x(device, xn0);
    let x1 = gemm::load_x(device, xn1);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    let mut yq1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    gemm::proc_tile_q_pair(device, &x0, &x1, q0, &sq, &mut yq0, &mut yq1, 0);
    gemm::proc_tile_q_pair(device, &x0, &x1, q1, &sq, &mut yq0, &mut yq1, 1);
    gemm::proc_tile_q_pair(device, &x0, &x1, q2, &sq, &mut yq0, &mut yq1, 2);
    gemm::proc_tile_q_pair(device, &x0, &x1, q3, &sq, &mut yq0, &mut yq1, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    let mut yk1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv_pair(device, &x0, &x1, k0, &sk, &mut yk0, &mut yk1, 0);
    gemm::proc_tile_kv_pair(device, &x0, &x1, k1, &sk, &mut yk0, &mut yk1, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv0: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    let mut yv1: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv_pair(device, &x0, &x1, v0, &sv, &mut yv0, &mut yv1, 0);
    gemm::proc_tile_kv_pair(device, &x0, &x1, v1, &sv, &mut yv0, &mut yv1, 1);

    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_direct(device, yq0, yk0, yv0, &q_w, &k_w, cos0, sin0, q_out0, index0, k_cache, v_cache);
    sliding_post_direct(device, yq1, yk1, yv1, &q_w, &k_w, cos1, sin1, q_out1, index1, k_cache, v_cache);
}
