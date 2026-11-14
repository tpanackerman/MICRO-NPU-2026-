//! Sliding-layer chunk QKV body (stages 0-2).  See `ops_qkvc::sliding_qkv_chunk` for the contract.
use furiosa_opt_std::prelude::*;

use super::{gemm, post, xnorm};
use crate::Chip;
use crate::axes::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    xnorm::input_norm(device, x, input_rms_weight, xn);
    sliding_qkv_tail(device, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// `sliding_qkv` for half `$HALF` of a 128-row residual `[Rh = 2, Tc, Kc, Ke]` (outputs stay 64-row tensors).
macro_rules! sliding_qkv_half_fn {
    ($hname:ident, $norm:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $hname(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
        ) {
            xnorm::$norm(device, x, input_rms_weight, xn);
            sliding_qkv_tail(device, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
sliding_qkv_half_fn!(sliding_qkv_h0, input_norm_h0);
sliding_qkv_half_fn!(sliding_qkv_h1, input_norm_h1);

#[allow(clippy::too_many_arguments)]
fn sliding_qkv_tail(
    device: &mut Device,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    let x_dm = gemm::load_x(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    gemm::proc_tile_q(device, &x_dm, q0, &sq, &mut yq, 0);
    gemm::proc_tile_q(device, &x_dm, q1, &sq, &mut yq, 1);
    gemm::proc_tile_q(device, &x_dm, q2, &sq, &mut yq, 2);
    gemm::proc_tile_q(device, &x_dm, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv(device, &x_dm, k0, &sk, &mut yk, 0);
    gemm::proc_tile_kv(device, &x_dm, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq2, Rt]> = DmTensor::new();
    gemm::proc_tile_kv(device, &x_dm, v0, &sv, &mut yv, 0);
    gemm::proc_tile_kv(device, &x_dm, v1, &sv, &mut yv, 1);

    let (cos_lo, cos_hi, sin_lo, sin_hi) = post::load_rope(device, cos, sin);
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);

    // q: head = group (Gg % 8 = (Ns % 4, Gs)); re-shard to slice (token, Ns % 4), element [Gs, Ds]
    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Tc, Ds]> = unsafe { yq.reshape() };
    let q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = yq.to_dm(&mut device.tdma);
    let q_n = post::norm_q(device, &q_tok, &q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    // k: group = (head, half); element [Dh, Dq] = [Ds]
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Tc, Dq]> = unsafe { yk.reshape() };
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = yk.to_dm(&mut device.tdma);
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, &k_w);
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
