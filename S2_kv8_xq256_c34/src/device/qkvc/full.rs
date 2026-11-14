//! Full-attention-layer chunk QKV body (stages 0-2).  See `ops_qkvc::full_qkv_chunk` for the contract.
use furiosa_opt_std::prelude::*;

use super::{gemm, post_full, xnorm};
use crate::Chip;
use crate::axes::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
) {
    xnorm::input_norm(device, x, input_rms_weight, xn);
    full_qkv_tail(device, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// `full_qkv` for half `$HALF` of a 128-row residual `[Rh = 2, Tc, Kc, Ke]` (outputs stay 64-row tensors).
macro_rules! full_qkv_half_fn {
    ($hname:ident, $norm:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $hname(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
        ) {
            xnorm::$norm(device, x, input_rms_weight, xn);
            full_qkv_tail(device, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
full_qkv_half_fn!(full_qkv_h0, input_norm_h0);
full_qkv_half_fn!(full_qkv_h1, input_norm_h1);

#[allow(clippy::too_many_arguments)]
fn full_qkv_tail(
    device: &mut Device,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
) {
    type QCl = m![Gf / 8];
    type QSl = post_full::QSl;
    type KCl = post_full::KCl;
    type KSl = post_full::KSl;

    let x_dm = gemm::load_x(device, xn);

    let sq = gemm::load_s_qf(device, sq);
    let k0 = gemm::load_w_k1(device, wk);
    let t0 = gemm::load_w_qf(device, wq, 0);
    let t1 = gemm::load_w_qf(device, wq, 1);
    let t2 = gemm::load_w_qf(device, wq, 2);
    let t3 = gemm::load_w_qf(device, wq, 3);
    let t4 = gemm::load_w_qf(device, wq, 4);
    let t5 = gemm::load_w_qf(device, wq, 5);
    let t6 = gemm::load_w_qf(device, wq, 6);
    let t7 = gemm::load_w_qf(device, wq, 7);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq8, Rt]> = DmTensor::new();
    gemm::proc_tile_qf(device, &x_dm, t0, &sq, &mut yq, 0);
    gemm::proc_tile_qf(device, &x_dm, t1, &sq, &mut yq, 1);
    gemm::proc_tile_qf(device, &x_dm, t2, &sq, &mut yq, 2);
    gemm::proc_tile_qf(device, &x_dm, t3, &sq, &mut yq, 3);
    gemm::proc_tile_qf(device, &x_dm, t4, &sq, &mut yq, 4);
    gemm::proc_tile_qf(device, &x_dm, t5, &sq, &mut yq, 5);
    gemm::proc_tile_qf(device, &x_dm, t6, &sq, &mut yq, 6);
    gemm::proc_tile_qf(device, &x_dm, t7, &sq, &mut yq, 7);
    let yk = gemm::proc_k1(device, &x_dm, k0, sk);

    // q: head = group (Gg % 8 = (Gf / 2 % 4, Gf % 2)); slice (token, Gf / 2 % 4), element [Gf % 2, Df]
    let (qc_lo, qc_hi, qs_lo, qs_hi) = post_full::load_rope::<QCl, QSl>(device, cos, sin);
    let q_w = post_full::load_norm_weight::<QCl, QSl>(device, q_rms_weight);
    let yq: DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Tc, Df]> = unsafe { yq.reshape() };
    let q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Df]> = yq.to_dm(&mut device.tdma);
    let q_n = post_full::norm_q(device, &q_tok, &q_w);
    let q_r = post_full::rope_q(device, &q_n, &qc_lo, &qc_hi, &qs_lo, &qs_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    // k / v: one head; slice (token % 32) of cluster token / 32, element [Df] (16 groups x 32 rows from both clusters)
    let (kc_lo, kc_hi, ks_lo, ks_hi) = post_full::load_rope::<KCl, KSl>(device, cos, sin);
    let k_w = post_full::load_norm_weight::<KCl, KSl>(device, k_rms_weight);
    let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Gg, Rk]> = yk.to_dm(&mut device.tdma);
    let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Df]> = unsafe { k_tok.reshape() };
    let k_n = post_full::norm_k(device, &k_tok, &k_w);
    let k_r = post_full::rope_k(device, &k_n, &kc_lo, &kc_hi, &ks_lo, &ks_hi);
    k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut());
    let v_n = post_full::norm_v(device, &k_tok);
    v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}
