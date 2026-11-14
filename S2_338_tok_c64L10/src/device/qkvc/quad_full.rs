//! R13 qkv256: QUAD full-attention-layer QKV for the 256-row hybrid pass (`full_qkv_quad`): ONE launch per full layer instead of four 64-row `full_qkv` launches, every Q / K weight tile
//! decoded and loaded into the TRF once for 256 tokens. Same per-token arithmetic as `full::full_qkv` (bit-identical): the input norm is `quad::input_norm_quad`, the contraction is the
//! 256-token form of `gemm::tile_body`, the post stage (q / k / v norms, partial RoPE) runs per 64-token block on `post_full` with the low-half rope tables only (cos[i + 256] == cos[i],
//! sin[i + 256] == -sin[i] in the host tables; the high half of the rotation subtracts instead of adds, see `quad.rs`).
use furiosa_opt_std::prelude::*;

use super::quad::input_norm_quad;
use super::{gemm, post_full};
use crate::Chip;
use crate::axes::*;

pub(super) type QCl = m![Gf / 8];
type QSl = post_full::QSl;
type KCl = post_full::KCl;
type KSl = post_full::KSl;
type RopeVrf<Cl, Sl> = post_full::RopeVrf<Cl, Sl>;

/// Decode (Main), TRF load (Sub), 256-token contraction + inter-slice reduce + row scale (Main) of one weight tile; the result is committed by the caller.
macro_rules! body_quad {
    ($device:ident, $xj:ident, $w_f8:ident, $scale_view:expr, $R:ident) => {{
        let scale_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]> = $device
            .sub
            .begin($scale_view)
            .fetch::<m![$R / 8], m![$R % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![$R / 8], m![$R % 8]>()
            .to_vrf();

        let w_bf: DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]> = $device
            .main
            .begin($w_f8.view())
            .fetch::<m![$R, Ke / 32], m![Ke % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![$R, Ke / 32, Ke % 32 / 8], m![Ke % 8]>()
            .cast::<bf16, m![Ke % 8 # 16]>()
            .commit_trim::<m![Ke % 8]>()
            .commit();
        let w_trf: TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]> = $device
            .sub
            .begin(w_bf.view())
            .fetch::<m![$R % 8, $R / 8, Ke / 16], m![Ke % 16]>()
            .collect::<m![$R % 8, $R / 8, Ke / 16], m![Ke % 16]>()
            .to_trf();

        $device
            .main
            .begin($xj.view())
            .fetch::<m![Ub, Tc, Ke / 16], m![Ke % 16]>()
            .collect::<m![Ub, Tc, Ke / 16], m![Ke % 16]>()
            .contract_outer::<m![Ub, Tc, Ke / 32, $R / 8], m![Ke % 32], _, _, _>(&w_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Ub, Tc, $R / 8]>()
            .contract_lane::<m![Ub, Tc, $R / 8], m![$R % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<gemm::SlOut, m![Ub, Tc, $R / 8]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Ub, Tc, $R / 8, $R / 4 % 2], m![$R % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_concat::<m![Ub, Tc, $R / 8], m![$R % 8]>()
            .vector_final()
            .cast::<bf16, m![$R % 8 # 16]>()
            .commit_trim::<m![$R % 8]>()
    }};
}

/// Low-half cos / sin of block `h` as f32 VRFs [Df = 256] for the (Cl, Sl) token layout (`post_full::load_rope` without the high halves).
pub(super) fn load_rope_block<Cl: M, Sl: M>(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    h: usize,
) -> (RopeVrf<Cl, Sl>, RopeVrf<Cl, Sl>) {
    let cos_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = cos.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Df]>(h).to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = sin.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Df]>(h).to_dm(&mut device.tdma);
    macro_rules! half {
        ($t:ident) => {
            device
                .sub
                .begin($t.view().tile::<m![Df], 256, m![Df = 256 # 512]>(0))
                .fetch::<m![1], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Df = 256 / 8], m![Df = 256 % 8]>()
                .to_vrf()
        };
    }
    (half!(cos_dm), half!(sin_dm))
}

/// q RoPE with the low-half tables only (`post_full::rope_q` with `cos_hi = cos_lo`, `sin_hi = -sin_lo`).
pub(super) fn rope_q_s<Cl: M, Sl: M>(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]>,
    cos: &RopeVrf<Cl, Sl>,
    sin_lo: &RopeVrf<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    q.view().tile::<m![Df], 256, m![Gf % 2, Df = 256 # 512]>($own),
                    q.view().tile::<m![Df], 256, m![Gf % 2, Df = 256 # 512]>($other),
                )
                .fetch::<m![Gf % 2, Iz], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf % 2, Iz, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Gf % 2, Df = 256 / 8]>()
                .vector_narrow_split::<m![Gf % 2, Df = 256 / 4], m![Df = 256 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Gf % 2, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Gf % 2, Df = 256 #{!} 512]>($own));
        };
    }
    zip_half!(0, 256, FpBinaryOp::AddF);
    zip_half!(256, 0, FpBinaryOp::SubF);
    out
}

/// k RoPE with the low-half tables only.
pub(super) fn rope_k_s<Cl: M, Sl: M>(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, Cl, Sl, m![Df]>,
    cos: &RopeVrf<Cl, Sl>,
    sin_lo: &RopeVrf<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    k.view().tile::<m![Df], 256, m![Df = 256 # 512]>($own),
                    k.view().tile::<m![Df], 256, m![Df = 256 # 512]>($other),
                )
                .fetch::<m![Iz], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Df = 256 / 8]>()
                .vector_narrow_split::<m![Df = 256 / 4], m![Df = 256 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Df = 256 #{!} 512]>($own));
        };
    }
    zip_half_k!(0, 256, FpBinaryOp::AddF);
    zip_half_k!(256, 0, FpBinaryOp::SubF);
    out
}

/// qkv256: joint input norm + ONE 256-token contraction per weight tile + the per-block q / k / v post stage of a FULL layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv_quad(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
) {
    full_qkv_quad_v(device, x.view(), input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// R14 qkvre: the same body with `x` as a view (a fused launch passes the layer output).
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv_quad_v(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
) {
    input_norm_quad(device, x, input_rms_weight, xn);
    let xj: DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Ub, Tc, Ke]> = xn.to_dm(&mut device.tdma);

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
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq8, Rt]> = DmTensor::new();
    macro_rules! qtile {
        ($t:ident, $i:literal) => {
            body_quad!(device, xj, $t, sq.view().tile::<m![Rq8], 1, m![Rq8 = 1 # 8, Rt]>($i), Rt)
                .commit_view(yq.view_mut().tile::<m![Rq8], 1, m![Ub, Tc, Rq8 = 1 #{!} 8, Rt]>($i));
        };
    }
    qtile!(t0, 0);
    qtile!(t1, 1);
    qtile!(t2, 2);
    qtile!(t3, 3);
    qtile!(t4, 4);
    qtile!(t5, 5);
    qtile!(t6, 6);
    qtile!(t7, 7);
    let scale_k: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rk]> = sk.to_dm(&mut device.tdma);
    let yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rk]> = body_quad!(device, xj, k0, scale_k.view(), Rk).commit();

    let q_w = post_full::load_norm_weight::<QCl, QSl>(device, q_rms_weight);
    let k_w = post_full::load_norm_weight::<KCl, KSl>(device, k_rms_weight);
    let yq: DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Ub, Tc, Df]> = unsafe { yq.reshape() };

    macro_rules! post_block {
        ($h:literal) => {{
            // q: head = group; slice (token, Gf / 2 % 4), element [Gf % 2, Df]
            let (qc, qs) = load_rope_block::<QCl, QSl>(device, cos, sin, $h);
            let mut q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Df]> = DmTensor::new();
            yq.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Df]>($h).to_dm_view(&mut device.tdma, q_tok.view_mut());
            let q_n = post_full::norm_q(device, &q_tok, &q_w);
            let q_r = rope_q_s(device, &q_n, &qc, &qs);
            q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Gf, Df]>($h));

            // k / v: one head; slice (token % 32) of cluster token / 32, element [Df]
            let (kc, ks) = load_rope_block::<KCl, KSl>(device, cos, sin, $h);
            let mut k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Gg, Rk]> = DmTensor::new();
            yk.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Rk]>($h).to_dm_view(&mut device.tdma, k_tok.view_mut());
            let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Df]> = unsafe { k_tok.reshape() };
            let k_n = post_full::norm_k(device, &k_tok, &k_w);
            let k_r = rope_k_s(device, &k_n, &kc, &ks);
            k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Df]>($h));
            let v_n = post_full::norm_v(device, &k_tok);
            v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Df]>($h));
        }};
    }
    post_block!(0);
    post_block!(1);
    post_block!(2);
    post_block!(3);
}
