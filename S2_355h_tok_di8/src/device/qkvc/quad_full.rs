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

    // e2b: two 64-token blocks per op
    post_pair_quad(device, &yq, &yk, cos, sin, &q_w, &k_w, q_out, k_out, v_out, 0);
    post_pair_quad(device, &yq, &yk, cos, sin, &q_w, &k_w, q_out, k_out, v_out, 2);
}

// ---------------------------------------------------------------------------------------------------- e2b: the post stage for TWO 64-token units per op
pub(super) type RopeVrf2<Cl, Sl> = VrfTensor<f32, Chip, Cl, Sl, m![Ub = 2, Df = 256]>;

/// Low-half cos / sin VRFs [Ub = 2, Df = 256] of blocks `h0`, `h0 + 1` (one DMA each instead of one per block).
pub(super) fn load_rope_pair<Cl: M, Sl: M>(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    h0: usize,
) -> (RopeVrf2<Cl, Sl>, RopeVrf2<Cl, Sl>) {
    let cos_dm: DmTensor<bf16, Chip, Cl, Sl, m![Ub = 2, Df]> = cos.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc, Df]>(h0).to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, Cl, Sl, m![Ub = 2, Df]> = sin.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc, Df]>(h0).to_dm(&mut device.tdma);
    macro_rules! half {
        ($t:ident) => {
            device
                .sub
                .begin($t.view().tile::<m![Df], 256, m![Ub = 2, Df = 256 # 512]>(0))
                .fetch::<m![Ub = 2], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Ub = 2, Df = 256 / 8], m![Df = 256 % 8]>()
                .to_vrf()
        };
    }
    (half!(cos_dm), half!(sin_dm))
}

/// `rope_k_s` on [Ub = 2, Df] rows with per-row tables (row r of the data uses row r of cos / sin: the same products as the per-block form).
pub(super) fn rope_k_pair<Cl: M, Sl: M>(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, Cl, Sl, m![Ub = 2, Df]>,
    cos: &RopeVrf2<Cl, Sl>,
    sin_lo: &RopeVrf2<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Ub = 2, Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Ub = 2, Df]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    k.view().tile::<m![Df], 256, m![Ub = 2, Df = 256 # 512]>($own),
                    k.view().tile::<m![Df], 256, m![Ub = 2, Df = 256 # 512]>($other),
                )
                .fetch::<m![Ub = 2, Iz], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Ub = 2, Iz, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Ub = 2, Df = 256 / 8]>()
                .vector_narrow_split::<m![Ub = 2, Df = 256 / 4], m![Df = 256 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Ub = 2, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Ub = 2, Df = 256 #{!} 512]>($own));
        };
    }
    zip_half_k!(0, 256, FpBinaryOp::AddF);
    zip_half_k!(256, 0, FpBinaryOp::SubF);
    out
}

/// `rope_q_s` on both heads and both units at once: q rows [Gf % 2, Ub = 2, Df], the per-unit tables [Ub = 2, Df = 256] broadcast over the leading head axis
/// (as the [Df = 256] tables of `rope_q_s` broadcast over its [Gf % 2] rows).
pub(super) fn rope_q_pair<Cl: M, Sl: M>(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Ub = 2, Df]>,
    cos: &RopeVrf2<Cl, Sl>,
    sin_lo: &RopeVrf2<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Ub = 2, Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Ub = 2, Df]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    q.view().tile::<m![Df], 256, m![Gf % 2, Ub = 2, Df = 256 # 512]>($own),
                    q.view().tile::<m![Df], 256, m![Gf % 2, Ub = 2, Df = 256 # 512]>($other),
                )
                .fetch::<m![Gf % 2, Ub = 2, Iz], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf % 2, Ub = 2, Iz, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Gf % 2, Ub = 2, Df = 256 / 8]>()
                .vector_narrow_split::<m![Gf % 2, Ub = 2, Df = 256 / 4], m![Df = 256 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Gf % 2, Ub = 2, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Gf % 2, Ub = 2, Df = 256 #{!} 512]>($own));
        };
    }
    zip_half!(0, 256, FpBinaryOp::AddF);
    zip_half!(256, 0, FpBinaryOp::SubF);
    out
}

/// The q / k / v post stage of blocks `h0`, `h0 + 1` in one pass over [Ub = 2] rows, from the re-laid q / k rows of exactly those two blocks.
/// The k / v section runs first and drops its rope VRFs before the q section loads its own (the slice VRF holds ~8 KiB: 2 norm weights + one 2 x 2 KiB rope pair).
#[allow(clippy::too_many_arguments)]
fn post_pair_core(
    device: &mut Device,
    q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Ub = 2, Df]>,
    k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Ub = 2, Gg, Rk]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_w: &VrfTensor<f32, Chip, QCl, QSl, m![Df]>,
    k_w: &VrfTensor<f32, Chip, KCl, KSl, m![Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    h0: usize,
) {
    {
        // k / v: one head; slice (token % 32) of cluster token / 32, rows = the two units
        let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Ub = 2, Df]> = unsafe { k_tok.reshape() };
        let (kc, ks) = load_rope_pair::<KCl, KSl>(device, cos, sin, h0);
        let k_n = post_full::norm_k_u2(device, &k_tok, k_w);
        let k_r = rope_k_pair(device, &k_n, &kc, &ks);
        k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut().tile::<m![Ub], 2, m![Ub = 2 #{!} 4, Tc, Df]>(h0));
        let v_n = post_full::norm_v_u2(device, &k_tok);
        v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut().tile::<m![Ub], 2, m![Ub = 2 #{!} 4, Tc, Df]>(h0));
    }
    {
        // q: head = group; slice (token, Gf / 2 % 4), rows = (Gf % 2, unit) head-outer; the norm runs on the 4 flat rows, the rope per head on its 2 unit rows
        let q_flat: DmTensor<bf16, Chip, QCl, QSl, m![Ub, Df]> = unsafe { q_tok.reshape() };
        let q_n = post_full::norm_q_u4(device, &q_flat, q_w);
        let q_n: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Ub = 2, Df]> = unsafe { q_n.reshape() };
        let (qc, qs) = load_rope_pair::<QCl, QSl>(device, cos, sin, h0);
        let q_r = rope_q_pair(device, &q_n, &qc, &qs);
        q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut().tile::<m![Ub], 2, m![Ub = 2 #{!} 4, Tc, Gf, Df]>(h0));
    }
}

/// `post_pair_core` on blocks `h0`, `h0 + 1` of the quad's [Ub, ...] q / k results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_pair_quad(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Ub, Tc, Df]>,
    yk: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rk]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_w: &VrfTensor<f32, Chip, QCl, QSl, m![Df]>,
    k_w: &VrfTensor<f32, Chip, KCl, KSl, m![Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    h0: usize,
) {
    let mut k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Ub = 2, Gg, Rk]> = DmTensor::new();
    yk.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc, Rk]>(h0).to_dm_view(&mut device.tdma, k_tok.view_mut());
    let mut q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Ub = 2, Df]> = DmTensor::new();
    yq.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc, Df]>(h0).to_dm_view(&mut device.tdma, q_tok.view_mut());
    post_pair_core(device, q_tok, k_tok, cos, sin, q_w, k_w, q_out, k_out, v_out, h0);
}

/// `post_pair_core` on one 128-token half ([Ub = 2] = its two blocks) of the e4m3 quad (`quadhf8`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_pair_half(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Ub = 2, Tc, Df]>,
    yk: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub = 2, Tc, Rk]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_w: &VrfTensor<f32, Chip, QCl, QSl, m![Df]>,
    k_w: &VrfTensor<f32, Chip, KCl, KSl, m![Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    h0: usize,
) {
    let mut k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Ub = 2, Gg, Rk]> = DmTensor::new();
    yk.view().to_dm_view(&mut device.tdma, k_tok.view_mut());
    let mut q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Ub = 2, Df]> = DmTensor::new();
    yq.view().to_dm_view(&mut device.tdma, q_tok.view_mut());
    post_pair_core(device, q_tok, k_tok, cos, sin, q_w, k_w, q_out, k_out, v_out, h0);
}
