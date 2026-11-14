//! R14 w8qkv (LOSSY, compile-time switch `crate::qkv8`): `quadhf::full_qkv_quad_h` (full-layer quad QKV, half-outer) with the e4m3 x e4m3 contraction of `quad8`:
//! per-token e4m3 x_n (`quad8::input_norm_quad8`), the 9 NATIVE e4m3 weight tiles kept in DM (no f8 -> bf16 decode), e4m3 TRF loads, 64 K per packet.
//! q, k and v (v = norm_v of the k projection) all pass through per-head RMSNorms, which cancel the per-token factor 1 / s (see quad8.rs).
use furiosa_opt_std::prelude::*;

use super::quad8::input_norm_quad8;
use super::quad_full::{QCl, load_rope_block, rope_k_s, rope_q_s};
use super::{gemm, post_full};
use crate::Chip;
use crate::axes::*;

type QSl = post_full::QSl;
type KCl = post_full::KCl;
type KSl = post_full::KSl;

/// Half `j` (blocks 2j, 2j+1) of the e4m3 xq, replicated to the GEMM layout: [Rh, Tc, Ke] per slice (one DMA).
fn load_x_half8(device: &mut Device, xq: &HbmTensor<f8e4m3, Chip, super::quad8::XqPair>, j: usize) -> DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]> {
    let x: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Ub = 2, Tc, Ke]> = xq.view().tile::<m![Ub], 2, m![Ub = 2 # 4, Tc / 2, Kc, Tc % 2, Ke]>(2 * j).to_dm(&mut device.tdma);
    unsafe { x.reshape() }
}

/// One e4m3 tile (DM) against one 128-token e4m3 half: scale VRF + e4m3 TRF load (Sub), e4m3 contraction (64 K / packet), inter-slice reduce, row scale, bf16.
macro_rules! con_half {
    ($device:ident, $xh:ident, $w_bf:ident, $scale_view:expr, $R:ident) => {{
        let scale_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]> = $device
            .sub
            .begin($scale_view)
            .fetch::<m![$R / 8], m![$R % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![$R / 8], m![$R % 8]>()
            .to_vrf();
        let w_trf: TrfTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]> = $device
            .sub
            .begin($w_bf.view())
            .fetch::<m![$R % 8, $R / 8, Ke / 32], m![Ke % 32]>()
            .collect::<m![$R % 8, $R / 8, Ke / 32], m![Ke % 32]>()
            .to_trf();
        $device
            .main
            .begin($xh.view())
            .fetch::<m![Rh, Tc, Ke / 32], m![Ke % 32]>()
            .collect::<m![Rh, Tc, Ke / 32], m![Ke % 32]>()
            .contract_outer::<m![Rh, Tc, Ke / 64, $R / 8], m![Ke % 64], _, _, _>(&w_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![Rh, Tc, $R / 8]>()
            .contract_lane::<m![Rh, Tc, $R / 8], m![$R % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<gemm::SlOut, m![Rh, Tc, $R / 8]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rh, Tc, $R / 8, $R / 4 % 2], m![$R % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_concat::<m![Rh, Tc, $R / 8], m![$R % 8]>()
            .vector_final()
            .cast::<bf16, m![$R % 8 # 16]>()
            .commit_trim::<m![$R % 8]>()
    }};
}

/// QHF e4m3 body: `quadhf::full_qkv_quad_h` with the e4m3 scratch `xq`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv_quad_h8(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xq: &mut HbmTensor<f8e4m3, Chip, super::quad8::XqPair>,
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
    input_norm_quad8(device, x, input_rms_weight, xq);
    let xa = load_x_half8(device, xq, 0);
    let xb = load_x_half8(device, xq, 1);

    let sq = gemm::load_s_qf(device, sq);
    let scale_k: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rk]> = sk.to_dm(&mut device.tdma);
    let dk = gemm::load_w_k1(device, wk);
    let d0 = gemm::load_w_qf(device, wq, 0);
    let d1 = gemm::load_w_qf(device, wq, 1);
    let d2 = gemm::load_w_qf(device, wq, 2);
    let d3 = gemm::load_w_qf(device, wq, 3);
    let d4 = gemm::load_w_qf(device, wq, 4);
    let d5 = gemm::load_w_qf(device, wq, 5);
    let d6 = gemm::load_w_qf(device, wq, 6);
    let d7 = gemm::load_w_qf(device, wq, 7);

    let q_w = post_full::load_norm_weight::<QCl, QSl>(device, q_rms_weight);
    let k_w = post_full::load_norm_weight::<KCl, KSl>(device, k_rms_weight);

    macro_rules! half {
        ($x:ident, $h0:literal) => {{
            let yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rk]> = con_half!(device, $x, dk, scale_k.view(), Rk).commit();
            let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq8, Rt]> = DmTensor::new();
            macro_rules! qtile {
                ($t:ident, $i:literal) => {
                    con_half!(device, $x, $t, sq.view().tile::<m![Rq8], 1, m![Rq8 = 1 # 8, Rt]>($i), Rt)
                        .commit_view(yq.view_mut().tile::<m![Rq8], 1, m![Rh, Tc, Rq8 = 1 #{!} 8, Rt]>($i));
                };
            }
            qtile!(d0, 0);
            qtile!(d1, 1);
            qtile!(d2, 2);
            qtile!(d3, 3);
            qtile!(d4, 4);
            qtile!(d5, 5);
            qtile!(d6, 6);
            qtile!(d7, 7);
            let yq: DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Rh, Tc, Df]> = unsafe { yq.reshape() };
            macro_rules! post_block {
                ($hh:literal) => {{
                    let h = $h0 + $hh;
                    let (kc, ks) = load_rope_block::<KCl, KSl>(device, cos, sin, h);
                    let mut k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Gg, Rk]> = DmTensor::new();
                    yk.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Rk]>($hh).to_dm_view(&mut device.tdma, k_tok.view_mut());
                    let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Df]> = unsafe { k_tok.reshape() };
                    let k_n = post_full::norm_k(device, &k_tok, &k_w);
                    let k_r = rope_k_s(device, &k_n, &kc, &ks);
                    k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Df]>(h));
                    let v_n = post_full::norm_v(device, &k_tok);
                    v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Df]>(h));

                    let (qc, qs) = load_rope_block::<QCl, QSl>(device, cos, sin, h);
                    let mut q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Df]> = DmTensor::new();
                    yq.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Df]>($hh).to_dm_view(&mut device.tdma, q_tok.view_mut());
                    let q_n = post_full::norm_q(device, &q_tok, &q_w);
                    let q_r = rope_q_s(device, &q_n, &qc, &qs);
                    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Gf, Df]>(h));
                }};
            }
            post_block!(0);
            post_block!(1);
        }};
    }
    half!(xa, 0);
    half!(xb, 2);
}
