//! R14 w8qkv (LOSSY, compile-time switch `crate::qkv8`): the sliding quad QKV of `quadp::sliding_qkv_quad_p` with an e4m3 x e4m3 contraction on the NATIVE e4m3 checkpoint
//! weights. What changes against the bf16 quad:
//! - the input RMSNorm output x_n (bf16, unchanged arithmetic) is quantised per TOKEN to e4m3 in the token-sharded producer layout (one token per slice: amax is an intra-slice
//!   reduce): s = sqrt(max_j x_n_j^2) / 448 + tiny, xq = e4m3(x_n / s) (the fold's `quant_part` arithmetic, all-intra-slice);
//! - xq goes to HBM (token-major, as the bf16 xn) and is replicated to the K-chunk slices as e4m3 (half the bytes; 128 B pieces);
//! - every weight tile goes DM -> TRF as e4m3 (no Main f8 -> bf16 decode pass, no DM commit, no Sub re-fetch of a bf16 copy) and is contracted 64 K per packet;
//! - the per-token factor 1 / s is NOT applied: q, k AND v all pass through a per-head RMSNorm in the post stage (q_norm, k_norm with weight, v_norm without), which cancels a
//!   per-token factor up to EPS (y' = y / s, s ~ amax(x_n) / 448: the post-norm inputs only grow, so EPS weighs less than in the bf16 path).
//! The post stage (`quadp::p4::post`) and the epilogue (inter-slice reduce, per-row checkpoint scale, bf16) are the bf16 quad's.
use furiosa_opt_std::prelude::*;

use super::post::{PCl, PSl};
use super::quadp::p4;
use super::{gemm, post};
use crate::axes::*;
use crate::device::f8::quant::{QINV, TINY};
use crate::{Chip, EPS};

type TokCl = m![Tc / 32];
type TokSl4 = m![Tc % 32, Ub, 1 # 2];

const H_F32: f32 = H::SIZE as f32;

/// Joint input norm of the four 64-token blocks (`quad::input_norm_quad`, same bf16 x_n) + per-token e4m3 quantiser, into the e4m3 scratch `xq = [Ub, Tc, Kc, Ke]` (chunks 30, 31 stay zero).
pub(crate) fn input_norm_quad8(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xq: &mut HbmTensor<f8e4m3, Chip, m![Ub, Tc, Kc, Ke]>,
) {
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Kc = 30, Ke]> = x
        .tile::<m![Kc], 30, m![Ub, Tc, Kc = 30 # 32, Ke]>(0)
        .to_dm(&mut device.tdma);
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![H]> = unsafe { x_tok.reshape() };

    let mean_square: DmTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .main
        .begin(x_tok.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl4, m![H]> = rms_weight.to_dm(&mut device.tdma);
    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq6, Hr640]> = unsafe { weight_dm.reshape() };
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq6, Hr640]> = unsafe { x_tok.reshape() };

    let mut xn_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq6, Hr640]> = DmTensor::new();
    macro_rules! chunk {
        ($i:literal) => {{
            let w_vrf: VrfTensor<f32, Chip, TokCl, TokSl4, m![Hr640]> = device
                .sub
                .begin(weight_dm.view().tile::<m![Hq6], 1, m![Hq6 = 1 # 6, Hr640]>($i))
                .fetch::<m![Hr640 / 16], m![Hr640 % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Hr640 / 8], m![Hr640 % 8]>()
                .to_vrf();
            device
                .main
                .begin(x_tok.view().tile::<m![Hq6], 1, m![Hq6 = 1 # 6, Hr640]>($i))
                .fetch::<m![Hr640 / 16], m![Hr640 % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Hr640 / 8], m![Hr640 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Hr640 / 4], m![Hr640 % 4]>()
                .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
                .vector_widen_concat::<m![Hr640 / 8], m![Hr640 % 8]>()
                .vector_final()
                .cast::<bf16, m![Hr640 % 8 # 16]>()
                .commit_trim::<m![Hr640 % 8]>()
                .commit_view(xn_tok.view_mut().tile::<m![Hq6], 1, m![Hq6 = 1 #{!} 6, Hr640]>($i));
        }};
    }
    chunk!(0);
    chunk!(1);
    chunk!(2);
    chunk!(3);
    chunk!(4);
    chunk!(5);

    // per-token e4m3 quantiser of x_n (one token per slice: every reduce is intra-slice)
    let xn_h: DmTensor<bf16, Chip, TokCl, TokSl4, m![H]> = unsafe { xn_tok.reshape() };
    let amax2: DmTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .main
        .begin(xn_h.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s: DmTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .main
        .begin(amax2.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QINV)
        .vector_fp_binary(FpBinaryOp::AddF, TINY)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let s_vrf: VrfTensor<f32, Chip, TokCl, TokSl4, m![1 # 8]> = device
        .sub
        .begin(s.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let xq_tok: DmTensor<f8e4m3, Chip, TokCl, TokSl4, m![H]> = device
        .main
        .begin(xn_h.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &s_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let xq_view: DmTensorView<'_, f8e4m3, Chip, TokCl, TokSl4, m![Kc = 30, Ke]> = unsafe { xq_tok.view().reshape() };
    xq_view.to_hbm_view(&mut device.tdma, xq.view_mut().tile::<m![Kc], 30, m![Ub, Tc, Kc = 30 #{!} 32, Ke]>(0));
}

/// One row tile for all four token blocks: e4m3 TRF straight from the DMA'd tile, e4m3 x e4m3 contraction (64 K per packet), bf16 quad epilogue.
macro_rules! tile_quad8_fn {
    ($name:ident, $Rq:ident, $nrq:literal, $R:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $name(
            device: &mut Device,
            xj: &DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Ub, Tc, Ke]>,
            w_f8: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]>,
            scale_dm: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![$Rq, $R]>,
            y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, $Rq, $R]>,
            i: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]> = device
                .sub
                .begin(scale_dm.view().tile::<m![$Rq], 1, m![$Rq = 1 # $nrq, $R]>(i))
                .fetch::<m![$R / 8], m![$R % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$R / 8], m![$R % 8]>()
                .to_vrf();

            let w_trf: TrfTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]> = device
                .sub
                .begin(w_f8.view())
                .fetch::<m![$R % 8, $R / 8, Ke / 32], m![Ke % 32]>()
                .collect::<m![$R % 8, $R / 8, Ke / 32], m![Ke % 32]>()
                .to_trf();

            device
                .main
                .begin(xj.view())
                .fetch::<m![Ub, Tc, Ke / 32], m![Ke % 32]>()
                .collect::<m![Ub, Tc, Ke / 32], m![Ke % 32]>()
                .contract_outer::<m![Ub, Tc, Ke / 64, $R / 8], m![Ke % 64], _, _, _>(&w_trf)
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
                .commit_view(y.view_mut().tile::<m![$Rq], 1, m![Ub, Tc, $Rq = 1 #{!} $nrq, $R]>(i));
        }
    };
}
tile_quad8_fn!(tile_q_quad8, Rq4, 4, Rt);
tile_quad8_fn!(tile_kv_quad8, Rq2, 2, Rt);

/// `quadp::sliding_qkv_quad_p` with the e4m3 activation / e4m3 TRF contraction (same signature, `xq` = the e4m3 scratch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_quad_p8(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xq: &mut HbmTensor<f8e4m3, Chip, m![Ub, Tc, Kc, Ke]>,
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
    kidx: &HbmTensor<i32, Chip, m![Ub, Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    input_norm_quad8(device, x, input_rms_weight, xq);
    let xj: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Ub, Tc, Ke]> = xq.to_dm(&mut device.tdma);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_quad8(device, &xj, q0, &sq, &mut yq, 0);
    tile_q_quad8(device, &xj, q1, &sq, &mut yq, 1);
    tile_q_quad8(device, &xj, q2, &sq, &mut yq, 2);
    tile_q_quad8(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad8(device, &xj, k0, &sk, &mut yk, 0);
    tile_kv_quad8(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad8(device, &xj, v0, &sv, &mut yv, 0);
    tile_kv_quad8(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, PCl, m![Ns % 4, Gs, 1 # 32], m![Ub, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    let cos_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ub, Ds]> = cos.to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ub, Ds]> = sin.to_dm(&mut device.tdma);
    p4::post(device, &yq, &yk, &yv, &q_w, &k_w, &cos_dm, &sin_dm, q_out.view_mut(), kidx, k_cache, v_cache);
}
