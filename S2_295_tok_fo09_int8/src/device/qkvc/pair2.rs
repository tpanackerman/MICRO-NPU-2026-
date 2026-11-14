//! R10 qkvchain variants of the PAIR chunk QKV (`pair::sliding_qkv_pair`). All are bit-identical by construction to the production pair kernel (same per-token / per-element arithmetic;
//! only the number of passes / launch of the same VE sequences changes).
//!
//! (joint input norm) ONE input RMSNorm pass over all 128 tokens of the pass (token (rh, tc) on slice (tc % 32, rh) of cluster tc / 32) instead of two 64-token passes: one x load, one rms-weight load, half the
//!     norm Main passes. The two xn blocks go to ONE joint scratch `xn = [Rh, Tc, Kc, Ke]` (single handle: the host would pass one zero-initialised 1 MB scratch instead of xn + xn1); the GEMM stage is unchanged.
//! sliding_qkv_pair_j: joint norm + ONE 128-token contraction per weight tile (x of both blocks in one DM tensor [Rh, Tc, Ke]): one TRF load and one contraction pass per tile instead of two.
use furiosa_opt_std::prelude::*;

use super::pair::sliding_post;
use super::{gemm, post};
use crate::axes::*;
use crate::{Chip, EPS};

type TokCl = m![Tc / 32];
type TokSl2 = m![Tc % 32, Rh, 1 # 4];

const H_F32: f32 = H::SIZE as f32;

/// Joint input norm of both 64-token blocks of `x = [Rh, Tc, Kc, Ke]` into ONE scratch `xn = [Rh, Tc, Kc, Ke]` (chunks 30, 31 stay zero).
fn input_norm_joint(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
) {
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl2, m![Kc = 30, Ke]> = x
        .view()
        .tile::<m![Kc], 30, m![Rh, Tc, Kc = 30 # 32, Ke]>(0)
        .to_dm(&mut device.tdma);
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl2, m![H]> = unsafe { x_tok.reshape() };

    let mean_square: DmTensor<f32, Chip, TokCl, TokSl2, m![1 # 8]> = device
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

    let rms: DmTensor<f32, Chip, TokCl, TokSl2, m![1 # 8]> = device
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
    let rms_vrf: VrfTensor<f32, Chip, TokCl, TokSl2, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl2, m![H]> = rms_weight.to_dm(&mut device.tdma);
    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl2, m![Hq, Hr]> = unsafe { weight_dm.reshape() };
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl2, m![Hq, Hr]> = unsafe { x_tok.reshape() };

    let mut xn_tok: DmTensor<bf16, Chip, TokCl, TokSl2, m![Hq, Hr]> = DmTensor::new();
    macro_rules! chunk {
        ($i:literal) => {{
            let w_vrf: VrfTensor<f32, Chip, TokCl, TokSl2, m![Hr]> = device
                .sub
                .begin(weight_dm.view().tile::<m![Hq], 1, m![Hq = 1 # 3, Hr]>($i))
                .fetch::<m![Hr / 16], m![Hr % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Hr / 8], m![Hr % 8]>()
                .to_vrf();
            device
                .main
                .begin(x_tok.view().tile::<m![Hq], 1, m![Hq = 1 # 3, Hr]>($i))
                .fetch::<m![Hr / 16], m![Hr % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Hr / 8], m![Hr % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Hr / 4], m![Hr % 4]>()
                .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
                .vector_widen_concat::<m![Hr / 8], m![Hr % 8]>()
                .vector_final()
                .cast::<bf16, m![Hr % 8 # 16]>()
                .commit_trim::<m![Hr % 8]>()
                .commit_view(xn_tok.view_mut().tile::<m![Hq], 1, m![Hq = 1 #{!} 3, Hr]>($i));
        }};
    }
    chunk!(0);
    chunk!(1);
    chunk!(2);

    let xn_view: DmTensorView<'_, bf16, Chip, TokCl, TokSl2, m![Kc = 30, Ke]> = unsafe { xn_tok.view().reshape() };
    xn_view.to_hbm_view(&mut device.tdma, xn.view_mut().tile::<m![Kc], 30, m![Rh, Tc, Kc = 30 #{!} 32, Ke]>(0));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// qb: joint 128-token contraction.

/// Both normalised blocks in ONE DM tensor [Rh, Tc, Ke] per slice (one DMA).
fn load_x_joint(device: &mut Device, xn: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]> {
    xn.to_dm(&mut device.tdma)
}

/// One row tile for both token blocks in ONE contraction pass over [Rh, Tc] = 128 tokens (same arithmetic per token as `gemm::tile_body`).
macro_rules! tile_joint_fn {
    ($name:ident, $Rq:ident, $nrq:literal, $R:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            device: &mut Device,
            xj: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
            w_f8: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]>,
            scale_dm: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![$Rq, $R]>,
            y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, $Rq, $R]>,
            i: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]> = device
                .sub
                .begin(scale_dm.view().tile::<m![$Rq], 1, m![$Rq = 1 # $nrq, $R]>(i))
                .fetch::<m![$R / 8], m![$R % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$R / 8], m![$R % 8]>()
                .to_vrf();

            let w_bf: DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]> = device
                .main
                .begin(w_f8.view())
                .fetch::<m![$R, Ke / 32], m![Ke % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![$R, Ke / 32, Ke % 32 / 8], m![Ke % 8]>()
                .cast::<bf16, m![Ke % 8 # 16]>()
                .commit_trim::<m![Ke % 8]>()
                .commit();
            let w_trf: TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]> = device
                .sub
                .begin(w_bf.view())
                .fetch::<m![$R % 8, $R / 8, Ke / 16], m![Ke % 16]>()
                .collect::<m![$R % 8, $R / 8, Ke / 16], m![Ke % 16]>()
                .to_trf();

            device
                .main
                .begin(xj.view())
                .fetch::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .collect::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .contract_outer::<m![Rh, Tc, Ke / 32, $R / 8], m![Ke % 32], _, _, _>(&w_trf)
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
                .commit_view(y.view_mut().tile::<m![$Rq], 1, m![Rh, Tc, $Rq = 1 #{!} $nrq, $R]>(i));
        }
    };
}
tile_joint_fn!(tile_q_joint, Rq4, 4, Rt);
tile_joint_fn!(tile_kv_joint, Rq2, 2, Rt);

/// Post stage of block `h` of the joint buffers: re-shard straight from the combined [Rh, Tc, Rq, Rt] buffers (view of block h), then the unchanged per-head norms / RoPE / stores.
#[allow(clippy::too_many_arguments)]
fn sliding_post_j(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]>,
    yk: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    yv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    h: usize,
) {
    let (cos_lo, cos_hi, sin_lo, sin_hi) = post::load_rope(device, cos, sin);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Ds]>(h).to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yk.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(h).to_dm_view(&mut device.tdma, k_tok.view_mut());
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = post::rope_k(device, &k_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    let k_r: DmTensor<bf16, Chip, post::PCl, post::StageSl, m![Ds]> = unsafe { k_r.reshape() };
    k_r.view().to_hbm_view(&mut device.tdma, k_out.view_mut());

    let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yv.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(h).to_dm_view(&mut device.tdma, v_tok.view_mut());
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    let v_n: DmTensor<bf16, Chip, post::PCl, post::StageSl, m![Ds]> = unsafe { v_n.reshape() };
    v_n.view().to_hbm_view(&mut device.tdma, v_out.view_mut());
}

/// qb: joint input norm + joint contraction + post straight from the combined buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_pair_j(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
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
    input_norm_joint(device, x, input_rms_weight, xn);
    let xj = load_x_joint(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_joint(device, &xj, q0, &sq, &mut yq, 0);
    tile_q_joint(device, &xj, q1, &sq, &mut yq, 1);
    tile_q_joint(device, &xj, q2, &sq, &mut yq, 2);
    tile_q_joint(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint(device, &xj, k0, &sk, &mut yk, 0);
    tile_kv_joint(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint(device, &xj, v0, &sv, &mut yv, 0);
    tile_kv_joint(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_j(device, &yq, &yk, &yv, &q_w, &k_w, cos0, sin0, q_out0, k_out0, v_out0, 0);
    sliding_post_j(device, &yq, &yk, &yv, &q_w, &k_w, cos1, sin1, q_out1, k_out1, v_out1, 1);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// qbd: qb on top of KVD (`pair::sliding_qkv_pair_direct`): K / V scattered straight into the flat ring caches.

/// KVD post stage of block `h` of the joint buffers (as `sliding_post_j`, K / V scattered into the ring caches with `index` = byte offsets of the cache rows of the 64 tokens).
#[allow(clippy::too_many_arguments)]
fn sliding_post_jd(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]>,
    yk: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    yv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    h: usize,
) {
    let (cos_lo, cos_hi, sin_lo, sin_hi) = post::load_rope(device, cos, sin);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Ds]>(h).to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut());

    let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yk.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(h).to_dm_view(&mut device.tdma, k_tok.view_mut());
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = post::rope_k(device, &k_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    k_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, k_r);

    let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yv.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Dq]>(h).to_dm_view(&mut device.tdma, v_tok.view_mut());
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    v_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, v_n);
}

/// qbd: joint input norm + joint contraction + KVD post (K / V scatter into the ring caches).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_pair_direct_j(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
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
    input_norm_joint(device, x, input_rms_weight, xn);
    let xj = load_x_joint(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_joint(device, &xj, q0, &sq, &mut yq, 0);
    tile_q_joint(device, &xj, q1, &sq, &mut yq, 1);
    tile_q_joint(device, &xj, q2, &sq, &mut yq, 2);
    tile_q_joint(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint(device, &xj, k0, &sk, &mut yk, 0);
    tile_kv_joint(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint(device, &xj, v0, &sv, &mut yv, 0);
    tile_kv_joint(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rh, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rh, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_jd(device, &yq, &yk, &yv, &q_w, &k_w, cos0, sin0, q_out0, index0, k_cache, v_cache, 0);
    sliding_post_jd(device, &yq, &yk, &yv, &q_w, &k_w, cos1, sin1, q_out1, index1, k_cache, v_cache, 1);
}
