//! R13 qkv256: QUAD sliding-layer QKV for the 256-row hybrid pass (`sliding_qkv_quad_direct`): ONE launch per sliding layer for all four 64-row blocks. Same per-token arithmetic as
//! `pair2::sliding_qkv_pair_direct_j` (qbd), so results are bit-identical; what changes is the amount of work that is shared:
//! - ONE joint input RMSNorm pass over the 256 tokens (token (ub, tc) on slice (tc % 32, ub) of cluster tc / 32; 128 live slices per cluster),
//! - every weight tile (Q 4 + K 2 + V 2 per layer) is DMA'd, decoded (f8 -> bf16, Main) and loaded into the TRF ONCE and contracted against all 256 tokens (`[Ub, Tc]`),
//! - the rope tables are loaded once, the post stage (per-head norms, RoPE, K / V scatter into the ring caches) runs per 64-token block on the unchanged code.
use furiosa_opt_std::prelude::*;

use super::{gemm, post};
use crate::axes::*;
use crate::{Chip, EPS};

type TokCl = m![Tc / 32];
type TokSl4 = m![Tc % 32, Ub, 1 # 2];

const H_F32: f32 = H::SIZE as f32;

/// Joint input norm of the four 64-token blocks of `x = [Ub, Tc, Kc, Ke]` into ONE scratch `xn = [Ub, Tc, Kc, Ke]` (chunks 30, 31 stay zero).
pub(super) fn input_norm_quad(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
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

    let xn_view: DmTensorView<'_, bf16, Chip, TokCl, TokSl4, m![Kc = 30, Ke]> = unsafe { xn_tok.view().reshape() };
    xn_view.to_hbm_view(&mut device.tdma, xn.view_mut().tile::<m![Kc], 30, m![Ub, Tc, Kc = 30 #{!} 32, Ke]>(0));
}

/// All four normalised blocks in ONE DM tensor [Ub, Tc, Ke] per slice (one DMA).
pub(super) fn load_x_quad(device: &mut Device, xn: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Ub, Tc, Ke]> {
    xn.to_dm(&mut device.tdma)
}

/// One row tile for all four token blocks in ONE contraction pass over [Ub, Tc] = 256 tokens (same arithmetic per token as `gemm::tile_body`).
macro_rules! tile_quad_fn {
    ($name:ident, $Rq:ident, $nrq:literal, $R:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(super) fn $name(
            device: &mut Device,
            xj: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Ub, Tc, Ke]>,
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
                .commit_view(y.view_mut().tile::<m![$Rq], 1, m![Ub, Tc, $Rq = 1 #{!} $nrq, $R]>(i));
        }
    };
}
tile_quad_fn!(tile_q_quad, Rq4, 4, Rt);
tile_quad_fn!(tile_kv_quad, Rq2, 2, Rt);

/// cos / sin of block `h` as f32 VRFs [Ds = 128]: ONLY the low halves. The host tables satisfy cos[i + 128] == cos[i] and sin[i + 128] == -sin[i] bit for bit (`qkvc::rope_rows_for_positions`:
/// the two halves are the same value with sign -1 / +1 on the sine), so the high half of the rotation reuses the low-half VRFs with a subtract instead of an add: half the VRF bytes of
/// `post::load_rope` (four 128-float VRFs per block do not fit the 8 KiB VRF once four blocks are scheduled).
pub(super) fn load_rope_block(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    h: usize,
) -> (post::RopeVrf, post::RopeVrf) {
    let cos_dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = cos.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Ds]>(h).to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = sin.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Ds]>(h).to_dm(&mut device.tdma);
    macro_rules! half {
        ($t:ident, $start:literal) => {
            device
                .sub
                .begin($t.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>($start))
                .fetch::<m![1], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf()
        };
    }
    let cos_lo: post::RopeVrf = half!(cos_dm, 0);
    let sin_lo: post::RopeVrf = half!(sin_dm, 0);
    (cos_lo, sin_lo)
}

/// q RoPE with the low-half tables only (`post::rope_q` with `cos_hi = cos_lo`, `sin_hi = -sin_lo`: the high half is `own * cos - other * sin_lo`).
pub(super) fn rope_q_s(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]>,
    cos: &post::RopeVrf,
    sin_lo: &post::RopeVrf,
) -> DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> {
    let mut out: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>($own),
                    q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>($other),
                )
                .fetch::<m![Iz, Gs], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Gs, Ds = 128 / 8]>()
                .vector_narrow_split::<m![Gs, Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half!(0, 128, FpBinaryOp::AddF);
    zip_half!(128, 0, FpBinaryOp::SubF);
    out
}

/// k RoPE with the low-half tables only (see `rope_q_s`).
pub(super) fn rope_k_s(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &post::RopeVrf,
    sin_lo: &post::RopeVrf,
) -> DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> {
    let mut out: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $op:expr) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>($own),
                    k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>($other),
                )
                .fetch::<m![Iz], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Ds = 128 / 8]>()
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half_k!(0, 128, FpBinaryOp::AddF);
    zip_half_k!(128, 0, FpBinaryOp::SubF);
    out
}

/// KVD post stage of block `h` of the joint buffers: q stored into block `h` of `q_out`, k / v scattered into the ring caches with `index` (cache row byte offsets of the 64 tokens).
#[allow(clippy::too_many_arguments)]
fn sliding_post_quad(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Ub, Tc, Ds]>,
    yk: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]>,
    yv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    h: usize,
) {
    let (cos_lo, sin_lo) = load_rope_block(device, cos, sin, h);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Ds]>(h).to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = rope_q_s(device, &q_n, &cos_lo, &sin_lo);
    q_r.view().to_hbm_view(&mut device.tdma, q_out.view_mut().tile::<m![Ub], 1, m![Ub = 1 #{!} 4, Tc, Ns, Gs, Ds]>(h));

    let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yk.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Dq]>(h).to_dm_view(&mut device.tdma, k_tok.view_mut());
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = rope_k_s(device, &k_n, &cos_lo, &sin_lo);
    k_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, k_r);

    let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    yv.view().tile::<m![Ub], 1, m![Ub = 1 # 4, Tc, Dq]>(h).to_dm_view(&mut device.tdma, v_tok.view_mut());
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    v_cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, v_n);
}

/// qkv256: joint input norm + ONE 256-token contraction per weight tile + KVD post (K / V scatter into the ring caches) of the four 64-token blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_quad_direct(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
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
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    input_norm_quad(device, x.view(), input_rms_weight, xn);
    let xj = load_x_quad(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_quad(device, &xj, q0, &sq, &mut yq, 0);
    tile_q_quad(device, &xj, q1, &sq, &mut yq, 1);
    tile_q_quad(device, &xj, q2, &sq, &mut yq, 2);
    tile_q_quad(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad(device, &xj, k0, &sk, &mut yk, 0);
    tile_kv_quad(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad(device, &xj, v0, &sv, &mut yv, 0);
    tile_kv_quad(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Ub, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index0, k_cache, v_cache, 0);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index1, k_cache, v_cache, 1);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index2, k_cache, v_cache, 2);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index3, k_cache, v_cache, 3);
}

pub(crate) fn sliding_qkv_quad_direct_v(
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
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    input_norm_quad(device, x, input_rms_weight, xn);
    let xj = load_x_quad(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_quad(device, &xj, q0, &sq, &mut yq, 0);
    tile_q_quad(device, &xj, q1, &sq, &mut yq, 1);
    tile_q_quad(device, &xj, q2, &sq, &mut yq, 2);
    tile_q_quad(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad(device, &xj, k0, &sk, &mut yk, 0);
    tile_kv_quad(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_quad(device, &xj, v0, &sv, &mut yv, 0);
    tile_kv_quad(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Ub, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index0, k_cache, v_cache, 0);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index1, k_cache, v_cache, 1);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index2, k_cache, v_cache, 2);
    sliding_post_quad(device, &yq, &yk, &yv, &q_w, &k_w, cos, sin, q_out, index3, k_cache, v_cache, 3);
}
