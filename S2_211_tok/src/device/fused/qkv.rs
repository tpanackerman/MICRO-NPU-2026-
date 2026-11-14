//! QKV of the fused sliding layer: `pair2::sliding_qkv_quad_direct_j` (quad2 branch: both 128-row pairs, every weight tile streamed / decoded / TRF-loaded once,
//! the 4 post units serialized by an exact-zero pin) with its operands carved from the fused kernel's tensors: x from the residual, the normalised x and
//! q in kernel-internal HBM tensors, the scales / norm weights from the small-weight blob, k / v from the grouped `[Wkv, ..]` tensor, cos / sin from the table region.
//! Same arithmetic as the production pair QKV.
use furiosa_opt_std::prelude::*;

use super::{Sb15, Sb16, Sb30, Sb240, Sw, Tb, Tt14, Wkv, sw, tb};
use crate::Chip;
use crate::axes::*;
use crate::device::qkvc::{gemm, post};
use crate::{EPS, ops_ffnc::Tk256};

type TokCl = m![Tc / 32];
type TokSl2 = m![Tc % 32, Rh, 1 # 4];
const H_F32: f32 = H::SIZE as f32;

/// Joint input norm of pair `w` (rows 128 w ..) of the residual into pair `w` of the scratch `xn` (`pair2::input_norm_joint_w`).
fn input_norm_pair(device: &mut Device, res_in: &HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>, w: usize, small: &HbmTensor<bf16, Chip, m![Sw]>, xn: &mut HbmTensor<bf16, Chip, m![Pw, Rh, Tc, Kc, Ke]>) {
    let x: HbmTensorView<'_, bf16, Chip, m![Pw, Rh, Tc, Kc, Ke]> = unsafe { res_in.view().reshape() };
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl2, m![Kc = 30, Ke]> = x
        .tile::<m![Pw], 1, m![Pw = 1 # 2, Rh, Tc, Kc, Ke]>(w)
        .tile::<m![Kc], 30, m![Pw = 1 # 2, Rh, Tc, Kc = 30 # 32, Ke]>(0)
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

    let norms: HbmTensorView<'_, bf16, Chip, m![Sb16, H]> = unsafe { small.view().reshape() };
    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl2, m![H]> =
        norms.tile::<m![Sb16], 1, m![Sb16 = 1 # 16, H]>(sw::INPUT_NORM).to_dm(&mut device.tdma);
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
    xn_view.to_hbm_view(
        &mut device.tdma,
        xn.view_mut()
            .tile::<m![Pw], 1, m![Pw = 1 #{!} 2, Rh, Tc, Kc, Ke]>(w)
            .tile::<m![Kc], 30, m![Pw = 1 #{!} 2, Rh, Tc, Kc = 30 #{!} 32, Ke]>(0),
    );
}

/// Pair `w` of the normalised x in ONE DM tensor [Rh, Tc, Ke] per slice.
fn load_x_pair(device: &mut Device, xn: &HbmTensor<bf16, Chip, m![Pw, Rh, Tc, Kc, Ke]>, w: usize) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]> {
    xn.view().tile::<m![Pw], 1, m![Pw = 1 # 2, Rh, Tc, Kc, Ke]>(w).to_dm(&mut device.tdma)
}

/// One row tile for BOTH pairs (`pair2::tile_joint2_fn!`).
macro_rules! tile_joint2_fn {
    ($name:ident, $Rq:ident, $nrq:literal, $R:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            device: &mut Device,
            xa: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
            xb: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
            w_f8: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]>,
            scale_dm: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![$Rq, $R]>,
            y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Pw, Rh, Tc, $Rq, $R]>,
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

            macro_rules! con {
                ($xj:ident, $p:literal) => {
                    device
                        .main
                        .begin($xj.view())
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
                        .commit_view(
                            y.view_mut()
                                .tile::<m![Pw], 1, m![Pw = 1 #{!} 2, Rh, Tc, $Rq, $R]>($p)
                                .tile::<m![$Rq], 1, m![Pw = 1 #{!} 2, Rh, Tc, $Rq = 1 #{!} $nrq, $R]>(i),
                        )
                };
            }
            con!(xa, 0);
            con!(xb, 1);
        }
    };
}
tile_joint2_fn!(tile_q_joint2, Rq4, 4, Rt);
tile_joint2_fn!(tile_kv_joint2, Rq2, 2, Rt);

type PostZero = VrfTensor<f32, Chip, post::PCl, post::PSl, m![1 # 8]>;

/// Exact +0.0 per post slice from a unit's rotated K (`pair2::zero_from_k`).
fn zero_from_k(device: &mut Device, k: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]>) -> PostZero {
    device
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// Exact +0.0 per post slice from head 0 of a unit's rotated q.
fn zero_from_q(device: &mut Device, q: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]>) -> PostZero {
    device
        .main
        .begin(q.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(0))
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// Exact +0.0 from a unit's rotated k, + `zq` (the zero of its rotated q): exists only once both rotations are done.
fn zero_from_k_and(device: &mut Device, k: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]>, zq: &PostZero) -> PostZero {
    device
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, zq)
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// One rope half as an f32 VRF, + the exact zero (`pair2::rope_half_pinned`).
fn rope_half_pinned(device: &mut Device, t: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]>, start: usize, zero: &PostZero) -> post::RopeVrf {
    device
        .main
        .begin(t.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(start))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// One norm weight `[Ds]` of the small blob as an f32 VRF (`post::load_norm_weight`).
fn norm_weight(device: &mut Device, small: &HbmTensor<bf16, Chip, m![Sw]>, slot: usize) -> VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]> {
    let v: HbmTensorView<'_, bf16, Chip, m![Sb240, Ds]> = unsafe { small.view().reshape() };
    let weight_dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = v.tile::<m![Sb240], 1, m![Sb240 = 1 # 240, Ds]>(slot).to_dm(&mut device.tdma);
    device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// cos or sin rows of unit `u` from the table region.
fn rope_rows(device: &mut Device, tab: &HbmTensor<bf16, Chip, m![Tb]>, slot: usize) -> DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> {
    let v: HbmTensorView<'_, bf16, Chip, m![Tt14, Tc, Ds]> = unsafe { tab.view().reshape() };
    v.tile::<m![Tt14], 1, m![Tt14 = 1 # 48, Tc, Ds]>(slot).to_dm(&mut device.tdma)
}

macro_rules! rope_vrfs_of {
    ($device:ident, $cos:ident, $sin:ident) => {
        post::rope_vrfs($device, &$cos, &$sin)
    };
    ($device:ident, $cos:ident, $sin:ident, $zero:ident) => {
        (
            rope_half_pinned($device, &$cos, 0, $zero),
            rope_half_pinned($device, &$cos, 128, $zero),
            rope_half_pinned($device, &$sin, 0, $zero),
            rope_half_pinned($device, &$sin, 128, $zero),
        )
    };
}

/// KVD post stage of unit (w, h) (`pair2::post_jd4_fn!`): q -> `q_out[Rh = wave w, Pw = unit h]`, k / v scattered into the ring caches.
macro_rules! post_fn {
    ($name:ident $(, $zero:ident)?) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            device: &mut Device,
            yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Pw, Rh, Tc, Ds]>,
            yk: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Pw, Rh, Tc, Dq]>,
            yv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Pw, Rh, Tc, Dq]>,
            q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
            k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
            tab: &HbmTensor<bf16, Chip, m![Tb]>,
            q_out: &mut HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]>,
            index: &HbmTensor<i32, Chip, m![Tc]>,
            k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            w: usize,
            h: usize,
            $($zero: &PostZero,)?
        ) -> PostZero {
            let cos_dm = rope_rows(device, tab, tb::COS + 2 * w + h);
            let sin_dm = rope_rows(device, tab, tb::SIN + 2 * w + h);
            let (cos_lo, cos_hi, sin_lo, sin_hi) = rope_vrfs_of!(device, cos_dm, sin_dm $(, $zero)?);

            let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
            yq.view()
                .tile::<m![Pw], 1, m![Pw = 1 # 2, Rh, Tc, Ds]>(w)
                .tile::<m![Rh], 1, m![Pw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
                .to_dm_view(&mut device.tdma, q_tok.view_mut());
            let q_n = post::norm_q(device, &q_tok, q_w);
            let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
            let zq = zero_from_q(device, &q_r);
            q_r.view().to_hbm_view(
                &mut device.tdma,
                q_out
                    .view_mut()
                    .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Pw, Tc, Ns, Gs, Ds]>(w)
                    .tile::<m![Pw], 1, m![Rh = 1 #{!} 2, Pw = 1 #{!} 2, Tc, Ns, Gs, Ds]>(h),
            );

            let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
            yk.view()
                .tile::<m![Pw], 1, m![Pw = 1 # 2, Rh, Tc, Dq]>(w)
                .tile::<m![Rh], 1, m![Pw = 1 # 2, Rh = 1 # 2, Tc, Dq]>(h)
                .to_dm_view(&mut device.tdma, k_tok.view_mut());
            let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
            let k_n = post::norm_k(device, &k_tok, k_w);
            let k_r = post::rope_k(device, &k_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
            let next_zero = zero_from_k_and(device, &k_r, &zq);
            k_r.dma_scatter::<m![Tc], _, _>(index, k_cache);

            let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
            yv.view()
                .tile::<m![Pw], 1, m![Pw = 1 # 2, Rh, Tc, Dq]>(w)
                .tile::<m![Rh], 1, m![Pw = 1 # 2, Rh = 1 # 2, Tc, Dq]>(h)
                .to_dm_view(&mut device.tdma, v_tok.view_mut());
            let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
            let v_n = post::norm_v(device, &v_tok);
            v_n.dma_scatter::<m![Tc], _, _>(index, v_cache);
            next_zero
        }
    };
}
post_fn!(post_first);
post_fn!(post_pinned, zero);

/// QKV of the 256 rows: q -> `q_out` (`[Rh = wave, Pw = unit of the wave, ..]`, the layout of the attention q), k / v -> the ring caches (rows `index{u}` of unit u = 2 w + h).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qkv(
    device: &mut Device,
    res_in: &HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]>,
    small: &HbmTensor<bf16, Chip, m![Sw]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    wkv: &HbmTensor<f8e4m3, Chip, m![Wkv, Rq2, Gg, Kc, Rt, Ke]>,
    tab: &HbmTensor<bf16, Chip, m![Tb]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    index2: &HbmTensor<i32, Chip, m![Tc]>,
    index3: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    // normalised x of both pairs: a kernel-internal HBM scratch (written, then read back with the row-group broadcast)
    let mut xn: HbmTensor<bf16, Chip, m![Pw, Rh, Tc, Kc, Ke]> = HbmTensor::new();
    input_norm_pair(device, res_in, 0, small, &mut xn);
    let xa = load_x_pair(device, &xn, 0);
    input_norm_pair(device, res_in, 1, small, &mut xn);
    let xb = load_x_pair(device, &xn, 1);

    let s15: HbmTensorView<'_, bf16, Chip, m![Sb15, Rq4, Gg, Rt]> = unsafe { small.view().reshape() };
    let sq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rq4, Rt]> = s15.tile::<m![Sb15], 1, m![Sb15 = 1 # 15, Rq4, Gg, Rt]>(sw::SQ).to_dm(&mut device.tdma);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Pw, Rh, Tc, Rq4, Rt]> = DmTensor::new();
    tile_q_joint2(device, &xa, &xb, q0, &sq, &mut yq, 0);
    tile_q_joint2(device, &xa, &xb, q1, &sq, &mut yq, 1);
    tile_q_joint2(device, &xa, &xb, q2, &sq, &mut yq, 2);
    tile_q_joint2(device, &xa, &xb, q3, &sq, &mut yq, 3);

    macro_rules! kv_tile {
        ($k:literal, $i:literal) => {{
            let t: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![Rt, Ke]> = wkv
                .view()
                .tile::<m![Wkv], 1, m![Wkv = 1 # 2, Rq2, Gg, Kc, Rt, Ke]>($k)
                .tile::<m![Rq2], 1, m![Wkv = 1 # 2, Rq2 = 1 # 2, Gg, Kc, Rt, Ke]>($i)
                .to_dm(&mut device.tdma);
            t
        }};
    }
    let s30: HbmTensorView<'_, bf16, Chip, m![Sb30, Rq2, Gg, Rt]> = unsafe { small.view().reshape() };
    let sk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rq2, Rt]> = unsafe { small.view().reshape::<Chip, m![Sb30, Rq2, Gg, Rt]>() }.tile::<m![Sb30], 1, m![Sb30 = 1 # 30, Rq2, Gg, Rt]>(sw::SK).to_dm(&mut device.tdma);
    let k0 = kv_tile!(0, 0);
    let k1 = kv_tile!(0, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Pw, Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint2(device, &xa, &xb, k0, &sk, &mut yk, 0);
    tile_kv_joint2(device, &xa, &xb, k1, &sk, &mut yk, 1);

    let sv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rq2, Rt]> = s30.tile::<m![Sb30], 1, m![Sb30 = 1 # 30, Rq2, Gg, Rt]>(sw::SV).to_dm(&mut device.tdma);
    let v0 = kv_tile!(1, 0);
    let v1 = kv_tile!(1, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Pw, Rh, Tc, Rq2, Rt]> = DmTensor::new();
    tile_kv_joint2(device, &xa, &xb, v0, &sv, &mut yv, 0);
    tile_kv_joint2(device, &xa, &xb, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Pw, Rh, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Pw, Rh, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Pw, Rh, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = norm_weight(device, small, sw::Q_NORM);
    let k_w = norm_weight(device, small, sw::K_NORM);
    // the 4 units one after the other: unit u + 1 builds its rope VRFs only after unit u rotated its K (exact-zero pin)
    let z0 = post_first(device, &yq, &yk, &yv, &q_w, &k_w, tab, q_out, index0, k_cache, v_cache, 0, 0);
    let z1 = post_pinned(device, &yq, &yk, &yv, &q_w, &k_w, tab, q_out, index1, k_cache, v_cache, 0, 1, &z0);
    let z2 = post_pinned(device, &yq, &yk, &yv, &q_w, &k_w, tab, q_out, index2, k_cache, v_cache, 1, 0, &z1);
    let _ = post_pinned(device, &yq, &yk, &yv, &q_w, &k_w, tab, q_out, index3, k_cache, v_cache, 1, 1, &z2);
}
