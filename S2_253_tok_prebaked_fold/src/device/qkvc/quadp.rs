//! R14 qkvre QP ("merged post"): the KVD post stage of the sliding quad QKV for SEVERAL 64-token blocks per pass (`X` = Ub: all four blocks; `X` = Rh: one 128-token half).
//! b11 runs the post once per 64-token block: per block 3 DM->DM reshards (~1.7k fixed static each), 2 scatters, 1 q store, 2 rope-table loads and ~14 VE passes whose cost is mostly
//! the per-pass fixed overhead (345-410 static cycles for one token per slice). The quad is DMA-bound (lean: io 133k of a 165-170k makespan), so the fixed DMA costs are time.
//! QP keeps the SAME per-token arithmetic (every reduce is over Ds of one (token, head); every product / sum is the b11 one, bit-identical) and moves X tokens per slice:
//! element q [Gs, X, Ds], k / v [X, Ds]; ONE reshard per tensor, ONE scatter per cache (key [X, Tc] = a joint host index table), ONE q store, ONE cos / sin table load.
use furiosa_opt_std::prelude::*;

use super::post::{PCl, PSl};
use crate::axes::*;
use crate::{Chip, EPS};

const DS_F32: f32 = Ds::SIZE as f32;

macro_rules! post_x {
    ($m:ident, $X:ident) => {
        pub(crate) mod $m {
            use super::*;

            pub(crate) type RopeVrfX = VrfTensor<f32, Chip, PCl, PSl, m![$X, Ds = 128]>;

            /// cos / sin low halves of the X tokens of the slice as f32 VRFs [X, Ds = 128] (the host tables satisfy cos[i + 128] == cos[i], sin[i + 128] == -sin[i]; see quad.rs).
            pub(crate) fn rope_vrfs(
                device: &mut Device,
                cos_dm: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
                sin_dm: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
            ) -> (RopeVrfX, RopeVrfX) {
                macro_rules! half {
                    ($t:ident) => {
                        device
                            .sub
                            .begin($t.view().tile::<m![Ds], 128, m![$X, Ds = 128 # 256]>(0))
                            .fetch::<m![$X], m![Ds = 128]>()
                            .fetch_cast::<f32>()
                            .collect::<m![$X, Ds = 128 / 8], m![Ds = 128 % 8]>()
                            .to_vrf()
                    };
                }
                (half!(cos_dm), half!(sin_dm))
            }

            /// q: RMSNorm of the (head, token) rows of the slice with weight (`post::norm_q` with the token axis added; per-row reduce over Ds unchanged).
            pub(crate) fn norm_q(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]>,
                weight_vrf: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
            ) -> DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]> {
                let mean_square: DmTensor<f32, Chip, PCl, PSl, m![Gs, $X / 2, $X % 2 # 8]> = device
                    .main
                    .begin(x.view())
                    .fetch::<m![Gs, $X, Ds / 16], m![Ds % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gs, $X, Ds / 8], m![Ds % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gs, $X, Ds / 4], m![Ds % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                    .vector_intra_slice_reduce::<Ds, m![Gs, $X], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(DS_F32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_clip(ClipBinaryOpF32::Add, EPS)
                    .vector_final()
                    .transpose::<m![Gs, $X / 2], m![$X % 2 # 8]>()
                    .commit_trim::<m![$X % 2 # 8]>()
                    .commit();
                let rms: DmTensor<f32, Chip, PCl, PSl, m![Gs, $X / 2, $X % 2 # 8]> = device
                    .main
                    .begin(mean_square.view())
                    .fetch::<m![Gs, $X / 2], m![$X % 2 # 8]>()
                    .collect::<m![Gs, $X / 2], m![$X % 2 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![$X % 2 # 4]>()
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_widen_pad::<m![$X % 2 # 8]>()
                    .vector_final()
                    .commit_trim::<m![$X % 2 # 8]>()
                    .commit();
                let rms_vrf: VrfTensor<f32, Chip, PCl, PSl, m![Gs, $X / 2, $X % 2 # 8]> = device
                    .sub
                    .begin(rms.view())
                    .fetch::<m![Gs, $X / 2], m![$X % 2 # 8]>()
                    .collect::<m![Gs, $X / 2], m![$X % 2 # 8]>()
                    .to_vrf();
                device
                    .main
                    .begin(x.view())
                    .fetch::<m![Gs, $X, Ds / 16], m![Ds % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gs, $X, Ds / 8], m![Ds % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gs, $X, Ds / 4], m![Ds % 4]>()
                    .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
                    .vector_widen_concat::<m![Gs, $X, Ds / 8], m![Ds % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Ds % 8 # 16]>()
                    .commit_trim::<m![Ds % 8]>()
                    .commit()
            }

            /// k / v: per-token root mean square of the slice's head rows [X, Ds] as a VRF [X # 8].
            fn root_mean_square(device: &mut Device, x: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>) -> VrfTensor<f32, Chip, PCl, PSl, m![$X / 2, $X % 2 # 8]> {
                let mean_square: DmTensor<f32, Chip, PCl, PSl, m![$X / 2, $X % 2 # 8]> = device
                    .main
                    .begin(x.view())
                    .fetch::<m![$X, Ds / 16], m![Ds % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$X, Ds / 8], m![Ds % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$X, Ds / 4], m![Ds % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                    .vector_intra_slice_reduce::<Ds, m![$X], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(DS_F32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_clip(ClipBinaryOpF32::Add, EPS)
                    .vector_final()
                    .transpose::<m![$X / 2], m![$X % 2 # 8]>()
                    .commit_trim::<m![$X % 2 # 8]>()
                    .commit();
                let rms: DmTensor<f32, Chip, PCl, PSl, m![$X / 2, $X % 2 # 8]> = device
                    .main
                    .begin(mean_square.view())
                    .fetch::<m![$X / 2], m![$X % 2 # 8]>()
                    .collect::<m![$X / 2], m![$X % 2 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![$X % 2 # 4]>()
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_widen_pad::<m![$X % 2 # 8]>()
                    .vector_final()
                    .commit_trim::<m![$X % 2 # 8]>()
                    .commit();
                device
                    .sub
                    .begin(rms.view())
                    .fetch::<m![$X / 2], m![$X % 2 # 8]>()
                    .collect::<m![$X / 2], m![$X % 2 # 8]>()
                    .to_vrf()
            }

            pub(crate) fn norm_k(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
                weight_vrf: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
            ) -> DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> {
                let rms_vrf = root_mean_square(device, x);
                device
                    .main
                    .begin(x.view())
                    .fetch::<m![$X, Ds / 16], m![Ds % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$X, Ds / 8], m![Ds % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$X, Ds / 4], m![Ds % 4]>()
                    .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
                    .vector_widen_concat::<m![$X, Ds / 8], m![Ds % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Ds % 8 # 16]>()
                    .commit_trim::<m![Ds % 8]>()
                    .commit()
            }

            pub(crate) fn norm_v(device: &mut Device, x: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>) -> DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> {
                let rms_vrf = root_mean_square(device, x);
                device
                    .main
                    .begin(x.view())
                    .fetch::<m![$X, Ds / 16], m![Ds % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$X, Ds / 8], m![Ds % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$X, Ds / 4], m![Ds % 4]>()
                    .vector_fp_div(&rms_vrf)
                    .vector_widen_concat::<m![$X, Ds / 8], m![Ds % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Ds % 8 # 16]>()
                    .commit_trim::<m![Ds % 8]>()
                    .commit()
            }

            /// q RoPE (`quad::rope_q_s` with the token axis: low-half tables, the high half subtracts).
            pub(crate) fn rope_q(
                device: &mut Device,
                q: &DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]>,
                cos: &RopeVrfX,
                sin_lo: &RopeVrfX,
            ) -> DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]> {
                let mut out: DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]> = DmTensor::new();
                macro_rules! zip_half {
                    ($own:literal, $other:literal, $op:expr) => {
                        device
                            .main
                            .begin_interleaved::<Iz, _, _, _, _, _>(
                                q.view().tile::<m![Ds], 128, m![Gs, $X, Ds = 128 # 256]>($own),
                                q.view().tile::<m![Ds], 128, m![Gs, $X, Ds = 128 # 256]>($other),
                            )
                            .fetch::<m![Gs, $X, Iz], m![Ds = 128]>()
                            .fetch_cast::<f32>()
                            .collect::<m![Gs, $X, Iz, Ds = 128 / 8], m![Ds = 128 % 8]>()
                            .vector_init()
                            .vector_intra_slice_unzip::<Iz, m![Gs, $X, Ds = 128 / 8]>()
                            .vector_narrow_split::<m![Gs, $X, Ds = 128 / 4], m![Ds = 128 % 4]>()
                            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                            .vector_fp_zip($op)
                            .vector_widen_concat::<m![Gs, $X, Ds = 128 / 8], m![Ds = 128 % 8]>()
                            .vector_final()
                            .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                            .commit_trim::<m![Ds = 128 % 8]>()
                            .commit_view(out.view_mut().tile::<m![Ds], 128, m![Gs, $X, Ds = 128 #{!} 256]>($own));
                    };
                }
                zip_half!(0, 128, FpBinaryOp::AddF);
                zip_half!(128, 0, FpBinaryOp::SubF);
                out
            }

            /// k RoPE (`quad::rope_k_s` with the token axis).
            pub(crate) fn rope_k(
                device: &mut Device,
                k: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
                cos: &RopeVrfX,
                sin_lo: &RopeVrfX,
            ) -> DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> {
                let mut out: DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> = DmTensor::new();
                macro_rules! zip_half_k {
                    ($own:literal, $other:literal, $op:expr) => {
                        device
                            .main
                            .begin_interleaved::<Iz, _, _, _, _, _>(
                                k.view().tile::<m![Ds], 128, m![$X, Ds = 128 # 256]>($own),
                                k.view().tile::<m![Ds], 128, m![$X, Ds = 128 # 256]>($other),
                            )
                            .fetch::<m![$X, Iz], m![Ds = 128]>()
                            .fetch_cast::<f32>()
                            .collect::<m![$X, Iz, Ds = 128 / 8], m![Ds = 128 % 8]>()
                            .vector_init()
                            .vector_intra_slice_unzip::<Iz, m![$X, Ds = 128 / 8]>()
                            .vector_narrow_split::<m![$X, Ds = 128 / 4], m![Ds = 128 % 4]>()
                            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), cos, ())
                            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), sin_lo)
                            .vector_fp_zip($op)
                            .vector_widen_concat::<m![$X, Ds = 128 / 8], m![Ds = 128 % 8]>()
                            .vector_final()
                            .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                            .commit_trim::<m![Ds = 128 % 8]>()
                            .commit_view(out.view_mut().tile::<m![Ds], 128, m![$X, Ds = 128 #{!} 256]>($own));
                    };
                }
                zip_half_k!(0, 128, FpBinaryOp::AddF);
                zip_half_k!(128, 0, FpBinaryOp::SubF);
                out
            }

            /// The merged post of the X blocks held in `yq` / `yk` / `yv` ([X, Tc, ..] per GEMM output slice): ONE reshard per tensor, ONE scatter per cache (index = byte offsets
            /// [X, Tc]), ONE q store into `q_out` (a view of the X blocks of the q stage).
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn post(
                device: &mut Device,
                yq: &DmTensor<bf16, Chip, PCl, m![Ns % 4, Gs, 1 # 32], m![$X, Tc, Ds]>,
                yk: &DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![$X, Tc, Dq]>,
                yv: &DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![$X, Tc, Dq]>,
                q_w: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
                k_w: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
                cos_dm: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
                sin_dm: &DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]>,
                q_out: HbmTensorViewMut<'_, bf16, Chip, m![$X, Tc, Ns, Gs, Ds]>,
                index: &HbmTensor<i32, Chip, m![$X, Tc]>,
                k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
                v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            ) {
                let (cos_lo, sin_lo) = rope_vrfs(device, cos_dm, sin_dm);

                let mut k_tok: DmTensor<bf16, Chip, PCl, PSl, m![$X, Dh, Dq]> = DmTensor::new();
                yk.view().to_dm_view(&mut device.tdma, k_tok.view_mut());
                let k_tok: DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> = unsafe { k_tok.reshape() };
                let k_n = norm_k(device, &k_tok, k_w);
                let k_r = rope_k(device, &k_n, &cos_lo, &sin_lo);
                k_r.dma_scatter::<m![$X, Tc], _, _>(index, k_cache);

                let mut v_tok: DmTensor<bf16, Chip, PCl, PSl, m![$X, Dh, Dq]> = DmTensor::new();
                yv.view().to_dm_view(&mut device.tdma, v_tok.view_mut());
                let v_tok: DmTensor<bf16, Chip, PCl, PSl, m![$X, Ds]> = unsafe { v_tok.reshape() };
                let v_n = norm_v(device, &v_tok);
                v_n.dma_scatter::<m![$X, Tc], _, _>(index, v_cache);

                let mut q_tok: DmTensor<bf16, Chip, PCl, PSl, m![Gs, $X, Ds]> = DmTensor::new();
                yq.view().to_dm_view(&mut device.tdma, q_tok.view_mut());
                let q_n = norm_q(device, &q_tok, q_w);
                let q_r = rope_q(device, &q_n, &cos_lo, &sin_lo);
                q_r.view().to_hbm_view(&mut device.tdma, q_out);
            }
        }
    };
}
post_x!(p4, Ub);

/// QP4 quad: the b11 joint quad (`quad::sliding_qkv_quad_direct_v`: joint norm, ONE xn replicate, 8 tiles x 256 tokens) with the merged post of the four blocks (`p4::post`).
/// `kidx` = the four ring row-offset tables [Ub, Tc] in ONE host table (same values as kidx0..3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_quad_p(
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
    kidx: &HbmTensor<i32, Chip, m![Ub, Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    use super::{gemm, post, quad};
    quad::input_norm_quad(device, x, input_rms_weight, xn);
    let xj = quad::load_x_quad(device, xn);

    let sq = gemm::load_s_q(device, sq);
    let q0 = gemm::load_w_q(device, wq, 0);
    let q1 = gemm::load_w_q(device, wq, 1);
    let q2 = gemm::load_w_q(device, wq, 2);
    let q3 = gemm::load_w_q(device, wq, 3);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq4, Rt]> = DmTensor::new();
    quad::tile_q_quad(device, &xj, q0, &sq, &mut yq, 0);
    quad::tile_q_quad(device, &xj, q1, &sq, &mut yq, 1);
    quad::tile_q_quad(device, &xj, q2, &sq, &mut yq, 2);
    quad::tile_q_quad(device, &xj, q3, &sq, &mut yq, 3);

    let sk = gemm::load_s_kv(device, sk);
    let k0 = gemm::load_w_kv(device, wk, 0);
    let k1 = gemm::load_w_kv(device, wk, 1);
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    quad::tile_kv_quad(device, &xj, k0, &sk, &mut yk, 0);
    quad::tile_kv_quad(device, &xj, k1, &sk, &mut yk, 1);

    let sv = gemm::load_s_kv(device, sv);
    let v0 = gemm::load_w_kv(device, wv, 0);
    let v1 = gemm::load_w_kv(device, wv, 1);
    let mut yv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Ub, Tc, Rq2, Rt]> = DmTensor::new();
    quad::tile_kv_quad(device, &xj, v0, &sv, &mut yv, 0);
    quad::tile_kv_quad(device, &xj, v1, &sv, &mut yv, 1);

    let yq: DmTensor<bf16, Chip, PCl, m![Ns % 4, Gs, 1 # 32], m![Ub, Tc, Ds]> = unsafe { yq.reshape() };
    let yk: DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yk.reshape() };
    let yv: DmTensor<bf16, Chip, PCl, m![Ns % 4, Dh, 1 # 32], m![Ub, Tc, Dq]> = unsafe { yv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    let cos_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ub, Ds]> = cos.to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ub, Ds]> = sin.to_dm(&mut device.tdma);
    p4::post(device, &yq, &yk, &yv, &q_w, &k_w, &cos_dm, &sin_dm, q_out.view_mut(), kidx, k_cache, v_cache);
}
