//! Native T = 256 chunk QKV for the 256-row hybrid pass (`host/runtime/hybrid256.rs`): ONE launch per layer for the four 64-row units instead of two pair launches (sliding) or
//! four 64-row launches (full). Every weight tile is streamed, decoded and loaded into the TRF ONCE and contracted against the two 128-row waves (the `layer_*_t256p` pattern of
//! `device/ffnc`: one TRF tile, two contractions committed through tiles of the SAME output handle, so the program order (tile, wave) is kept). Per token the arithmetic is the one of
//! `pair2::sliding_qkv_pair_direct_j` / `full::full_qkv` (same input norm, same contraction and reduce per 128-token block, same post passes), so the results are expected to be
//! bit-identical; `bin/qkv_check.rs` compares them bytewise on hardware.
//!
//! x = the residual viewed as `[Rw, Rh, Tc, Kc, Ke]` (row = (rw * 2 + rh) * 64 + tc, unit u = 2 * rw + rh). The input norm runs ONCE over the 256 tokens, token (rw, rh, tc) on slice
//! (tc % 32, rw, rh) of cluster tc / 32 (128 live slices per cluster; the pair kernel used 64).
//!
//! Sliding post stage (`sliding_post_q`): q as in `pair2`; k and v share ONE norm chain: the GEMM commits K into tiles 0, 1 and V into tiles 2, 3 of one `[.., Rq4, Rt]` buffer, one
//! DM -> DM re-shard gives `kv_tok = [Dummy2, Ds]` per slice, one mean-square pass, one sqrt pass, one rms VRF and one normalize pass with the weight rows `[k_norm; 1.0]` (V has no
//! weight; x / rms * 1.0 == x / rms exactly). RoPE runs on the K row only; the V row is copied out (DM -> DM) for its scatter. 3 Main passes and 1 Sub pass fewer per block.
use furiosa_opt_std::prelude::*;

use super::{gemm, post, post_full};
use crate::axes::*;
use crate::{Chip, EPS};

type TokCl = m![Tc / 32];
/// token (rw, rh, tc) on slice (tc % 32, rw, rh) of cluster tc / 32
type TokSl4 = m![Tc % 32, Rw, Rh, 1 # 2];

const H_F32: f32 = H::SIZE as f32;
const DS_F32: f32 = Ds::SIZE as f32;

/// Input RMSNorm of the 256 rows of `x = [Rw, Rh, Tc, Kc, Ke]` into the zero-initialised scratch `xn` of the same type (chunks 30, 31 stay zero). Per token identical to
/// `pair2::input_norm_joint` / `xnorm::input_norm` (one intra-slice reduce over H, f32, one bf16 rounding).
fn input_norm_quad(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
) {
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Kc = 30, Ke]> = x
        .view()
        .tile::<m![Kc], 30, m![Rw, Rh, Tc, Kc = 30 # 32, Ke]>(0)
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
    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq, Hr]> = unsafe { weight_dm.reshape() };
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq, Hr]> = unsafe { x_tok.reshape() };

    let mut xn_tok: DmTensor<bf16, Chip, TokCl, TokSl4, m![Hq, Hr]> = DmTensor::new();
    macro_rules! chunk {
        ($i:literal) => {{
            let w_vrf: VrfTensor<f32, Chip, TokCl, TokSl4, m![Hr]> = device
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

    let xn_view: DmTensorView<'_, bf16, Chip, TokCl, TokSl4, m![Kc = 30, Ke]> = unsafe { xn_tok.view().reshape() };
    xn_view.to_hbm_view(&mut device.tdma, xn.view_mut().tile::<m![Kc], 30, m![Rw, Rh, Tc, Kc = 30 #{!} 32, Ke]>(0));
}

/// Wave `w` (128 tokens) of the normalised activations, chunk-per-slice and replicated to the 8 row groups of each cluster (the `pair2::load_x_joint` layout).
fn load_x_wave(device: &mut Device, xn: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>, w: usize) -> DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]> {
    xn.view().tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Kc, Ke]>(w).to_dm(&mut device.tdma)
}

/// Scale VRF + bf16 decode + TRF load of row tile `i` (first half of `pair2::tile_joint_fn`), shared by the two waves.
macro_rules! dec_fn {
    ($name:ident, $Rq:ident, $nrq:literal, $R:ident) => {
        fn $name(
            device: &mut Device,
            w_f8: DmTensor<f8e4m3, Chip, gemm::Cl, gemm::Sl, m![$R, Ke]>,
            scale_dm: &DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![$Rq, $R]>,
            i: usize,
        ) -> (TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]>, VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]>) {
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
            (w_trf, scale_vrf)
        }
    };
}
dec_fn!(dec_q, Rq4, 4, Rt);
dec_fn!(dec_kv, Rq2, 2, Rt);
dec_fn!(dec_qf, Rq8, 8, Rt);

/// Contraction of one TRF tile against wave `w` (128 tokens), reduce over the 32 chunk slices, row scale, bf16, committed into tile (w, i) of `y` (second half of
/// `pair2::tile_joint_fn`, same per-token arithmetic).
macro_rules! con_fn {
    ($name:ident, $Ry:ident, $nry:literal, $R:ident) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            device: &mut Device,
            xj: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
            w_trf: &TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![$R % 8], m![$R / 8, Ke]>,
            scale_vrf: &VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![$R]>,
            y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, $Ry, $R]>,
            w: usize,
            i: usize,
        ) {
            device
                .main
                .begin(xj.view())
                .fetch::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .collect::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
                .contract_outer::<m![Rh, Tc, Ke / 32, $R / 8], m![Ke % 32], _, _, _>(w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Rh, Tc, $R / 8]>()
                .contract_lane::<m![Rh, Tc, $R / 8], m![$R % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<gemm::SlOut, m![Rh, Tc, $R / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Rh, Tc, $R / 8, $R / 4 % 2], m![$R % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale_vrf)
                .vector_widen_concat::<m![Rh, Tc, $R / 8], m![$R % 8]>()
                .vector_final()
                .cast::<bf16, m![$R % 8 # 16]>()
                .commit_trim::<m![$R % 8]>()
                .commit_view(
                    y.view_mut()
                        .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, $Ry, $R]>(w)
                        .tile::<m![$Ry], 1, m![Rw = 1 #{!} 2, Rh, Tc, $Ry = 1 #{!} $nry, $R]>(i),
                );
        }
    };
}
con_fn!(con4, Rq4, 4, Rt);
con_fn!(con8, Rq8, 8, Rt);

/// The single-tile full-layer K projection (`gemm::proc_k1`) for wave `w`, committed into tile `w` of `yk = [Rw, Rh, Tc, Rk]`.
fn con_k1(
    device: &mut Device,
    xj: &DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rh, Tc, Ke]>,
    w_trf: &TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rk % 8], m![Rk / 8, Ke]>,
    scale_vrf: &VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![Rk]>,
    y: &mut DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, Rk]>,
    w: usize,
) {
    device
        .main
        .begin(xj.view())
        .fetch::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
        .collect::<m![Rh, Tc, Ke / 16], m![Ke % 16]>()
        .contract_outer::<m![Rh, Tc, Ke / 32, Rk / 8], m![Ke % 32], _, _, _>(w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Rh, Tc, Rk / 8]>()
        .contract_lane::<m![Rh, Tc, Rk / 8], m![Rk % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<gemm::SlOut, m![Rh, Tc, Rk / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rh, Tc, Rk / 8, Rk / 4 % 2], m![Rk % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scale_vrf)
        .vector_widen_concat::<m![Rh, Tc, Rk / 8], m![Rk % 8]>()
        .vector_final()
        .cast::<bf16, m![Rk % 8 # 16]>()
        .commit_trim::<m![Rk % 8]>()
        .commit_view(y.view_mut().tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Rk]>(w));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// sliding post stage

/// cos / sin halves of unit (w, h) of the 4-unit sliding tables `[Rw, Rh, Tc, Ds]` as f32 VRFs (`post::load_rope` on a tile view).
#[allow(dead_code)]
fn load_rope_s(
    device: &mut Device,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    w: usize,
    h: usize,
) -> (post::RopeVrf, post::RopeVrf, post::RopeVrf, post::RopeVrf) {
    let cos_dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = cos4
        .view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Ds]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
        .to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = sin4
        .view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Ds]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
        .to_dm(&mut device.tdma);
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
    let cos_hi: post::RopeVrf = half!(cos_dm, 128);
    let sin_lo: post::RopeVrf = half!(sin_dm, 0);
    let sin_hi: post::RopeVrf = half!(sin_dm, 128);
    (cos_lo, cos_hi, sin_lo, sin_hi)
}

/// cos / sin halves of unit (w, h) of the 4-unit full tables `[Rw, Rh, Tc, Df]` (`post_full::load_rope` on a tile view).
fn load_rope_f<Cl: M, Sl: M>(
    device: &mut Device,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    w: usize,
    h: usize,
) -> (post_full::RopeVrf<Cl, Sl>, post_full::RopeVrf<Cl, Sl>, post_full::RopeVrf<Cl, Sl>, post_full::RopeVrf<Cl, Sl>) {
    let cos_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = cos4
        .view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Df]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Df]>(h)
        .to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = sin4
        .view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Df]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Df]>(h)
        .to_dm(&mut device.tdma);
    macro_rules! half {
        ($t:ident, $start:literal) => {
            device
                .sub
                .begin($t.view().tile::<m![Df], 256, m![Df = 256 # 512]>($start))
                .fetch::<m![1], m![Df = 256]>()
                .fetch_cast::<f32>()
                .collect::<m![Df = 256 / 8], m![Df = 256 % 8]>()
                .to_vrf()
        };
    }
    (half!(cos_dm, 0), half!(cos_dm, 256), half!(sin_dm, 0), half!(sin_dm, 256))
}

/// k / v rows of one slice: RMSNorm of both heads in one chain, weight rows `[k_norm; 1.0]` (`post::norm_q` with the head axis `Dummy2` and a per-head weight).
#[allow(dead_code)]
fn norm_kv(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dummy2, Ds]>,
    weight_vrf: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Dummy2, Ds]>,
) -> DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dummy2, Ds]> {
    let mean_square: DmTensor<f32, Chip, post::PCl, post::PSl, m![Dummy2 % 2 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Dummy2, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Dummy2, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Dummy2, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![1], m![Dummy2 % 2 # 8]>()
        .commit_trim::<m![Dummy2 % 2 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, post::PCl, post::PSl, m![Dummy2 % 2 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Dummy2 % 2 # 8]>()
        .collect::<m![1], m![Dummy2 % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Dummy2 % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![Dummy2 % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![Dummy2 % 2 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, post::PCl, post::PSl, m![Dummy2 % 2 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Dummy2 % 2 # 8]>()
        .collect::<m![1], m![Dummy2 % 2 # 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![Dummy2, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Dummy2, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Dummy2, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Dummy2, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// k RoPE on row 0 of `kv = [Dummy2, Ds]` (`post::rope_k` reading through a tile view).
#[allow(dead_code)]
fn rope_k_row(
    device: &mut Device,
    kv: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dummy2, Ds]>,
    cos_lo: &post::RopeVrf,
    cos_hi: &post::RopeVrf,
    sin_lo: &post::RopeVrf,
    sin_hi: &post::RopeVrf,
) -> DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> {
    let mut out: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
            device
                .main
                .begin_interleaved::<Iz, _, _, _, _, _>(
                    kv.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0).tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>($own),
                    kv.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(0).tile::<m![Ds], 128, m![Dummy2 = 1 # 2, Ds = 128 # 256]>($other),
                )
                .fetch::<m![Iz], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Iz, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<Iz, m![Ds = 128 / 8]>()
                .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half_k!(0, 128, cos_lo, sin_lo);
    zip_half_k!(128, 0, cos_hi, sin_hi);
    out
}

/// Post stage of block (w, h) = unit 2w + h: q exactly as `pair2::sliding_post_jd`; k / v through the shared chain (`norm_kv`), K and V scattered into the ring caches.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn sliding_post_q(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rw, Rh, Tc, Ds]>,
    ykv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rw, Rh, Tc, Dummy2, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    kv_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Dummy2, Ds]>,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    q_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    w: usize,
    h: usize,
) {
    let (cos_lo, cos_hi, sin_lo, sin_hi) = load_rope_s(device, cos4, sin4, w, h);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Ds]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
        .to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = post::rope_q(device, &q_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    q_r.view().to_hbm_view(
        &mut device.tdma,
        q_all
            .view_mut()
            .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Ns, Gs, Ds]>(w)
            .tile::<m![Rh], 1, m![Rw = 1 #{!} 2, Rh = 1 #{!} 2, Tc, Ns, Gs, Ds]>(h),
    );

    let mut kv_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dummy2, Dh, Dq]> = DmTensor::new();
    ykv.view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Dummy2, Dq]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Dummy2, Dq]>(h)
        .to_dm_view(&mut device.tdma, kv_tok.view_mut());
    let kv_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dummy2, Ds]> = unsafe { kv_tok.reshape() };
    let kv_n = norm_kv(device, &kv_tok, kv_w);
    let k_r = rope_k_row(device, &kv_n, &cos_lo, &cos_hi, &sin_lo, &sin_hi);
    k_r.dma_scatter::<m![Tc], _, _>(index, k_cache);
    let mut v_n: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = DmTensor::new();
    kv_n.view().tile::<m![Dummy2], 1, m![Dummy2 = 1 # 2, Ds]>(1).to_dm_view(&mut device.tdma, v_n.view_mut());
    v_n.dma_scatter::<m![Tc], _, _>(index, v_cache);
}

/// cos / sin of unit (w, h) as TWO f32 VRFs of [Ds = 128] (the high halves). The host tables (`host::qkvc::rope_rows_for_positions`) have cos_lo == cos_hi bit for bit and
/// sin_lo == -sin_hi (bf16 rounding is sign-symmetric), so the low half of the zip RoPE can use (cos, sin) with a SUBTRACT: x_lo * cos + x_hi * (-sin) == x_lo * cos - x_hi * sin
/// exactly in IEEE arithmetic. Half the rope VRF of `post::load_rope` (1 KiB per slice per block instead of 2), which keeps the four post blocks inside the 8 KiB VRF.
fn load_rope_s2(
    device: &mut Device,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    w: usize,
    h: usize,
) -> (post::RopeVrf, post::RopeVrf) {
    macro_rules! hi_half {
        ($t:ident) => {{
            let dm: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = $t
                .view()
                .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Ds]>(w)
                .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
                .to_dm(&mut device.tdma);
            let v: post::RopeVrf = device
                .sub
                .begin(dm.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
                .fetch::<m![1], m![Ds = 128]>()
                .fetch_cast::<f32>()
                .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .to_vrf();
            v
        }};
    }
    let c = hi_half!(cos4);
    let s = hi_half!(sin4);
    (c, s)
}

/// `post::rope_q` with the two-table rope: low half own * cos - other * sin, high half own * cos + other * sin.
fn rope_q2(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]>,
    c: &post::RopeVrf,
    s: &post::RopeVrf,
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), s)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half!(0, 128, FpBinaryOp::SubF);
    zip_half!(128, 0, FpBinaryOp::AddF);
    out
}

/// `post::rope_k` with the two-table rope.
fn rope_k2(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]>,
    c: &post::RopeVrf,
    s: &post::RopeVrf,
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), s)
                .vector_fp_zip($op)
                .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half_k!(0, 128, FpBinaryOp::SubF);
    zip_half_k!(128, 0, FpBinaryOp::AddF);
    out
}

/// Post stage of block (w, h) with SEPARATE k / v norm chains (the passes of `pair2::sliding_post_jd`, k and v read through the K / V tiles of the shared GEMM buffer). Used instead of
/// `sliding_post_q` because the shared chain's 2 KiB weight VRF pushes the 4-block post stage over the 8 KiB VRF (scheduler: "Cannot find evict target for Vrf").
#[allow(clippy::too_many_arguments)]
fn sliding_post_sep(
    device: &mut Device,
    yq: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rw, Rh, Tc, Ds]>,
    ykv: &DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rw, Rh, Tc, Dummy2, Dq]>,
    q_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    k_w: &VrfTensor<f32, Chip, post::PCl, post::PSl, m![Ds]>,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    q_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ns, Gs, Ds]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    w: usize,
    h: usize,
) {
    let (c, sn) = load_rope_s2(device, cos4, sin4, w, h);

    let mut q_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Gs, Ds]> = DmTensor::new();
    yq.view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Ds]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Ds]>(h)
        .to_dm_view(&mut device.tdma, q_tok.view_mut());
    let q_n = post::norm_q(device, &q_tok, q_w);
    let q_r = rope_q2(device, &q_n, &c, &sn);
    q_r.view().to_hbm_view(
        &mut device.tdma,
        q_all
            .view_mut()
            .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Ns, Gs, Ds]>(w)
            .tile::<m![Rh], 1, m![Rw = 1 #{!} 2, Rh = 1 #{!} 2, Tc, Ns, Gs, Ds]>(h),
    );

    let mut k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    ykv.view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Dummy2, Dq]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Dummy2, Dq]>(h)
        .tile::<m![Dummy2], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Dummy2 = 1 # 2, Dq]>(0)
        .to_dm_view(&mut device.tdma, k_tok.view_mut());
    let k_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { k_tok.reshape() };
    let k_n = post::norm_k(device, &k_tok, k_w);
    let k_r = rope_k2(device, &k_n, &c, &sn);
    k_r.dma_scatter::<m![Tc], _, _>(index, k_cache);

    let mut v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Dh, Dq]> = DmTensor::new();
    ykv.view()
        .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Dummy2, Dq]>(w)
        .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Dummy2, Dq]>(h)
        .tile::<m![Dummy2], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Dummy2 = 1 # 2, Dq]>(1)
        .to_dm_view(&mut device.tdma, v_tok.view_mut());
    let v_tok: DmTensor<bf16, Chip, post::PCl, post::PSl, m![Ds]> = unsafe { v_tok.reshape() };
    let v_n = post::norm_v(device, &v_tok);
    v_n.dma_scatter::<m![Tc], _, _>(index, v_cache);
}

/// Native T = 256 sliding QKV with the KVD-256 scatter: `x` = the 256 residual rows `[Rw, Rh, Tc, Kc, Ke]`, `xn` its zero-initialised scratch, per-unit cos / sin rows, per-unit
/// q outputs (the `q_s` pieces) and cache row offsets, `kv_rms_weight` = `[k_norm; 1.0]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sliding_qkv_quad_direct(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    q_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    index2: &HbmTensor<i32, Chip, m![Tc]>,
    index3: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    input_norm_quad(device, x, input_rms_weight, xn);
    let x0 = load_x_wave(device, xn, 0);
    let x1 = load_x_wave(device, xn, 1);

    let sq = gemm::load_s_q(device, sq);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, Rq4, Rt]> = DmTensor::new();
    macro_rules! qtile {
        ($i:literal) => {{
            let wt = gemm::load_w_q(device, wq, $i);
            let (t, s) = dec_q(device, wt, &sq, $i);
            con4(device, &x0, &t, &s, &mut yq, 0, $i);
            con4(device, &x1, &t, &s, &mut yq, 1, $i);
        }};
    }
    qtile!(0);
    qtile!(1);
    qtile!(2);
    qtile!(3);

    // K -> tiles 0, 1 and V -> tiles 2, 3 of ONE buffer (the shared k / v norm chain reads both through one re-shard)
    let sk = gemm::load_s_kv(device, sk);
    let sv = gemm::load_s_kv(device, sv);
    let mut ykv: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, Rq4, Rt]> = DmTensor::new();
    macro_rules! kvtile {
        ($w:ident, $s:ident, $src:literal, $dst:literal) => {{
            let wt = gemm::load_w_kv(device, $w, $src);
            let (t, s) = dec_kv(device, wt, &$s, $src);
            con4(device, &x0, &t, &s, &mut ykv, 0, $dst);
            con4(device, &x1, &t, &s, &mut ykv, 1, $dst);
        }};
    }
    kvtile!(wk, sk, 0, 0);
    kvtile!(wk, sk, 1, 1);
    kvtile!(wv, sv, 0, 2);
    kvtile!(wv, sv, 1, 3);

    let yq: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Gs, 1 # 32], m![Rw, Rh, Tc, Ds]> = unsafe { yq.reshape() };
    let ykv: DmTensor<bf16, Chip, post::PCl, m![Ns % 4, Dh, 1 # 32], m![Rw, Rh, Tc, Dummy2, Dq]> = unsafe { ykv.reshape() };
    let q_w = post::load_norm_weight(device, q_rms_weight);
    let k_w = post::load_norm_weight(device, k_rms_weight);
    sliding_post_sep(device, &yq, &ykv, &q_w, &k_w, cos4, sin4, q_all, index0, k_cache, v_cache, 0, 0);
    sliding_post_sep(device, &yq, &ykv, &q_w, &k_w, cos4, sin4, q_all, index1, k_cache, v_cache, 0, 1);
    sliding_post_sep(device, &yq, &ykv, &q_w, &k_w, cos4, sin4, q_all, index2, k_cache, v_cache, 1, 0);
    sliding_post_sep(device, &yq, &ykv, &q_w, &k_w, cos4, sin4, q_all, index3, k_cache, v_cache, 1, 1);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// full layers

/// Native T = 256 full-layer QKV: the 256 residual rows, one input norm, 8 Q tiles + 1 K tile each decoded once for both waves, then the unchanged `full::full_qkv_tail` post passes
/// per 64-row unit (q [Tc, Gf, Df], k / v [Tc, Df] = the stage pieces of unit u).
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_qkv_quad(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    sin4: &HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    q_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Gf, Df]>,
    k_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    v_all: &mut HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
) {
    type QCl = m![Gf / 8];
    type QSl = post_full::QSl;
    type KCl = post_full::KCl;
    type KSl = post_full::KSl;

    input_norm_quad(device, x, input_rms_weight, xn);
    let x0 = load_x_wave(device, xn, 0);
    let x1 = load_x_wave(device, xn, 1);

    let sq = gemm::load_s_qf(device, sq);
    let mut yq: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, Rq8, Rt]> = DmTensor::new();
    macro_rules! qtile {
        ($i:literal) => {{
            let wt = gemm::load_w_qf(device, wq, $i);
            let (t, s) = dec_qf(device, wt, &sq, $i);
            con8(device, &x0, &t, &s, &mut yq, 0, $i);
            con8(device, &x1, &t, &s, &mut yq, 1, $i);
        }};
    }
    qtile!(0);
    qtile!(1);
    qtile!(2);
    qtile!(3);
    qtile!(4);
    qtile!(5);
    qtile!(6);
    qtile!(7);

    // K: one tile of 32 rows per group (the `gemm::proc_k1` decode), contracted against both waves
    let mut yk: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rw, Rh, Tc, Rk]> = DmTensor::new();
    {
        let k_f8 = gemm::load_w_k1(device, wk);
        let sk_dm: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Rk]> = sk.to_dm(&mut device.tdma);
        let sk_vrf: VrfTensor<f32, Chip, gemm::Cl, gemm::SlOut, m![Rk]> = device
            .sub
            .begin(sk_dm.view())
            .fetch::<m![Rk / 8], m![Rk % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rk / 8], m![Rk % 8]>()
            .to_vrf();
        let k_bf: DmTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rk, Ke]> = device
            .main
            .begin(k_f8.view())
            .fetch::<m![Rk, Ke / 32], m![Ke % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![Rk, Ke / 32, Ke % 32 / 8], m![Ke % 8]>()
            .cast::<bf16, m![Ke % 8 # 16]>()
            .commit_trim::<m![Ke % 8]>()
            .commit();
        let k_trf: TrfTensor<bf16, Chip, gemm::Cl, gemm::Sl, m![Rk % 8], m![Rk / 8, Ke]> = device
            .sub
            .begin(k_bf.view())
            .fetch::<m![Rk % 8, Rk / 8, Ke / 16], m![Ke % 16]>()
            .collect::<m![Rk % 8, Rk / 8, Ke / 16], m![Ke % 16]>()
            .to_trf();
        con_k1(device, &x0, &k_trf, &sk_vrf, &mut yk, 0);
        con_k1(device, &x1, &k_trf, &sk_vrf, &mut yk, 1);
    }

    let yq: DmTensor<bf16, Chip, QCl, m![Gf / 2 % 4, Gf % 2, 1 # 32], m![Rw, Rh, Tc, Df]> = unsafe { yq.reshape() };
    let q_w = post_full::load_norm_weight::<QCl, QSl>(device, q_rms_weight);
    let k_w = post_full::load_norm_weight::<KCl, KSl>(device, k_rms_weight);
    macro_rules! unit {
        ($w:literal, $h:literal) => {{
            let (qc_lo, qc_hi, qs_lo, qs_hi) = load_rope_f::<QCl, QSl>(device, cos4, sin4, $w, $h);
            let mut q_tok: DmTensor<bf16, Chip, QCl, QSl, m![Gf % 2, Df]> = DmTensor::new();
            yq.view()
                .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Df]>($w)
                .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Df]>($h)
                .to_dm_view(&mut device.tdma, q_tok.view_mut());
            let q_n = post_full::norm_q(device, &q_tok, &q_w);
            let q_r = post_full::rope_q(device, &q_n, &qc_lo, &qc_hi, &qs_lo, &qs_hi);
            q_r.view().to_hbm_view(
                &mut device.tdma,
                q_all
                    .view_mut()
                    .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Gf, Df]>($w)
                    .tile::<m![Rh], 1, m![Rw = 1 #{!} 2, Rh = 1 #{!} 2, Tc, Gf, Df]>($h),
            );

            let (kc_lo, kc_hi, ks_lo, ks_hi) = load_rope_f::<KCl, KSl>(device, cos4, sin4, $w, $h);
            let mut k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Gg, Rk]> = DmTensor::new();
            yk.view()
                .tile::<m![Rw], 1, m![Rw = 1 # 2, Rh, Tc, Rk]>($w)
                .tile::<m![Rh], 1, m![Rw = 1 # 2, Rh = 1 # 2, Tc, Rk]>($h)
                .to_dm_view(&mut device.tdma, k_tok.view_mut());
            let k_tok: DmTensor<bf16, Chip, KCl, KSl, m![Df]> = unsafe { k_tok.reshape() };
            let k_n = post_full::norm_k(device, &k_tok, &k_w);
            let k_r = post_full::rope_k(device, &k_n, &kc_lo, &kc_hi, &ks_lo, &ks_hi);
            k_r.view().to_hbm_view(
                &mut device.tdma,
                k_all
                    .view_mut()
                    .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Df]>($w)
                    .tile::<m![Rh], 1, m![Rw = 1 #{!} 2, Rh = 1 #{!} 2, Tc, Df]>($h),
            );
            let v_n = post_full::norm_v(device, &k_tok);
            v_n.view().to_hbm_view(
                &mut device.tdma,
                v_all
                    .view_mut()
                    .tile::<m![Rw], 1, m![Rw = 1 #{!} 2, Rh, Tc, Df]>($w)
                    .tile::<m![Rh], 1, m![Rw = 1 #{!} 2, Rh = 1 #{!} 2, Tc, Df]>($h),
            );
        }};
    }
    unit!(0, 0);
    unit!(0, 1);
    unit!(1, 0);
    unit!(1, 1);
}
