//! Stage 0 of the chunk QKV: input RMSNorm of x[Tc, H] with one token per slice, written to a pitch-4096 HBM scratch.
//!
//! Token t lives on slice t % 32 of cluster t / 32 (`TokSl`, 32 live slices at stride 8), the whole hidden vector H = 3840 in the slice,
//! so the mean square is one intra-slice reduce (no inter-slice reduce).  Arithmetic identical to `shared::rmsnorm::normalize`:
//! `x / sqrt(mean(x^2) + EPS) * w`, f32, one bf16 rounding.  The 15 KB f32 weight does not fit the 8 KiB VRF, so the normalize pass
//! runs on 3 chunks of 1280 elements.  The result goes to `xn[Tc, Kc, Ke]` (K padded to 32 chunks of 128; chunks 30 and 31 are
//! never written and must be zero, the host zero-initialises the scratch once), which the GEMM stage reads chunk-major.
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::{Chip, EPS};

pub(crate) type TokCl = m![Tc / 32];
pub(crate) type TokSl = m![Tc % 32, 1 # 8];

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn input_norm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
) {
    // The residual stream is the zero padded `[T, 4096]` tensor shared with the AO / FFN kernels: read its 30 real chunks through a tile view.
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![Kc = 30, Ke]> = x
        .view()
        .tile::<m![Kc], 30, m![Tc, Kc = 30 # 32, Ke]>(0)
        .to_dm(&mut device.tdma);
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![H]> = unsafe { x_tok.reshape() };
    norm_rest(device, x_tok, rms_weight, xn);
}

/// Same for half `$HALF` (0 / 1) of a 128-row residual `[Rh = 2, Tc, Kc, Ke]` (T = 128 chunks: two 64-row QKV passes over one residual tensor).
macro_rules! input_norm_half_fn {
    ($fname:ident, $HALF:literal) => {
        pub(crate) fn $fname(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
            rms_weight: &HbmTensor<bf16, Chip, m![H]>,
            xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
        ) {
            let x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![Kc = 30, Ke]> = x
                .view()
                .tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Kc, Ke]>($HALF)
                .tile::<m![Kc], 30, m![Rh = 1 # 2, Tc, Kc = 30 # 32, Ke]>(0)
                .to_dm(&mut device.tdma);
            let x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![H]> = unsafe { x_tok.reshape() };
            norm_rest(device, x_tok, rms_weight, xn);
        }
    };
}
input_norm_half_fn!(input_norm_h0, 0);
input_norm_half_fn!(input_norm_h1, 1);

fn norm_rest(
    device: &mut Device,
    x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
) {

    let mean_square: DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
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

    let rms: DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
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
    let rms_vrf: VrfTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![H]> = rms_weight.to_dm(&mut device.tdma);
    // a view's tile start is resolved against the base tensor's own mapping, so the chunk axis must be in the tensor type
    let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![Hq, Hr]> = unsafe { weight_dm.reshape() };
    let x_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![Hq, Hr]> = unsafe { x_tok.reshape() };

    let mut xn_tok: DmTensor<bf16, Chip, TokCl, TokSl, m![Hq, Hr]> = DmTensor::new();
    macro_rules! chunk {
        ($i:literal) => {{
            let w_vrf: VrfTensor<f32, Chip, TokCl, TokSl, m![Hr]> = device
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

    let xn_view: DmTensorView<'_, bf16, Chip, TokCl, TokSl, m![Kc = 30, Ke]> = unsafe { xn_tok.view().reshape() };
    xn_view.to_hbm_view(
        &mut device.tdma,
        xn.view_mut().tile::<m![Kc], 30, m![Tc, Kc = 30 #{!} 32, Ke]>(0),
    );
}
