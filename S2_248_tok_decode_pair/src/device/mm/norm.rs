//! Token-per-slice normalisations of the batched image / audio encoders (the row-wise stages that used to run on the host).
//!
//! Layout of every stage (modelled on `device/qkvc/xnorm.rs`): token `t` of the 64-row tile lives on slice `t % 32` of cluster `t / 32` (`TokSl`, 32 live slices
//! at stride 8 per cluster) with its WHOLE row in the slice, so every mean / variance is one intra-slice reduce and there is no cross-slice reduce.
//! The arithmetic is the per-row device kernels of `device/vision/layernorm.rs` (`normalize_patch` = LN over Rv, `normalize` = LN over Mv) and
//! `device/vision/projection.rs::rmsnorm_without_weight`, f32 with one bf16 rounding at the end of every stage, i.e. the rounding points of the host pipeline of
//! `host/mm.rs`.  The per-row kernels cannot be reused as they are: their final pass tiles the plain `Rv` / `Mv` axis, and under the token-per-slice mapping a
//! view's tile start is resolved against the base tensor's own mapping (carve panic, see `qkvc/xnorm.rs`), so the chunk axis is part of the tensor type here.
//!
//! HBM interface of the stages (T = 64 rows per tile):
//! * `ln1_tile`:   pixels `[Tm64, Rv]` -> LN1 -> `xn1[Tm64, Mc, Ev]` (K = 6912 padded to 32 x 224; the 256 pad columns are never written, the host zero-initialises
//!   the scratch once) = the dense GEMM's input, read token-major / chunk-sliced like `qkvc/gemm.rs::load_x`;
//! * `ln23_tile`:  dense output `y[Tm64, Mv]` -> LN2 -> + position tile -> LN3 -> rmsnorm -> `xn2[Tm64, Mc, Ep]` (30 real chunks of 128) = the projection's input;
//! * `rms_audio_tile`: frames `[Tm64, Aa]` -> rmsnorm (no weight) -> `xn[Tm64, Mc, Ea]` (20 real chunks of 32) = the audio projection's input.
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::{Chip, EPS};

pub(crate) type TokCl = m![Tm64 / 32];
pub(crate) type TokSl = m![Tm64 % 32, 1 # 8];

const RV_F32: f32 = Rv::SIZE as f32;
const MV_F32: f32 = Mv::SIZE as f32;
const AA_F32: f32 = Aa::SIZE as f32;

fn scalar_vrf(
    device: &mut Device,
    x: &DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]>,
) -> VrfTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> {
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn sqrt_scalar(
    device: &mut Device,
    x: &DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]>,
) -> DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// LayerNorm over axis `$A` of `$N` elements with weight and bias, one row per slice: the arithmetic of `device/vision/layernorm.rs` with the final pass walking
/// the row in `$NQ` chunks `[$Q, $R]` of the tensor type.
macro_rules! layer_norm_fn {
    ($name:ident, $A:ident, $Q:ident, $R:ident, $NQ:tt, $N:expr) => {
        pub(crate) fn $name(
            device: &mut Device,
            x: &DmTensor<bf16, Chip, TokCl, TokSl, m![$A]>,
            weight: &HbmTensor<bf16, Chip, m![$A]>,
            bias: &HbmTensor<bf16, Chip, m![$A]>,
        ) -> DmTensor<bf16, Chip, TokCl, TokSl, m![$A]> {
            let mean: DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
                .main
                .begin(x.view())
                .fetch::<m![$A / 16], m![$A % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$A / 8], m![$A % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$A / 4], m![$A % 4]>()
                .vector_intra_slice_reduce::<$A, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div($N)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let mean_vrf = scalar_vrf(device, &mean);

            let centred: DmTensor<f32, Chip, TokCl, TokSl, m![$A]> = device
                .main
                .begin(x.view())
                .fetch::<m![$A / 16], m![$A % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$A / 8], m![$A % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$A / 4], m![$A % 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &mean_vrf)
                .vector_widen_concat::<m![$A / 8], m![$A % 8]>()
                .vector_final()
                .commit_trim::<m![$A % 8]>()
                .commit();

            let var: DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
                .main
                .begin(centred.view())
                .fetch::<m![$A / 8], m![$A % 8]>()
                .collect::<m![$A / 8], m![$A % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$A / 4], m![$A % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<$A, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div($N)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, EPS)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let std_dev = sqrt_scalar(device, &var);
            let std_vrf = scalar_vrf(device, &std_dev);

            let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![$A]> = weight.to_dm(&mut device.tdma);
            let bias_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![$A]> = bias.to_dm(&mut device.tdma);
            let weight_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![$Q, $R]> = unsafe { weight_dm.reshape() };
            let bias_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![$Q, $R]> = unsafe { bias_dm.reshape() };
            let centred: DmTensor<f32, Chip, TokCl, TokSl, m![$Q, $R]> = unsafe { centred.reshape() };

            let mut output: DmTensor<bf16, Chip, TokCl, TokSl, m![$Q, $R]> = DmTensor::new();
            for i in 0..$NQ {
                let centred_tile = centred.view().tile::<m![$Q], 1, m![$Q = 1 # $NQ, $R]>(i);
                let weight_tile = weight_dm.view().tile::<m![$Q], 1, m![$Q = 1 # $NQ, $R]>(i);
                let bias_tile = bias_dm.view().tile::<m![$Q], 1, m![$Q = 1 # $NQ, $R]>(i);

                let weight_vrf: VrfTensor<f32, Chip, TokCl, TokSl, m![$R]> = device
                    .sub
                    .begin(weight_tile)
                    .fetch::<m![1], m![$R]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$R / 8], m![$R % 8]>()
                    .to_vrf();
                let bias_vrf: VrfTensor<f32, Chip, TokCl, TokSl, m![$R]> = device
                    .sub
                    .begin(bias_tile)
                    .fetch::<m![1], m![$R]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$R / 8], m![$R % 8]>()
                    .to_vrf();

                device
                    .main
                    .begin(centred_tile)
                    .fetch::<m![1], m![$R]>()
                    .collect::<m![$R / 8], m![$R % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$R / 4], m![$R % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
                    .vector_fp_div(&std_vrf)
                    .vector_widen_concat::<m![$R / 8], m![$R % 8]>()
                    .vector_clip(ClipBinaryOpF32::Add, &bias_vrf)
                    .vector_final()
                    .cast::<bf16, m![$R % 8 # 16]>()
                    .commit_trim::<m![$R % 8]>()
                    .commit_view(output.view_mut().tile::<m![$Q], 1, m![$Q = 1 #{!} $NQ, $R]>(i));
            }
            unsafe { output.reshape() }
        }
    };
}
layer_norm_fn!(ln_rv, Rv, Lq8, Lr864, 8, RV_F32);
layer_norm_fn!(ln_mv, Mv, Mq8, Mr480, 8, MV_F32);

/// `x + residual` rounded to bf16 over the 3840 columns, one row per slice (`shared::residual::add_vision` with the chunk axis in the tensor type).
fn add_mv(
    device: &mut Device,
    x: DmTensor<bf16, Chip, TokCl, TokSl, m![Mv]>,
    residual: DmTensor<bf16, Chip, TokCl, TokSl, m![Mv]>,
) -> DmTensor<bf16, Chip, TokCl, TokSl, m![Mv]> {
    let x: DmTensor<bf16, Chip, TokCl, TokSl, m![Mq8, Mr480]> = unsafe { x.reshape() };
    let residual: DmTensor<bf16, Chip, TokCl, TokSl, m![Mq8, Mr480]> = unsafe { residual.reshape() };
    let mut output: DmTensor<bf16, Chip, TokCl, TokSl, m![Mq8, Mr480]> = DmTensor::new();
    for i in 0..8 {
        let x_tile = x.view().tile::<m![Mq8], 1, m![Mq8 = 1 # 8, Mr480]>(i);
        let residual_tile = residual.view().tile::<m![Mq8], 1, m![Mq8 = 1 # 8, Mr480]>(i);
        let residual_vrf: VrfTensor<f32, Chip, TokCl, TokSl, m![Mr480]> = device
            .sub
            .begin(residual_tile)
            .fetch::<m![1], m![Mr480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mr480 / 8], m![Mr480 % 8]>()
            .to_vrf();
        device
            .main
            .begin(x_tile)
            .fetch::<m![1], m![Mr480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mr480 / 8], m![Mr480 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
            .vector_final()
            .cast::<bf16, m![Mr480 % 8 # 16]>()
            .commit_trim::<m![Mr480 % 8]>()
            .commit_view(output.view_mut().tile::<m![Mq8], 1, m![Mq8 = 1 #{!} 8, Mr480]>(i));
    }
    unsafe { output.reshape() }
}

/// LN1 over the 6912 pixels of every patch row, written into the first 6912 columns of the zero padded `[Tm64, 7168]` dense input.
pub(crate) fn ln1_tile(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Rv]>,
    w: &HbmTensor<bf16, Chip, m![Rv]>,
    b: &HbmTensor<bf16, Chip, m![Rv]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
) {
    let x_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![Rv]> = x.to_dm(&mut device.tdma);
    let y = ln_rv(device, &x_dm, w, b);
    y.view().to_hbm_view(
        &mut device.tdma,
        xn.view_mut().tile::<m![Rv # 7168], 6912, m![Tm64, Rv = 6912 #{!} 7168]>(0),
    );
}

/// `x / sqrt(mean(x^2) + EPS)` rounded to bf16 (rmsnorm without weight) over axis `$A` of `$N` elements, one row per slice.
macro_rules! rms_nowt_fn {
    ($name:ident, $A:ident, $N:expr) => {
        pub(crate) fn $name(
            device: &mut Device,
            x: &DmTensor<bf16, Chip, TokCl, TokSl, m![$A]>,
        ) -> DmTensor<bf16, Chip, TokCl, TokSl, m![$A]> {
            let mean_square: DmTensor<f32, Chip, TokCl, TokSl, m![1 # 8]> = device
                .main
                .begin(x.view())
                .fetch::<m![$A / 16], m![$A % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$A / 8], m![$A % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$A / 4], m![$A % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<$A, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div($N)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, EPS)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let rms = sqrt_scalar(device, &mean_square);
            let rms_vrf = scalar_vrf(device, &rms);
            device
                .main
                .begin(x.view())
                .fetch::<m![$A / 16], m![$A % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$A / 8], m![$A % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$A / 4], m![$A % 4]>()
                .vector_fp_div(&rms_vrf)
                .vector_widen_concat::<m![$A / 8], m![$A % 8]>()
                .vector_final()
                .cast::<bf16, m![$A % 8 # 16]>()
                .commit_trim::<m![$A % 8]>()
                .commit()
        }
    };
}
rms_nowt_fn!(rms_mv, Mv, MV_F32);
rms_nowt_fn!(rms_aa, Aa, AA_F32);

/// Dense output rows -> LN2 -> + position row -> LN3 -> rmsnorm (no weight), written as the projection's input `[Tm64, Mc, Ep]` (30 real chunks of 128).
pub(crate) fn ln23_tile(
    device: &mut Device,
    y: HbmTensorView<'_, bf16, Chip, m![Tm64, Mv]>,
    pos: &HbmTensor<bf16, Chip, m![Tm64, Mv]>,
    w2: &HbmTensor<bf16, Chip, m![Mv]>,
    b2: &HbmTensor<bf16, Chip, m![Mv]>,
    w3: &HbmTensor<bf16, Chip, m![Mv]>,
    b3: &HbmTensor<bf16, Chip, m![Mv]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
) {
    let y_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![Mv]> = y.to_dm(&mut device.tdma);
    let a = ln_mv(device, &y_dm, w2, b2);
    let pos_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![Mv]> = pos.to_dm(&mut device.tdma);
    let s = add_mv(device, a, pos_dm);
    let c = ln_mv(device, &s, w3, b3);
    let r = rms_mv(device, &c);
    let rv: DmTensorView<'_, bf16, Chip, TokCl, TokSl, m![Mc = 30, Ep]> = unsafe { r.view().reshape() };
    rv.to_hbm_view(&mut device.tdma, xn.view_mut().tile::<m![Mc], 30, m![Tm64, Mc = 30 #{!} 32, Ep]>(0));
}

/// Audio frames -> rmsnorm (no weight) -> the audio projection's input `[Tm64, Mc, Ea]` (20 real chunks of 32).
pub(crate) fn rms_audio_tile(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Aa]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
) {
    let x_dm: DmTensor<bf16, Chip, TokCl, TokSl, m![Aa]> = x.to_dm(&mut device.tdma);
    let r = rms_aa(device, &x_dm);
    let rv: DmTensorView<'_, bf16, Chip, TokCl, TokSl, m![Mc = 20, Ea]> = unsafe { r.view().reshape() };
    rv.to_hbm_view(&mut device.tdma, xn.view_mut().tile::<m![Mc], 20, m![Tm64, Mc = 20 #{!} 32, Ea]>(0));
}
