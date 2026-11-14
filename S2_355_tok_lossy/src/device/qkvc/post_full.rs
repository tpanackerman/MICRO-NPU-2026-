//! Stage 2 of the chunk QKV for FULL-attention layers (Df = 512, partial rotary through the baseline cos / sin tables): per-head q norm,
//! k norm, v norm (V is the K projection normalised without weight) and RoPE, on token-per-slice layouts.  Generic over the cluster / slice
//! mapping so the same passes serve both layouts (q: cluster Gf / 8, slice (Tc, Gf / 2 % 4), element [Gf % 2, Df]; k, v: cluster Tc / 32,
//! slice (Tc % 32, 1 # 8), element [Df]).  Arithmetic = `full/rmsnorm.rs` (note: q and k multiply by the weight first, then divide by the
//! rms) and the two-group zip RoPE with Df / 2 = 256 element halves.
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::{Chip, EPS};

/// q layout: cluster Gf / 8, slice (token, Gf / 2 % 4), element [Gf % 2, Df]; k / v layout: cluster Tc / 32, slice (Tc % 32, 1 # 8), element [Df].
pub(crate) type QSl = m![Tc, Gf / 2 % 4];
pub(crate) type KCl = m![Tc / 32];
pub(crate) type KSl = m![Tc % 32, 1 # 8];

const DF_F32: f32 = Df::SIZE as f32;

pub(crate) type RopeVrf<Cl, Sl> = VrfTensor<f32, Chip, Cl, Sl, m![Df = 256]>;

/// cos / sin halves of the chunk rows as f32 VRFs: (cos_lo, cos_hi, sin_lo, sin_hi), each [Df = 256].
pub(crate) fn load_rope<Cl: M, Sl: M>(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
) -> (RopeVrf<Cl, Sl>, RopeVrf<Cl, Sl>, RopeVrf<Cl, Sl>, RopeVrf<Cl, Sl>) {
    let cos_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = cos.to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = sin.to_dm(&mut device.tdma);
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

pub(crate) fn load_norm_weight<Cl: M, Sl: M>(device: &mut Device, rms_weight: &HbmTensor<bf16, Chip, m![Df]>) -> VrfTensor<f32, Chip, Cl, Sl, m![Df]> {
    let weight_dm: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = rms_weight.to_dm(&mut device.tdma);
    device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf()
}

/// q: RMSNorm of the two heads of the slice (`full::rmsnorm::normalize_query` on [Gf % 2, Df]).
pub(crate) fn norm_q<Cl: M, Sl: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]>,
    weight_vrf: &VrfTensor<f32, Chip, Cl, Sl, m![Df]>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]> {
    let mean_square: DmTensor<f32, Chip, Cl, Sl, m![Gf % 2 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gf % 2, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![Gf % 2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DF_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![1], m![Gf % 2 # 8]>()
        .commit_trim::<m![Gf % 2 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cl, Sl, m![Gf % 2 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Gf % 2 # 8]>()
        .collect::<m![1], m![Gf % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Gf % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![Gf % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![Gf % 2 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Gf % 2 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Gf % 2 # 8]>()
        .collect::<m![1], m![Gf % 2 # 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![Gf % 2, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Gf % 2, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

/// k / v: scalar root mean square of the slice's head (`full::rmsnorm::root_mean_square`).
fn root_mean_square<Cl: M, Sl: M>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![Df]>) -> VrfTensor<f32, Chip, Cl, Sl, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DF_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cl, Sl, m![1 # 8]> = device
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

    device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn norm_k<Cl: M, Sl: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cl, Sl, m![Df]>,
    weight_vrf: &VrfTensor<f32, Chip, Cl, Sl, m![Df]>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Df]> {
    let rms_vrf = root_mean_square(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

pub(crate) fn norm_v<Cl: M, Sl: M>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![Df]>) -> DmTensor<bf16, Chip, Cl, Sl, m![Df]> {
    let rms_vrf = root_mean_square(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

/// q RoPE, two zip passes over the Df / 2 = 256 element halves of the slice's two heads.
pub(crate) fn rope_q<Cl: M, Sl: M>(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]>,
    cos_lo: &RopeVrf<Cl, Sl>,
    cos_hi: &RopeVrf<Cl, Sl>,
    sin_lo: &RopeVrf<Cl, Sl>,
    sin_hi: &RopeVrf<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Gf % 2, Df]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Gf % 2, Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Gf % 2, Df = 256 #{!} 512]>($own));
        };
    }
    zip_half!(0, 256, cos_lo, sin_lo);
    zip_half!(256, 0, cos_hi, sin_hi);
    out
}

/// k RoPE, two zip passes.
pub(crate) fn rope_k<Cl: M, Sl: M>(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, Cl, Sl, m![Df]>,
    cos_lo: &RopeVrf<Cl, Sl>,
    cos_hi: &RopeVrf<Cl, Sl>,
    sin_lo: &RopeVrf<Cl, Sl>,
    sin_hi: &RopeVrf<Cl, Sl>,
) -> DmTensor<bf16, Chip, Cl, Sl, m![Df]> {
    let mut out: DmTensor<bf16, Chip, Cl, Sl, m![Df]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Df = 256 / 8], m![Df = 256 % 8]>()
                .vector_final()
                .cast::<bf16, m![Df = 256 % 8 # 16]>()
                .commit_trim::<m![Df = 256 % 8]>()
                .commit_view(out.view_mut().tile::<m![Df], 256, m![Df = 256 #{!} 512]>($own));
        };
    }
    zip_half_k!(0, 256, cos_lo, sin_lo);
    zip_half_k!(256, 0, cos_hi, sin_hi);
    out
}

// ---------------------------------------------------------------------------------------------------- e2b: two 64-token units per op (rows Ub = 2)
// Same per-row arithmetic as `norm_k` / `norm_v` (and `norm_q`, which already carries [Gf % 2] rows): each row has its own intra-slice mean-square
// reduce and its own rms; the weight is broadcast over the rows. Halves the op count of the full-QKV post path (image bytes + serial small-op latency).
macro_rules! norm_rows {
    ($rms:ident, $norm_w:ident, $norm_nw:ident, $($R:tt)*) => {
        // per-row scalars stay [rows, 1 # 8] (the `prenorm_k` form): the [1, rows # 8] transpose of `norm_q` only serves <= 2 rows
        fn $rms<Cl: M, Sl: M>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![$($R)*, Df]>) -> VrfTensor<f32, Chip, Cl, Sl, m![$($R)*, 1 # 8]> {
            let mean_square: DmTensor<f32, Chip, Cl, Sl, m![$($R)*, 1 # 8]> = device
                .main
                .begin(x.view())
                .fetch::<m![$($R)*, Df / 16], m![Df % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$($R)*, Df / 8], m![Df % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$($R)*, Df / 4], m![Df % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<Df, m![$($R)*], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(DF_F32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, EPS)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let rms: DmTensor<f32, Chip, Cl, Sl, m![$($R)*, 1 # 8]> = device
                .main
                .begin(mean_square.view())
                .fetch::<m![$($R)*], m![1 # 8]>()
                .collect::<m![$($R)*], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            device
                .sub
                .begin(rms.view())
                .fetch::<m![$($R)*], m![1 # 8]>()
                .collect::<m![$($R)*], m![1 # 8]>()
                .to_vrf()
        }
        /// weight first, then divide by the rms (q / k)
        pub(crate) fn $norm_w<Cl: M, Sl: M>(
            device: &mut Device,
            x: &DmTensor<bf16, Chip, Cl, Sl, m![$($R)*, Df]>,
            weight_vrf: &VrfTensor<f32, Chip, Cl, Sl, m![Df]>,
        ) -> DmTensor<bf16, Chip, Cl, Sl, m![$($R)*, Df]> {
            let rms_vrf = $rms(device, x);
            device
                .main
                .begin(x.view())
                .fetch::<m![$($R)*, Df / 16], m![Df % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$($R)*, Df / 8], m![Df % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$($R)*, Df / 4], m![Df % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
                .vector_fp_div(&rms_vrf)
                .vector_widen_concat::<m![$($R)*, Df / 8], m![Df % 8]>()
                .vector_final()
                .cast::<bf16, m![Df % 8 # 16]>()
                .commit_trim::<m![Df % 8]>()
                .commit()
        }
        /// divide by the rms only (v)
        #[allow(dead_code)]
        pub(crate) fn $norm_nw<Cl: M, Sl: M>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![$($R)*, Df]>) -> DmTensor<bf16, Chip, Cl, Sl, m![$($R)*, Df]> {
            let rms_vrf = $rms(device, x);
            device
                .main
                .begin(x.view())
                .fetch::<m![$($R)*, Df / 16], m![Df % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$($R)*, Df / 8], m![Df % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$($R)*, Df / 4], m![Df % 4]>()
                .vector_fp_div(&rms_vrf)
                .vector_widen_concat::<m![$($R)*, Df / 8], m![Df % 8]>()
                .vector_final()
                .cast::<bf16, m![Df % 8 # 16]>()
                .commit_trim::<m![Df % 8]>()
                .commit()
        }
    };
}
norm_rows!(root_mean_square_u2, norm_k_u2, norm_v_u2, Ub = 2);
norm_rows!(root_mean_square_u4, norm_q_u4, norm_v_u4, Ub);
