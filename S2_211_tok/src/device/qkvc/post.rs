//! Stage 2 of the chunk QKV (sliding layers): per-head q / k RMSNorm, the v norm and RoPE, one (token, head group) per slice.
//!
//! The GEMM leaves each head (or half head) on one slice with all Tc tokens; a DM -> DM re-shard (`to_dm`, 512 B pieces) moves it to
//! the layout `PSl`: cluster c = heads 4c..4c+3 (`Ns / 4`), slice = (token, Ns % 4) (all 256 slices of the cluster), element = the
//! head (k, v: [Ds]) or the two heads of one KV group (q: [Gs, Ds]).  Each slice then runs exactly the single-token baseline
//! passes (`refimpl/sliding/rmsnorm.rs` norms, the two-group zip RoPE of `sliding/rope_h4.rs`, bit-identical products), on
//! its own token.  cos / sin come as per-chunk tables [Tc, Ds] (rows p0..p0+Tc-1 of the baseline `ws.cos_s` / `ws.sin_s`).
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::{Chip, EPS};

/// Post-stage cluster: heads 4c..4c+3.
pub(crate) type PCl = m![Ns / 4];
/// (token, head % 4) per slice.
pub(crate) type PSl = m![Tc, Ns % 4];
/// The same slices labelled with the attention staging token axis `Ts = 64` (k_stage / v_stage are `m![Ts = 64, Ns, Ds]`).
pub(crate) type StageSl = m![Ts = 64, Ns % 4];

const DS_F32: f32 = Ds::SIZE as f32;

pub(crate) type RopeVrf = VrfTensor<f32, Chip, PCl, PSl, m![Ds = 128]>;

/// cos / sin halves as f32 VRFs: (cos_lo, cos_hi, sin_lo, sin_hi), each [Ds = 128].
pub(crate) fn load_rope(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
) -> (RopeVrf, RopeVrf, RopeVrf, RopeVrf) {
    let cos_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ds]> = cos.to_dm(&mut device.tdma);
    let sin_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ds]> = sin.to_dm(&mut device.tdma);
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
    let cos_lo: RopeVrf = half!(cos_dm, 0);
    let cos_hi: RopeVrf = half!(cos_dm, 128);
    let sin_lo: RopeVrf = half!(sin_dm, 0);
    let sin_hi: RopeVrf = half!(sin_dm, 128);
    (cos_lo, cos_hi, sin_lo, sin_hi)
}

/// `load_rope` from cos / sin rows already in DM (the 256-row QKV loads them through views of the 4-unit tables).
pub(crate) fn rope_vrfs(
    device: &mut Device,
    cos_dm: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>,
    sin_dm: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>,
) -> (RopeVrf, RopeVrf, RopeVrf, RopeVrf) {
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
    let cos_lo: RopeVrf = half!(cos_dm, 0);
    let cos_hi: RopeVrf = half!(cos_dm, 128);
    let sin_lo: RopeVrf = half!(sin_dm, 0);
    let sin_hi: RopeVrf = half!(sin_dm, 128);
    (cos_lo, cos_hi, sin_lo, sin_hi)
}

pub(crate) fn load_norm_weight(device: &mut Device, rms_weight: &HbmTensor<bf16, Chip, m![Ds]>) -> VrfTensor<f32, Chip, PCl, PSl, m![Ds]> {
    let weight_dm: DmTensor<bf16, Chip, PCl, PSl, m![Ds]> = rms_weight.to_dm(&mut device.tdma);
    device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

/// q: RMSNorm of the two heads of the slice, with weight (`refimpl::sliding::rmsnorm::normalize_query` on [Gs, Ds]).
pub(crate) fn norm_q(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, PCl, PSl, m![Gs, Ds]>,
    weight_vrf: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
) -> DmTensor<bf16, Chip, PCl, PSl, m![Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, PCl, PSl, m![Gs % 2 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![1], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, PCl, PSl, m![Gs % 2 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Gs % 2 # 8]>()
        .collect::<m![1], m![Gs % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Gs % 2 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![Gs % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![Gs % 2 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, PCl, PSl, m![Gs % 2 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Gs % 2 # 8]>()
        .collect::<m![1], m![Gs % 2 # 8]>()
        .to_vrf();

    device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// k / v: scalar root mean square of the slice's head (`refimpl::sliding::rmsnorm::root_mean_square` with one head).
fn root_mean_square(device: &mut Device, x: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>) -> VrfTensor<f32, Chip, PCl, PSl, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, PCl, PSl, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, PCl, PSl, m![1 # 8]> = device
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

pub(crate) fn norm_k(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>,
    weight_vrf: &VrfTensor<f32, Chip, PCl, PSl, m![Ds]>,
) -> DmTensor<bf16, Chip, PCl, PSl, m![Ds]> {
    let rms_vrf = root_mean_square(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn norm_v(device: &mut Device, x: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>) -> DmTensor<bf16, Chip, PCl, PSl, m![Ds]> {
    let rms_vrf = root_mean_square(device, x);
    device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// q RoPE, two zip passes (`sliding/rope_h4.rs::apply_rope_q_zip`).
pub(crate) fn rope_q(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, PCl, PSl, m![Gs, Ds]>,
    cos_lo: &RopeVrf,
    cos_hi: &RopeVrf,
    sin_lo: &RopeVrf,
    sin_hi: &RopeVrf,
) -> DmTensor<bf16, Chip, PCl, PSl, m![Gs, Ds]> {
    let mut out: DmTensor<bf16, Chip, PCl, PSl, m![Gs, Ds]> = DmTensor::new();
    macro_rules! zip_half {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $c, ())
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), $s)
                .vector_fp_zip(FpBinaryOp::AddF)
                .vector_widen_concat::<m![Gs, Ds = 128 / 8], m![Ds = 128 % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds = 128 % 8 # 16]>()
                .commit_trim::<m![Ds = 128 % 8]>()
                .commit_view(out.view_mut().tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>($own));
        };
    }
    zip_half!(0, 128, cos_lo, sin_lo);
    zip_half!(128, 0, cos_hi, sin_hi);
    out
}

/// k RoPE, two zip passes.
pub(crate) fn rope_k(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, PCl, PSl, m![Ds]>,
    cos_lo: &RopeVrf,
    cos_hi: &RopeVrf,
    sin_lo: &RopeVrf,
    sin_hi: &RopeVrf,
) -> DmTensor<bf16, Chip, PCl, PSl, m![Ds]> {
    let mut out: DmTensor<bf16, Chip, PCl, PSl, m![Ds]> = DmTensor::new();
    macro_rules! zip_half_k {
        ($own:literal, $other:literal, $c:ident, $s:ident) => {
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
