use super::rope_h4::HeadCluster as Cluster;
use super::rope_h4::KvHeadsAcrossSlices as Heads;
use crate::axes::{Ds, Gs};
use crate::{Chip, EPS};
use furiosa_opt_std::prelude::*;

const DS_F32: f32 = Ds::SIZE as f32;

/// Normalize each head on its own slice; the reduction axis remains Ds = 256.
/// q with its per-channel scale folded in (see `normalize_value_scaled`): mean square of s * q,
/// then w * q * s / rms -> bf16. Drops the separate scale pass and its bf16 rounding.
pub(crate) fn normalize_query_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// k with its per-channel scale folded in.
pub(crate) fn normalize_key_scaled(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `normalize_key_scaled` with the norm weight already loaded (qd).
pub(crate) fn normalize_key_scaled_dm(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// V with its scale folded in and the sqrt pass writing the rms VRF directly (Main -> Sub VRF).
pub(crate) fn normalize_value_scaled_sv(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
        .to_vrf(&mut device.sub);
    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `normalize_key_scaled` whose mean square adds an exact-zero VRF (a pure ordering dependency).
pub(crate) fn normalize_key_scaled_z(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
    zero: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// A head-layout [Ds] bf16 tensor -> f32 VRF through a Main vector pass (x * 1.0): not eager, unlike a Sub to_vrf.
pub(crate) fn heads_vrf_main(
    device: &mut Device,
    t: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> {
    device.main
        .begin(t.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `heads_vrf_main` for [Gs, Ds].
pub(crate) fn heads_vrf_main_g(
    device: &mut Device,
    t: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> {
    device.main
        .begin(t.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0f32)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub)
}

/// `normalize_key_scaled` with the scale VRF written by a Main pass.
pub(crate) fn normalize_key_scaled_e(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale = heads_vrf_main(device, scale);
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `normalize_query_scaled` with the scale VRF written by a Main pass.
pub(crate) fn normalize_query_scaled_e(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let scale = heads_vrf_main_g(device, scale);
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// `normalize_key_scaled` whose FINAL pass (the one streaming the norm weight) adds an exact-zero VRF from
/// the Q contraction: only the norm-weight load is pinned behind the Q weight; the scale load and the
/// mean square stay free to run earlier.
pub(crate) fn normalize_key_scaled_z2(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
    zero: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// qkv11 SQF (V10-E311): `normalize_value_scaled_sv` without the sqrt pass. The mean-square pass writes
/// `ms + EPS` straight to the VRF (Main -> Sub VRF); the final pass stashes `x * scale`, replaces the stream
/// with the per-slice `ms` (`FmaF` in `Mode021`: stream * 0 + ms), takes its sqrt on FpFpu and divides the
/// stash by it on the separate FpDiv stage (`Mode10`: operand / stream).
pub(crate) fn normalize_value_scaled_sqf(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let ms_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_ternary_with_mode(FpTernaryOp::FmaF, TernaryArgMode::Mode021, (&ms_vrf, 0.0f32))
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
/// V12 ksq2 (qkvD2): `normalize_key_scaled_z2o` split in two. Part 1 (scale VRF = scale * 1.0, mean square, rms,
/// rms VRF, k VRF) has no k_rms / q dependency; part 2 is the final pass streaming k_rms (pinned to `zero`).
pub(crate) fn normalize_key_p1(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Ds]>,
    one: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
) -> (
    VrfTensor<f32, Chip, Cluster, Heads, m![Ds]>,
    DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    VrfTensor<f32, Chip, Cluster, Heads, m![Ds]>,
) {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .main
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), one)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
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
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    (scale, rms, rms_vrf, x_vrf)
}

/// V12 ksq2: part 2 of the split K normalize (the final pass streaming k_rms, pinned to `zero`).
pub(crate) fn normalize_key_p2(
    device: &mut Device,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
    zero: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    scale: &VrfTensor<f32, Chip, Cluster, Heads, m![Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    x_vrf: &VrfTensor<f32, Chip, Cluster, Heads, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> {
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Ds]> = weight.to_dm(&mut device.tdma);
    device.main
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale)
        .vector_fp_binary(FpBinaryOp::DivF, rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

// ---- V17 QKV qf7 (engineer qkv18): the q normalize of `normalize_query_scaled` split in two parts (part 1: the q
// scale as a Main pass x exact 1.0, mean square, rms; part 2: the q-norm weight, gated on an exact zero), as V16's
// `normalize_key_p1` / `normalize_key_p2`. Values identical. ----
pub(crate) fn normalize_query_p1(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    scale: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    one: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
) -> (
    VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
    DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]>,
    VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]>,
    VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
) {
    let scale: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .main
        .begin(scale.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), one)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let mean_square: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .main
        .begin(mean_square.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]> = device
        .sub
        .begin(rms.view())
        .fetch::<m![Gs], m![1 # 8]>()
        .collect::<m![Gs], m![1 # 8]>()
        .to_vrf();
    let x_vrf: VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]> = device
        .sub
        .begin(x.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();
    (scale, rms, rms_vrf, x_vrf)
}

/// Part 2 of the split q normalize: the final pass streaming q_rms (loaded here), pinned to `zero`.
pub(crate) fn normalize_query_p2(
    device: &mut Device,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
    zero: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    scale: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]>,
    x_vrf: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    let weight: DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> = weight.to_dm(&mut device.tdma);
    normalize_query_p2_dm(device, &weight, zero, scale, rms, x_vrf)
}

/// Part 2 with q_rms already in DM (`[Gs, Ds]`, the DMA replicates the `[Ds]` row over Gs).
pub(crate) fn normalize_query_p2_dm(
    device: &mut Device,
    weight: &DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]>,
    zero: &VrfTensor<f32, Chip, Cluster, Heads, m![1 # 8]>,
    scale: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
    rms: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, 1 # 8]>,
    x_vrf: &VrfTensor<f32, Chip, Cluster, Heads, m![Gs, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Heads, m![Gs, Ds]> {
    device.main
        .begin(weight.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, zero)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), scale)
        .vector_fp_binary(FpBinaryOp::DivF, rms)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
