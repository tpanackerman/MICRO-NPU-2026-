//! Baseline bodies of the three Stage 1 kernels, compiled as a reference path.
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::refimpl::layout::{self, Cluster, Replicated, Slice};
use crate::device::refimpl::{shared, sliding};
use crate::Chip;

#[device(chip = 1)]
pub fn sliding_project_qkv_ref(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(device, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::projection::project_query(device, &x, q_weight, q_weight_scale);
    let (k, v) = sliding::projection::project_key_value(device, &x, k_weight, v_weight, k_weight_scale, v_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_key(device, &k, k_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_value(device, &v);

    let (q, k) = sliding::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn sliding_attention_output_ref(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]> = layout::broadcast_sliding_heads(device, &x);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        sliding::projection::project_output(device, &x, o_weight, o_weight_scale);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(device, &x, post_attn_rms_weight);

    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut device.tdma);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(device, &x, &residual);
    residual.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn decoder_feedforward_ref(
    device: &mut Device,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &residual, pre_ff_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::mlp::feedforward(
        device,
        x,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    );

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(device, &x, post_ff_rms_weight);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(device, &x, &residual);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        shared::residual::scale_by_layer_gate(device, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}

