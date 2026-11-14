
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::layout::{self, Cluster, Replicated, Slice};
use crate::device::{full, shared, sliding};
use crate::{Chip, EMBED_SCALE, LOGIT_SOFTCAP};

#[device(chip = 1)]
pub fn embed_token(
    device: &mut Device,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = embedding_table.dma_gather_scaled(offset);
    let row: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = row.to_dm(&mut device.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = device
        .main
        .begin(row.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), EMBED_SCALE)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn sliding_project_qkv(
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
    // `cos` / `sin` are part of the fixed skeleton signature; the RoPE rows are computed on chip.
    let _ = cos;
    let _ = sin;
    // V13 qkvB13 (x15k2): K-split-15 weight layout (15 x 256 B H-chunks per row across a 16-slot slice group; the
    // DMA windows walk whole rows) with qkvD2's ksq2 ordering: K lands first and its gather, rope-row staging and
    // K-norm part 1 run during the Q weight DMA (src/device/sliding/qkv_k15b.rs, experiments/qkvB13_notes.md;
    // same-job labs vs V12: Arena 91814 -362, 91889 -429 / E[max2] -513).
    let weights = sliding::qkv_k15b::load_weights_k15(device, q_weight, k_weight, v_weight);
    // V16 qkv17 g6z: input RMS weight loaded once per PE + on-chip fan-out, the K scale ahead of K, and the zip RoPE
    // (Arena 100276 -853, 100292 -814 vs x15k2; src/device/sliding/qkv_k15b.rs `normalize_input_k15g6`,
    // `qkv_k15_ksq2_zip`; experiments/qkv17_progress.md).
    // V17 qkv18 qf7: Q-first weight order (Q, K, V): the RoPE rows run under the Q stream, the K contraction is pinned
    // on the q rms and the K scale (Arena 100939 -636, 100958 -413 vs V16 g6z; src/device/sliding/qkv_k15b.rs
    // `normalize_input_k15g6m18`, `qkv18_qf7`; experiments/qkv18_progress.md).
    // V18 qkv19 q19n2: qf7 with the two-pass V normalize (sqrt folded into the final pass, V10-E311 form; INFO labs
    // 102090 -336, 102103 -392, 102114 -185 vs qf7, pooled with 101974 -224 SE 48; `qkv19_qf7n2`;
    // experiments/qkv19_progress.md).
    // Q800 vtail: qkv19_qf7n2 with the V tail as two packed switch + VE passes (gather + mean square, gather +
    // normalize) over a 4-consecutive-channel V partial (src/device/sliding/qkv_k15b.rs `qkv63_vtail`).
    let (x_trf, mask, mdm) = sliding::qkv_k15b::normalize_input_k15x4f(device, x, input_rms_weight);
    let (q, k, v) = sliding::qkv_k15b::qkv63_vtail(
        device, &x_trf, &mask, &mdm, weights, q_weight_scale, k_weight_scale, v_weight_scale, rope_offset,
        q_rms_weight, k_rms_weight,
    );

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn full_project_qkv(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(device, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::projection::project_query(device, &x, q_weight, q_weight_scale);
    let k_raw: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> =
        full::projection::project_key(device, &x, k_weight, k_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = full::rmsnorm::normalize_query(device, &q, q_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_value(device, &k_raw);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_key(device, &k_raw, k_rms_weight);

    let (q, k) = full::rope::apply_rope(device, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn sliding_attention(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    sliding::attention::attend(device, q, k, v, mask, out_hbm);
}

#[device(chip = 1)]
pub fn sliding_attention_output(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let x = sliding::projection::load_output_input(device, x);

    // Channel scale, post-attention RMSNorm and residual add are fused into the projection's
    // epilogue, which ends in a 32-slice layout that is stored to HBM directly.
    let out: DmTensor<bf16, Chip, Cluster, sliding::projection::AoEpilogueSlices, m![H % 120]> =
        sliding::projection::project_output_fused(device, &x, o_weight, o_weight_scale, post_attn_rms_weight, residual_hbm);
    out.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_first_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_first_page(device, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_page(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_next_page(device, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_output(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![Gf]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::attention::divide_by_softmax_sum(device, &x, running_sum);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> = layout::broadcast_full_heads(device, &x);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        full::projection::project_output(device, &x, o_weight, o_weight_scale);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(device, &x, post_attn_rms_weight);

    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut device.tdma);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(device, &x, &residual);
    residual.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn decoder_feedforward(
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
    let x = shared::mlp::normalize_input(device, residual_hbm, pre_ff_rms_weight);
    // Post-FFN RMSNorm, residual add and layer gate are fused into the FFN's epilogue, which
    // ends in an 8-slice layout that is stored to HBM directly.
    let out: DmTensor<bf16, Chip, Cluster, shared::mlp::EpilogueSlices, shared::mlp::EpilogueElem> = shared::mlp::feedforward(
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
        post_ff_rms_weight,
        layer_scalar,
        residual_hbm,
    );
    out.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn final_norm_and_logits(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![W]>,
) {
    let x: DmTensor<bf16, Chip, shared::lm_head::Cluster, Slice, m![H]> = input.to_dm(&mut device.tdma);
    let x = shared::rmsnorm::normalize(device, &x, rms_weight);

    let logits = shared::lm_head::logits(device, &x, lm_head_weight);

    let scaled: DmTensor<
        f32,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = device
        .main
        .begin(logits.view())
        .fetch::<m![W / 16 % 32], m![W % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_div(LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .commit_trim::<m![W % 8]>()
        .commit();

    let capped: DmTensor<
        bf16,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = device
        .main
        .begin(scaled.view())
        .fetch::<m![W / 8 % 64], m![W % 8]>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit();

    capped.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
