//! Batched decode attention entry points (B rows from different requests per launch). See RESULTS.md.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::dattn;

/// Batched sliding-layer decode attention (B = Bd rows, two clusters = row parity).
#[device(chip = 1)]
pub fn sliding_decode_attention(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
) {
    dattn::sliding::sliding_decode_attend8(device, q, k_cache, v_cache, index, mask, out);
}

/// Appends the Bd new k / v rows ([Bd, Ns, Ds], staging layout) into the unified flat sliding caches.
/// `index[b]` = byte offset of cache row `slot_b * 1024 + position_b % 1024` (`dattn_host::sliding_append_index`).
#[device(chip = 1)]
pub fn sliding_decode_append(
    device: &mut Device,
    k_new: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    dattn::append::scatter_sliding_rows(device, k_new, index, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new, index, v_cache);
}

/// Same kernel for B = 16 rows (Bd16, 8 rows per cluster).
#[device(chip = 1)]
pub fn sliding_decode_attention_16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
) {
    dattn::sliding::sliding_decode_attend16(device, q, k_cache, v_cache, index, mask, out);
}

#[device(chip = 1)]
pub fn sliding_decode_append_16(
    device: &mut Device,
    k_new: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd16]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    dattn::append::scatter_sliding_rows16(device, k_new, index, k_cache);
    dattn::append::scatter_sliding_rows16(device, v_new, index, v_cache);
}

/// Batched full-layer decode attention, 5-page bucket (positions < 2560), 8 rows.
#[device(chip = 1)]
pub fn full_decode_attention_5(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
    index: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    mask: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
) {
    dattn::full::full_decode_attend8_5(device, q, k_pool, v_pool, index, mask, out);
}

/// Two 8-row groups (16 rows) in one launch, 5-page bucket: group `a` and group `b` are independent (own q / index / mask / out).
#[device(chip = 1)]
pub fn full_decode_attention_5_x2(
    device: &mut Device,
    q_a: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    q_b: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
    index_a: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    index_b: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    mask_a: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    mask_b: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    out_a: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    out_b: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
) {
    dattn::full::full_decode_attend8_5(device, q_a, k_pool, v_pool, index_a, mask_a, out_a);
    dattn::full::full_decode_attend8_5(device, q_b, k_pool, v_pool, index_b, mask_b, out_b);
}

/// Appends the Bd new k / v rows of a full layer ([Bd, Df]) into the flat page pools (`dattn_host::full_append_index`).
#[device(chip = 1)]
pub fn full_decode_append(
    device: &mut Device,
    k_new: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    index: &HbmTensor<i32, Chip, m![Bd]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
) {
    dattn::append::scatter_full_rows(device, k_new, index, k_pool);
    dattn::append::scatter_full_rows(device, v_new, index, v_pool);
}

/// Two 8-row append groups in one launch (rows 0..8 and 8..16 of a 16-row step).
#[device(chip = 1)]
pub fn full_decode_append_x2(
    device: &mut Device,
    k_new_a: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new_a: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    index_a: &HbmTensor<i32, Chip, m![Bd]>,
    k_new_b: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new_b: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    index_b: &HbmTensor<i32, Chip, m![Bd]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
) {
    dattn::append::scatter_full_rows(device, k_new_a, index_a, k_pool);
    dattn::append::scatter_full_rows(device, v_new_a, index_a, v_pool);
    dattn::append::scatter_full_rows(device, k_new_b, index_b, k_pool);
    dattn::append::scatter_full_rows(device, v_new_b, index_b, v_pool);
}

/// Sliding layers: two 8-row groups (16 rows) in one launch.
#[device(chip = 1)]
pub fn sliding_decode_attention_x2(
    device: &mut Device,
    q_a: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    q_b: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index_a: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    index_b: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_a: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    mask_b: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out_a: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    out_b: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
) {
    dattn::sliding::sliding_decode_attend8(device, q_a, k_cache, v_cache, index_a, mask_a, out_a);
    dattn::sliding::sliding_decode_attend8(device, q_b, k_cache, v_cache, index_b, mask_b, out_b);
}

/// Sliding layers, B = 8: appends the new k / v rows into the caches and then attends (one launch instead of two).
#[device(chip = 1)]
pub fn sliding_decode_attention_fused(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_index: &HbmTensor<i32, Chip, m![Bd]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
) {
    dattn::append::scatter_sliding_rows(device, k_new, append_index, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new, append_index, v_cache);
    dattn::sliding::sliding_decode_attend8(device, q, k_cache, v_cache, index, mask, out);
}

/// Sliding layers, 16 rows as two 8-row groups: appends the new k / v rows of both groups, then attends both groups.
#[device(chip = 1)]
pub fn sliding_decode_attention_fused_x2(
    device: &mut Device,
    q_a: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    q_b: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new_a: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_a: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    k_new_b: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_b: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_index_a: &HbmTensor<i32, Chip, m![Bd]>,
    append_index_b: &HbmTensor<i32, Chip, m![Bd]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index_a: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    index_b: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_a: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    mask_b: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out_a: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    out_b: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
) {
    dattn::append::scatter_sliding_rows(device, k_new_a, append_index_a, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_a, append_index_a, v_cache);
    dattn::append::scatter_sliding_rows(device, k_new_b, append_index_b, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_b, append_index_b, v_cache);
    dattn::sliding::sliding_decode_attend8(device, q_a, k_cache, v_cache, index_a, mask_a, out_a);
    dattn::sliding::sliding_decode_attend8(device, q_b, k_cache, v_cache, index_b, mask_b, out_b);
}

/// Full layers, 5-page bucket, 8 rows: appends the new k / v rows into the pools, then attends.
#[device(chip = 1)]
pub fn full_decode_attention_5_fused(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    k_new: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    append_index: &HbmTensor<i32, Chip, m![Bd]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    index: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    mask: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
) {
    dattn::append::scatter_full_rows(device, k_new, append_index, k_pool);
    dattn::append::scatter_full_rows(device, v_new, append_index, v_pool);
    dattn::full::full_decode_attend8_5(device, q, k_pool, v_pool, index, mask, out);
}

/// Full layers, 5-page bucket, 16 rows as two 8-row groups: appends both groups, then attends both groups.
#[device(chip = 1)]
pub fn full_decode_attention_5_fused_x2(
    device: &mut Device,
    q_a: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    q_b: &HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    k_new_a: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new_a: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    k_new_b: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    v_new_b: &HbmTensor<bf16, Chip, m![Bd, Df]>,
    append_index_a: &HbmTensor<i32, Chip, m![Bd]>,
    append_index_b: &HbmTensor<i32, Chip, m![Bd]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    index_a: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    index_b: &HbmTensor<i32, Chip, m![Bd, Kx5]>,
    mask_a: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    mask_b: &HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>,
    out_a: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
    out_b: &mut HbmTensor<bf16, Chip, m![Bd, Gf, Df]>,
) {
    dattn::append::scatter_full_rows(device, k_new_a, append_index_a, k_pool);
    dattn::append::scatter_full_rows(device, v_new_a, append_index_a, v_pool);
    dattn::append::scatter_full_rows(device, k_new_b, append_index_b, k_pool);
    dattn::append::scatter_full_rows(device, v_new_b, append_index_b, v_pool);
    dattn::full::full_decode_attend8_5(device, q_a, k_pool, v_pool, index_a, mask_a, out_a);
    dattn::full::full_decode_attend8_5(device, q_b, k_pool, v_pool, index_b, mask_b, out_b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Chunked-prefill kernels on the unified flat caches (same math as ops_chunk, the ring is gathered from `[Rf, Ns, Ds]`).
// ---------------------------------------------------------------------------------------------------------------------

/// `ring_index[t]` = byte offset of ring row t of the request: `(slot * 1024 + t) * Ns * Ds * 2` (`dattn_host::sliding_gather_index(&[slot], 1)`).
#[device(chip = 1)]
pub fn sliding_chunk_attention_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Tx]>,
    k_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    mask: &HbmTensor<bf16, Chip, m![Ts % 32, Tc, Seg, Ts / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
) {
    dattn::chunk_flat::sliding_chunk_attend_flat(device, q, k_cache, v_cache, ring_index, k_stage, v_stage, mask, out);
}

/// Appends the staged chunk rows into the flat cache; `index[c]` = byte offset of cache row `slot * 1024 + (p0 + c) % 1024`.
#[device(chip = 1)]
pub fn sliding_chunk_append_flat(
    device: &mut Device,
    k_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Ts = 64]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    dattn::chunk_flat::append_chunk_rows_flat(device, k_stage, index, k_cache);
    dattn::chunk_flat::append_chunk_rows_flat(device, v_stage, index, v_cache);
}

/// Appends a full-layer chunk ([Tc, Df]) into the flat page pool; `index[c]` = byte offset of pool row `page * 512 + (p0 % 512) + c`.
#[device(chip = 1)]
pub fn full_chunk_append_flat(
    device: &mut Device,
    k_chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
) {
    dattn::chunk_flat::append_full_chunk_rows_flat(device, k_chunk, index, k_pool);
    dattn::chunk_flat::append_full_chunk_rows_flat(device, v_chunk, index, v_pool);
}

/// Full-layer chunk attention over at most two pages (positions < 1024) on the flat pool viewed as `[Pq, R8, Df]` (8-row blocks).
/// `index[hg][seg][bk]` = byte offset (`block * 8 * Df * 2`) of block `bk` (8 rows) of logical page `seg` of the request, identical for all 8 head groups
/// (`dattn_host::full_chunk_block_index`); mask = `dattn_host::full_chunk_mask_flat(2, p0)`.
#[device(chip = 1)]
pub fn full_chunk_attention_2_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    dattn::full_chunk_flat::full_chunk_attend_2_flat(device, q, k_pool, v_pool, index, mask, out);
}

/// Four-page variant (positions < 2048); mask = `dattn_host::full_chunk_mask_flat(4, p0)`.
#[device(chip = 1)]
pub fn full_chunk_attention_4_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg4, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf64]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    dattn::full_chunk_flat::full_chunk_attend_4_flat(device, q, k_pool, v_pool, index, mask, out);
}

/// Sliding layers, B = 16 rows with native Bd16-shaped tensors: appends the new k / v rows, then attends (one launch).
#[device(chip = 1)]
pub fn sliding_decode_attention_fused_16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_new: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    v_new: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    append_index: &HbmTensor<i32, Chip, m![Bd16]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
) {
    dattn::append::scatter_sliding_rows16(device, k_new, append_index, k_cache);
    dattn::append::scatter_sliding_rows16(device, v_new, append_index, v_cache);
    dattn::sliding::sliding_decode_attend16(device, q, k_cache, v_cache, index, mask, out);
}
