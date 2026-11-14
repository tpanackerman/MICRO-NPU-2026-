//! Ragged attention entry points: ONE launch per layer for a hybrid batch (prefill segments of 64 rows + decode rows), KV append fused.
//! See RESULTS.md. Sliding layers only in this file's first part; the full-layer variants follow.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::{dattn, rpa};

/// Prefill-only ragged attention, S = 2 segments (one wave of two units, one per cluster) + the append of the staged rows.
#[device(chip = 1)]
pub fn sliding_ragged_s2(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P128]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_128(device, q_p, k_cache, v_cache, ring_index, k_stage, v_stage, stage_index, mask_p, pen_p, out_p);
    rpa::prefill::ragged_append_128(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_128(device, v_stage, append_p, v_cache);
}

/// Mixed ragged attention, S = 2 prefill segments + B = 16 decode rows (native Bd16 tensors): prefill attention reads the rings, then all appends,
/// then the decode attention (reads its rings after its own append).
#[device(chip = 1)]
pub fn sliding_ragged_s2_b16(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P128]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd16]>,
    index_d: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    out_d: &mut HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_128(device, q_p, k_cache, v_cache, ring_index, k_stage, v_stage, stage_index, mask_p, pen_p, out_p);
    rpa::prefill::ragged_append_128(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_128(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows16(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows16(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend16(device, q_d, k_cache, v_cache, index_d, mask_d, out_d);
}

/// Mixed, S = 2 prefill segments + B = 8 decode rows.
#[device(chip = 1)]
pub fn sliding_ragged_s2_b8(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P128]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out_d: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_128(device, q_p, k_cache, v_cache, ring_index, k_stage, v_stage, stage_index, mask_p, pen_p, out_p);
    rpa::prefill::ragged_append_128(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_128(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend8(device, q_d, k_cache, v_cache, index_d, mask_d, out_d);
}

/// Mixed, S = 2 prefill segments + B = 8 decode rows, for the HYBRID pass layout: the 8 decode rows are the LAST 8 rows of the 128-row prefill output tensor. The prefill part stores all
/// 128 rows (the last 8 are junk), then the decode part stores its 8 rows into the same rows THROUGH THE SAME HANDLE `out_p` (a tile of its view), so the compiler orders the two
/// stores (write-after-write on one handle). Everything else is `sliding_ragged_s2_b8`; there is no separate `out_d`.
#[device(chip = 1)]
pub fn sliding_ragged_s2_b8_h(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P128]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_128_r8(device, q_p, k_cache, v_cache, ring_index, k_stage, v_stage, stage_index, mask_p, pen_p, out_p);
    // The decode rows' scatters and the decode attention come BEFORE the prefill appends: the prefill rows belong to other request slots than the decode rows (the host guarantees it), so
    // the final caches and both outputs are the same, and the scheduler overlaps the prefill appends with the tail of the launch (static -2.5%, hardware -3%).
    dattn::append::scatter_sliding_rows(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend8_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p);
    rpa::prefill::ragged_append_128(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_128(device, v_stage, append_p, v_cache);
}

/// Mixed, S = 4 prefill segments (two waves of two units) + B = 16 decode rows.
/// NOTE (penalty port): both waves share ONE per-key penalty `pen_p0` (24-element argument tuple limit). Build it as the intersection (a slot is penalised only if it is
/// invisible to both waves), which is always safe. The hybrid pass does not use this variant.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b16(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index0: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    ring_index1: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index0: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    stage_index1: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P256]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd16]>,
    index_d: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    out_d: &mut HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p0, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows16(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows16(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend16(device, q_d, k_cache, v_cache, index_d, mask_d, out_d);
}

/// Mixed, S = 4 prefill segments + B = 8 decode rows.
/// NOTE (penalty port): both waves share ONE per-key penalty `pen_p0` (24-element argument tuple limit). Build it as the intersection (a slot is penalised only if it is
/// invisible to both waves), which is always safe. The hybrid pass does not use this variant.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b8(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index0: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    ring_index1: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index0: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    stage_index1: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P256]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    out_d: &mut HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p0, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend8(device, q_d, k_cache, v_cache, index_d, mask_d, out_d);
}

/// Prefill-only ragged attention, S = 4 segments (two waves) + the append of the staged rows.
#[device(chip = 1)]
pub fn sliding_ragged_s4(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    ring_index0: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    ring_index1: &HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage_index0: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    stage_index1: &HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    k_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    append_p: &HbmTensor<i32, Chip, m![P256]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p1, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
}
