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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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

/// KVD: Mixed, S = 2 prefill segments + B = 8 decode rows for the hybrid pass, with the K / V of the launch's rows ALREADY written into the ring caches by the QKV launch
/// (`ops_qkvc::sliding_qkv_pair_direct`): no stage tensors, no appends here. Prefill keys = one window gather of 1024 ring rows + 128 chunk rows per unit; the decode rows read their
/// 1024-row windows (their new key was scattered by the QKV launch). Output through the same handle as `sliding_ragged_s2_b8_h`.
#[device(chip = 1)]
pub fn sliding_ragged_s2_b8_kvd(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill_kvd::ragged_prefill_pair_kvd(device, q_p, k_cache, v_cache, win_index, mask_p, pen_p, out_p);
    dattn::sliding::sliding_decode_attend8_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p);
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
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
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p1, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
}

// ---- N3 entries for the 256-row pass (t256b, rebased onto fin8: stage-based path) ----
/// 256-row pass, 8 decode rows: prefill (two waves, S = 4) + KV appends + decode attention in ONE launch. The decode rows are the LAST 8 rows of unit 3 (wave 1, unit 1): their
/// attention output is stored into `out_p1` THROUGH THE SAME HANDLE as the wave-1 prefill store (the compiler orders the two stores).
#[device(chip = 1)]
pub fn sliding_ragged_s4_b8_h(
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
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256_r8(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p1, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend8_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p1);
}

/// 256-row pass, 16 decode rows (the last 16 rows of unit 3), otherwise as `sliding_ragged_s4_b8_h`.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b16_h(
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
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    v_new_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]>,
    append_d: &HbmTensor<i32, Chip, m![Bd16]>,
    index_d: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    k_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill::ragged_prefill_pair_256(device, q_p0, k_cache, v_cache, ring_index0, k_stage, v_stage, stage_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill::ragged_prefill_pair_256_r16(device, q_p1, k_cache, v_cache, ring_index1, k_stage, v_stage, stage_index1, mask_p1, pen_p1, out_p1);
    rpa::prefill::ragged_append_256(device, k_stage, append_p, k_cache);
    rpa::prefill::ragged_append_256(device, v_stage, append_p, v_cache);
    dattn::append::scatter_sliding_rows16(device, k_new_d, append_d, k_cache);
    dattn::append::scatter_sliding_rows16(device, v_new_d, append_d, v_cache);
    dattn::sliding::sliding_decode_attend16_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p1);
}

// ---- KVD-256 entries (QKV scattered K / V into the ring; each unit reads ONE 1280-row window of the ring) ----
/// 256-row pass, 8 decode rows (the last 8 rows of unit 3): S = 4 prefill units (two waves) + decode attention in ONE launch, no stage tensors and no appends (the QKV launches
/// `sliding_qkv_pair_direct` already wrote every K / V row of the pass into the ring caches). The decode rows' output is stored through the same handle as the wave-1 prefill store.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b8_kvd(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index0: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    win_index1: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256(device, q_p0, k_cache, v_cache, win_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_r8(device, q_p1, k_cache, v_cache, win_index1, mask_p1, pen_p1, out_p1);
    dattn::sliding::sliding_decode_attend8_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p1);
}

/// 256-row pass, 16 decode rows (the last 16 rows of unit 3), otherwise as `sliding_ragged_s4_b8_kvd`.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b16_kvd(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index0: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    win_index1: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
) {
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256(device, q_p0, k_cache, v_cache, win_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_r16(device, q_p1, k_cache, v_cache, win_index1, mask_p1, pen_p1, out_p1);
    dattn::sliding::sliding_decode_attend16_view(device, q_d, k_cache, v_cache, index_d, mask_d, out_p1);
}

// ---- race-free decode part of the KVD launches (R12 layouthaz, from race4): the decode rows' K / V window gather is the BLOCK-granular UNSCALED gather of `dattn::sliding_blk` (index = raw
// 32-row block numbers staged on chip by `to_dm`: no table-load command, no table-load / gather race). K / V of the decode rows are already in the ring caches (the QKV launch scattered them),
// so the block view of the caches (`k_blk` / `v_blk`, the same buffers as `k_cache` / `v_cache`, read only here) needs no ordering inside the launch. The prefill units still use the scaled window gather.

/// `sliding_ragged_s2_b8_kvd` with the race-free decode part.
#[device(chip = 1)]
pub fn sliding_ragged_s2_b8_kvd_blk(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Sb]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Rw32, Sb40]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    k_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    v_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
) {
    rpa::prefill_kvd::ragged_prefill_pair_kvd(device, q_p, k_cache, v_cache, win_index, mask_p, pen_p, out_p);
    dattn::sliding_blk::sliding_decode_attend8_blk_view(device, q_d, k_blk, v_blk, index_d, mask_d, out_p);
}

/// `sliding_ragged_s4_b8_kvd` with the race-free decode part.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b8_kvd_blk(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index0: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    win_index1: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd, Sb]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd, Rw32, Sb40]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    k_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    v_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
) {
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256(device, q_p0, k_cache, v_cache, win_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_r8(device, q_p1, k_cache, v_cache, win_index1, mask_p1, pen_p1, out_p1);
    dattn::sliding_blk::sliding_decode_attend8_blk_view(device, q_d, k_blk, v_blk, index_d, mask_d, out_p1);
}

/// `sliding_ragged_s4_b16_kvd` with the race-free decode part.
#[device(chip = 1)]
pub fn sliding_ragged_s4_b16_kvd_blk(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    win_index0: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    win_index1: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
    q_d: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    index_d: &HbmTensor<i32, Chip, m![Bd16, Sb]>,
    mask_d: &HbmTensor<bf16, Chip, m![Bd16, Rw32, Sb40]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    k_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    v_blk: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
) {
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256(device, q_p0, k_cache, v_cache, win_index0, mask_p0, pen_p0, out_p0);
    rpa::prefill_kvd256::ragged_prefill_pair_kvd256_r16(device, q_p1, k_cache, v_cache, win_index1, mask_p1, pen_p1, out_p1);
    dattn::sliding_blk::sliding_decode_attend16_blk_view(device, q_d, k_blk, v_blk, index_d, mask_d, out_p1);
}
