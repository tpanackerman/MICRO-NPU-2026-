//! Host-side helpers for the chunked-prefill attention kernels in `ops_chunk` (masks and append indices).
//!
//! A chunk is `Tc` (= 64) consecutive prompt positions `p0 .. p0 + Tc` of one request (`p0` a multiple of `Tc`, so a chunk
//! never wraps the sliding ring and never crosses a full-attention page). All 40 sliding layers share one sliding mask and
//! one append index per chunk; all 8 full layers share one full mask.
use furiosa_opt_std::prelude::*;

use crate::axes::{Tc, Tf, Ts};

fn one() -> bf16 {
    bf16::from_f32(1.0)
}
fn zero() -> bf16 {
    bf16::from_f32(0.0)
}
/// Score penalty of a key slot that can never be visible (`MASK_PEN`).
fn hidden() -> bf16 {
    bf16::from_f32(crate::MASK_PEN)
}

/// Position held by sliding-ring row `r` before the chunk starting at `p0` is appended (`None`: never written).
pub fn sliding_ring_pos(p0: usize, r: usize) -> Option<usize> {
    if p0 == 0 {
        return None;
    }
    let ts = Ts::SIZE as i64;
    let p = p0 as i64;
    let kpos = p - 1 - ((p - 1 - r as i64).rem_euclid(ts));
    if kpos >= 0 { Some(kpos as usize) } else { None }
}

/// Mask for `sliding_chunk_attention`: `m![Ts % 32, Tc, Seg, Ts / 32]` bf16 0/1, flat index
/// `((lane * Tc + t) * 2 + seg) * 32 + slot`. Segment 0 is the ring (ring row `slot * 32 + lane`, visible to query row `t`
/// iff the position it holds is `>= p0 + t - 1023`), segment 1 is the chunk's own staged keys (key `c = slot * 32 + lane`,
/// visible iff `c <= t`; only slots 0 and 1 exist for `Tc = 64`).
pub fn sliding_mask(p0: usize) -> Vec<bf16> {
    sliding_mask_rows(p0, Tc::SIZE)
}

/// `sliding_mask` for a partial chunk of `n` real rows: query rows `t >= n` (padding) get the mask of row `n - 1`, so they see exactly the keys row
/// `n - 1` sees and compute exactly what it computes (clones: their duplicate KV appends rewrite identical values, and their cache rows stay exact).
pub fn sliding_mask_rows(p0: usize, n: usize) -> Vec<bf16> {
    let tc = Tc::SIZE;
    let ts = Ts::SIZE;
    let mut mask = vec![zero(); 32 * tc * 2 * 32];
    for lane in 0..32 {
        for t_row in 0..tc {
            let t = t_row.min(n - 1);
            for slot in 0..32 {
                let r = slot * 32 + lane;
                let low = (p0 + t) as i64 - (ts as i64 - 1);
                if let Some(pos) = sliding_ring_pos(p0, r) {
                    if pos as i64 >= low {
                        mask[((lane * tc + t_row) * 2) * 32 + slot] = one();
                    }
                }
                if slot < tc / 32 && slot * 32 + lane <= t {
                    mask[((lane * tc + t_row) * 2 + 1) * 32 + slot] = one();
                }
            }
        }
    }
    mask
}

/// Per-key additive score penalty for `sliding_chunk_attention` (`m![Ts % 32, Seg, Ts / 32]` bf16, flat `(lane * 2 + seg) * 32 + slot`): `MASK_PEN` for the key slots that no row
/// of the chunk can ever see, 0 elsewhere. Segment 0 (ring row `slot * 32 + lane`): rows the request has not written yet (`sliding_ring_pos == None`: they hold a previous
/// request's keys or zeros). Segment 1 (staged chunk keys): the zero-filled slots `>= Tc / 32`. Added to the scores before the row max, so the max (and every bf16 rounding of
/// `exp(s - max)` after it) is independent of the slot history. Keys that are masked for SOME rows only (out-of-window ring rows, chunk keys after the query row) are
/// deterministic functions of the request and stay as they are.
pub fn sliding_pen(p0: usize) -> Vec<bf16> {
    let tc = Tc::SIZE;
    let mut pen = vec![zero(); 32 * 2 * 32];
    for lane in 0..32 {
        for slot in 0..32 {
            if sliding_ring_pos(p0, slot * 32 + lane).is_none() {
                pen[(lane * 2) * 32 + slot] = hidden();
            }
            if slot >= tc / 32 {
                pen[(lane * 2 + 1) * 32 + slot] = hidden();
            }
        }
    }
    pen
}

/// Append index for `sliding_chunk_append` (`m![Ts = 64]`): byte offset of ring row `(p0 + c) % 1024` (one row = Ns * Ds * 2 B).
pub fn sliding_append_index(p0: usize) -> Vec<i32> {
    let row_bytes = crate::axes::Ns::SIZE * crate::axes::Ds::SIZE * 2;
    (0..Tc::SIZE).map(|c| (((p0 + c) % Ts::SIZE) * row_bytes) as i32).collect()
}

/// Mask for `full_chunk_attention_{2,4}d` with `npages` (2 or 4) key pages passed in logical page order (page `s` holds
/// positions `s * 512 ..`): `m![Tf % 32, Tc, Kf]` bf16 0/1, flat index `(lane * Tc + t) * (16 * npages) + kf` with
/// `kf = seg * 16 + slot * 8 + g`; key `j = slot * 256 + g * 32 + lane` of page `seg` has position `seg * 512 + j` and is
/// visible to query row `t` iff `position <= p0 + t`. Pages after the last needed one (all positions > p0 + Tc - 1) get an
/// all-zero mask; the host may pass any valid page tensor for them (for example page 0).
pub fn full_mask(npages: usize, p0: usize) -> Vec<bf16> {
    full_mask_rows(npages, p0, Tc::SIZE)
}

/// `full_mask` for a partial chunk of `n` real rows (padding rows `t >= n` get the mask of row `n - 1`, see `sliding_mask_rows`).
pub fn full_mask_rows(npages: usize, p0: usize, n: usize) -> Vec<bf16> {
    let tc = Tc::SIZE;
    let tf = Tf::SIZE;
    let nk = 16 * npages;
    let mut mask = vec![zero(); 32 * tc * nk];
    for lane in 0..32 {
        for t_row in 0..tc {
            let t = t_row.min(n - 1);
            for kf in 0..nk {
                let (seg, slot, g) = (kf / 16, (kf % 16) / 8, kf % 8);
                let j = slot * 256 + g * 32 + lane;
                if seg * tf + j <= p0 + t {
                    mask[(lane * tc + t_row) * nk + kf] = one();
                }
            }
        }
    }
    mask
}

/// Per-key additive score penalty for `full_chunk_attention_{2,4}` (`m![Tf % 32, Kf]` bf16, flat `lane * nk + kf`, `nk = 16 * npages`) for a chunk whose first `n` rows
/// are real (positions `p0 .. p0 + n`): `MASK_PEN` for the key slots at positions `>= p0 + n` (rows of the pages that the request has not written yet: stale rows of a previous
/// request, zeros, or the page passed in place of an unneeded later page), 0 elsewhere (see `sliding_pen`).
pub fn full_pen(npages: usize, p0: usize, n: usize) -> Vec<bf16> {
    let tf = Tf::SIZE;
    let nk = 16 * npages;
    let mut pen = vec![zero(); 32 * nk];
    for lane in 0..32 {
        for kf in 0..nk {
            let (seg, slot, g) = (kf / 16, (kf % 16) / 8, kf % 8);
            let j = slot * 256 + g * 32 + lane;
            if seg * tf + j >= p0 + n {
                pen[lane * nk + kf] = hidden();
            }
        }
    }
    pen
}

/// Append index for `full_chunk_append` (`m![Tc]`): byte offset of page row `(p0 % 512) + c` (one row = Df * 2 B).
pub fn full_append_index(p0: usize) -> Vec<i32> {
    let row_bytes = crate::axes::Df::SIZE * 2;
    (0..Tc::SIZE).map(|c| (((p0 % Tf::SIZE) + c) * row_bytes) as i32).collect()
}
