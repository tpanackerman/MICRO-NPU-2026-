//! Host helpers for the ragged attention kernels in `ops_rpa` (the "segment table": index tensors and masks of a launch).
//!
//! ## Flat sliding cache layout (changed w.r.t. dattn): ring() = 1280 rows per request slot
//! Per sliding layer K and V are `[Rf, Ns, Ds]` with `Rf = (Rn + 1) * ring()` rows (slot `Rn` = scratch slot). Position `p` of the request in slot `s` lives in row
//! `s * ring() + p % ring()`. A ring of `ring() = 1280 >= 1024 + 256` rows (instead of 1024) makes an append of up to 256 new rows harmless for every key a query of the same
//! launch still needs (the rows it overwrites hold positions `p0 + c - 1280 < p0 - 1023`), so appends and gathers of one launch may execute in ANY order. (The compiler does
//! NOT order a gather before a later scatter to the same tensor: write-after-read is not tracked; a 1024-row ring made the mixed kernel wrong at p0 >= 1024.)
//! Gather indices / masks are windows over POSITIONS, not ring rows (the decode kernel just gathers 1024 rows named by an index and applies a mask).
//!
//! ## Prefill part
//! A launch of the S-segment bucket (S = 2 or 4) carries up to S prefill *units* of 64 rows, grouped into *runs*: a run is the new rows `p0 .. p0 + n`
//! (n <= 256 = Cx) of ONE request (sliding slot `slot`), occupying `ceil(n / 64)` consecutive units. Several runs (different requests) can share a launch; unused
//! units are padding (masked, appended to the scratch slot). Unit `u` runs on cluster `u % 2` of wave `u / 2`; the staged k / v rows of the launch are ONE tensor
//! `[S * 64, Ns, Ds]` whose rows are the runs' new keys in unit order (row `64 * first_unit(run) + c` = new key of position `p0 + c`).
//! Keys of a unit (run, segment j): ring slot i = position `p0 - 1024 + i` (only if >= 0), chunk key c = stage row of position `p0 + c`.
//!
//! ## Decode part
//! `decode_*` below (bd = 8 or 16 rows; padding rows read the scratch slot).
use furiosa_opt_std::prelude::*;

use crate::axes::{Cx, Ds, Ns, Rn, Rf, Tc, Ts, Tx};

/// Rows per request slot of the flat sliding caches (see the module doc): `axes::Rf / (Rn + 1)`.
pub fn ring() -> usize {
    Rf::SIZE / (Rn::SIZE + 1)
}

fn one() -> bf16 {
    bf16::from_f32(1.0)
}
fn zero() -> bf16 {
    bf16::from_f32(0.0)
}
/// Penalty of a key that may be visible / of a key that is never visible (`MASK_PEN`): see `lib.rs`.
fn visible() -> bf16 {
    bf16::from_f32(0.0)
}
fn hidden() -> bf16 {
    bf16::from_f32(crate::MASK_PEN)
}

/// Bytes of one cache row (`Ns * Ds * 2`).
pub fn row_bytes() -> usize {
    Ns::SIZE * Ds::SIZE * 2
}
/// Byte offset of the cache row holding position `pos` of the request in `slot`.
pub fn row_offset(slot: usize, pos: usize) -> i32 {
    ((slot * ring() + pos % ring()) * row_bytes()) as i32
}
/// Cache row of window slot `i` of a window that starts at position `start` (may be negative): the ring row `(start + i) mod RING` of the slot.
/// Entries that do not hold a position of the request (before 0 / after the newest key) still name a DISTINCT row of the slot (stale or never written, masked):
/// a gather whose padding entries all repeat one row is several times slower on hardware (hot spot), so padding never uses a single row.
pub fn window_row(slot: usize, start: i64, i: usize) -> usize {
    slot * ring() + (start + i as i64).rem_euclid(ring() as i64) as usize
}
/// Row of a padding row / padding unit's window slot `i` (scratch slot rows, distinct).
fn scratch_row(i: usize) -> usize {
    Rn::SIZE * ring() + i
}
/// Row padding rows of an append write to (scratch slot row 0).
fn scratch_write_offset() -> i32 {
    ((Rn::SIZE * ring()) * row_bytes()) as i32
}

// ------------------------------------------------------------------------------------------------------------------
// Decode rows
// ------------------------------------------------------------------------------------------------------------------

/// Position held by window slot `i` (0..1024) of a decode row whose NEW token is at position `p` (`None` = not a position: before 0).
pub fn decode_window_pos(p: usize, i: usize) -> Option<usize> {
    let start = p.saturating_sub(Tx::SIZE - 1);
    let pos = start + i;
    if pos <= p { Some(pos) } else { None }
}

/// Gather index `m![Bd, Tx]` (bd = 8 or 16): window slot `i` of row b = ring row of position `max(0, p - 1023) + i` (slots beyond `p` name the stale rows
/// that follow, masked). `slots` / `positions` list the real rows; padding rows (up to `bd`) read distinct scratch rows.
pub fn decode_gather_index(slots: &[usize], positions: &[usize], bd: usize) -> Vec<i32> {
    assert!(slots.len() == positions.len() && slots.len() <= bd);
    let mut v = Vec::with_capacity(bd * Tx::SIZE);
    for b in 0..bd {
        for i in 0..Tx::SIZE {
            let r = if b < slots.len() { window_row(slots[b], positions[b].saturating_sub(Tx::SIZE - 1) as i64, i) } else { scratch_row(i) };
            v.push((r * row_bytes()) as i32);
        }
    }
    v
}

/// ADDITIVE penalty `m![Bd, Tx % 32, Tx / 32]` (bf16: 0 = visible, `MASK_PEN` = masked; flat `(b * 32 + lane) * 32 + slot`, window slot `i = slot * 32 + lane`): visible iff `i` holds a position `<= p`
/// (the new key is appended in the same launch, so position `p` itself is visible). Padding rows see only slot 0.
pub fn decode_mask(positions: &[usize], bd: usize) -> Vec<bf16> {
    assert!(positions.len() <= bd);
    let mut m = vec![hidden(); bd * 32 * 32];
    for b in 0..bd {
        for lane in 0..32 {
            for slot in 0..32 {
                let i = slot * 32 + lane;
                let vis = if b < positions.len() { decode_window_pos(positions[b], i).is_some() } else { i == 0 };
                if vis {
                    m[(b * 32 + lane) * 32 + slot] = visible();
                }
            }
        }
    }
    m
}

/// Append index `m![Bd]`: row of position `positions[b]` of row b; padding rows write scratch row 0.
pub fn decode_append_index(slots: &[usize], positions: &[usize], bd: usize) -> Vec<i32> {
    assert_eq!(slots.len(), positions.len());
    (0..bd).map(|b| if b < slots.len() { row_offset(slots[b], positions[b]) } else { scratch_write_offset() }).collect()
}

// ------------------------------------------------------------------------------------------------------------------
// Prefill units
// ------------------------------------------------------------------------------------------------------------------

/// `n` new rows of the request in sliding slot `slot`, at positions `p0 .. p0 + n` (`1 <= n <= 256`; p0 need not be aligned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub slot: usize,
    pub p0: usize,
    pub n: usize,
}

impl Run {
    /// Number of 64-row units (segments) the run occupies.
    pub fn units(&self) -> usize {
        self.n.div_ceil(Tc::SIZE)
    }
}

/// Position held by ring-part slot `i` (0..1024) of the units of a run starting at `p0` (`None`: before position 0).
pub fn unit_ring_pos(p0: usize, i: usize) -> Option<usize> {
    (p0 + i).checked_sub(Ts::SIZE)
}

/// Host-built tables of the prefill part of one launch (all per-wave vectors have `s / 2` entries; wave `w` = units `2w`, `2w + 1`).
pub struct PrefillTables {
    /// Segments of the bucket (2 or 4).
    pub s: usize,
    /// `(run index, segment index inside the run)` of each unit, `None` = padding unit.
    pub unit: Vec<Option<(usize, usize)>>,
    /// First unit (= first 64-row stage block) of each run.
    pub first_unit: Vec<usize>,
    /// Per wave `[Pw, Tx]` i32: gather index of the ring part (slot i = position `p0 - 1024 + i`).
    pub ring_index: Vec<Vec<i32>>,
    /// Per wave `[Pw, Cx]` i32: byte offset of the stage row of chunk key c (padding entries `c >= n` and padding units name DISTINCT stage rows, masked and penalised: see `stage_row`).
    pub stage_index: Vec<Vec<i32>>,
    /// Per wave `[Pw, Ts % 32, Tc, Seg, Ts / 32]` bf16 visibility masks.
    pub mask: Vec<Vec<bf16>>,
    /// Per wave `[Pw, Ts % 32, Seg, Ts / 32]` bf16 additive per-key penalties (`unit_pen`).
    pub pen: Vec<Vec<bf16>>,
    /// `[s * 64]` i32: cache row byte offset of each staged row (padding rows: scratch row 0).
    pub append_index: Vec<i32>,
}

/// Mask `[lane = 32, t = 64, seg = 2, slot = 32]` (flat `((lane * 64 + t) * 2 + seg) * 32 + slot`) of unit (run, seg `j`): segment 0 = ring slot
/// `i = slot * 32 + lane` (position `p0 - 1024 + i`; visible iff that position exists and lies in the window of the query row at `p0 + 64 j + t`,
/// i.e. `i >= 64 j + t + 1`), segment 1 = chunk key `c = slot * 32 + lane` (slot < 8; visible iff `c <= 64 j + t` and `c < n`).
pub fn unit_mask(run: &Run, j: usize) -> Vec<bf16> {
    let tc = Tc::SIZE;
    let mut mask = vec![zero(); 32 * tc * 2 * 32];
    for lane in 0..32 {
        for t in 0..tc {
            for slot in 0..32 {
                let i = slot * 32 + lane;
                if unit_ring_pos(run.p0, i).is_some() && i >= 64 * j + t + 1 {
                    mask[((lane * tc + t) * 2) * 32 + slot] = one();
                }
                if slot < Cx::SIZE / 32 {
                    let c = slot * 32 + lane;
                    if c <= 64 * j + t && c < run.n {
                        mask[((lane * tc + t) * 2 + 1) * 32 + slot] = one();
                    }
                }
            }
        }
    }
    mask
}

/// Per-key additive score penalty `[lane = 32, seg = 2, slot = 32]` (flat `(lane * 2 + seg) * 32 + slot`) of the units of a run: `MASK_PEN` for the slots no row of the unit can see:
/// ring slots before position 0 (they name stale rows of the slot), staged keys `c >= n` (they name other stage rows, see `stage_row`) and the zero-filled slots `>= Cx / 32`.
/// Added to the scores before the row max, so the max never depends on stale ring rows (see `lib.rs::MASK_PEN`).
pub fn unit_pen(run: &Run) -> Vec<bf16> {
    let mut pen = vec![zero(); 32 * 2 * 32];
    for lane in 0..32 {
        for slot in 0..32 {
            let i = slot * 32 + lane;
            if unit_ring_pos(run.p0, i).is_none() {
                pen[(lane * 2) * 32 + slot] = hidden();
            }
            if slot >= Cx::SIZE / 32 || i >= run.n {
                pen[(lane * 2 + 1) * 32 + slot] = hidden();
            }
        }
    }
    pen
}

/// Gather index `[Tx]` of the ring part of a run's units.
pub fn unit_ring_index(run: &Run) -> Vec<i32> {
    (0..Tx::SIZE).map(|i| (window_row(run.slot, run.p0 as i64 - Ts::SIZE as i64, i) * row_bytes()) as i32).collect()
}

/// Stage row named by entry `k` of the stage index of a unit whose run starts at stage row `first` (`s * 64` stage rows in all). Entries `k < n` are the run's keys
/// (`first + k`); the padding entries `k >= n` (and the entries of a padding unit) are masked AND penalised (`unit_pen`: they cannot reach the row max, and `p = 0` kills their
/// values), so they may name ANY finite stage row: they name consecutive rows (mod the stage size), never one row repeated (a gather whose index entries repeat one row
/// is several times slower on hardware: hot spot, same rule as `window_row`).
fn stage_row(s: usize, first: usize, k: usize) -> usize {
    (first + k) % (s * Tc::SIZE)
}

/// Builds the prefill tables of the S-segment bucket for `runs` (in unit order). Panics when the runs do not fit.
pub fn prefill_tables(s: usize, runs: &[Run]) -> PrefillTables {
    assert!(s == 2 || s == 4);
    let rb = row_bytes();
    let tc = Tc::SIZE;
    let mut first_unit = vec![];
    let mut unit: Vec<Option<(usize, usize)>> = vec![None; s];
    let mut u0 = 0;
    for (ri, r) in runs.iter().enumerate() {
        assert!(r.n >= 1 && r.n <= Cx::SIZE && r.slot < Rn::SIZE, "bad run {r:?}");
        assert!(runs.iter().enumerate().all(|(rj, o)| rj == ri || o.slot != r.slot), "two runs of one slot in a launch");
        first_unit.push(u0);
        for j in 0..r.units() {
            assert!(u0 + j < s, "runs need more than {s} units");
            unit[u0 + j] = Some((ri, j));
        }
        u0 += r.units();
    }
    let waves = s / 2;
    let (mut ring_index, mut stage_index, mut mask, mut pen) = (vec![], vec![], vec![], vec![]);
    for w in 0..waves {
        let (mut ri_w, mut si_w, mut mk_w, mut pn_w) = (vec![], vec![], vec![], vec![]);
        for c in 0..2 {
            let u = 2 * w + c;
            match unit[u] {
                Some((ri, j)) => {
                    let run = &runs[ri];
                    ri_w.extend(unit_ring_index(run));
                    for k in 0..Cx::SIZE {
                        si_w.push((stage_row(s, 64 * first_unit[ri], k) * rb) as i32);
                    }
                    mk_w.extend(unit_mask(run, j));
                    pn_w.extend(unit_pen(run));
                }
                None => {
                    ri_w.extend((0..Tx::SIZE).map(|i| (scratch_row(i) * row_bytes()) as i32));
                    si_w.extend((0..Cx::SIZE).map(|k| (stage_row(s, 64 * u, k) * rb) as i32));
                    mk_w.extend(vec![zero(); 32 * tc * 2 * 32]);
                    pn_w.extend(vec![hidden(); 32 * 2 * 32]);
                }
            }
        }
        ring_index.push(ri_w);
        stage_index.push(si_w);
        mask.push(mk_w);
        pen.push(pn_w);
    }
    let mut append_index = vec![scratch_write_offset(); s * tc];
    for (ri, run) in runs.iter().enumerate() {
        for c in 0..run.n {
            append_index[64 * first_unit[ri] + c] = row_offset(run.slot, run.p0 + c);
        }
    }
    PrefillTables { s, unit, first_unit, ring_index, stage_index, mask, pen, append_index }
}

// ------------------------------------------------------------------------------------------------------------------
// Full layers: prefill units (64-row aligned chunks of any request; consecutive segments of one request are just consecutive units: the pool already holds the
// rows of the segments before them because the staged rows of the whole launch are scattered first) + the existing decode helpers `dattn_host::full_*`.
// ------------------------------------------------------------------------------------------------------------------
use crate::axes::{Df, Pn};
use crate::dattn_host::{full_chunk_block_index, full_chunk_mask_flat, full_chunk_pen_flat};

/// One 64-row prefill unit of a full layer: the request's physical page table (at least `(p0 + n - 1) / 512 + 1` entries), the unit's first position
/// (`p0 % 8 == 0`: the staged rows are scattered as whole 8-row blocks) and its number of real rows `n` (`1..=64`; the rows after them are padding: the batch's decode rows or clones).
#[derive(Clone, Debug)]
pub struct FullUnit {
    pub table: Vec<usize>,
    pub p0: usize,
    pub n: usize,
}

/// Host-built tables of the prefill launch of a full layer (bucket of `s` units, `npages` = 2 or 4 key pages per unit).
pub struct FullPrefillTables {
    pub s: usize,
    pub npages: usize,
    /// Per wave `[Pw, Hg, Seg(4), Bk]` i32 block gather indices (the unit's block list replicated over the 8 head groups).
    pub index: Vec<Vec<i32>>,
    /// Per wave `[Pw, Tf % 32, Tc, Kf(64)]` bf16 masks.
    pub mask: Vec<Vec<bf16>>,
    /// Per wave `[Pw, Tf % 32, Kf(64)]` bf16 additive per-key penalties (`dattn_host::full_chunk_pen_flat`: positions `>= p0 + n` of each unit).
    pub pen: Vec<Vec<bf16>>,
    /// `[s * 8]` i32: byte offset of the pool block each staged 8-row block is scattered to (blocks without a real row, and padding units: distinct trash-page blocks).
    pub append_index: Vec<i32>,
}

/// Full-layer pools hold `Pn + 2` pages (`axes::Pr = (Pn + 2) * 512`): page `Pn` is the ZERO page that unallocated logical pages of every row gather (never written, so a real
/// row's softmax maximum never sees junk), page `Pn + 1` is the TRASH page that padding rows / units append to. (With one scratch page, the junk k row appended by a padding row
/// sat inside the keys every real row with unallocated pages gathers: a cross-request leak through the max, fatal when its score exceeds the real maximum by tens.)
pub fn full_trash_page() -> usize {
    Pn::SIZE + 1
}

/// Append index `m![Bd]` of the fused full decode launch: byte offset of the pool row of position `positions[b]`; padding rows write trash row 0.
pub fn full_decode_append_index(page_tables: &[Vec<usize>], positions: &[usize], bd: usize) -> Vec<i32> {
    assert_eq!(page_tables.len(), positions.len());
    let rb = Df::SIZE * 2;
    (0..bd)
        .map(|b| {
            if b < positions.len() {
                ((page_tables[b][positions[b] / 512] * 512 + positions[b] % 512) * rb) as i32
            } else {
                (full_trash_page() * 512 * rb) as i32
            }
        })
        .collect()
}

/// Builds the tables of the full-layer prefill launch for `units` (in order; fewer than `s` units leaves padding units).
pub fn full_prefill_tables(s: usize, npages: usize, units: &[FullUnit]) -> FullPrefillTables {
    assert!(s == 2 || s == 4);
    assert!(npages == 2 || npages == 4);
    assert!(units.len() <= s);
    let tc = Tc::SIZE;
    let blk_bytes = 8 * Df::SIZE * 2;
    let kf = 16 * npages;
    let (mut index, mut mask, mut pen) = (vec![], vec![], vec![]);
    for w in 0..s / 2 {
        let (mut ix, mut mk, mut pn) = (vec![], vec![], vec![]);
        for c in 0..2 {
            match units.get(2 * w + c) {
                Some(u) => {
                    assert!(
                        u.p0 % 8 == 0 && u.n >= 1 && u.n <= tc && (u.p0 + u.n - 1) / 512 < npages && u.table.len() > (u.p0 + u.n - 1) / 512,
                        "bad full unit {u:?}"
                    );
                    ix.extend(full_chunk_block_index(&u.table, npages));
                    mk.extend(full_chunk_mask_flat(npages, u.p0));
                    pn.extend(full_chunk_pen_flat(npages, u.p0, u.n));
                }
                None => {
                    ix.extend(full_chunk_block_index(&[], npages));
                    mk.extend(vec![zero(); 32 * tc * kf]);
                    pn.extend(vec![hidden(); 32 * kf]);
                }
            }
        }
        index.push(ix);
        mask.push(mk);
        pen.push(pn);
    }
    let mut append_index = Vec::with_capacity(s * 8);
    for u in 0..s {
        for b in 0..8 {
            let block = match units.get(u) {
                Some(un) if 8 * b < un.n => {
                    let pos = un.p0 + 8 * b;
                    un.table[pos / 512] * 64 + (pos % 512) / 8
                }
                _ => full_trash_page() * 64 + 8 * u + b,
            };
            append_index.push((block * blk_bytes) as i32);
        }
    }
    FullPrefillTables { s, npages, index, mask, pen, append_index }
}

// ------------------------------------------------------------------------------------------------------------------
// KVD (QKV scatters K / V into the ring; the N3 launch gathers ring + chunk keys back in ONE window gather of `Kw = 1024 + 128` rows)
// ------------------------------------------------------------------------------------------------------------------

/// Gather index `[Kw]` of the key window of a run's units: window row `i` = ring row of position `p0 - 1024 + i` (`i < 1024` = the ring part, `i >= 1024` = the chunk keys c = i - 1024:
/// positions `p0 .. p0 + 127`, written by the QKV launch of the same pass; entries `c >= n` name the following ring rows, stale but finite and hidden by the mask / penalty tables).
pub fn unit_win_index(run: &Run) -> Vec<i32> {
    (0..crate::axes::Kw::SIZE).map(|i| (window_row(run.slot, run.p0 as i64 - Ts::SIZE as i64, i) * row_bytes()) as i32).collect()
}

/// KVD tables of an 8-decode-row hybrid pass (`bd == 8`): the window index `[Pw, Kw]` of the two units and the cache row byte offsets of the 128 pass rows split in the two 64-row QKV
/// blocks (prefill rows: position `p0 + c` of their slot; the 8 decode rows = the last 8 rows: position of the rider; every other row: the scratch row, like the stage appends).
pub fn kvd_tables(runs: &[Run], dslots: &[usize], dpos: &[usize], bd: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    assert_eq!(bd, 8, "KVD tables exist for the 8-decode-row bucket only");
    assert!(dslots.len() == dpos.len() && dslots.len() <= bd);
    let pt = prefill_tables(2, runs);
    let mut win = Vec::with_capacity(2 * crate::axes::Kw::SIZE);
    for u in 0..2 {
        match pt.unit[u] {
            Some((ri, _)) => win.extend(unit_win_index(&runs[ri])),
            None => win.extend((0..crate::axes::Kw::SIZE).map(|i| (scratch_row(i) * row_bytes()) as i32)),
        }
    }
    let mut idx = pt.append_index.clone();
    let tc = Tc::SIZE;
    for b in 0..bd {
        let r = 2 * tc - bd + b;
        assert_eq!(idx[r], scratch_write_offset(), "a decode row of the pass overlaps a prefill row");
        if b < dslots.len() {
            idx[r] = row_offset(dslots[b], dpos[b]);
        }
    }
    (win, idx[..tc].to_vec(), idx[tc..].to_vec())
}

/// KVD-256 tables of a 256-row hybrid pass with `bd` (8 or 16) decode rows: the window gather indices `[Pw, Kw5]` of the two waves (unit u of the pass = window `u % 2` of wave `u / 2`: window row
/// `i` = ring row of position `p0 - 1024 + i`, `i >= 1024` = the run's chunk keys c = i - 1024 written by the QKV launch of the same pass; entries past the run name the following ring rows, stale but
/// finite and hidden by the mask / penalty tables, which are the stage path's) and the cache row byte offsets of the four 64-row QKV blocks (prefill rows: position `p0 + c` of their slot; the last `bd`
/// rows: the position of the rider; every other row: the scratch row, like the stage appends).
pub fn kvd256_tables(runs: &[Run], dslots: &[usize], dpos: &[usize], bd: usize) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    assert!(bd == 8 || bd == 16, "KVD-256 tables exist for the 8 / 16 decode-row buckets");
    assert!(dslots.len() == dpos.len() && dslots.len() <= bd);
    let pt = prefill_tables(4, runs);
    let kw = crate::axes::Kw5::SIZE;
    let mut win = vec![Vec::with_capacity(2 * kw); 2];
    for u in 0..4 {
        match pt.unit[u] {
            Some((ri, _)) => win[u / 2].extend((0..kw).map(|i| (window_row(runs[ri].slot, runs[ri].p0 as i64 - Ts::SIZE as i64, i) * row_bytes()) as i32)),
            None => win[u / 2].extend((0..kw).map(|i| (scratch_row(i) * row_bytes()) as i32)),
        }
    }
    let mut idx = pt.append_index.clone();
    let tc = Tc::SIZE;
    assert_eq!(idx.len(), 4 * tc);
    for b in 0..bd {
        let r = 4 * tc - bd + b;
        assert_eq!(idx[r], scratch_write_offset(), "a decode row of the pass overlaps a prefill row");
        if b < dslots.len() {
            idx[r] = row_offset(dslots[b], dpos[b]);
        }
    }
    (win, (0..4).map(|u| idx[u * tc..(u + 1) * tc].to_vec()).collect())
}
