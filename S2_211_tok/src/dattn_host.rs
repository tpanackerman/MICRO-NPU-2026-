//! Host helpers for the batched decode attention kernels in `ops_dattn` (index tensors and masks).
//!
//! One launch handles `Bd` (= 8 or 16) decode rows, row `b` belonging to the request that owns slot `slots[b]` of the unified
//! sliding cache `[Rf, Ns, Ds]` (Rf = (Rn + 1) * Ts rows; request slot `r < Rn` owns rows `r * 1024 .. (r + 1) * 1024`, position `p`
//! lives in row `r * 1024 + p % 1024`; slot `Rn` is a scratch slot used by padding rows). The index and mask tensors are built once per decode step and shared by all 40
//! sliding layers.
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Ns, Rn, Ts, Tx};

fn one() -> bf16 {
    bf16::from_f32(1.0)
}
fn zero() -> bf16 {
    bf16::from_f32(0.0)
}
/// Penalty of a key that may be visible.
fn visible() -> bf16 {
    bf16::from_f32(0.0)
}
/// Penalty of a key that is never visible (stale / unwritten row): `MASK_PEN`.
fn hidden() -> bf16 {
    bf16::from_f32(crate::MASK_PEN)
}

/// Bytes of one cache row of the sliding caches (`Ns * Ds * 2`).
pub fn sliding_row_bytes() -> usize {
    Ns::SIZE * Ds::SIZE * 2
}

/// Gather index `m![Bd, Tx]` (i32; `bd` = 8 or 16, the bucket of the launch): byte offset of the ring row `t` of the request in slot `slots[b]`, i.e.
/// `(slots[b] * Ts + t) * Ns * Ds * 2`. Rows beyond `slots.len()` (padding rows) read the scratch slot `Rn`.
pub fn sliding_gather_index(slots: &[usize], bd: usize) -> Vec<i32> {
    assert!(slots.len() <= bd);
    let rb = sliding_row_bytes();
    let mut v = Vec::with_capacity(bd * Tx::SIZE);
    for b in 0..bd {
        let slot = if b < slots.len() { slots[b] } else { Rn::SIZE };
        assert!(slot <= Rn::SIZE);
        for t in 0..Tx::SIZE {
            v.push(((slot * Ts::SIZE + t) * rb) as i32);
        }
    }
    v
}

/// Additive visibility penalty `m![Bd, Tx % 32, Tx / 32]` (bf16: 0 = visible, `MASK_PEN` = masked, added to the scores before the max), flat `(b * 32 + lane) * 32 + slot`, ring row `t = slot * 32 + lane`.
/// The kernel reads the ring AFTER the new key of position `p` was appended at row `p % 1024`, so row `t` is visible iff
/// `p >= 1023` (the ring holds exactly the window) or `t <= p`. Padding rows (`positions.len() < Bd`) see only row 0.
pub fn sliding_mask(positions: &[usize], bd: usize) -> Vec<bf16> {
    assert!(positions.len() <= bd);
    let ts = Ts::SIZE;
    let mut m = vec![hidden(); bd * 32 * 32];
    for b in 0..bd {
        let p = if b < positions.len() { positions[b] } else { 0 };
        for lane in 0..32 {
            for slot in 0..32 {
                let t = slot * 32 + lane;
                if p + 1 >= ts || t <= p {
                    m[(b * 32 + lane) * 32 + slot] = visible();
                }
            }
        }
    }
    m
}

/// Append index `m![Bd]` (i32) for `sliding_decode_append`: byte offset of the row `slots[b] * 1024 + positions[b] % 1024`.
/// Padding rows (`slots.len() < bd`) write row 0 of the scratch slot `Rn`.
pub fn sliding_append_index(slots: &[usize], positions: &[usize], bd: usize) -> Vec<i32> {
    assert_eq!(slots.len(), positions.len());
    let rb = sliding_row_bytes();
    let mut v = Vec::with_capacity(bd);
    for b in 0..bd {
        if b < slots.len() {
            v.push(((slots[b] * Ts::SIZE + positions[b] % Ts::SIZE) * rb) as i32);
        } else {
            v.push((Rn::SIZE * Ts::SIZE * rb) as i32);
        }
    }
    v
}

// ------------------------------------------------------------------------------------------------------------------
// Full-attention layers. The page pool of a layer is ONE flat tensor `[Pr, Df]` (Pr = (Pn + 1) * 512 rows: Pn physical pages
// plus a scratch page `Pn` used by padding rows and by the not yet allocated logical pages of a request). A request owns
// the physical pages `page_table[0..]` (logical page `i` holds positions `i * 512 .. (i + 1) * 512`).
// ------------------------------------------------------------------------------------------------------------------
use crate::axes::{Df, Pn, Tf};

/// Bytes of one pool row (`Df * 2`).
pub fn full_row_bytes() -> usize {
    Df::SIZE * 2
}

fn full_page_of(page_table: &[usize], logical: usize) -> usize {
    if logical < page_table.len() { page_table[logical] } else { Pn::SIZE }
}

/// Gather index `m![Bd, Kx]` (i32) for a bucket of `kx` key rows (`kx = 512 * pages`): byte offset of the pool row holding
/// position `j` of row `b`, i.e. `(physical_page(j / 512) * 512 + j % 512) * Df * 2`. Logical pages the request has not
/// allocated (and all pages of padding rows) read the scratch page `Pn`.
pub fn full_gather_index(page_tables: &[Vec<usize>], bd: usize, kx: usize) -> Vec<i32> {
    assert!(page_tables.len() <= bd && kx % Tf::SIZE == 0);
    let rb = full_row_bytes();
    let mut v = Vec::with_capacity(bd * kx);
    for b in 0..bd {
        let empty: Vec<usize> = vec![];
        let pt = if b < page_tables.len() { &page_tables[b] } else { &empty };
        for j in 0..kx {
            v.push(((full_page_of(pt, j / Tf::SIZE) * Tf::SIZE + j % Tf::SIZE) * rb) as i32);
        }
    }
    v
}

/// Additive visibility penalty `m![Bd, Kx % L, Kx / L]` (bf16: 0 = visible, `MASK_PEN` = masked; L = `lanes` = 64 for the 8-row kernels, 32 for the 16-row kernels), flat `(b * L + lane) * (kx / L) + slot`, key `j = slot * L + lane`:
/// visible iff `j <= position` (the new key of the step is appended before the attention). Padding rows see only key 0.
pub fn full_mask(positions: &[usize], bd: usize, kx: usize, lanes: usize) -> Vec<bf16> {
    assert!(positions.len() <= bd && kx % lanes == 0);
    let sf = kx / lanes;
    let mut m = vec![hidden(); bd * lanes * sf];
    for b in 0..bd {
        let p = if b < positions.len() { positions[b] } else { 0 };
        for lane in 0..lanes {
            for slot in 0..sf {
                if slot * lanes + lane <= p {
                    m[(b * lanes + lane) * sf + slot] = visible();
                }
            }
        }
    }
    m
}

/// Append index `m![Bd]` (i32) for `full_decode_append`: byte offset of the pool row of position `positions[b]` of row `b`.
/// Padding rows write row 0 of the scratch page.
pub fn full_append_index(page_tables: &[Vec<usize>], positions: &[usize], bd: usize) -> Vec<i32> {
    assert_eq!(page_tables.len(), positions.len());
    let rb = full_row_bytes();
    (0..bd)
        .map(|b| {
            if b < positions.len() {
                (full_page_of(&page_tables[b], positions[b] / Tf::SIZE) * Tf::SIZE + positions[b] % Tf::SIZE) as i32 * rb as i32
            } else {
                (Pn::SIZE * Tf::SIZE * rb) as i32
            }
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------------------------
// Chunked prefill on the flat caches (kernels `*_chunk_*_flat`). A chunk is Tc = 64 consecutive positions p0 .. p0 + 63
// (p0 a multiple of 64), masks come from `chunk_host::{sliding_mask, full_mask}` (unchanged).
// ------------------------------------------------------------------------------------------------------------------
use crate::axes::Tc;

/// Append index `m![Ts = 64]` for `sliding_chunk_append_flat`: byte offset of cache row `slot * 1024 + (p0 + c) % 1024`.
pub fn sliding_chunk_append_index(slot: usize, p0: usize) -> Vec<i32> {
    let rb = sliding_row_bytes();
    (0..Tc::SIZE).map(|c| ((slot * Ts::SIZE + (p0 + c) % Ts::SIZE) * rb) as i32).collect()
}

/// Ring index `m![Tx]` for `sliding_chunk_attention_flat`: byte offset of ring row `t` of the request in `slot`.
pub fn sliding_chunk_ring_index(slot: usize) -> Vec<i32> {
    sliding_gather_index(&[slot], 1)
}

/// Append index `m![Tc]` for `full_chunk_append_flat`: byte offset of pool row `page_table[p0 / 512] * 512 + p0 % 512 + c`.
pub fn full_chunk_append_index(page_table: &[usize], p0: usize) -> Vec<i32> {
    let rb = full_row_bytes();
    let page = full_page_of(page_table, p0 / Tf::SIZE);
    (0..Tc::SIZE).map(|c| ((page * Tf::SIZE + p0 % Tf::SIZE + c) * rb) as i32).collect()
}

/// Block index `m![Hg, Seg, Bk]` (i32, Hg = 8 identical copies) for `full_chunk_attention_{2,4}_flat` over `npages` logical pages of a request:
/// byte offset of the 8-row block `bk` (rows `bk * 8 .. bk * 8 + 8`) of logical page `seg`, i.e. `(physical_page(seg) * 64 + bk) * 8 * Df * 2`.
/// Logical pages the request has not allocated read the scratch page `Pn`.
pub fn full_chunk_block_index(page_table: &[usize], npages: usize) -> Vec<i32> {
    let blk_bytes = 8 * Df::SIZE * 2;
    let mut v = Vec::with_capacity(8 * npages * 64);
    for _hg in 0..8 {
        for seg in 0..npages {
            let page = full_page_of(page_table, seg);
            for bk in 0..64 {
                v.push(((page * 64 + bk) * blk_bytes) as i32);
            }
        }
    }
    v
}

/// Mask `m![Tf % 32, Tc, Kf]` (`Kf64` for 4 pages) of the flat full-layer chunk kernels, flat `(lane * Tc + t) * (16 * npages) + kf`,
/// `kf = seg * 16 + a * 8 + r`: the key in lane `lane`, slot `(a, r)` of logical page `seg` has position `seg * 512 + lane * 16 + a * 8 + r`
/// and is visible to query row `t` iff `position <= p0 + t`. (This is NOT the mask of the page-argument kernels in `chunk_host`: the key
/// to lane assignment differs.)
pub fn full_chunk_mask_flat(npages: usize, p0: usize) -> Vec<bf16> {
    let tc = Tc::SIZE;
    let nk = 16 * npages;
    let mut mask = vec![zero(); 32 * tc * nk];
    for lane in 0..32 {
        for t in 0..tc {
            for kf in 0..nk {
                let (seg, a, r) = (kf / 16, (kf % 16) / 8, kf % 8);
                let pos = seg * Tf::SIZE + lane * 16 + a * 8 + r;
                if pos <= p0 + t {
                    mask[(lane * tc + t) * nk + kf] = one();
                }
            }
        }
    }
    mask
}

/// Per-key additive penalty `m![Tf % 32, Kf]` (`Kf64` for 4 pages), flat `lane * (16 * npages) + kf`, of the flat full-layer chunk kernels for a unit whose first `n` rows
/// (positions `p0 .. p0 + n`) are real: `MASK_PEN` for the key slots at positions `>= p0 + n` (same key to position map as `full_chunk_mask_flat`), 0 elsewhere. Those rows
/// hold stale keys of a previous request, zeros, or the keys of the padding rows (decode rows of the batch) that the launch scattered together with the real rows.
pub fn full_chunk_pen_flat(npages: usize, p0: usize, n: usize) -> Vec<bf16> {
    let nk = 16 * npages;
    let mut pen = vec![zero(); 32 * nk];
    for lane in 0..32 {
        for kf in 0..nk {
            let (seg, a, r) = (kf / 16, (kf % 16) / 8, kf % 8);
            let pos = seg * Tf::SIZE + lane * 16 + a * 8 + r;
            if pos >= p0 + n {
                pen[lane * nk + kf] = bf16::from_f32(crate::MASK_PEN);
            }
        }
    }
    pen
}

// ------------------------------------------------------------------------------------------------------------------
// Allocation helpers (one zeroed tensor per layer and K / V) and the two views of the page pool.
// ------------------------------------------------------------------------------------------------------------------
use crate::Chip;
use crate::axes::{Pq, Pr, R8, Rf};

/// Zeroed unified sliding K (or V) cache of one layer: `[Rf, Ns, Ds]` = (Rn + 1) request slots x 1024 rows (slot `Rn` is the scratch slot).
pub async fn alloc_sliding_cache(device: &mut Device) -> Result<HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>, Error> {
    HostTensor::<bf16, m![Rf, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await
}

/// Zeroed flat page pool K (or V) of one full layer: `[Pr, Df]` = (Pn + 1) pages x 512 rows (page `Pn` is the scratch page).
pub async fn alloc_full_pool(device: &mut Device) -> Result<HbmTensor<bf16, Chip, m![Pr, Df]>, Error> {
    HostTensor::<bf16, m![Pr, Df]>::zero().to_hbm(&mut device.pdma).await
}

/// The same pool bytes viewed as 8-row blocks `[Pq, R8, Df]` (the type `full_chunk_attention_{2,4}_flat` takes). Free (no data movement).
pub fn pool_as_blocks(pool: HbmTensor<bf16, Chip, m![Pr, Df]>) -> HbmTensor<bf16, Chip, m![Pq, R8, Df]> {
    // SAFETY: [Pr, Df] -> [Pq, R8, Df] only regroups the row axis (Pr = Pq * R8), the wire order is unchanged.
    unsafe { pool.reshape() }
}

/// Inverse of `pool_as_blocks`.
pub fn pool_as_rows(blocks: HbmTensor<bf16, Chip, m![Pq, R8, Df]>) -> HbmTensor<bf16, Chip, m![Pr, Df]> {
    // SAFETY: see `pool_as_blocks`.
    unsafe { blocks.reshape() }
}

// ------------------------------------------------------------------------------------------------------------------
// One decode step (up to 16 rows = two 8-row groups): builds the per-step index / mask tensors once and launches the fused
// append + attention kernels of every layer.
// ------------------------------------------------------------------------------------------------------------------
use crate::axes::{Bd, Gf, Gs, Kx5};
use crate::ops_dattn;

/// Per-step device tensors of up to 16 decode rows (`groups` = 1 for <= 8 rows, 2 for 9..=16 rows); rows beyond the active ones are padding.
pub struct DecodeStep {
    groups: usize,
    s_index: Vec<HbmTensor<i32, Chip, m![Bd, Tx]>>,
    s_mask: Vec<HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>>,
    s_append: Vec<HbmTensor<i32, Chip, m![Bd]>>,
    f_index: Vec<HbmTensor<i32, Chip, m![Bd, Kx5]>>,
    f_mask: Vec<HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>>,
    f_append: Vec<HbmTensor<i32, Chip, m![Bd]>>,
}

impl DecodeStep {
    /// `slots[b]` = sliding request slot of row b (< Rn, distinct), `positions[b]` = position of the new token of row b (< 2560 for the full layers),
    /// `tables[b]` = physical full-attention pages of row b (at least `positions[b] / 512 + 1` entries, all < Pn).
    pub async fn new(device: &mut Device, slots: &[usize], positions: &[usize], tables: &[Vec<usize>]) -> Result<Self, Error> {
        let n = slots.len();
        assert!(n >= 1 && n <= 16 && positions.len() == n && tables.len() == n);
        let groups = if n <= 8 { 1 } else { 2 };
        let mut st = DecodeStep {
            groups,
            s_index: vec![],
            s_mask: vec![],
            s_append: vec![],
            f_index: vec![],
            f_mask: vec![],
            f_append: vec![],
        };
        for g in 0..groups {
            let r = 8 * g..(8 * g + 8).min(n);
            let (sl, po, ta) = (&slots[r.clone()], &positions[r.clone()], &tables[r.clone()]);
            st.s_index.push(HostTensor::<i32, m![Bd, Tx]>::from_vec(sliding_gather_index(sl, 8)).to_hbm(&mut device.pdma).await?);
            st.s_mask.push(HostTensor::<bf16, m![Bd, Tx % 32, Tx / 32]>::from_vec(sliding_mask(po, 8)).to_hbm(&mut device.pdma).await?);
            st.s_append.push(HostTensor::<i32, m![Bd]>::from_vec(sliding_append_index(sl, po, 8)).to_hbm(&mut device.pdma).await?);
            st.f_index.push(HostTensor::<i32, m![Bd, Kx5]>::from_vec(full_gather_index(ta, 8, Kx5::SIZE)).to_hbm(&mut device.pdma).await?);
            st.f_mask.push(HostTensor::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>::from_vec(full_mask(po, 8, Kx5::SIZE, 64)).to_hbm(&mut device.pdma).await?);
            st.f_append.push(HostTensor::<i32, m![Bd]>::from_vec(full_append_index(ta, po, 8)).to_hbm(&mut device.pdma).await?);
        }
        Ok(st)
    }

    /// Number of 8-row groups (the length of the per-group tensor slices the layer methods take).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// One sliding layer: appends `k_new` / `v_new` into the layer's flat caches, then attends. All per-group slices have `groups()` entries.
    pub async fn sliding_layer(
        &self,
        device: &mut Device,
        q: &[HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>],
        k_new: &[HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>],
        v_new: &[HbmTensor<bf16, Chip, m![Bd, Ns, Ds]>],
        k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
        v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
        out: &mut [HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]>],
    ) -> Result<(), Error> {
        if self.groups == 1 {
            launch(
                ops_dattn::sliding_decode_attention_fused,
                (device, &q[0], &k_new[0], &v_new[0], &self.s_append[0], k_cache, v_cache, &self.s_index[0], &self.s_mask[0], &mut out[0]),
            )
            .await
        } else {
            let (oa, ob) = out.split_at_mut(1);
            launch(
                ops_dattn::sliding_decode_attention_fused_x2,
                (
                    device,
                    &q[0],
                    &q[1],
                    &k_new[0],
                    &v_new[0],
                    &k_new[1],
                    &v_new[1],
                    &self.s_append[0],
                    &self.s_append[1],
                    k_cache,
                    v_cache,
                    &self.s_index[0],
                    &self.s_index[1],
                    &self.s_mask[0],
                    &self.s_mask[1],
                    &mut oa[0],
                    &mut ob[0],
                ),
            )
            .await
        }
    }

    /// One full layer (5-page bucket): appends `k_new` / `v_new` into the layer's flat page pools, then attends.
    pub async fn full_layer(
        &self,
        device: &mut Device,
        q: &[HbmTensor<bf16, Chip, m![Bd, Gf, Df]>],
        k_new: &[HbmTensor<bf16, Chip, m![Bd, Df]>],
        v_new: &[HbmTensor<bf16, Chip, m![Bd, Df]>],
        k_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
        v_pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
        out: &mut [HbmTensor<bf16, Chip, m![Bd, Gf, Df]>],
    ) -> Result<(), Error> {
        if self.groups == 1 {
            launch(
                ops_dattn::full_decode_attention_5_fused,
                (device, &q[0], &k_new[0], &v_new[0], &self.f_append[0], k_pool, v_pool, &self.f_index[0], &self.f_mask[0], &mut out[0]),
            )
            .await
        } else {
            let (oa, ob) = out.split_at_mut(1);
            launch(
                ops_dattn::full_decode_attention_5_fused_x2,
                (
                    device,
                    &q[0],
                    &q[1],
                    &k_new[0],
                    &v_new[0],
                    &k_new[1],
                    &v_new[1],
                    &self.f_append[0],
                    &self.f_append[1],
                    k_pool,
                    v_pool,
                    &self.f_index[0],
                    &self.f_index[1],
                    &self.f_mask[0],
                    &self.f_mask[1],
                    &mut oa[0],
                    &mut ob[0],
                ),
            )
            .await
        }
    }
}
