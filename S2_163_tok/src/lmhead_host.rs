//! Host side of the device argmax (see `RESULTS.md`): the constant tables the kernels read, and the decoding of the per-slice keys they write.
//!
//! Dependencies are only `axes` and the std prelude, so the lean `~/ldev` kernel-development tree can build it.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Fl1, W};

/// Number of slices (cluster x 256); each leaves one key per row.
pub const SLICES: usize = 512;
/// One-row kernels write each key in a 2-word (8 B) cell (DM rows must be a multiple of 8 B): `[GROUPS, 1 # 2]`; the 8-row kernels write `[GROUPS, ROWS]`.
pub const KEY_STRIDE: usize = 8;

/// Position of vocabulary id `w` in a slice-contiguous table `[c, v, a, r]` (`c = w / 8192 % 2`, `v = w / 32 % 256`, `a = w / 16384`, `r = w % 32`):
/// every slice then reads one contiguous 2 KB run.
pub fn slice_layout_index(w: usize) -> usize {
    let (a, c, v, r) = (w / 16384, w / 8192 % 2, w / 32 % 256, w % 32);
    ((c * 256 + v) * 16 + a) * 32 + r
}

fn table_from(f: impl Fn(usize) -> i32) -> Vec<i32> {
    let mut table = vec![0i32; W::SIZE];
    for w in 0..W::SIZE {
        table[slice_layout_index(w)] = f(w);
    }
    table
}

/// `P[w] = 2 * ((15 - a) * 32 + (31 - r)) + 1` (`a = w / 16384`, `r = w % 32`): odd, larger for lower `w` inside a slice.
pub fn payload_table() -> Vec<i32> {
    table_from(|w| {
        let (a, r) = (w / 16384, w % 32);
        2 * ((15 - a) * 32 + (31 - r)) as i32 + 1
    })
}

/// `S[w] = 31` at a banned id (the key then becomes `i32::MIN`), else 0.
pub fn shift_table(banned: &[usize]) -> Vec<i32> {
    table_from(|w| if banned.contains(&w) { 31 } else { 0 })
}

/// Rows per batched kernel launch (the TRF lanes).
pub const ROWS: usize = 8;

/// `rows8` logits tile layout `[c, v, a, r, row]` (`r = w % 32`, `v = w / 32 % 256`) of a `[ROWS][W]` logits matrix, as read by `argmax_logits_rows8`
/// (the layout the batched LM-head contraction leaves in the slices).
pub fn to_rows8_layout(logits: &[Vec<bf16>]) -> Vec<bf16> {
    assert_eq!(logits.len(), ROWS);
    let mut out = vec![bf16::from_bits(0); ROWS * W::SIZE];
    for (row, values) in logits.iter().enumerate() {
        for (w, &value) in values.iter().enumerate() {
            let (a, c, v, r) = (w / 16384, w / 8192 % 2, w / 32 % 256, w % 32);
            out[(((c * 256 + v) * 16 + a) * 32 + r) * ROWS + row] = value;
        }
    }
    out
}

/// Order-preserving i32 image of a bf16 logit, as the kernel computes it (low 16 bits zero).
pub fn order_key(logit: bf16) -> i32 {
    let bits = logit.to_f32().to_bits() as i32;
    bits ^ ((bits >> 31) & 0x7FFF_0000)
}

/// Picks the greedy token of one row from its `SLICES` keys (`keys[g]`, `g = c * 256 + v`): the largest logit, ties to the lowest id.
/// Returns `None` if every id is banned (all keys `i32::MIN`).
pub fn decode_row(keys: impl Iterator<Item = i32>) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    let mut seen = 0;
    for (g, key) in keys.enumerate() {
        seen += 1;
        if key == i32::MIN {
            continue;
        }
        let (c, v) = (g / 256, g % 256);
        let value = key & !0xFFFF;
        let p = ((key & 0xFFFF) >> 1) as usize;
        let (a, r) = (15 - p / 32, 31 - p % 32);
        let w = a * 16384 + c * 8192 + v * 32 + r;
        if best.is_none_or(|(bv, bw)| value > bv || (value == bv && w < bw)) {
            best = Some((value, w));
        }
    }
    assert_eq!(seen, SLICES);
    best.map(|(_, w)| w)
}

/// Row `row` of `rows` from the output as read back from the device: `[SLICES, 1 # 2]` for one row, `[SLICES, rows]` for the batched kernels.
pub fn decode_keys(keys: &[i32], rows: usize, row: usize) -> Option<usize> {
    let stride = if rows == 1 { KEY_STRIDE } else { 1 };
    assert_eq!(keys.len(), SLICES * rows * stride);
    decode_row((0..SLICES).map(|g| keys[(g * rows + row) * stride]))
}

/// Row `row` of ban variant `variant` from the two-variant output of the ban-variant kernels, as read back from the device: `[2, SLICES, 1 # 2]` for one row, `[2, SLICES, rows]` for the batched
/// kernels (one contiguous block per variant).
pub fn decode_variant_keys(keys: &[i32], variant: usize, rows: usize, row: usize) -> Option<usize> {
    let stride = if rows == 1 { KEY_STRIDE } else { 1 };
    let block = SLICES * rows * stride;
    assert_eq!(keys.len(), 2 * block);
    decode_keys(&keys[variant * block..(variant + 1) * block], rows, row)
}

/// The host semantics the kernels reproduce (`sampling::argmax`): max by `f32::total_cmp`, ties to the lowest id, banned ids excluded,
/// the unbanned argmax if everything is banned.
pub fn reference_argmax(logits: &[bf16], banned: &[usize]) -> usize {
    let logits: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
    let best = |allowed: &dyn Fn(usize) -> bool| {
        (0..logits.len())
            .filter(|&i| allowed(i))
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
    };
    best(&|i| !banned.contains(&i)).or_else(|| best(&|_| true)).unwrap()
}

/// Uploads a slice-layout i32 table (see `payload_table`, `shift_table`).
pub async fn upload_table(
    device: &mut Device,
    table: Vec<i32>,
) -> Result<HbmTensor<i32, Chip, m![W / 8192 % 2, W / 32 % 256, Fl1]>, Error> {
    HostTensor::<i32, m![W / 8192 % 2, W / 32 % 256, Fl1]>::from_vec(table)
        .to_hbm(&mut device.pdma)
        .await
}
