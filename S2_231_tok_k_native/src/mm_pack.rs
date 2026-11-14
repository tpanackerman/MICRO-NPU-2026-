//! Host-side packing for the batched encoder GEMMs (`device/mm/gemm.rs`).  Pure functions over raw bf16 (u16 bits / LE bytes).
//!
//! Layouts (all zero padded):
//! * weight tile `W_i[g, c, r, e]`:  row `n = g * rows_per_group + tile * rt + r`, column `k = c * e_len + e` of the row-major `W[n, k]`;
//! * activation `x[c, t, e]` (chunk-major): `x[c][t][e] = row_t[c * e_len + e]` (zero beyond `k_len`, zero rows beyond the real rows).

/// Number of row groups and K chunks of the mapping (axes `Mg`, `Mc`).
pub const GROUPS: usize = 16;
pub const CHUNKS: usize = 32;

/// Packs one row tile of a row-major bf16 weight `w[n_rows, k_len]` (LE bytes) into `[GROUPS, CHUNKS, rt, e_len]` bytes.
pub fn pack_weight_tile(w: &[u8], n_rows: usize, k_len: usize, tile: usize, rt: usize, e_len: usize) -> Vec<u8> {
    assert_eq!(w.len(), n_rows * k_len * 2);
    let rows_per_group = n_rows / GROUPS;
    assert_eq!(rows_per_group * GROUPS, n_rows);
    assert!((tile + 1) * rt <= rows_per_group);
    assert!(k_len <= CHUNKS * e_len);
    let mut out = vec![0u8; GROUPS * CHUNKS * rt * e_len * 2];
    for g in 0..GROUPS {
        for c in 0..CHUNKS {
            let k0 = c * e_len;
            if k0 >= k_len {
                continue;
            }
            let take = (k_len - k0).min(e_len);
            for r in 0..rt {
                let n = g * rows_per_group + tile * rt + r;
                let src = (n * k_len + k0) * 2;
                let dst = (((g * CHUNKS + c) * rt) + r) * e_len * 2;
                out[dst..dst + take * 2].copy_from_slice(&w[src..src + take * 2]);
            }
        }
    }
    out
}

/// Bias tile `[GROUPS, rt]` (bf16 LE bytes) for the same row assignment as `pack_weight_tile`.
pub fn pack_bias_tile(b: &[u8], n_rows: usize, tile: usize, rt: usize) -> Vec<u8> {
    assert_eq!(b.len(), n_rows * 2);
    let rows_per_group = n_rows / GROUPS;
    let mut out = Vec::with_capacity(GROUPS * rt * 2);
    for g in 0..GROUPS {
        for r in 0..rt {
            let n = g * rows_per_group + tile * rt + r;
            out.extend_from_slice(&b[n * 2..n * 2 + 2]);
        }
    }
    out
}

/// Chunk-major activation `[CHUNKS, t_tile, e_len]` (bf16 LE bytes) from `rows` row-major activation rows of `k_len` bf16 bits;
/// rows beyond `rows.len() / k_len` (up to `t_tile`) and columns beyond `k_len` are zero.
pub fn pack_x_chunk_major(rows: &[u16], k_len: usize, t_tile: usize, e_len: usize) -> Vec<u8> {
    #[cfg(not(target_endian = "little"))]
    compile_error!("pack_x_chunk_major assumes a little-endian host");
    let n_rows = rows.len() / k_len;
    assert!(n_rows <= t_tile && k_len <= CHUNKS * e_len);
    // SAFETY: u16 -> u8 view of the same memory (little-endian host: the bf16 LE byte image).
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len() * 2) };
    let mut out = vec![0u8; CHUNKS * t_tile * e_len * 2];
    for c in 0..CHUNKS {
        let k0 = c * e_len;
        if k0 >= k_len {
            break;
        }
        let take = (k_len - k0).min(e_len);
        for t in 0..n_rows {
            let src = (t * k_len + k0) * 2;
            let dst = ((c * t_tile + t) * e_len) * 2;
            out[dst..dst + take * 2].copy_from_slice(&bytes[src..src + take * 2]);
        }
    }
    out
}

/// Group-major tile output `y_i[g, t, r]` (`[GROUPS, t_tile, rt]` bf16 LE bytes) scattered into row-major `out[t, n]` (`n_rows`
/// columns, first `rows` rows), with `n = g * rows_per_group + tile * rt + r`.
pub fn unpack_group_major_tile(y: &[u8], t_tile: usize, rt: usize, rows: usize, n_rows: usize, tile: usize, out: &mut [u16]) {
    let rows_per_group = n_rows / GROUPS;
    for g in 0..GROUPS {
        for t in 0..rows {
            let src = &y[((g * t_tile + t) * rt) * 2..((g * t_tile + t + 1) * rt) * 2];
            let dst = &mut out[t * n_rows + g * rows_per_group + tile * rt..][..rt];
            for (o, b) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *o = u16::from_le_bytes([b[0], b[1]]);
            }
        }
    }
}
