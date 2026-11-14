//! Load-time weight repack for the chunk AO / FFN kernels (`ops_ffnc`).  Pure byte permutations of the checkpoint tensors (no value changes) plus
//! the HBM upload.  The original tensors stay untouched (the token-serial decode path still uses them); this module adds a second, packed copy.
//!
//! Layouts (all "tile" tensors are contiguous per slice: one run per (row group g, K chunk c)):
//! - AO (fp8 e4m3 o_proj, n = 3840 rows, k = 4096 (sliding) / 8192 (full)): 3 tiles of 80 rows per group of 240; `W_i[g, c, rr, e] = W[g * 240 + i * 80 + rr, c * ck + e]`
//!   with chunk `ck` = 128 (sliding) / 256 (full) and 32 chunks; `S_i[g, rr] = scale[g * 240 + i * 80 + rr]`.
//! - FFN up / gate (NVFP4: f4e2m1 `[15360, 3840 / 2]` bytes, f8e4m3 block scales `[15360, 240]`, global scale): 4 tiles of 240 rows per group of 960;
//!   `W_i[g, c, rr, e] = W[g * 960 + i * 240 + rr, c * 128 + e]`, chunks 30 and 31 are zero (K 3840 -> 4096); `S_i[g, c, rr, b] = S[row, c * 8 + b]`.
//! - FFN down (f4 `[3840, 15360 / 2]`, scales `[3840, 960]`): 5 tiles of 48 rows per group of 240; the K axis (the up/gate row index L) is cut into 32 chunks
//!   of 2 halves of 240 real columns: `k = (c / 2) * 960 + (c % 2) * 480 + h * 240 + r` (r < 240), each half padded to 256 slots with 16 zero columns:
//!   `W_i[g, c, rr, h, r] = Wd[g * 240 + i * 48 + rr, k]`, `S_i[g, c, rr, h, b] = Sd[row, k / 16]` (b < 15, block 15 zero).
//!   This is exactly the layout in which `ffn_a` writes its GeGLU output `h[c, h, t, r]`.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::H;
use crate::ops_ffnc::{C32, C64, E32, E256, Fb8, Fb16, Fh2, Fi4, Fi5, G32, Gd8, Rt48, Rt80, Rt96, Rt240, Rt256};

pub const N_H: usize = 3840;
pub const N_L: usize = 15360;

/// AO weight tile `i` (0..3): fp8 bytes `w[3840, k]` -> `[16, 32, 80, ck]`.
pub fn pack_ao_w(w: &[u8], k: usize, ck: usize, i: usize) -> Vec<u8> {
    assert_eq!(w.len(), N_H * k);
    assert_eq!(32 * ck, k);
    let mut out = vec![0u8; 16 * 32 * 80 * ck];
    for g in 0..16 {
        for c in 0..32 {
            for rr in 0..80 {
                let src = (g * 240 + i * 80 + rr) * k + c * ck;
                let dst = ((g * 32 + c) * 80 + rr) * ck;
                out[dst..dst + ck].copy_from_slice(&w[src..src + ck]);
            }
        }
    }
    out
}

/// AO row scale tile `i`: bf16 bytes `[3840]` -> `[16, 80]`.
pub fn pack_ao_scale(scale: &[u8], i: usize) -> Vec<u8> {
    assert_eq!(scale.len(), N_H * 2);
    let mut out = Vec::with_capacity(16 * 80 * 2);
    for g in 0..16 {
        for rr in 0..80 {
            let n = g * 240 + i * 80 + rr;
            out.extend_from_slice(&scale[n * 2..n * 2 + 2]);
        }
    }
    out
}

/// FFN up / gate weight tile `i` (0..4): f4 bytes `[15360, 1920]` -> `[16, 32, 240, 128]` f4 (64 bytes per row and chunk).
pub fn pack_up_w(w4: &[u8], i: usize) -> Vec<u8> {
    assert_eq!(w4.len(), N_L * N_H / 2);
    let mut out = vec![0u8; 16 * 32 * 240 * 64];
    for g in 0..16 {
        for c in 0..30 {
            for rr in 0..240 {
                let n = g * 960 + i * 240 + rr;
                let src = n * (N_H / 2) + c * 64;
                let dst = ((g * 32 + c) * 240 + rr) * 64;
                out[dst..dst + 64].copy_from_slice(&w4[src..src + 64]);
            }
        }
    }
    out
}

/// FFN up / gate block-scale tile `i`: f8 bytes `[15360, 240]` -> `[16, 32, 240, 8]`.
pub fn pack_up_scale(s: &[u8], i: usize) -> Vec<u8> {
    assert_eq!(s.len(), N_L * N_H / 16);
    let mut out = vec![0u8; 16 * 32 * 240 * 8];
    for g in 0..16 {
        for c in 0..30 {
            for rr in 0..240 {
                let n = g * 960 + i * 240 + rr;
                let src = n * (N_H / 16) + c * 8;
                let dst = ((g * 32 + c) * 240 + rr) * 8;
                out[dst..dst + 8].copy_from_slice(&s[src..src + 8]);
            }
        }
    }
    out
}

/// First K column of (chunk c, half h) of the down projection.
fn down_k0(c: usize, h: usize) -> usize {
    (c / 2) * 960 + (c % 2) * 480 + h * 240
}

/// FFN down weight tile `i` (0..5): f4 bytes `[3840, 7680]` -> `[16, 32, 48, 2, 256]` f4 (128 bytes per row, chunk half).
pub fn pack_down_w(w4: &[u8], i: usize) -> Vec<u8> {
    assert_eq!(w4.len(), N_H * N_L / 2);
    let mut out = vec![0u8; 16 * 32 * 48 * 2 * 128];
    for g in 0..16 {
        for c in 0..32 {
            for rr in 0..48 {
                let n = g * 240 + i * 48 + rr;
                for h in 0..2 {
                    let k0 = down_k0(c, h);
                    let src = n * (N_L / 2) + k0 / 2;
                    let dst = (((g * 32 + c) * 48 + rr) * 2 + h) * 128;
                    out[dst..dst + 120].copy_from_slice(&w4[src..src + 120]);
                }
            }
        }
    }
    out
}

/// FFN down block-scale tile `i`: f8 bytes `[3840, 960]` -> `[16, 32, 48, 2, 16]` (block 15 of every half is zero).
pub fn pack_down_scale(s: &[u8], i: usize) -> Vec<u8> {
    assert_eq!(s.len(), N_H * N_L / 16);
    let mut out = vec![0u8; 16 * 32 * 48 * 2 * 16];
    for g in 0..16 {
        for c in 0..32 {
            for rr in 0..48 {
                let n = g * 240 + i * 48 + rr;
                for h in 0..2 {
                    let k0 = down_k0(c, h);
                    let src = n * (N_L / 16) + k0 / 16;
                    let dst = (((g * 32 + c) * 48 + rr) * 2 + h) * 16;
                    out[dst..dst + 15].copy_from_slice(&s[src..src + 15]);
                }
            }
        }
    }
    out
}

/// Pre-FF norm weight bf16 bytes `[3840]` -> zero padded `[4096]`.
pub fn pad_norm(w: &[u8]) -> Vec<u8> {
    assert_eq!(w.len(), N_H * 2);
    let mut out = vec![0u8; 4096 * 2];
    out[..w.len()].copy_from_slice(w);
    out
}

/// Interleaves the tiles of one tensor family as `[g, c, tile, rows..]`: `block` = bytes of one (g, c) block of one tile.
fn merge_tiles(tiles: &[Vec<u8>], block: usize) -> Vec<u8> {
    let n = tiles.len();
    let mut out = vec![0u8; 512 * n * block];
    for gc in 0..512 {
        for (i, t) in tiles.iter().enumerate() {
            out[(gc * n + i) * block..(gc * n + i + 1) * block].copy_from_slice(&t[gc * block..(gc + 1) * block]);
        }
    }
    out
}

/// Tile-merged up / gate weights `[16, 32, 4, 240, 128]` f4 (`ffn_chunk`: one tensor per family, tile selected by a tile view).
pub fn pack_up_w_merged(w4: &[u8]) -> Vec<u8> {
    merge_tiles(&(0..4).map(|i| pack_up_w(w4, i)).collect::<Vec<_>>(), 240 * 64)
}
pub fn pack_up_scale_merged(s: &[u8]) -> Vec<u8> {
    merge_tiles(&(0..4).map(|i| pack_up_scale(s, i)).collect::<Vec<_>>(), 240 * 8)
}
/// First K column of chunk `c` (0..64) of the 8-row-group down layout: `c = g * 4 + (c2 * 2 + h)` covers the 240 real columns of half `h` of chunk `c2` of up/gate row group `g`.
fn down8_k0(c: usize) -> usize {
    (c / 4) * 960 + (c % 4) * 240
}

/// Down weights in the 8-ROW-GROUP layout `[8, 64, 5, 96, 256]` f4 (K split over 64 slices, 4 row groups per cluster, tile-merged): row `n = (gd / 4) * 1920 + (gd % 4) * 480 + i * 96 + rr`,
/// chunk `c` = columns `down8_k0(c) + [0, 240)` padded to 256 slots with 16 zero columns (128 bytes per row and chunk). The `h` hop of the FFN stays in the GeGLU chunk layout,
/// only the replication (8x instead of 16x) and the reduction tree change.
pub fn pack_down_w_merged(w4: &[u8]) -> Vec<u8> {
    assert_eq!(w4.len(), N_H * N_L / 2);
    let mut out = vec![0u8; 8 * 64 * 5 * 96 * 128];
    for gd in 0..8 {
        for c in 0..64 {
            let k0 = down8_k0(c);
            for i in 0..5 {
                for rr in 0..96 {
                    let n = (gd / 4) * 1920 + (gd % 4) * 480 + i * 96 + rr;
                    let src = n * (N_L / 2) + k0 / 2;
                    let dst = (((gd * 64 + c) * 5 + i) * 96 + rr) * 128;
                    out[dst..dst + 120].copy_from_slice(&w4[src..src + 120]);
                }
            }
        }
    }
    out
}
/// Down block scales in the 8-row-group layout `[8, 64, 5, 96, 16]` f8 (block 15 of every chunk is zero).
pub fn pack_down_scale_merged(s: &[u8]) -> Vec<u8> {
    assert_eq!(s.len(), N_H * N_L / 16);
    let mut out = vec![0u8; 8 * 64 * 5 * 96 * 16];
    for gd in 0..8 {
        for c in 0..64 {
            let k0 = down8_k0(c);
            for i in 0..5 {
                for rr in 0..96 {
                    let n = (gd / 4) * 1920 + (gd % 4) * 480 + i * 96 + rr;
                    let src = n * (N_L / 16) + k0 / 16;
                    let dst = (((gd * 64 + c) * 5 + i) * 96 + rr) * 16;
                    out[dst..dst + 15].copy_from_slice(&s[src..src + 15]);
                }
            }
        }
    }
    out
}

/// Packed FFN weights of one layer in the tile-merged layout used by `ffn_chunk` (6 tensors instead of 26).
pub struct PackedFfnM {
    pub up_w: HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub up_s: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    pub gate_w: HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub gate_s: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    pub down_w: HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    pub down_s: HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    pub pre_norm: HbmTensor<bf16, Chip, m![C32, E32]>,
    /// scales of `S2_FP8=fold` (None when S2_FP8=off)
    pub fold: Option<FoldFfn>,
}

/// Scale tensors of the lossy fold FFN (`S2_FP8=fold`, see `device/f8/ffn.rs`): the NVFP4 block scales divided by 8 (exact in e4m3, same packed layouts as the bf16 ones) and 8 x the reciprocal global scales.
pub struct FoldFfn {
    pub up_s: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    pub gate_s: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    pub down_s: HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    pub up_gs: HbmTensor<f32, Chip, m![1]>,
    pub gate_gs: HbmTensor<f32, Chip, m![1]>,
    pub down_gs: HbmTensor<f32, Chip, m![1]>,
    /// R14 fixedcost PREBAKE: the e4m3 weights the fold decodes every pass, decoded ONCE at load by `ops_f8::prebake_{up,down}8` (bit-identical); only for layers with `fp8::ffn_fold`.
    pub w8: Option<FoldW8>,
}

/// R14 fixedcost PREBAKE: pre-decoded e4m3 FFN weights of one folded layer (same packed layouts as the f4 tensors).
pub struct FoldW8 {
    pub up: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub gate: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub down: HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
}

/// Uploads the tile-merged FFN tensors (same inputs as `upload_ffn`).
pub async fn upload_ffn_merged(
    device: &mut Device,
    up_w4: &[u8],
    up_s: &[u8],
    gate_w4: &[u8],
    gate_s: &[u8],
    down_w4: &[u8],
    down_s: &[u8],
    pre_norm: &[u8],
) -> Result<PackedFfnM, Box<dyn std::error::Error>> {
    Ok(PackedFfnM {
        up_w: up(device, pack_up_w_merged(up_w4)).await?,
        up_s: up(device, pack_up_scale_merged(up_s)).await?,
        gate_w: up(device, pack_up_w_merged(gate_w4)).await?,
        gate_s: up(device, pack_up_scale_merged(gate_s)).await?,
        down_w: up(device, pack_down_w_merged(down_w4)).await?,
        down_s: up(device, pack_down_scale_merged(down_s)).await?,
        pre_norm: up(device, pad_norm(pre_norm)).await?,
        fold: None,
    })
}

/// Packed AO weights of one layer (`E` = `E32` for sliding, `E256` for full layers).
pub struct PackedAo<E: AxisName> {
    pub w: [HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E]>; 3],
    pub s: [HbmTensor<bf16, Chip, m![G32, Rt80]>; 3],
}

/// Packed FFN weights of one layer.
pub struct PackedFfn {
    pub up_w: [HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>; 4],
    pub up_s: [HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>; 4],
    pub gate_w: [HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>; 4],
    pub gate_s: [HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>; 4],
    pub down_w: [HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>; 5],
    pub down_s: [HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>; 5],
    /// zero padded pre-FF norm weight `[C32, E32]`
    pub pre_norm: HbmTensor<bf16, Chip, m![C32, E32]>,
}

async fn up<D: Scalar, E: M>(device: &mut Device, bytes: Vec<u8>) -> Result<HbmTensor<D, Chip, E>, Box<dyn std::error::Error>>
where
    D: MaterializableScalar,
{
    Ok(HostTensor::<D, E>::from_buf(bytes).to_hbm(&mut device.pdma).await?)
}

/// Uploads the packed AO tiles from checkpoint bytes (`w` = `[3840, k]` e4m3, `scale` = `[3840]` bf16).
pub async fn upload_ao<E: AxisName>(
    device: &mut Device,
    w: &[u8],
    k: usize,
    ck: usize,
    scale: &[u8],
) -> Result<PackedAo<E>, Box<dyn std::error::Error>> {
    let w0 = up(device, pack_ao_w(w, k, ck, 0)).await?;
    let w1 = up(device, pack_ao_w(w, k, ck, 1)).await?;
    let w2 = up(device, pack_ao_w(w, k, ck, 2)).await?;
    let s0 = up(device, pack_ao_scale(scale, 0)).await?;
    let s1 = up(device, pack_ao_scale(scale, 1)).await?;
    let s2 = up(device, pack_ao_scale(scale, 2)).await?;
    Ok(PackedAo { w: [w0, w1, w2], s: [s0, s1, s2] })
}

/// Uploads the packed FFN tiles (`*_w4` f4 bytes, `*_s` f8 scale bytes as stored in the checkpoint, `pre_norm` bf16 bytes `[3840]`).
pub async fn upload_ffn(
    device: &mut Device,
    up_w4: &[u8],
    up_s: &[u8],
    gate_w4: &[u8],
    gate_s: &[u8],
    down_w4: &[u8],
    down_s: &[u8],
    pre_norm: &[u8],
) -> Result<PackedFfn, Box<dyn std::error::Error>> {
    macro_rules! tiles {
        ($n:expr, $f:expr) => {{
            let mut v = Vec::new();
            for i in 0..$n {
                v.push(up(device, $f(i)).await?);
            }
            v.try_into().ok().unwrap()
        }};
    }
    Ok(PackedFfn {
        up_w: tiles!(4, |i| pack_up_w(up_w4, i)),
        up_s: tiles!(4, |i| pack_up_scale(up_s, i)),
        gate_w: tiles!(4, |i| pack_up_w(gate_w4, i)),
        gate_s: tiles!(4, |i| pack_up_scale(gate_s, i)),
        down_w: tiles!(5, |i| pack_down_w(down_w4, i)),
        down_s: tiles!(5, |i| pack_down_scale(down_s, i)),
        pre_norm: up(device, pad_norm(pre_norm)).await?,
    })
}

/// Bytes of HBM taken by the packed copy of one layer (for the report).
pub fn packed_bytes_ao(k: usize) -> usize {
    16 * 32 * 80 * (k / 32) * 3 + 16 * 80 * 2 * 3
}
pub fn packed_bytes_ffn() -> usize {
    2 * (16 * 32 * 240 * 64 * 4 + 16 * 32 * 240 * 8 * 4) + 16 * 32 * 48 * 2 * 128 * 5 + 16 * 32 * 48 * 2 * 16 * 5 + 4096 * 2
}

#[allow(dead_code)]
type _Unused = (H, E256);
