//! Host side of the chunk QKV kernels (`ops_qkvc`): weight packing into the tile layout the kernels stream, the per-chunk RoPE rows, and the
//! scratch buffer.  See RESULTS.md (qkvc tree) for the layouts.
//!
//! Packed projection weight of `n` rows, `nt` row tiles of 64 rows per row group: `W[nt, G = n / (nt * 64), 32 chunks, 64 rows, 128]`
//! (fp8 bytes), natural row `r = g * (nt * 64) + i * 64 + rr` -> `W[i, g, c, rr, e] = w[r][c * 128 + e]` for `c < 30` (zero for the 2 padding
//! chunks).  The row scale is packed `S[nt, G, 64]` (bf16).  Q (4096 rows): nt = 4; K, V (2048 rows): nt = 2.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub const CHUNK: usize = 64;

/// Packs the natural fp8 matrix `w[n][k = 3840]` (bytes) into `W[nt, G, 32, rt, 128]` (rt = 64; the full-layer K projection uses nt = 1, rt = 32).
pub fn pack_weight_tiles(w: &[u8], n: usize, k: usize, nt: usize, rt: usize) -> Vec<u8> {
    let rows = nt * rt;
    let groups = n / rows;
    let mut out = vec![0u8; nt * groups * 32 * rt * 128];
    for i in 0..nt {
        for g in 0..groups {
            for c in 0..k / 128 {
                for rr in 0..rt {
                    let src = (g * rows + i * rt + rr) * k + c * 128;
                    let dst = (((i * groups + g) * 32 + c) * rt + rr) * 128;
                    out[dst..dst + 128].copy_from_slice(&w[src..src + 128]);
                }
            }
        }
    }
    out
}

/// Packs the natural bf16 row scales `scale[n]` (little-endian bytes) into `S[nt, G, rt]`.
pub fn pack_scale_tiles(scale: &[u8], n: usize, nt: usize, rt: usize) -> Vec<u8> {
    let rows = nt * rt;
    let groups = n / rows;
    let mut out = vec![0u8; nt * groups * rt * 2];
    for i in 0..nt {
        for g in 0..groups {
            for rr in 0..rt {
                let src = (g * rows + i * rt + rr) * 2;
                let dst = ((i * groups + g) * rt + rr) * 2;
                out[dst..dst + 2].copy_from_slice(&scale[src..src + 2]);
            }
        }
    }
    out
}

/// Sliding-layer packed projection weights (all three are tile-major, K padded to 32 chunks with zeros).
pub struct SlidingChunkWeights {
    pub wq: HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    pub sq: HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    pub wk: HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    pub sk: HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    pub wv: HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    pub sv: HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
}

/// Reads layer `layer` (a sliding layer) of the model's safetensors and uploads the packed chunk weights.
pub async fn load_sliding_chunk_weights(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<SlidingChunkWeights, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}.self_attn");
    let get = |name: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> { Ok(tensors.tensor(&format!("{prefix}.{name}"))?.data().to_vec()) };
    let (q, k, v) = (get("q_proj.weight")?, get("k_proj.weight")?, get("v_proj.weight")?);
    let (qs, ks, vs) = (get("q_proj.weight_scale")?, get("k_proj.weight_scale")?, get("v_proj.weight_scale")?);
    let h = 3840;
    let wq = HostTensor::<f8e4m3, m![Rq4, Gg, Kc, Rt, Ke]>::from_buf(pack_weight_tiles(&q, 4096, h, 4, 64)).to_hbm(&mut device.pdma).await?;
    let sq = HostTensor::<bf16, m![Rq4, Gg, Rt]>::from_buf(pack_scale_tiles(&qs, 4096, 4, 64)).to_hbm(&mut device.pdma).await?;
    let wk = HostTensor::<f8e4m3, m![Rq2, Gg, Kc, Rt, Ke]>::from_buf(pack_weight_tiles(&k, 2048, h, 2, 64)).to_hbm(&mut device.pdma).await?;
    let sk = HostTensor::<bf16, m![Rq2, Gg, Rt]>::from_buf(pack_scale_tiles(&ks, 2048, 2, 64)).to_hbm(&mut device.pdma).await?;
    let wv = HostTensor::<f8e4m3, m![Rq2, Gg, Kc, Rt, Ke]>::from_buf(pack_weight_tiles(&v, 2048, h, 2, 64)).to_hbm(&mut device.pdma).await?;
    let sv = HostTensor::<bf16, m![Rq2, Gg, Rt]>::from_buf(pack_scale_tiles(&vs, 2048, 2, 64)).to_hbm(&mut device.pdma).await?;
    Ok(SlidingChunkWeights { wq, sq, wk, sk, wv, sv })
}

/// Full-attention-layer packed projection weights: Q 8192 rows (8 tiles of 64 rows per head group), K 512 rows (16 groups of 32 rows, one tile).
pub struct FullChunkWeights {
    pub wq: HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    pub sq: HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    pub wk: HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    pub sk: HbmTensor<bf16, Chip, m![Gg, Rk]>,
}

/// Reads layer `layer` (a full-attention layer) of the model's safetensors and uploads the packed chunk weights.
pub async fn load_full_chunk_weights(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<FullChunkWeights, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}.self_attn");
    let get = |name: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> { Ok(tensors.tensor(&format!("{prefix}.{name}"))?.data().to_vec()) };
    let (q, k, qs, ks) = (get("q_proj.weight")?, get("k_proj.weight")?, get("q_proj.weight_scale")?, get("k_proj.weight_scale")?);
    let h = 3840;
    let wq = HostTensor::<f8e4m3, m![Rq8, Gg, Kc, Rt, Ke]>::from_buf(pack_weight_tiles(&q, 8192, h, 8, 64)).to_hbm(&mut device.pdma).await?;
    let sq = HostTensor::<bf16, m![Rq8, Gg, Rt]>::from_buf(pack_scale_tiles(&qs, 8192, 8, 64)).to_hbm(&mut device.pdma).await?;
    let wk = HostTensor::<f8e4m3, m![Gg, Kc, Rk, Ke]>::from_buf(pack_weight_tiles(&k, 512, h, 1, 32)).to_hbm(&mut device.pdma).await?;
    let sk = HostTensor::<bf16, m![Gg, Rk]>::from_buf(pack_scale_tiles(&ks, 512, 1, 32)).to_hbm(&mut device.pdma).await?;
    Ok(FullChunkWeights { wq, sq, wk, sk })
}

/// cos / sin rows of the baseline runtime tables for the given POSITIONS (`dim` = Ds 256 / Df 512, same formula as `runtime::rope_data`): two
/// `[positions.len(), dim]` bf16 little-endian byte vectors, row i = table row `positions[i]`.  The chunk QKV kernels take exactly this per-row
/// layout, so a prefill chunk (positions p0..p0 + 63) and a decode batch (one position per request) use the same kernels; build it once per
/// iteration and share it between all layers of the same type.
pub fn rope_rows_for_positions(dim: usize, theta: f32, rotated_dim: usize, positions: &[usize]) -> (Vec<u8>, Vec<u8>) {
    let half = dim / 2;
    let rotated_half = rotated_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * dim * 2);
    let mut sin = Vec::with_capacity(positions.len() * dim * 2);
    let bf = |v: f32| {
        let bits = v.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16;
        (rounded as u16).to_le_bytes()
    };
    for &pos in positions {
        for sign in [-1.0f32, 1.0f32] {
            for i in 0..half {
                if i < rotated_half {
                    let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                    cos.extend_from_slice(&bf(freq.cos()));
                    sin.extend_from_slice(&bf(sign * freq.sin()));
                } else {
                    cos.extend_from_slice(&bf(1.0));
                    sin.extend_from_slice(&bf(0.0));
                }
            }
        }
    }
    (cos, sin)
}

/// Rows `p0..p0 + count` of the tables (see `rope_rows_for_positions`); `count = E` gives the whole `ws.cos_s`-style table.
pub fn rope_rows_bytes(dim: usize, theta: f32, rotated_dim: usize, p0: usize, count: usize) -> (Vec<u8>, Vec<u8>) {
    let positions: Vec<usize> = (p0..p0 + count).collect();
    rope_rows_for_positions(dim, theta, rotated_dim, &positions)
}

/// The zero-initialised normalised-activation scratch `xn[T, Kc, Ke]` for T = 64 (the 2 padding chunks must stay zero: the kernels never write them).
pub async fn new_xn_scratch(device: &mut Device) -> Result<HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>, Box<dyn std::error::Error>> {
    Ok(HostTensor::<bf16, m![Tc, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?)
}

/// The same scratch for the T = 8 bucket.
pub async fn new_xn_scratch_t8(device: &mut Device) -> Result<HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>, Box<dyn std::error::Error>> {
    Ok(HostTensor::<bf16, m![Tb, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?)
}
