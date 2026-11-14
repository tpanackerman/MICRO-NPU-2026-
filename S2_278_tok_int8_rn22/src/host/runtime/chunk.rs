//! The chunk path of `Decode::run_chunk`: 64 consecutive prompt positions of one request through all 48 layers with the chunk kernels
//! (qkvc QKV, attn attention + append, ffnc AO + FFN), instead of 64 token-serial passes.
//!
//! Residual stream: one zero padded `[64, 4096]` bf16 HBM tensor (`Res`, columns 3840.. stay zero), ping-pong between the AO output (`res2`) and
//! the FFN output (`res`). Per chunk the host builds and uploads: the embedding rows (text rows from the memory-mapped checkpoint table, scaled
//! like `ops::embed_token`; soft rows from the host copy `SoftEmbeddings::fetch_host`), the per-row cos / sin tables, the attention masks and
//! the KV append indices; those are shared by all layers of a type.
//!
//! A partial chunk (n < 64 rows, the tail of a prompt) is padded with CLONES of its last row: same embedding, same position, same append index
//! AND the same attention mask as row n-1 (`chunk_host::*_mask_rows`), so every padded row computes exactly what row n-1 computes in every layer
//! (it sees the same keys), is never visible to rows < n, and the duplicate KV appends rewrite identical values.

use super::*;
use crate::axes::{Kc, Ke, Kf, Kf64, Rh, Seg, Tc};
use crate::device::ffnc::E256;
use crate::host::chunkw::ChunkLayer;
use crate::host::qkvc as qk;
use crate::mm_norm::{bf_to_f32, f32_to_bf};
use crate::device::ffnc::Tk128;
use crate::{chunk_host, ops_chunk, ops_ffnc, ops_qkvc, ops_t128};

type Res = HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>;
/// A 128-row residual `[128, 4096]` seen as two 64-row halves (the T = 128 chunk path).
type Res128 = HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>;

/// Rows per chunk (the `Tc` of the kernels).
pub(super) const ROWS: usize = 64;
/// Rows of the largest chunk (two 64-row halves).
pub(super) const ROWS128: usize = 128;
/// Chunks must end below this position: the chunk full attention covers key pages 0..3 only.
pub(super) const LIMIT: usize = 2048;
const WIDTH: usize = 4096;

pub(super) struct ChunkEngine {
    res2: Res,
    xn: Res,
    q_s: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_stage: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    attn_s: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    q_f: HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_f: HbmTensor<bf16, Chip, m![Tc, Df]>,
    attn_f: HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    // T = 128 path: second attention output buffers, the concatenated attention inputs and the second residual
    attn_s2: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    attn_f2: HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    cat_s: HbmTensor<bf16, Chip, m![Rh, Tc, Ns, Gs, Ds]>,
    cat_f: HbmTensor<bf16, Chip, m![Rh, Tc, Gf, Df]>,
    res128_out: Res128,
    /// cos / sin rows of the runtime tables for positions `0..LIMIT` (bf16 LE bytes).
    cos_s: Vec<u8>,
    sin_s: Vec<u8>,
    cos_f: Vec<u8>,
    sin_f: Vec<u8>,
}

async fn zero<D: MaterializableScalar + num_traits::Zero, E: M>(device: &mut Device) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await
}

/// Test switch: return the final hidden row of this chunk row instead of the last one (`usize::MAX` = the last row).
pub(super) static DEBUG_ROW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Test switch (`runtime::force_token_serial`): run every chunk through the token-serial fallback.
pub(super) static FORCE_SERIAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether `run_chunk` can take this chunk through the kernels.
pub(super) fn eligible(weights: &Weights, pos0: usize, rows: &[Row<'_>]) -> bool {
    !FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        && !weights.chunk.is_empty()
        && !rows.is_empty()
        && rows.len() <= ROWS128
        // the rows must stay inside one 512-position page; any start position works (a decode row is a chunk of ONE row at its position)
        && pos0 % Tf::SIZE + rows.len() <= Tf::SIZE
        && pos0 + rows.len() <= LIMIT
        && rows.iter().all(|row| match row.soft {
            Some((embeddings, token)) => embeddings.host_row(token).is_some(),
            None => true,
        })
}

/// `dst` = the bf16 row `src` times `EMBED_SCALE`, rounded to nearest even (a NaN becomes 0x7FC0), all little-endian bf16. Eight lanes at a time with AVX2 when the CPU has it; the result is
/// bit for bit the scalar loop's (all 65536 inputs checked), the lanes only run the same IEEE multiply and the same integer rounding.
pub(super) fn scale_row(dst: &mut [u8], src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (just checked); `scale_row_avx2` only reads `src` and writes `dst` inside their lengths.
            unsafe { scale_row_avx2(dst, src) };
            return;
        }
    }
    scale_row_scalar(dst, src);
}

fn scale_row_scalar(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)) {
        let value = bf_to_f32(u16::from_le_bytes([s[0], s[1]])) * crate::EMBED_SCALE;
        d.copy_from_slice(&f32_to_bf(value).to_le_bytes());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_row_avx2(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len()) / 2;
    // SAFETY (all intrinsics below): every access is `2 * i .. 2 * i + 16` bytes with `i + 8 <= n`, inside both slices.
    unsafe {
        let scale = _mm256_set1_ps(crate::EMBED_SCALE);
        let c7fff = _mm256_set1_epi32(0x7FFF);
        let one = _mm256_set1_epi32(1);
        let qnan = _mm256_set1_epi32(0x7FC0);
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm_loadu_si128(src.as_ptr().add(2 * i) as *const __m128i);
            let v = _mm256_mul_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)), scale);
            let b = _mm256_castps_si256(v);
            let lsb = _mm256_and_si256(_mm256_srli_epi32(b, 16), one);
            let rounded = _mm256_srli_epi32(_mm256_add_epi32(b, _mm256_add_epi32(c7fff, lsb)), 16);
            let nan = _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
            let r = _mm256_blendv_epi8(rounded, qnan, nan);
            let packed = _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
            _mm_storeu_si128(dst.as_mut_ptr().add(2 * i) as *mut __m128i, packed);
            i += 8;
        }
        scale_row_scalar(&mut dst[2 * i..2 * n], &src[2 * i..2 * n]);
    }
}

/// The embedding of one row (text row: the checkpoint table row scaled like `ops::embed_token`; soft row: its host copy) written into `dst` (`H` bf16 LE values).
pub(super) fn embed_row_into(weights: &Weights, row: &Row<'_>, dst: &mut [u8]) {
    match row.soft {
        Some((embeddings, token)) => dst.copy_from_slice(embeddings.host_row(token).expect("host copy of the soft row")),
        None => scale_row(dst, weights.embed_host.row(row.token)),
    }
}

/// The embedding of the text token `token` written into `dst` (same bytes as `embed_row_into` of a text row; R13 pre-upload, rows predicted by the worker).
pub(super) fn embed_text_into(weights: &Weights, token: usize, dst: &mut [u8]) {
    scale_row(dst, weights.embed_host.row(token));
}

/// The `[total, 4096]` embedding of `rows` (little-endian bf16 bytes, columns 3840.. zero): text rows from the checkpoint table scaled like `ops::embed_token`,
/// soft rows from their host copy; rows `rows.len()..total` repeat the last row.
pub(super) fn embed_rows(weights: &Weights, rows: &[Row<'_>], total: usize) -> Vec<u8> {
    let n = rows.len();
    let mut out = vec![0u8; total * WIDTH * 2];
    let width = H::SIZE * 2;
    for (c, row) in rows.iter().enumerate() {
        embed_row_into(weights, row, &mut out[c * WIDTH * 2..c * WIDTH * 2 + width]);
    }
    let last = (n - 1) * WIDTH * 2;
    for c in n..total {
        out.copy_within(last..last + width, c * WIDTH * 2);
    }
    out
}

impl ChunkEngine {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        let (cos_s, sin_s) = qk::rope_rows_bytes(Ds::SIZE, 10_000.0, Ds::SIZE, 0, LIMIT);
        let (cos_f, sin_f) = qk::rope_rows_bytes(Df::SIZE, 1_000_000.0, 128, 0, LIMIT);
        Ok(Self {
            res2: zero(device).await?,
            xn: zero(device).await?,
            q_s: zero(device).await?,
            k_stage: zero(device).await?,
            v_stage: zero(device).await?,
            attn_s: zero(device).await?,
            q_f: zero(device).await?,
            k_f: zero(device).await?,
            v_f: zero(device).await?,
            attn_f: zero(device).await?,
            attn_s2: zero(device).await?,
            attn_f2: zero(device).await?,
            cat_s: zero(device).await?,
            cat_f: zero(device).await?,
            res128_out: zero(device).await?,
            cos_s,
            sin_s,
            cos_f,
            sin_f,
        })
    }

    /// Rows of a table for the chunk positions (row `c` = position `pos0 + min(c, n - 1)`).
    fn rope(table: &[u8], dim: usize, pos0: usize, n: usize) -> Vec<u8> {
        let width = dim * 2;
        let mut out = Vec::with_capacity(ROWS * width);
        for c in 0..ROWS {
            let p = pos0 + c.min(n - 1);
            out.extend_from_slice(&table[p * width..(p + 1) * width]);
        }
        out
    }

    /// Runs the chunk through all layers. Returns the last row final residual (`H` bf16 LE bytes) if `want_hidden`.
    pub(super) async fn run(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        blocks: &Blocks,
        pos0: usize,
        rows: &[Row<'_>],
        want_hidden: bool,
    ) -> Result<Option<Vec<u8>>, Error> {
        let n = rows.len();
        let x = prof::timed!("c_embed", embed_rows(weights, rows, ROWS));
        let mut res: Res = prof::timed!("c_h2d", HostTensor::<bf16, m![Tc, Kc, Ke]>::from_buf(x).to_hbm(&mut device.pdma).await)?;

        // per-chunk inputs shared by all layers of a type
        let (cos_s, sin_s, cos_f, sin_f, s_mask, s_index, f_index, f_mask2, f_mask4, s_pen, f_pen2, f_pen4) = prof::timed!("c_h2d", async {
            let cos_s = HostTensor::<bf16, m![Tc, Ds]>::from_buf(Self::rope(&self.cos_s, Ds::SIZE, pos0, n)).to_hbm(&mut device.pdma).await?;
            let sin_s = HostTensor::<bf16, m![Tc, Ds]>::from_buf(Self::rope(&self.sin_s, Ds::SIZE, pos0, n)).to_hbm(&mut device.pdma).await?;
            let cos_f = HostTensor::<bf16, m![Tc, Df]>::from_buf(Self::rope(&self.cos_f, Df::SIZE, pos0, n)).to_hbm(&mut device.pdma).await?;
            let sin_f = HostTensor::<bf16, m![Tc, Df]>::from_buf(Self::rope(&self.sin_f, Df::SIZE, pos0, n)).to_hbm(&mut device.pdma).await?;
            let s_mask = HostTensor::<bf16, m![Ts % 32, Tc, Seg, Ts / 32]>::from_vec(chunk_host::sliding_mask_rows(pos0, n)).to_hbm(&mut device.pdma).await?;
            let pad = |mut index: Vec<i32>| {
                let last = index[n - 1];
                index[n..].iter_mut().for_each(|slot| *slot = last);
                index
            };
            let s_index = HostTensor::<i32, m![Ts = 64]>::from_vec(pad(chunk_host::sliding_append_index(pos0))).to_hbm(&mut device.pdma).await?;
            let f_index = HostTensor::<i32, m![Tc]>::from_vec(pad(chunk_host::full_append_index(pos0))).to_hbm(&mut device.pdma).await?;
            let page = pos0 / Tf::SIZE;
            let f_mask2 = if page <= 1 {
                Some(HostTensor::<bf16, m![Tf % 32, Tc, Kf]>::from_vec(chunk_host::full_mask_rows(2, pos0, n)).to_hbm(&mut device.pdma).await?)
            } else {
                None
            };
            let f_mask4 = if page > 1 {
                Some(HostTensor::<bf16, m![Tf % 32, Tc, Kf64]>::from_vec(chunk_host::full_mask_rows(4, pos0, n)).to_hbm(&mut device.pdma).await?)
            } else {
                None
            };
            let s_pen = HostTensor::<bf16, m![Ts % 32, Seg, Ts / 32]>::from_vec(chunk_host::sliding_pen(pos0)).to_hbm(&mut device.pdma).await?;
            let f_pen2 = if page <= 1 {
                Some(HostTensor::<bf16, m![Tf % 32, Kf]>::from_vec(chunk_host::full_pen(2, pos0, n)).to_hbm(&mut device.pdma).await?)
            } else {
                None
            };
            let f_pen4 = if page > 1 {
                Some(HostTensor::<bf16, m![Tf % 32, Kf64]>::from_vec(chunk_host::full_pen(4, pos0, n)).to_hbm(&mut device.pdma).await?)
            } else {
                None
            };
            Ok::<_, Error>((cos_s, sin_s, cos_f, sin_f, s_mask, s_index, f_index, f_mask2, f_mask4, s_pen, f_pen2, f_pen4))
        }
        .await)?;

        let page = pos0 / Tf::SIZE;
        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (k_ring, v_ring) = ws.kv_cache[index].as_sliding_mut();
                    let (k_ring, v_ring) = (&mut k_ring[blocks.sliding], &mut v_ring[blocks.sliding]);
                    prof::timed!("c_qkv_s", launch(
                        ops_qkvc::sliding_qkv_chunk,
                        (
                            device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                            &l.q_norm, &l.k_norm, &cos_s, &sin_s, &mut self.q_s, &mut self.k_stage, &mut self.v_stage,
                        ),
                    ).await)?;
                    prof::timed!("c_attn_s", launch(
                        ops_chunk::sliding_chunk_attention,
                        (device, &self.q_s, &*k_ring, &*v_ring, &self.k_stage, &self.v_stage, &s_mask, &s_pen, &mut self.attn_s),
                    ).await)?;
                    prof::timed!("c_append_s", launch(
                        ops_chunk::sliding_chunk_append,
                        (device, &self.k_stage, &self.v_stage, &s_index, &mut *k_ring, &mut *v_ring),
                    ).await)?;
                    let attn = std::mem::replace(&mut self.attn_s, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tc, Kc, Ke]> = unsafe { attn.reshape() };
                    prof::timed!("c_layer_s", launch(
                        ops_ffnc::layer_sliding_t64,
                        (
                            device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut self.res2,
                        ),
                    ).await)?;
                    self.attn_s = unsafe { attn.reshape() };
                    std::mem::swap(&mut res, &mut self.res2);
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (k_cache, v_cache) = ws.kv_cache[index].as_full_mut();
                    let pages = &blocks.full;
                    assert!(page < pages.len(), "position has no full-attention block");
                    prof::timed!("c_qkv_f", launch(
                        ops_qkvc::full_qkv_chunk,
                        (
                            device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                            &l.q_norm, &l.k_norm, &cos_f, &sin_f, &mut self.q_f, &mut self.k_f, &mut self.v_f,
                        ),
                    ).await)?;
                    prof::timed!("c_append_f", launch(
                        ops_chunk::full_chunk_append,
                        (device, &self.k_f, &self.v_f, &f_index, &mut k_cache[pages[page]], &mut v_cache[pages[page]]),
                    ).await)?;
                    // pages in logical order; pages after the current one carry an all-zero mask, any valid page will do
                    let physical = |logical: usize| pages[if logical <= page { logical } else { 0 }];
                    if let Some(mask) = &f_mask2 {
                        let (a, b) = (physical(0), physical(1));
                        prof::timed!("c_attn_f", launch(
                            ops_chunk::full_chunk_attention_2,
                            (device, &self.q_f, &k_cache[a], &k_cache[b], &v_cache[a], &v_cache[b], mask, f_pen2.as_ref().expect("two-page penalty"), &mut self.attn_f),
                        ).await)?;
                    } else {
                        let mask = f_mask4.as_ref().expect("four-page mask");
                        let p = [physical(0), physical(1), physical(2), physical(3)];
                        prof::timed!("c_attn_f", launch(
                            ops_chunk::full_chunk_attention_4,
                            (
                                device, &self.q_f, &k_cache[p[0]], &k_cache[p[1]], &k_cache[p[2]], &k_cache[p[3]],
                                &v_cache[p[0]], &v_cache[p[1]], &v_cache[p[2]], &v_cache[p[3]], mask, f_pen4.as_ref().expect("four-page penalty"), &mut self.attn_f,
                            ),
                        ).await)?;
                    }
                    let attn = std::mem::replace(&mut self.attn_f, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tc, Kc, E256]> = unsafe { attn.reshape() };
                    prof::timed!("c_layer_f", launch(
                        ops_ffnc::layer_full_t64,
                        (
                            device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut self.res2,
                        ),
                    ).await)?;
                    self.attn_f = unsafe { attn.reshape() };
                    std::mem::swap(&mut res, &mut self.res2);
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
        }

        if !want_hidden {
            return Ok(None);
        }
        let host: HostTensor<bf16, m![Tc, Kc, Ke]> = prof::timed!("c_d2h", res.to_host(&mut device.pdma).await)?;
        let bytes = host.into_inner().into_buf();
        let row = DEBUG_ROW.load(std::sync::atomic::Ordering::Relaxed).min(n - 1);
        let start = row * WIDTH * 2;
        Ok(Some(bytes[start..start + H::SIZE * 2].to_vec()))
    }

    /// The per-half inputs of a T = 128 chunk (half = 64 rows at positions `p0..`, `n` of them real, the rest clones of row `n - 1`).
    async fn half_inputs(&self, device: &mut Device, p0: usize, n: usize) -> Result<Half, Error> {
        let cos_s = HostTensor::<bf16, m![Tc, Ds]>::from_buf(Self::rope(&self.cos_s, Ds::SIZE, p0, n)).to_hbm(&mut device.pdma).await?;
        let sin_s = HostTensor::<bf16, m![Tc, Ds]>::from_buf(Self::rope(&self.sin_s, Ds::SIZE, p0, n)).to_hbm(&mut device.pdma).await?;
        let cos_f = HostTensor::<bf16, m![Tc, Df]>::from_buf(Self::rope(&self.cos_f, Df::SIZE, p0, n)).to_hbm(&mut device.pdma).await?;
        let sin_f = HostTensor::<bf16, m![Tc, Df]>::from_buf(Self::rope(&self.sin_f, Df::SIZE, p0, n)).to_hbm(&mut device.pdma).await?;
        let s_mask = HostTensor::<bf16, m![Ts % 32, Tc, Seg, Ts / 32]>::from_vec(chunk_host::sliding_mask_rows(p0, n)).to_hbm(&mut device.pdma).await?;
        let pad = |mut index: Vec<i32>| {
            let last = index[n - 1];
            index[n..].iter_mut().for_each(|slot| *slot = last);
            index
        };
        let s_index = HostTensor::<i32, m![Ts = 64]>::from_vec(pad(chunk_host::sliding_append_index(p0))).to_hbm(&mut device.pdma).await?;
        let f_index = HostTensor::<i32, m![Tc]>::from_vec(pad(chunk_host::full_append_index(p0))).to_hbm(&mut device.pdma).await?;
        let page = p0 / Tf::SIZE;
        let f_mask2 = if page <= 1 {
            Some(HostTensor::<bf16, m![Tf % 32, Tc, Kf]>::from_vec(chunk_host::full_mask_rows(2, p0, n)).to_hbm(&mut device.pdma).await?)
        } else {
            None
        };
        let f_mask4 = if page > 1 {
            Some(HostTensor::<bf16, m![Tf % 32, Tc, Kf64]>::from_vec(chunk_host::full_mask_rows(4, p0, n)).to_hbm(&mut device.pdma).await?)
        } else {
            None
        };
        let s_pen = HostTensor::<bf16, m![Ts % 32, Seg, Ts / 32]>::from_vec(chunk_host::sliding_pen(p0)).to_hbm(&mut device.pdma).await?;
        let f_pen2 = if page <= 1 {
            Some(HostTensor::<bf16, m![Tf % 32, Kf]>::from_vec(chunk_host::full_pen(2, p0, n)).to_hbm(&mut device.pdma).await?)
        } else {
            None
        };
        let f_pen4 = if page > 1 {
            Some(HostTensor::<bf16, m![Tf % 32, Kf64]>::from_vec(chunk_host::full_pen(4, p0, n)).to_hbm(&mut device.pdma).await?)
        } else {
            None
        };
        Ok(Half { cos_s, sin_s, cos_f, sin_f, s_mask, s_index, f_index, f_mask2, f_mask4, s_pen, f_pen2, f_pen4 })
    }

    /// The T = 128 chunk (65..=128 rows): per layer two 64-row QKV + attention + append passes (each half reads its rows of ONE 128-row residual through a
    /// tile view; the second half sees the first half keys in the cache), then ONE `layer_*_t128` launch (AO + FFN of all 128 rows: the FFN weights are
    /// streamed once per 128 rows instead of once per 64). Rows beyond `n` are clones of row `n - 1` (see the module doc).
    pub(super) async fn run128(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        blocks: &Blocks,
        pos0: usize,
        rows: &[Row<'_>],
        want_hidden: bool,
    ) -> Result<Option<Vec<u8>>, Error> {
        let n = rows.len();
        assert!(n > ROWS && n <= ROWS128);
        let x = prof::timed!("c_embed", embed_rows(weights, rows, ROWS128));
        let mut res: Res128 = prof::timed!("c_h2d", HostTensor::<bf16, m![Rh, Tc, Kc, Ke]>::from_buf(x).to_hbm(&mut device.pdma).await)?;
        // half 0: rows 0..64 (all real), half 1: rows 64..128 (n - 64 real)
        let halves = prof::timed!("c_h2d", async {
            let h0 = self.half_inputs(device, pos0, ROWS).await?;
            let h1 = self.half_inputs(device, pos0 + ROWS, n - ROWS).await?;
            Ok::<_, Error>([h0, h1])
        }
        .await)?;
        let page = pos0 / Tf::SIZE;
        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (k_ring, v_ring) = ws.kv_cache[index].as_sliding_mut();
                    let (k_ring, v_ring) = (&mut k_ring[blocks.sliding], &mut v_ring[blocks.sliding]);
                    for (half, inp) in halves.iter().enumerate() {
                        if half == 0 {
                            prof::timed!("c_qkv_s", launch(
                                ops_qkvc::sliding_qkv_chunk_h0,
                                (
                                    &mut *device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                                    &l.q_norm, &l.k_norm, &inp.cos_s, &inp.sin_s, &mut self.q_s, &mut self.k_stage, &mut self.v_stage,
                                ),
                            ).await)?;
                        } else {
                            prof::timed!("c_qkv_s", launch(
                                ops_qkvc::sliding_qkv_chunk_h1,
                                (
                                    &mut *device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                                    &l.q_norm, &l.k_norm, &inp.cos_s, &inp.sin_s, &mut self.q_s, &mut self.k_stage, &mut self.v_stage,
                                ),
                            ).await)?;
                        }
                        let attn_out = if half == 0 { &mut self.attn_s } else { &mut self.attn_s2 };
                        prof::timed!("c_attn_s", launch(
                            ops_chunk::sliding_chunk_attention,
                            (&mut *device, &self.q_s, &*k_ring, &*v_ring, &self.k_stage, &self.v_stage, &inp.s_mask, &inp.s_pen, attn_out),
                        ).await)?;
                        prof::timed!("c_append_s", launch(
                            ops_chunk::sliding_chunk_append,
                            (&mut *device, &self.k_stage, &self.v_stage, &inp.s_index, &mut *k_ring, &mut *v_ring),
                        ).await)?;
                    }
                    prof::timed!("c_cat", launch(ops_t128::cat_attn_s, (&mut *device, &self.attn_s, &self.attn_s2, &mut self.cat_s)).await)?;
                    let attn = std::mem::replace(&mut self.cat_s, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = unsafe { attn.reshape() };
                    let res_in: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = unsafe { std::mem::replace(&mut res, HbmTensor::new()).reshape() };
                    let mut res_out: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = unsafe { std::mem::replace(&mut self.res128_out, HbmTensor::new()).reshape() };
                    prof::timed!("c_layer_s", launch(
                        ops_ffnc::layer_sliding_t128,
                        (
                            &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res_in,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut res_out,
                        ),
                    ).await)?;
                    self.cat_s = unsafe { attn.reshape() };
                    res = unsafe { res_out.reshape() };
                    self.res128_out = unsafe { res_in.reshape() };
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (k_cache, v_cache) = ws.kv_cache[index].as_full_mut();
                    let pages = &blocks.full;
                    assert!(page < pages.len(), "position has no full-attention block");
                    for (half, inp) in halves.iter().enumerate() {
                        if half == 0 {
                            prof::timed!("c_qkv_f", launch(
                                ops_qkvc::full_qkv_chunk_h0,
                                (
                                    &mut *device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                                    &l.q_norm, &l.k_norm, &inp.cos_f, &inp.sin_f, &mut self.q_f, &mut self.k_f, &mut self.v_f,
                                ),
                            ).await)?;
                        } else {
                            prof::timed!("c_qkv_f", launch(
                                ops_qkvc::full_qkv_chunk_h1,
                                (
                                    &mut *device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                                    &l.q_norm, &l.k_norm, &inp.cos_f, &inp.sin_f, &mut self.q_f, &mut self.k_f, &mut self.v_f,
                                ),
                            ).await)?;
                        }
                        prof::timed!("c_append_f", launch(
                            ops_chunk::full_chunk_append,
                            (&mut *device, &self.k_f, &self.v_f, &inp.f_index, &mut k_cache[pages[page]], &mut v_cache[pages[page]]),
                        ).await)?;
                        let physical = |logical: usize| pages[if logical <= page { logical } else { 0 }];
                        let attn_out = if half == 0 { &mut self.attn_f } else { &mut self.attn_f2 };
                        if let Some(mask) = &inp.f_mask2 {
                            let (a, b) = (physical(0), physical(1));
                            prof::timed!("c_attn_f", launch(
                                ops_chunk::full_chunk_attention_2,
                                (&mut *device, &self.q_f, &k_cache[a], &k_cache[b], &v_cache[a], &v_cache[b], mask, inp.f_pen2.as_ref().expect("two-page penalty"), attn_out),
                            ).await)?;
                        } else {
                            let mask = inp.f_mask4.as_ref().expect("four-page mask");
                            let p = [physical(0), physical(1), physical(2), physical(3)];
                            prof::timed!("c_attn_f", launch(
                                ops_chunk::full_chunk_attention_4,
                                (
                                    &mut *device, &self.q_f, &k_cache[p[0]], &k_cache[p[1]], &k_cache[p[2]], &k_cache[p[3]],
                                    &v_cache[p[0]], &v_cache[p[1]], &v_cache[p[2]], &v_cache[p[3]], mask, inp.f_pen4.as_ref().expect("four-page penalty"), attn_out,
                                ),
                            ).await)?;
                        }
                    }
                    prof::timed!("c_cat", launch(ops_t128::cat_attn_f, (&mut *device, &self.attn_f, &self.attn_f2, &mut self.cat_f)).await)?;
                    let attn = std::mem::replace(&mut self.cat_f, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, E256]> = unsafe { attn.reshape() };
                    let res_in: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = unsafe { std::mem::replace(&mut res, HbmTensor::new()).reshape() };
                    let mut res_out: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = unsafe { std::mem::replace(&mut self.res128_out, HbmTensor::new()).reshape() };
                    prof::timed!("c_layer_f", launch(
                        ops_ffnc::layer_full_t128,
                        (
                            &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res_in,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut res_out,
                        ),
                    ).await)?;
                    self.cat_f = unsafe { attn.reshape() };
                    res = unsafe { res_out.reshape() };
                    self.res128_out = unsafe { res_in.reshape() };
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
        }
        if !want_hidden {
            return Ok(None);
        }
        let host: HostTensor<bf16, m![Rh, Tc, Kc, Ke]> = prof::timed!("c_d2h", res.to_host(&mut device.pdma).await)?;
        let bytes = host.into_inner().into_buf();
        let row = DEBUG_ROW.load(std::sync::atomic::Ordering::Relaxed).min(n - 1);
        let start = row * WIDTH * 2;
        Ok(Some(bytes[start..start + H::SIZE * 2].to_vec()))
    }
}

/// Per-half inputs of a T = 128 chunk (shared by all layers of a type).
struct Half {
    cos_s: HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin_s: HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos_f: HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin_f: HbmTensor<bf16, Chip, m![Tc, Df]>,
    s_mask: HbmTensor<bf16, Chip, m![Ts % 32, Tc, Seg, Ts / 32]>,
    s_index: HbmTensor<i32, Chip, m![Ts = 64]>,
    f_index: HbmTensor<i32, Chip, m![Tc]>,
    f_mask2: Option<HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>>,
    f_mask4: Option<HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf64]>>,
    s_pen: HbmTensor<bf16, Chip, m![Ts % 32, Seg, Ts / 32]>,
    f_pen2: Option<HbmTensor<bf16, Chip, m![Tf % 32, Kf]>>,
    f_pen4: Option<HbmTensor<bf16, Chip, m![Tf % 32, Kf64]>>,
}
