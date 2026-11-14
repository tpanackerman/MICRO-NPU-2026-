use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Df, Ds, E, Gf, Gs, H, Mv, Ns, Pv, Rv, Tf, Tm64, Ts, W};
use crate::host::audio::AudioFrames;
use crate::host::generate::SoftToken;
use crate::host::image::ImagePatches;
use crate::host::load::{FullLayer, Layer, Weights, SlidingLayer};
use crate::{Chip, ops, ops_audio, ops_vision};
use crate::host::prof;
use crate::ops_ref;
use std::cell::RefCell;

mod chunk;
mod decode;
mod hybrid;
mod hybrid256;
pub mod lq;
pub mod d13e;
pub use hybrid256::preup_enabled;

pub use hybrid::{DecDesc, DevRows, HintRow, HintRun, HybDec, HybRun, PassDesc, RunDesc, decode_bucket as hybrid_decode_bucket, fits as hybrid_fits, n2_enabled as hybrid_n2_enabled, prebuild_enabled as hybrid_prebuild_enabled};
/// Prefill rows of a hybrid pass must end below this position.
pub const HYBRID_PREFILL_LIMIT: usize = hybrid::PREFILL_LIMIT;
/// Rows of the hybrid batch.
pub const HYBRID_ROWS: usize = hybrid::ROWS;
/// Rows of the 256-row hybrid batch (`S2_PASS_ROWS=256`, `runtime/hybrid256.rs`).
pub const HYBRID256_ROWS: usize = hybrid256::ROWS;
pub use hybrid256::fits as hybrid256_fits;

/// The pass size when `S2_PASS_ROWS` is not set (build 6: 256).
pub const DEFAULT_PASS_ROWS: usize = 256;

static PASS_ROWS_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Largest hybrid pass the worker plans: 256 (DEFAULT since build 6: 240 prefill rows + 16 riders, capacity 18, see `plan256`; a plan below `S2_P256_MIN` rows still runs on the 128-row engine)
/// or 128 (`S2_PASS_ROWS=128` = the fin8 behaviour).
pub fn pass_rows() -> usize {
    let forced = PASS_ROWS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if forced != 0 {
        return forced;
    }
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| match std::env::var("S2_PASS_ROWS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(128) => 128,
        _ => DEFAULT_PASS_ROWS,
    })
}

/// Tests: force the pass size (`0` = back to the environment).
pub fn set_pass_rows(rows: usize) {
    PASS_ROWS_OVERRIDE.store(rows, std::sync::atomic::Ordering::Relaxed);
}

/// `S2_P256_FORCE=1` (tests): every hybrid pass runs on the 256-row engine when `pass_rows()` is 256 (default: only the plans that do not fit 128 rows).
fn force256() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| std::env::var("S2_P256_FORCE").map(|v| v == "1").unwrap_or(false))
}

/// Whether `run_rows` and `d` decode rows fit a hybrid pass of the configured size.
pub fn hybrid_fits_any(run_rows: &[usize], d: usize) -> bool {
    hybrid::fits(run_rows, d) || (pass_rows() >= 256 && hybrid256::fits(run_rows, d))
}

/// Whether the pass with these prefill runs and `d` decode rows is executed by the 256-row engine (instrumentation).
pub fn hybrid_is_256(run_rows: &[usize], d: usize) -> bool {
    is_256(run_rows, d)
}

/// Whether a pass with these runs / decode rows is executed by the 256-row engine.
fn is_256(run_rows: &[usize], d: usize) -> bool {
    pass_rows() >= 256 && (force256() || !hybrid::fits(run_rows, d)) && hybrid256::fits(run_rows, d)
}

pub use crate::host::mm::{EncodedRows, EncodedTile, encode_audio, encode_image};

/// Next-token logits of one request slot.
pub type Logits = HbmTensor<bf16, Chip, m![W]>;
/// Final residual-stream row of one request slot (the input of the final norm + LM head): `H` bf16 values as little-endian bytes ON THE HOST.
/// Rows travel through the host (7.5 KB each way, ~60 us) so the kernels that produce them (token-serial `[H]`, chunk `[64, 4096]`, batched
/// decode `[8, 4096]`) and the ones that consume them (`[H]` logits, `[Rw8, H]` argmax) need no layout adapters.
pub type Hidden = Vec<u8>;
/// One soft-token embedding: a row of text-embedding width, ready to replace a token embedding.
pub type SoftRow = HbmTensor<bf16, Chip, m![H]>;

/// The V17.8 QKV / AO / FFN kernels split activations into FP8 hi/lo with FIXED scales (16 / 128 /
/// 16+256), valid only for the Stage 1 fixture ranges; real activations can overflow (the FFN
/// produces garbage end to end). The baseline kernels are therefore the default; `GEMMA4_REF`
/// may list `qkvopt`, `aoopt`, `ffnopt` to opt into the V17.8 kernels for experiments.
#[allow(dead_code)]
fn use_ref(kernel: &str) -> bool {
    static FLAGS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let flags = FLAGS.get_or_init(|| std::env::var("GEMMA4_REF").unwrap_or_default());
    !flags.split(',').any(|k| k.strip_suffix("opt") == Some(kernel))
}

pub(crate) fn gather_pos_embedding_row(pos_embedding: &[bf16], col: usize, row: usize) -> Vec<bf16> {
    let mv = Mv::SIZE;
    let col = col.min(Pv::SIZE - 1);
    let row = row.min(Pv::SIZE - 1);
    let col_row = &pos_embedding[col * 2 * mv..col * 2 * mv + mv];
    let row_row = &pos_embedding[(row * 2 + 1) * mv..(row * 2 + 1) * mv + mv];
    col_row
        .iter()
        .zip(row_row)
        .map(|(&a, &b)| bf16::from_f32(a.to_f32() + b.to_f32()))
        .collect()
}

fn gather_pos_embedding(weights: &Weights, col: usize, row: usize) -> Vec<bf16> {
    gather_pos_embedding_row(&weights.vision.pos_embedding, col, row)
}

pub async fn encode_vision_patch(
    device: &mut Device,
    weights: &Weights,
    pixels: &[bf16],
    position: (usize, usize),
) -> Result<HbmTensor<bf16, Chip, m![H]>, Error> {
    assert_eq!(pixels.len(), Rv::SIZE, "vision patch must be Rv-wide");
    let x: HbmTensor<bf16, Chip, m![Rv]> = HostTensor::<bf16, m![Rv]>::from_vec(pixels.to_vec())
        .to_hbm(&mut device.pdma)
        .await?;

    let mut patch_out: HbmTensor<bf16, Chip, m![Mv]> =
        HostTensor::<bf16, m![Mv]>::zero().to_hbm(&mut device.pdma).await?;
    prof::timed!("patch_embed", launch(
        ops_vision::patch_embed,
        (
            device,
            &x,
            &weights.vision.patch_ln1_weight,
            &weights.vision.patch_ln1_bias,
            &weights.vision.patch_dense_weight,
            &weights.vision.patch_dense_bias,
            &weights.vision.patch_ln2_weight,
            &weights.vision.patch_ln2_bias,
            &mut patch_out,
        ),
    ).await)?;

    let (col, row) = position;
    let pos_embed: HbmTensor<bf16, Chip, m![Mv]> =
        HostTensor::<bf16, m![Mv]>::from_vec(gather_pos_embedding(weights, col, row))
            .to_hbm(&mut device.pdma)
            .await?;

    let mut normed: HbmTensor<bf16, Chip, m![Mv]> = HostTensor::<bf16, m![Mv]>::zero().to_hbm(&mut device.pdma).await?;
    prof::timed!("add_position_and_norm", launch(
        ops_vision::add_position_and_norm,
        (
            device,
            &patch_out,
            &pos_embed,
            &weights.vision.pos_norm_weight,
            &weights.vision.pos_norm_bias,
            &mut normed,
        ),
    ).await)?;

    let mut projected: HbmTensor<bf16, Chip, m![H]> =
        HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    prof::timed!("project_to_text_embedding", launch(
        ops_vision::project_to_text_embedding,
        (
            device,
            &unsafe { normed.reshape() },
            &weights.vision.embedding_projection_weight,
            &mut projected,
        ),
    ).await)?;

    Ok(projected)
}

pub async fn encode_audio_frame(
    device: &mut Device,
    weights: &Weights,
    samples: &[bf16],
) -> Result<HbmTensor<bf16, Chip, m![H]>, Error> {
    assert_eq!(samples.len(), Aa::SIZE, "audio frame must be Aa-wide");
    let input: HbmTensor<bf16, Chip, m![Aa]> = HostTensor::<bf16, m![Aa]>::from_vec(samples.to_vec())
        .to_hbm(&mut device.pdma)
        .await?;
    let mut output: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    prof::timed!("audio_project_frame", launch(
        ops_audio::audio_project_frame,
        (device, &input, &weights.audio.embedding_projection_weight, &mut output),
    ).await)?;
    Ok(output)
}

/// The original per-patch encoder (three launches per patch, all weights re-read per patch): the
/// reference for the batched `encode_image` and the `GEMMA4_MMENC=old` fallback. One [H] row per patch.
pub async fn encode_image_per_patch(
    device: &mut Device,
    weights: &Weights,
    image: &ImagePatches,
) -> Result<Vec<SoftRow>, Error> {
    let mut rows = Vec::with_capacity(image.pixels.len());
    for (pixels, &position) in image.pixels.iter().zip(&image.positions) {
        rows.push(encode_vision_patch(device, weights, pixels, position).await?);
    }
    Ok(rows)
}

/// The original per-frame audio encoder: one [H] row per frame.
pub async fn encode_audio_per_frame(device: &mut Device, weights: &Weights, audio: &AudioFrames) -> Result<Vec<SoftRow>, Error> {
    let mut rows = Vec::with_capacity(audio.frames.len());
    for frame in &audio.frames {
        rows.push(encode_audio_frame(device, weights, frame).await?);
    }
    Ok(rows)
}

/// The encoder output of one modality of one request.
enum SoftSource {
    /// Batched encoder output (`host::mm`): row-major [64, H] tiles.
    Tiles(EncodedRows),
    /// Per-patch encoder output: one [H] tensor per row, moved out when used.
    Rows(Vec<RefCell<Option<SoftRow>>>),
}

impl SoftSource {
    /// Every row as little-endian bf16 bytes on the host (`len() * H * 2` bytes).
    async fn to_host_bytes(&self, device: &mut Device) -> Result<Vec<u8>, Error> {
        if let SoftSource::Tiles(encoded) = self {
            return encoded.to_host_bytes(device).await;
        }
        let mut out = Vec::with_capacity(self.len() * H::SIZE * 2);
        match self {
            SoftSource::Tiles(_) => unreachable!("returned above"),
            SoftSource::Rows(rows) => {
                for cell in rows {
                    let cell = cell.borrow();
                    let row = cell.as_ref().expect("a soft embedding is read to the host before it is used");
                    let host: HostTensor<bf16, m![H]> = row.to_host(&mut device.pdma).await?;
                    out.extend_from_slice(&host.into_inner().into_buf());
                }
            }
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        match self {
            SoftSource::Tiles(encoded) => encoded.rows,
            SoftSource::Rows(rows) => rows.len(),
        }
    }

    async fn row(&self, device: &mut Device, index: usize) -> Result<SoftRow, Error> {
        match self {
            SoftSource::Tiles(encoded) => encoded.row(device, index).await,
            SoftSource::Rows(rows) => Ok(rows[index].borrow_mut().take().expect("a soft embedding is used once")),
        }
    }
}

/// All soft-token embeddings of one request, computed once at admission (`Model::embed_media`).
///
/// - `row(device, token)` is what the token-serial fallback consumes: one [H] tensor per soft token
///   (a tiny `mm_copy_row` launch out of the batched tile, or the per-patch result moved out).
/// - `tile(token)` is what a chunk kernel wants: the row-major [Tm64, H] tile of the batched encoder and the
///   row inside it, so a chunk can DMA rows straight from `tile` into its [n, H] activations (None for the
///   per-patch path).
pub struct SoftEmbeddings {
    image: Option<SoftSource>,
    audio: Option<SoftSource>,
    /// Host copies of the rows (bf16 LE bytes, row-major `[rows, H]`), filled by `fetch_host` for the chunk path, which assembles chunk inputs on the host.
    host_image: Vec<u8>,
    host_audio: Vec<u8>,
}

impl SoftEmbeddings {
    /// From the batched encoders' output.
    pub fn new(image: Option<EncodedRows>, audio: Option<EncodedRows>) -> Self {
        Self { image: image.map(SoftSource::Tiles), audio: audio.map(SoftSource::Tiles), host_image: Vec::new(), host_audio: Vec::new() }
    }

    /// From per-patch / per-frame rows (`GEMMA4_MMENC=old`).
    pub fn from_rows(image: Option<Vec<SoftRow>>, audio: Option<Vec<SoftRow>>) -> Self {
        let cells = |rows: Vec<SoftRow>| SoftSource::Rows(rows.into_iter().map(|row| RefCell::new(Some(row))).collect());
        Self { image: image.map(cells), audio: audio.map(cells), host_image: Vec::new(), host_audio: Vec::new() }
    }

    pub fn empty() -> Self {
        Self { image: None, audio: None, host_image: Vec::new(), host_audio: Vec::new() }
    }

    /// Copies every row to the host once (chunk path). Call before any `row` of the per-patch variant moves a row out.
    pub async fn fetch_host(&mut self, device: &mut Device) -> Result<(), Error> {
        if let Some(source) = &self.image {
            self.host_image = source.to_host_bytes(device).await?;
        }
        if let Some(source) = &self.audio {
            self.host_audio = source.to_host_bytes(device).await?;
        }
        Ok(())
    }

    /// The host copy of the embedding of `token` (bf16 LE bytes, `H` elements), if `fetch_host` ran.
    pub fn host_row(&self, token: SoftToken) -> Option<&[u8]> {
        let (rows, index) = match token {
            SoftToken::Image(index) => (&self.host_image, index),
            SoftToken::Audio(index) => (&self.host_audio, index),
        };
        let width = H::SIZE * 2;
        rows.get(index * width..(index + 1) * width)
    }

    fn source(&self, token: SoftToken) -> Option<(&SoftSource, usize)> {
        let (source, index) = match token {
            SoftToken::Image(index) => (self.image.as_ref()?, index),
            SoftToken::Audio(index) => (self.audio.as_ref()?, index),
        };
        (index < source.len()).then_some((source, index))
    }

    /// Whether `token` has an encoded row.
    pub fn contains(&self, token: SoftToken) -> bool {
        self.source(token).is_some()
    }

    /// The embedding of one soft token as its own [H] tensor (the per-patch variant gives it up: once per token).
    pub async fn row(&self, device: &mut Device, token: SoftToken) -> Result<SoftRow, Error> {
        let (source, index) = self.source(token).expect("soft token has an encoded row");
        source.row(device, index).await
    }

    /// The batched encoder's tile holding `token` and the row inside it.
    pub fn tile(&self, token: SoftToken) -> Option<(&EncodedTile, usize)> {
        match self.source(token)? {
            (SoftSource::Tiles(encoded), index) => Some((&encoded.tiles[index / crate::host::mm::TILE], index % crate::host::mm::TILE)),
            (SoftSource::Rows(_), _) => None,
        }
    }

    /// (image rows, audio rows)
    pub fn len(&self) -> (usize, usize) {
        (self.image.as_ref().map_or(0, SoftSource::len), self.audio.as_ref().map_or(0, SoftSource::len))
    }
}

/// One input row of a chunk or of a decode batch: a token id, or a soft-token embedding.
#[derive(Clone, Copy)]
pub struct Row<'a> {
    /// The token id (for a soft token, the placeholder id: unused).
    pub token: usize,
    pub soft: Option<(&'a SoftEmbeddings, SoftToken)>,
}

/// One request of a decode batch: its pending token at `pos`, with its own KV blocks.
pub struct DecodeRow<'a> {
    pub row: Row<'a>,
    pub pos: usize,
    pub blocks: &'a Blocks,
    /// Whether this row's next token is needed. Its final hidden row goes to `hidden[blocks.sliding]`.
    pub logits: bool,
}

enum KvCache {
    /// One window per active request, selected by `Blocks::sliding`.
    Sliding {
        k: Vec<HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>>,
        v: Vec<HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>>,
    },
    /// A pool of `Tf`-position blocks shared by all requests, addressed through `Blocks::full`.
    Full {
        k: Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
        v: Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
    },
}

impl KvCache {
    fn as_sliding_mut(
        &mut self,
    ) -> (
        &mut Vec<HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>>,
        &mut Vec<HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>>,
    ) {
        match self {
            KvCache::Sliding { k, v } => (k, v),
            KvCache::Full { .. } => panic!("sliding layer must have a sliding KV cache"),
        }
    }

    fn as_full_mut(
        &mut self,
    ) -> (
        &mut Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
        &mut Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
    ) {
        match self {
            KvCache::Full { k, v } => (k, v),
            KvCache::Sliding { .. } => panic!("full layer must have a full KV cache"),
        }
    }
}

/// Which physical KV storage backs one request: `full[i]` is the pool block holding its
/// positions `i * Tf..(i + 1) * Tf`, and `sliding` selects its sliding-window buffers.
#[derive(Clone, Debug, Default)]
pub struct Blocks {
    pub full: Vec<usize>,
    pub sliding: usize,
    /// Pages of the FLAT full-layer pool (`[Pr, Df]`) the request decodes in, logical order (empty: the request does not use the batched decode path).
    /// The sliding rows of the flat cache are `sliding * 1024 ..` (the request slot). Filled when the prompt is done (`Decode::migrate`).
    pub flat: Vec<usize>,
}

pub struct Workspace {
    cos_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    sin_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    cos_f: HbmTensor<bf16, Chip, m![E, Df]>,
    sin_f: HbmTensor<bf16, Chip, m![E, Df]>,
    mask_s: Vec<HbmTensor<f32, Chip, m![Ts]>>,
    mask_f_line: HbmTensor<f32, Chip, m![Tf]>,
    mask_f_diag: Vec<HbmTensor<f32, Chip, m![Tf]>>,
    kv_cache: Vec<KvCache>,
    capacity: usize,
    requests: usize,
    offset_kv_s: HbmTensor<i32, Chip, m![1]>,
    offset_kv_f: HbmTensor<i32, Chip, m![1]>,
    offset_rope_s: HbmTensor<i32, Chip, m![1]>,
    offset_rope_f: HbmTensor<i32, Chip, m![1]>,
    /// The flat KV caches of the batched decode kernels (None when disabled: more than `Rn` slots, `S2_DECODE_BATCH=0`, or no chunk path).
    flat: Option<decode::FlatCache>,
}

impl Workspace {
    /// Whether requests can decode through the batched path (flat caches allocated).
    pub fn has_flat(&self) -> bool {
        self.flat.is_some()
    }

    /// Context positions per request.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Requests whose KV the workspace can hold at once.
    pub fn requests(&self) -> usize {
        self.requests
    }

    /// Full-attention blocks needed to hold `positions` positions.
    pub fn blocks_for(positions: usize) -> usize {
        positions.div_ceil(Tf::SIZE)
    }

    /// `requests` concurrent requests of up to `capacity` positions each. The full-attention
    /// pool holds exactly `requests * blocks_for(capacity)` blocks per layer.
    pub async fn new(
        device: &mut Device,
        weights: &Weights,
        capacity: usize,
        requests: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if requests == 0 || !(1..=E::SIZE).contains(&capacity) {
            return Err(format!("need at least one request and 1..={} positions", E::SIZE).into());
        }
        let pool = requests * Self::blocks_for(capacity);
        let (cos_s, sin_s) = upload_rope(device, Ds::SIZE, 10_000.0, Ds::SIZE).await?;
        let (cos_f, sin_f) = upload_rope(device, Df::SIZE, 1_000_000.0, 128).await?;
        let mask_s = upload_mask_rows::<m![Ts], _>(device, Ts::SIZE, |r, c| if c <= r { 1.0 } else { 0.0 }).await?;
        let mask_f_line = HostTensor::<f32, m![Tf]>::from_vec(vec![1.0; Tf::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
        let mask_f_diag =
            upload_mask_rows::<m![Tf], _>(device, Tf::SIZE, |r, c| if c <= r { 1.0 } else { 0.0 }).await?;

        let mut kv_cache = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let cache = match layer {
                Layer::Sliding(_) => KvCache::Sliding {
                    k: zero_hbm_vec::<bf16, m![Ts, Ns, Ds]>(device, requests).await?,
                    v: zero_hbm_vec::<bf16, m![Ts, Ns, Ds]>(device, requests).await?,
                },
                Layer::Full(_) => KvCache::Full {
                    k: zero_hbm_vec::<bf16, m![Tf, Df]>(device, pool).await?,
                    v: zero_hbm_vec::<bf16, m![Tf, Df]>(device, pool).await?,
                },
            };
            kv_cache.push(cache);
        }

        let offset_kv_s = zero_offset(device).await?;
        let offset_kv_f = zero_offset(device).await?;
        let offset_rope_s = zero_offset(device).await?;
        let offset_rope_f = zero_offset(device).await?;
        let flat_started = std::time::Instant::now();
        let flat = if decode::enabled(requests, weights) { Some(decode::FlatCache::new(device, weights).await?) } else { None };
        eprintln!("S2 load: flat caches {:.0} ms", flat_started.elapsed().as_secs_f64() * 1e3);

        Ok(Self {
            cos_s,
            sin_s,
            cos_f,
            sin_f,
            mask_s,
            mask_f_line,
            mask_f_diag,
            kv_cache,
            capacity,
            requests,
            offset_kv_s,
            offset_kv_f,
            offset_rope_s,
            offset_rope_f,
            flat,
        })
    }

    /// A kernel writes each field wholesale before it is ever read, but the NPU backend still
    /// needs a real device allocation behind each `&mut` output argument at launch time, so
    /// every field is placed with a throwaway zero write rather than left as `HbmTensor::new()`
    /// (unplaced, for a kernel that returns its output instead of writing a parameter).
    pub async fn begin_decode(&mut self, device: &mut Device) -> Result<Decode, Error> {
        Ok(Decode {
            x: HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?,
            q_s: HostTensor::<bf16, m![Ns, Gs, Ds]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            q_f: HostTensor::<bf16, m![Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            attn_max: HostTensor::<f32, m![Gf]>::zero().to_hbm(&mut device.pdma).await?,
            attn_sum: HostTensor::<f32, m![Gf]>::zero().to_hbm(&mut device.pdma).await?,
            attn_s: HostTensor::<bf16, m![Ns, Gs, Ds]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            attn_f: HostTensor::<bf16, m![Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            chunk: None,
            decode: None,
            hybrid: None,
            hybrid256: None,
        })
    }
}

async fn set_position_offsets(device: &mut Device, ws: &mut Workspace, pos: usize) -> Result<(), Error> {
    let ps = pos % Ts::SIZE;
    let pf = pos % Tf::SIZE;

    let offset_kv_s: HostTensor<i32, m![1]> = HostTensor::from_vec([(ps * Ns::SIZE * Ds::SIZE * 2) as i32]);
    let offset_kv_f: HostTensor<i32, m![1]> = HostTensor::from_vec([(pf * Df::SIZE * 2) as i32]);

    let offset_rope_s: HostTensor<i32, m![1]> = HostTensor::from_vec([(pos * Ds::SIZE * 2) as i32]);
    let offset_rope_f: HostTensor<i32, m![1]> = HostTensor::from_vec([(pos * Df::SIZE * 2) as i32]);

    ws.offset_kv_s = offset_kv_s.to_hbm(&mut device.pdma).await?;
    ws.offset_kv_f = offset_kv_f.to_hbm(&mut device.pdma).await?;

    ws.offset_rope_s = offset_rope_s.to_hbm(&mut device.pdma).await?;
    ws.offset_rope_f = offset_rope_f.to_hbm(&mut device.pdma).await?;
    Ok(())
}

async fn zero_offset(device: &mut Device) -> Result<HbmTensor<i32, Chip, m![1]>, Error> {
    let zero: HostTensor<i32, m![1]> = HostTensor::from_vec([0]);
    zero.to_hbm(&mut device.pdma).await
}

async fn zero_hbm_vec<D: MaterializableScalar + num_traits::Zero, E: M>(
    device: &mut Device,
    count: usize,
) -> Result<Vec<HbmTensor<D, Chip, E>>, Error> {
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        tensors.push(HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await?);
    }
    Ok(tensors)
}

async fn upload_mask_rows<Elt: M, F>(
    device: &mut Device,
    width: usize,
    mut value: F,
) -> Result<Vec<HbmTensor<f32, Chip, Elt>>, Error>
where
    F: FnMut(usize, usize) -> f32,
{
    let mut rows = Vec::with_capacity(width);
    for r in 0..width {
        let row: Vec<f32> = (0..width).map(|c| value(r, c)).collect();
        rows.push(HostTensor::<f32, Elt>::from_vec(row).to_hbm(&mut device.pdma).await?);
    }
    Ok(rows)
}

async fn upload_rope<D: AxisName>(
    device: &mut Device,
    dim: usize,
    theta: f32,
    rotated_dim: usize,
) -> Result<(HbmTensor<bf16, Chip, m![E, D]>, HbmTensor<bf16, Chip, m![E, D]>), Error> {
    let (cos, sin) = rope_data(dim, theta, rotated_dim);
    Ok((
        HostTensor::<bf16, m![E, D]>::from_vec(cos)
            .to_hbm(&mut device.pdma)
            .await?,
        HostTensor::<bf16, m![E, D]>::from_vec(sin)
            .to_hbm(&mut device.pdma)
            .await?,
    ))
}

fn rope_data(dim: usize, theta: f32, rotated_dim: usize) -> (Vec<bf16>, Vec<bf16>) {
    let half = dim / 2;
    let rotated_half = rotated_dim / 2;
    let mut cos = Vec::with_capacity(E::SIZE * dim);
    let mut sin = Vec::with_capacity(E::SIZE * dim);
    for pos in 0..E::SIZE {
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(-freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
    }
    (cos, sin)
}

/// Debug: `S2_DEBUG_LAYERS=n` runs only the first n layers (both paths), to compare the residual stream layer by layer. Default: all layers.
pub(crate) fn debug_layers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("S2_DEBUG_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX))
}

/// Positions (prompt + new tokens) a request may use to decode through the batched path.
pub fn decode_limit() -> usize {
    decode::LIMIT
}

/// Tests: the chunk path returns the final hidden row of chunk row `row` (`None` = the last row) instead of the last one.
pub fn set_debug_row(row: Option<usize>) {
    chunk::DEBUG_ROW.store(row.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}

/// Tests: `true` sends every chunk through the token-serial fallback even when the chunk kernels could take it.
pub fn force_token_serial(on: bool) {
    chunk::FORCE_SERIAL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Final RMSNorm + LM head + soft-cap of one hidden row into `logits` (sampled requests and the reference path).
pub async fn final_logits(
    device: &mut Device,
    weights: &Weights,
    hidden: &Hidden,
    logits: &mut Logits,
) -> Result<(), Error> {
    let row = HostTensor::<bf16, m![H]>::from_buf(hidden.clone()).to_hbm(&mut device.pdma).await?;
    prof::timed!("final_norm_and_logits", launch(
        ops::final_norm_and_logits,
        (device, &row, &weights.final_norm, &weights.embedding_table, logits),
    ).await)
}

pub struct Decode {
    x: HbmTensor<bf16, Chip, m![H]>,
    q_s: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    q_f: HbmTensor<bf16, Chip, m![Gf, Df]>,
    attn_max: HbmTensor<f32, Chip, m![Gf]>,
    attn_sum: HbmTensor<f32, Chip, m![Gf]>,
    attn_s: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    attn_f: HbmTensor<bf16, Chip, m![Gf, Df]>,
    /// The chunk path state (scratch tensors, rope tables), created on first use.
    chunk: Option<chunk::ChunkEngine>,
    /// The batched decode path state, created on first use.
    decode: Option<decode::DecodeEngine>,
    /// The hybrid pass state (prefill runs + decode rows in one 128-row batch), created on first use.
    hybrid: Option<hybrid::HybridEngine>,
    /// The 256-row hybrid pass state, created on first use.
    hybrid256: Option<hybrid256::HybridEngine256>,
}

impl Decode {
    /// CHUNK INTEGRATION POINT. The device work for `rows.len()` CONSECUTIVE positions
    /// `pos0..pos0 + rows.len()` of ONE request: embed every row, run all 48 layers appending each
    /// row's K/V to the request's KV (`blocks`), and, if `hidden` is given, leave the LAST row's final
    /// residual-stream row there (the LM head runs in `Model::next_tokens`).
    ///
    /// Contract the scheduler guarantees (and `Model::prefill_chunk` checks):
    /// - the chunk does not cross a multiple of `Tf` (512), so it never wraps the 1024 sliding ring
    ///   and stays inside one full-attention page;
    /// - `blocks.full` covers position `pos0 + rows.len() - 1`;
    /// - rows are ordinary token rows or soft rows (`Row::soft`); a soft row's embedding is read
    ///   from the request's `SoftEmbeddings` (`row` or `tile`).
    ///
    /// Fallback body: the per-token kernels, one position at a time.
    pub async fn run_chunk(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        blocks: &Blocks,
        pos0: usize,
        rows: &[Row<'_>],
        hidden: Option<&mut Hidden>,
    ) -> Result<(), Error> {
        if chunk::eligible(weights, pos0, rows) {
            if self.chunk.is_none() {
                self.chunk = Some(chunk::ChunkEngine::new(device).await?);
            }
            let engine = self.chunk.as_mut().expect("chunk engine");
            let last = if rows.len() > chunk::ROWS {
                prof::timed!("chunk_kernels", engine.run128(device, ws, weights, blocks, pos0, rows, hidden.is_some()).await)?
            } else {
                prof::timed!("chunk_kernels", engine.run(device, ws, weights, blocks, pos0, rows, hidden.is_some()).await)?
            };
            if let (Some(bytes), Some(hidden)) = (last, hidden) {
                *hidden = bytes;
            }
            return Ok(());
        }
        let mut hidden = hidden;
        for (i, row) in rows.iter().enumerate() {
            let want = if i + 1 == rows.len() { hidden.take() } else { None };
            self.run_row(device, ws, weights, blocks, pos0 + i, row, want).await?;
        }
        Ok(())
    }

    /// HYBRID PASS: prefill runs and decode rows of one iteration in one 128-row batch (see `runtime/hybrid.rs`). Leaves the final hidden rows of the runs / decode rows
    /// that ask for them in `hidden[slot]`.
    pub async fn run_hybrid(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        runs: &[HybRun<'_>],
        decs: &[HybDec<'_>],
        hidden: &mut [Hidden],
        hint: Option<PassDesc>,
        hint_rows: Option<Vec<HintRun>>,
        hint_feed: Option<d13e::FeedPlan>,
        blind: bool,
        late_cb: Option<&mut dyn FnMut(&[usize])>,
    ) -> Result<(), Error> {
        // device rows left by an earlier pass are valid only for the `next_tokens` call that follows that pass
        if let Some(e) = self.hybrid.as_mut() {
            e.dev_rows = None;
        }
        if let Some(e) = self.hybrid256.as_mut() {
            e.dev_rows = None;
        }
        // a decode-only pass (`S2_DECODE_N2`) whose rows no `next_tokens` took before this pass: materialise them (never happens in the worker: rows without flat pages use `run_chunk`)
        if let Some(rows) = self.decode.as_mut().and_then(|e| e.dev_rows.take()) {
            rows.flush(device, hidden).await?;
        }
        let counts: Vec<usize> = runs.iter().map(|r| r.rows.len()).collect();
        let this_256 = is_256(&counts, decs.len());
        // the predicted NEXT pass goes to the engine that will run it (its tables are built on a thread while the device runs this pass)
        let (mut hint128, mut hint256) = (None, None);
        if let Some(h) = hint {
            let hcounts: Vec<usize> = h.runs.iter().map(|r| r.n).collect();
            if is_256(&hcounts, h.decs.len()) {
                hint256 = Some(h);
            } else {
                hint128 = Some(h);
            }
        }
        if self.hybrid.is_none() {
            self.hybrid = Some(hybrid::HybridEngine::new(device).await?);
        }
        if pass_rows() >= 256 && self.hybrid256.is_none() {
            self.hybrid256 = Some(hybrid256::HybridEngine256::new(device).await?);
        }
        if this_256 {
            if let (Some(h), true) = (hint128, hybrid::prebuild_enabled()) {
                self.hybrid.as_mut().expect("hybrid engine").spawn_prebuild(h);
            }
            let engine = self.hybrid256.as_mut().expect("256-row hybrid engine");
            engine.run(device, ws, weights, runs, decs, hidden, hint256, hint_rows, hint_feed, blind, late_cb).await
        } else {
            if let (Some(h), true) = (hint256, hybrid::prebuild_enabled()) {
                self.hybrid256.as_mut().expect("256-row hybrid engine").spawn_prebuild(h);
            }
            let engine = self.hybrid.as_mut().expect("hybrid engine");
            // R15 LQ: the 128-row engine launches synchronously and moves its own buffers; start it on an empty queue
            lq::drain()?;
            engine.run(device, ws, weights, runs, decs, hidden, hint128).await
        }
    }

    /// R14 d13e: see `HybridEngine256::d13e_feed_ready`.
    pub fn d13e_feed_ready(&self, lane_slots: &[usize]) -> bool {
        d13e::level() >= 2 && self.hybrid256.as_ref().is_some_and(|e| e.d13e_feed_ready(lane_slots))
    }

    /// R14 d13e: see `HybridEngine256::d13e_boundary`.
    pub async fn d13e_boundary(&mut self, device: &mut Device, fd: &d13e::FeedDev, keys16: &crate::ops_lmhead::Keys16x2, noread: bool) -> Result<Vec<usize>, Error> {
        self.hybrid256.as_mut().expect("256-row engine").d13e_boundary(device, fd, keys16, noread).await
    }

    /// R14 d13e P2: the request slots of the device rows the last 256-row pass left in its two res2 views (lanes), without taking them.
    pub fn d13e_dev_slots(&self) -> Option<Vec<usize>> {
        self.hybrid256.as_ref().and_then(|e| e.dev_rows.as_ref()).filter(|d| d.views.len() == 2).map(|d| d.slots.clone())
    }

    /// R14 d13e P2: see `HybridEngine256::d13e_blind_ready`.
    pub fn d13e_blind_ready(&self, desc: &PassDesc, requests: &[u64]) -> bool {
        d13e::level() >= 3 && self.hybrid256.as_ref().is_some_and(|e| e.d13e_blind_ready(desc, requests))
    }

    /// R14 d13e P2: see `HybridEngine256::d13e_late_take`.
    pub async fn d13e_late_take(&mut self, device: &mut Device) -> Result<Option<Vec<usize>>, Error> {
        match self.hybrid256.as_mut() {
            Some(e) => e.d13e_late_take(device).await,
            None => Ok(None),
        }
    }

    /// TAKES the decode rows of the last hybrid pass (or of the last decode-only pass, `S2_DECODE_N2`) whose hidden rows are still on the device (`S2_N2`), if any: they are valid only for the `next_tokens` call that follows the pass.
    pub fn take_hybrid_dev_rows(&mut self) -> Option<hybrid::DevRows> {
        let a = self.hybrid.as_mut().and_then(|engine| engine.dev_rows.take());
        let b = self.hybrid256.as_mut().and_then(|engine| engine.dev_rows.take());
        let c = self.decode.as_mut().and_then(|engine| engine.dev_rows.take());
        a.or(b).or(c)
    }

    /// BATCHED-DECODE INTEGRATION POINT. One pending token per request, each at its own position
    /// with its own KV blocks. Row `i` with `logits` leaves its final hidden row in
    /// `hidden[rows[i].blocks.sliding]`, or lends device views to `next_tokens` when `S2_DECODE_N2` is enabled (request slots are distinct, so the buffers never alias).
    ///
    /// Rows of requests that migrated into the flat caches (`Blocks::flat` set) run through the batched path, up to 8 per pass (T = 8 kernels, dattn batched
    /// attention). The other rows (long requests, positions >= 2048 without flat pages, `S2_DECODE_BATCH=0`) run one at a time as one-row chunks through the
    /// 64-row kernels (any position below 2048) or token-serially.
    pub async fn run_decode_batch(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        rows: &[DecodeRow<'_>],
        hidden: &mut [Hidden],
    ) -> Result<(), Error> {
        drop(self.take_hybrid_dev_rows());
        let mut batched: Vec<&DecodeRow<'_>> = Vec::new();
        for row in rows {
            let flat = ws.has_flat()
                && !row.blocks.flat.is_empty()
                && row.row.soft.is_none()
                && row.pos < decode::LIMIT
                && row.pos / Tf::SIZE < row.blocks.flat.len();
            if flat {
                batched.push(row);
                continue;
            }
            let want = row.logits.then(|| &mut hidden[row.blocks.sliding]);
            self.run_chunk(device, ws, weights, row.blocks, row.pos, std::slice::from_ref(&row.row), want).await?;
        }
        let group_count = batched.len().div_ceil(decode::ROWS16);
        for (index, group) in batched.chunks(decode::ROWS16).enumerate() {
            // only the last pass may lend views: later passes reuse its scratch buffers
            let keep_device = index + 1 == group_count && group.iter().all(|row| row.logits);
            if self.decode.is_none() {
                self.decode = Some(decode::DecodeEngine::new(device).await?);
            }
            let engine = self.decode.as_mut().expect("decode engine");
            let out = if group.len() > decode::ROWS {
                prof::timed!("decode_kernels", engine.run16(device, ws, weights, group, keep_device).await)?
            } else {
                prof::timed!("decode_kernels", engine.run(device, ws, weights, group, keep_device).await)?
            };
            if let Some(out) = out {
                for (row, bytes) in group.iter().zip(out) {
                    if row.logits { hidden[row.blocks.sliding] = bytes; }
                }
            } else {
                for row in group { hidden[row.blocks.sliding].clear(); }
            }
        }
        Ok(())
    }

    async fn run_row(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        blocks: &Blocks,
        pos: usize,
        row: &Row<'_>,
        hidden: Option<&mut Hidden>,
    ) -> Result<(), Error> {
        match row.soft {
            Some((embeddings, token)) => {
                self.x = embeddings.row(device, token).await?;
            }
            None => {
                // The device LM-head table is in stream order (no row gather): the row is embedded on the host, bit for bit like `ops::embed_token`
                // (the chunk path does the same for every text row).
                let mut bytes = vec![0u8; H::SIZE * 2];
                chunk::embed_row_into(weights, row, &mut bytes);
                self.x = prof::timed!("embed_token", HostTensor::<bf16, m![H]>::from_buf(bytes).to_hbm(&mut device.pdma).await)?;
            }
        }
        self.run_layers(device, ws, weights, blocks, pos, hidden).await
    }

    async fn run_layers(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        blocks: &Blocks,
        pos: usize,
        hidden: Option<&mut Hidden>,
    ) -> Result<(), Error> {
        assert!(pos < E::SIZE, "position exceeds runtime context");
        prof::timed!("set_offsets", set_position_offsets(device, ws, pos).await)?;
        for (layer_index, layer) in weights.layers.iter().enumerate().take(debug_layers()) {
            match layer {
                Layer::Sliding(layer) => self.sliding(device, ws, blocks, layer_index, layer, pos).await?,
                Layer::Full(layer) => self.full(device, ws, blocks, layer_index, layer, pos).await?,
            }
        }

        if let Some(hidden) = hidden {
            let host: HostTensor<bf16, m![H]> = prof::timed!("hidden_d2h", self.x.to_host(&mut device.pdma).await)?;
            *hidden = host.into_inner().into_buf();
        }
        Ok(())
    }

    async fn sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        blocks: &Blocks,
        layer_index: usize,
        layer: &SlidingLayer,
        pos: usize,
    ) -> Result<(), Error> {
        let mask_row = pos.min(Ts::SIZE - 1);
        let ring = blocks.sliding;
        let (k, v) = ws.kv_cache[layer_index].as_sliding_mut();
        let (k, v) = (&mut k[ring], &mut v[ring]);
        prof::timed!("sliding_project_qkv", { launch(
            ops_ref::sliding_project_qkv_ref,
            (
                device,
                &self.x,
                &layer.q_weight,
                &layer.k_weight,
                &layer.v_weight,
                &layer.q_weight_scale,
                &layer.k_weight_scale,
                &layer.v_weight_scale,
                &layer.input_norm,
                &layer.q_norm,
                &layer.k_norm,
                &ws.offset_kv_s,
                &ws.offset_rope_s,
                &ws.cos_s,
                &ws.sin_s,
                &mut *k,
                &mut *v,
                &mut self.q_s,
            ),
        ).await })?;
        prof::timed!("sliding_attention", launch(
            ops::sliding_attention,
            (device, &self.q_s, &*k, &*v, &ws.mask_s[mask_row], &mut self.attn_s),
        ).await)?;
        prof::timed!("sliding_attention_output", { launch(
            ops_ref::sliding_attention_output_ref,
            (
                device,
                &self.attn_s,
                &layer.post_attention_norm,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.x,
            ),
        ).await })?;
        self.mlp(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn full(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        blocks: &Blocks,
        layer_index: usize,
        layer: &FullLayer,
        pos: usize,
    ) -> Result<(), Error> {
        let page = pos / Tf::SIZE;
        let row = pos % Tf::SIZE;
        assert!(page < blocks.full.len(), "position has no full-attention block");
        let blocks = &blocks.full;
        let (k_cache, v_cache) = ws.kv_cache[layer_index].as_full_mut();
        prof::timed!("full_project_qkv", launch(
            ops::full_project_qkv,
            (
                device,
                &self.x,
                &layer.q_weight,
                &layer.k_weight,
                &layer.q_weight_scale,
                &layer.k_weight_scale,
                &layer.input_norm,
                &layer.q_norm,
                &layer.k_norm,
                &ws.offset_kv_f,
                &ws.offset_rope_f,
                &ws.cos_f,
                &ws.sin_f,
                &mut k_cache[blocks[page]],
                &mut v_cache[blocks[page]],
                &mut self.q_f,
            ),
        ).await)?;

        for kv_page in 0..=page {
            let mask = if kv_page == page {
                &ws.mask_f_diag[row]
            } else {
                &ws.mask_f_line
            };
            let args = (
                &mut *device,
                &self.q_f,
                &k_cache[blocks[kv_page]],
                &v_cache[blocks[kv_page]],
                mask,
                &mut self.attn_max,
                &mut self.attn_sum,
                &mut self.attn_f,
            );
            if kv_page == 0 {
                prof::timed!("full_attention_first_page", launch(ops::full_attention_first_page, args).await)?;
            } else {
                prof::timed!("full_attention_page", launch(ops::full_attention_page, args).await)?;
            }
        }

        prof::timed!("full_attention_output", launch(
            ops::full_attention_output,
            (
                device,
                &self.attn_f,
                &self.attn_sum,
                &layer.post_attention_norm,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.x,
            ),
        ).await)?;
        self.mlp(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn mlp(
        &mut self,
        device: &mut Device,
        mlp: &crate::host::load::MlpWeights,
        pre_feedforward_norm: &HbmTensor<bf16, Chip, m![H]>,
        post_feedforward_norm: &HbmTensor<bf16, Chip, m![H]>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    ) -> Result<(), Error> {
        prof::timed!("decoder_feedforward", { launch(
            ops_ref::decoder_feedforward_ref,
            (
                device,
                &mut self.x,
                pre_feedforward_norm,
                &mlp.up_weight_packed,
                &mlp.gate_weight_packed,
                &mlp.down_weight_packed,
                &mlp.up_weight_scale,
                &mlp.gate_weight_scale,
                &mlp.down_weight_scale,
                &mlp.up_global_scale,
                &mlp.gate_global_scale,
                &mlp.down_global_scale,
                post_feedforward_norm,
                layer_scalar,
            ),
        ).await })
    }
}
