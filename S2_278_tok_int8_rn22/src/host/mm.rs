//! Batched image / audio encoders: `encode_image` and `encode_audio` run the per-patch / per-frame encoder for `TILE` rows at a
//! time, streaming every weight matrix once per tile (kernels in `src/device/mm/gemm.rs`) instead of once per row.
//!
//! Division of work (all activations stay bf16 between stages; every stage rounds to bf16 exactly where the per-row kernels do):
//! * host: builds the tile inputs (pixels, position rows, audio frames) while the previous tile runs;
//! * device: ONE fused launch per tile of `TILE` rows (`device/mm/norm.rs`, `device/mm/gemm.rs`): vision patch LN1, dense (6912 -> 3840, + bias), LN2 + position embedding + LN3,
//!   rmsnorm, projection (3840 -> 3840); audio rmsnorm, projection (640 -> 3840).
//!
//! Output contract (`EncodedRows`): the encoder output is row-major `[rows, H]` bf16 (H = 3840) stored in `tiles`, tile `i` holding rows
//! `i * TILE .. (i + 1) * TILE` (the last tile is zero padded) as an `HbmTensor<bf16, Chip, m![Tm64, H]>`; `EncodedRows::row(i)` copies one
//! row into a fresh `HbmTensor<bf16, Chip, m![H]>` (one tiny launch), which is what the token-serial decoder consumes
//! (`Decode::run_with_embedding`).  A chunked-prefill consumer can read `tiles` directly (plain row-major bf16).
use std::time::Instant;

use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::load::Weights;
use crate::host::prof;
use crate::mm_norm::*;
use crate::mm_pack::*;
use crate::{Chip, ops_mm};

/// Rows per device tile (compile-time axis `Tm64`).
pub const TILE: usize = 64;

pub type EncodedTile = HbmTensor<bf16, Chip, m![Tm64, H]>;

/// Device-resident encoder output: `rows` row-major rows of `H` bf16 in tiles of `TILE` rows.
pub struct EncodedRows {
    pub tiles: Vec<EncodedTile>,
    pub rows: usize,
    /// Where the tiles go when this value is dropped (the pool of the device-norm pipeline, `MmWeights::pool`): an output tile is fully overwritten by its encode launch, so a recycled
    /// tile needs neither a fresh device allocation nor a zero upload.
    pub(crate) pool: Option<TilePool>,
}

pub(crate) type TilePool = std::rc::Rc<std::cell::RefCell<Vec<EncodedTile>>>;

/// Most tiles the pool keeps (a 30 s audio clip is 12 tiles of 0.5 MB; an image 5).
const POOL_MAX: usize = 48;

impl Drop for EncodedRows {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            let mut pool = pool.borrow_mut();
            for tile in self.tiles.drain(..) {
                if pool.len() < POOL_MAX {
                    pool.push(tile);
                }
            }
        }
    }
}

impl EncodedRows {
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Copies row `index` into its own `[H]` tensor (the layer kernels update it in place as the residual stream).
    pub async fn row(&self, device: &mut Device, index: usize) -> Result<HbmTensor<bf16, Chip, m![H]>, Error> {
        assert!(index < self.rows, "encoded row out of range");
        let tile = &self.tiles[index / TILE];
        let offset: HbmTensor<i32, Chip, m![1]> =
            HostTensor::<i32, m![1]>::from_vec([((index % TILE) * H::SIZE * 2) as i32]).to_hbm(&mut device.pdma).await?;
        let mut out: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
        launch(ops_mm::mm_copy_row, (device, tile, &offset, &mut out)).await?;
        Ok(out)
    }

    /// Every row as little-endian bf16 bytes on the host (`rows * H * 2` bytes), one D2H per tile into a recycled host buffer (`.output()`: the 0.5 MB pages stay mapped; measured on the
    /// worker 424 us per tile against 776 us for a fresh host tensor, upbench job 149330).
    pub async fn to_host_bytes(&self, device: &mut Device) -> Result<Vec<u8>, Error> {
        thread_local! {
            static HOST: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let tile_bytes = TILE * H::SIZE * 2;
        let mut out = Vec::with_capacity(self.rows * H::SIZE * 2);
        let mut buf = HOST.with(|h| std::mem::take(&mut *h.borrow_mut()));
        if buf.len() != tile_bytes {
            buf = vec![0u8; tile_bytes];
        }
        let mut result = Ok(());
        for (i, tile) in self.tiles.iter().enumerate() {
            let mut host = HostTensor::<bf16, m![Tm64, H]>::from_buf(buf);
            let r = tile.to_host(&mut device.pdma).output(&mut host).await;
            buf = host.into_inner().into_buf();
            if let Err(error) = r {
                result = Err(error);
                break;
            }
            let rows = (self.rows - i * TILE).min(TILE);
            out.extend_from_slice(&buf[..rows * H::SIZE * 2]);
        }
        HOST.with(|h| *h.borrow_mut() = buf);
        result.map(|()| out)
    }

    /// Reads every row back to the host as f32 (tests only).
    pub async fn to_host_f32(&self, device: &mut Device) -> Result<Vec<f32>, Error> {
        let mut out = Vec::with_capacity(self.rows * H::SIZE);
        for (i, tile) in self.tiles.iter().enumerate() {
            let host: HostTensor<bf16, m![Tm64, H]> = tile.to_host(&mut device.pdma).await?;
            let rows = (self.rows - i * TILE).min(TILE);
            let bytes = host.into_inner().into_buf();
            out.extend(bytes.chunks_exact(2).take(rows * H::SIZE).map(|b| bf_to_f32(u16::from_le_bytes([b[0], b[1]]))));
        }
        Ok(out)
    }
}

/// Persistent device scratch of the device-norm pipeline (`device/mm/norm.rs`): the dense input (its pad columns must stay zero: the stage kernels never write them), the dense output,
/// the projection input and the audio projection input.  One tile at a time: launches are serialised.
pub(crate) struct MmScratch {
    xn1: HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
    y: HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
    xn2: HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
    xna: HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
    /// Persistent device inputs of a tile (written with `.output()`: no fresh device allocation per transfer) and the host byte buffers they are filled from (recycled through
    /// `HostTensor::into_inner().into_buf()`, so the pages stay mapped and warm).
    px_dev: HbmTensor<bf16, Chip, m![Tm64, Rv]>,
    pos_dev: HbmTensor<bf16, Chip, m![Tm64, Mv]>,
    frames_dev: HbmTensor<bf16, Chip, m![Tm64, Aa]>,
    px_host: Vec<u8>,
    pos_host: Vec<u8>,
    frames_host: Vec<u8>,
}

/// Packed device weights of the media encoder.
pub(crate) struct MmWeights {
    /// Allocated on the first media request (`scratch_of`), NOT at load: seven persistent tensors (3.36 MiB) allocated before the 48 decoder layers shift every layer-weight address
    /// and cost the text path +0.64 ms per pass in `h_layer_s` (v1e result; see RESULTS.md).
    scratch: std::cell::RefCell<Option<MmScratch>>,
    pool: TilePool,
    dense_w: Vec<HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>>,
    dense_b: Vec<HbmTensor<bf16, Chip, m![Mg, Rt120]>>,
    proj_w: HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    audio_w: HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
}

/// The persistent scratch tensors, zero-initialised.
async fn new_scratch(device: &mut Device) -> Result<MmScratch, Error> {
    // The dense input is `[Tm64, 32 x 224]` for the GEMM and `[Tm64, Rv # 7168]` for the LN1 writer (6912 live columns): same bytes, one reshape.
    let xn1: HbmTensor<bf16, Chip, m![Tm64, Mc, Ev]> = HostTensor::<bf16, m![Tm64, Mc, Ev]>::zero().to_hbm(&mut device.pdma).await?;
    Ok(MmScratch {
        xn1: unsafe { xn1.reshape() },
        y: HostTensor::<bf16, m![Tm64, Mg, Dummy2, Rt120]>::zero().to_hbm(&mut device.pdma).await?,
        xn2: HostTensor::<bf16, m![Tm64, Mc, Ep]>::zero().to_hbm(&mut device.pdma).await?,
        xna: HostTensor::<bf16, m![Tm64, Mc, Ea]>::zero().to_hbm(&mut device.pdma).await?,
        px_dev: HostTensor::<bf16, m![Tm64, Rv]>::zero().to_hbm(&mut device.pdma).await?,
        pos_dev: HostTensor::<bf16, m![Tm64, Mv]>::zero().to_hbm(&mut device.pdma).await?,
        frames_dev: HostTensor::<bf16, m![Tm64, Aa]>::zero().to_hbm(&mut device.pdma).await?,
        px_host: vec![0u8; TILE * Rv::SIZE * 2],
        pos_host: vec![0u8; TILE * Mv::SIZE * 2],
        frames_host: vec![0u8; TILE * Aa::SIZE * 2],
    })
}

/// The scratch, allocated on first use.  Encodes are serialised by the caller (the guard is held across awaits, as before), so the check-then-fill cannot race.
async fn scratch_of<'a>(device: &mut Device, mm: &'a MmWeights) -> Result<std::cell::RefMut<'a, MmScratch>, Error> {
    if mm.scratch.borrow().is_none() {
        let fresh = new_scratch(device).await?;
        *mm.scratch.borrow_mut() = Some(fresh);
    }
    Ok(std::cell::RefMut::map(mm.scratch.borrow_mut(), |slot| slot.as_mut().expect("scratch allocated above")))
}

pub(crate) async fn load_mm(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<MmWeights, Box<dyn std::error::Error>> {
    let (mv, rv, ov, aa) = (Mv::SIZE, Rv::SIZE, Ov::SIZE, Aa::SIZE);
    let dense = tensors.tensor("model.vision_embedder.patch_dense.weight")?;
    let dense_bias = tensors.tensor("model.vision_embedder.patch_dense.bias")?;
    let proj = tensors.tensor("model.embed_vision.embedding_projection.weight")?;
    let audio = tensors.tensor("model.embed_audio.embedding_projection.weight")?;
    for (name, view, shape) in [
        ("dense", &dense, [mv, rv]),
        ("proj", &proj, [H::SIZE, ov]),
        ("audio", &audio, [H::SIZE, aa]),
    ] {
        if view.dtype() != safetensors::Dtype::BF16 || view.shape() != shape {
            return Err(format!("{name} weight: expected BF16 {shape:?}, got {:?} {:?}", view.dtype(), view.shape()).into());
        }
    }
    let mut dense_w = Vec::new();
    let mut dense_b = Vec::new();
    for tile in 0..2 {
        dense_w.push(
            HostTensor::<bf16, m![Mg, Mc, Rt120, Ev]>::from_buf(pack_weight_tile(dense.data(), mv, rv, tile, 120, Ev::SIZE))
                .to_hbm(&mut device.pdma)
                .await?,
        );
        dense_b.push(
            HostTensor::<bf16, m![Mg, Rt120]>::from_buf(pack_bias_tile(dense_bias.data(), mv, tile, 120))
                .to_hbm(&mut device.pdma)
                .await?,
        );
    }
    let proj_w = HostTensor::<bf16, m![Mg, Mc, Rt240, Ep]>::from_buf(pack_weight_tile(proj.data(), H::SIZE, ov, 0, 240, Ep::SIZE))
        .to_hbm(&mut device.pdma)
        .await?;
    let audio_w =
        HostTensor::<bf16, m![Mg, Mc, Rt240, Ea]>::from_buf(pack_weight_tile(audio.data(), H::SIZE, aa, 0, 240, Ea::SIZE))
            .to_hbm(&mut device.pdma)
            .await?;
    Ok(MmWeights {
        scratch: std::cell::RefCell::new(None),
        pool: TilePool::default(),
        dense_w,
        dense_b,
        proj_w,
        audio_w,
    })
}

// ---------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------
// Device-norm pipeline: every normalisation runs on the device (`device/mm/norm.rs`), a tile never leaves it.

// ---------------------------------------------------------------------------------------------------------------------
// Optimised device-norm pipeline (default).  Measured on the Arena worker (job 149265) the plain device-norm encoder spends per 64-row tile 1.24 ms in host prep (fresh `vec!` pages + scalar
// loops), 0.90 + 0.39 ms in the two fresh-allocation uploads and 0.3 ms on a zero upload for the output tile, against 0.28 ms of device work: the device part is 10 % of the encode wall time.
// This version keeps persistent device inputs written with `.output()`, recycles the host byte buffers, takes output tiles from a pool (a tile is fully overwritten by its launch) and builds
// the host inputs of tile k + 1 WHILE tile k executes on the device (`launch` submits and then awaits the completion, so a join of the launch and the host work overlaps them on one thread).

/// The raw bytes of a bf16 slice (little-endian host: the device byte order).
fn bf_slice_bytes(v: &[bf16]) -> &[u8] {
    assert_eq!(std::mem::size_of::<bf16>(), 2);
    // SAFETY: bf16 is a plain 16-bit scalar; the byte view covers exactly the slice and is only read.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Pixel tile: rows `start..start + rows` of the image (raw bf16 bytes, `[TILE, Rv]`), the remaining rows zero.
fn fill_px(buf: &mut [u8], image: &ImagePatches, start: usize, rows: usize) {
    let row_bytes = Rv::SIZE * 2;
    assert_eq!(buf.len(), TILE * row_bytes);
    for i in 0..rows {
        let src = &image.pixels[start + i];
        assert_eq!(src.len(), Rv::SIZE, "a patch has {} pixels", Rv::SIZE);
        buf[i * row_bytes..(i + 1) * row_bytes].copy_from_slice(bf_slice_bytes(src));
    }
    buf[rows * row_bytes..].fill(0);
}

/// `dst[i] = bf16(a[i] + b[i])` (f32 add of the two bf16 values, round to nearest even; NaN -> quiet NaN, which finite embeddings never produce).  Branch-free so it vectorises.
#[inline(always)]
fn add_bf16_rows(dst: &mut [u16], a: &[u16], b: &[u16]) {
    for ((d, &a), &b) in dst.iter_mut().zip(a).zip(b) {
        let sum = f32::from_bits((a as u32) << 16) + f32::from_bits((b as u32) << 16);
        let bits = sum.to_bits();
        let rounded = (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
        let nan = ((sum != sum) as u16).wrapping_neg();
        *d = (rounded & !nan) | (0x7FC0 & nan);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_bf16_rows_avx2(dst: &mut [u16], a: &[u16], b: &[u16]) {
    add_bf16_rows(dst, a, b)
}

fn add_bf16_rows_best(dst: &mut [u16], a: &[u16], b: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked above).
        return unsafe { add_bf16_rows_avx2(dst, a, b) };
    }
    add_bf16_rows(dst, a, b)
}

/// The bits of a bf16 slice as `u16` words (bf16 is a plain 16-bit scalar).
fn bf_bits(v: &[bf16]) -> &[u16] {
    assert_eq!(std::mem::size_of::<bf16>(), 2);
    assert_eq!(v.as_ptr() as usize % std::mem::align_of::<u16>(), 0);
    // SAFETY: same size and alignment, plain scalar, read only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u16, v.len()) }
}

/// Position tile `[TILE, Mv]`: row i = bf16(column embedding + row embedding) of patch `start + i` (what `gather_pos_embedding_row` returns), the remaining rows zero.
fn fill_pos(buf: &mut [u8], pos_embedding: &[bf16], positions: &[(usize, usize)], start: usize, rows: usize) {
    let mv = Mv::SIZE;
    assert_eq!(buf.len(), TILE * mv * 2);
    assert_eq!(buf.as_ptr() as usize % std::mem::align_of::<u16>(), 0);
    // SAFETY: `buf` is 2-byte aligned (asserted), its length is a multiple of 2 and `words` is the only live view of it.
    let words: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len() / 2) };
    let table = bf_bits(pos_embedding);
    for i in 0..rows {
        let (col, row) = positions[start + i];
        let (col, row) = (col.min(Pv::SIZE - 1), row.min(Pv::SIZE - 1));
        let a = &table[col * 2 * mv..col * 2 * mv + mv];
        let b = &table[(row * 2 + 1) * mv..(row * 2 + 1) * mv + mv];
        add_bf16_rows_best(&mut words[i * mv..(i + 1) * mv], a, b);
    }
    words[rows * mv..].fill(0);
}

/// Audio tile: frames `start..start + rows` (raw bf16 bytes, `[TILE, Aa]`), the remaining rows zero.
fn fill_frames(buf: &mut [u8], audio: &AudioFrames, start: usize, rows: usize) {
    let row_bytes = Aa::SIZE * 2;
    assert_eq!(buf.len(), TILE * row_bytes);
    for i in 0..rows {
        let src = &audio.frames[start + i];
        assert_eq!(src.len(), Aa::SIZE, "a frame has {} samples", Aa::SIZE);
        buf[i * row_bytes..(i + 1) * row_bytes].copy_from_slice(bf_slice_bytes(src));
    }
    buf[rows * row_bytes..].fill(0);
}

/// Uploads `host` into the persistent device tensor `dev` and gives the byte buffer back for the next fill.
async fn upload_into<E: M>(device: &mut Device, host: &mut Vec<u8>, dev: &mut HbmTensor<bf16, Chip, E>) -> Result<(), Error> {
    let tensor = HostTensor::<bf16, E>::from_buf(std::mem::take(host));
    let result = tensor.to_hbm(&mut device.pdma).output(dev).await;
    *host = tensor.into_inner().into_buf();
    result
}

/// An output tile: recycled from the pool, else a fresh allocation.
async fn output_tile(device: &mut Device, pool: &TilePool) -> Result<HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>, Error> {
    let recycled = pool.borrow_mut().pop();
    match recycled {
        // SAFETY: [Tm64, H] and [Tm64, Mg, Rt240] are the same row-major layout (H = 16 x 240).
        Some(tile) => Ok(unsafe { tile.reshape::<Chip, m![Tm64, Mg, Rt240]>() }),
        None => prof::timed!("mm_zero_out", HostTensor::<bf16, m![Tm64, Mg, Rt240]>::zero().to_hbm(&mut device.pdma).await),
    }
}

/// Launches the kernels of one image tile (one fused launch).
#[allow(clippy::too_many_arguments)]
async fn launch_image_tile(
    device: &mut Device,
    weights: &Weights,
    px: &HbmTensor<bf16, Chip, m![Tm64, Rv]>,
    pos: &HbmTensor<bf16, Chip, m![Tm64, Mv]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
    xn2: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
    out: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) -> Result<(), Error> {
    let (mm, vis) = (&weights.mm, &weights.vision);
    prof::timed!("mm_image_launch", launch(
        ops_mm::mm_image_tile,
        (
            &mut *device,
            px,
            pos,
            &vis.patch_ln1_weight,
            &vis.patch_ln1_bias,
            &mm.dense_w[0],
            &mm.dense_w[1],
            &mm.dense_b[0],
            &mm.dense_b[1],
            &vis.patch_ln2_weight,
            &vis.patch_ln2_bias,
            &vis.pos_norm_weight,
            &vis.pos_norm_bias,
            &mm.proj_w,
            xn1,
            y,
            xn2,
            out,
        ),
    )
    .await)
}

/// Launches the kernels of one audio tile (one fused launch).
async fn launch_audio_tile(
    device: &mut Device,
    weights: &Weights,
    frames: &HbmTensor<bf16, Chip, m![Tm64, Aa]>,
    xna: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
    out: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) -> Result<(), Error> {
    let mm = &weights.mm;
    prof::timed!("mma_launch", launch(ops_mm::mm_audio_tile, (&mut *device, frames, &mm.audio_w, xna, out)).await)
}

/// Encodes all patches of an image on the device (optimised pipeline, see above): per 64-row tile one upload of pixels and one of positions into persistent device tensors, one fused launch
/// into a pooled output tile, with the host inputs of the next tile built while the launch runs.
pub async fn encode_image(device: &mut Device, weights: &Weights, image: &ImagePatches) -> Result<EncodedRows, Error> {
    let started = Instant::now();
    let mm = &weights.mm;
    let total = image.pixels.len();
    let pos_embedding = &weights.vision.pos_embedding;
    let mut scratch = scratch_of(&mut *device, mm).await?;
    let MmScratch { xn1, y, xn2, px_dev, pos_dev, px_host, pos_host, .. } = &mut *scratch;
    let n_tiles = total.div_ceil(TILE);
    let mut tiles = Vec::with_capacity(n_tiles);
    if n_tiles > 0 {
        prof::timed!("mm_prep", {
            fill_px(px_host, image, 0, total.min(TILE));
            fill_pos(pos_host, pos_embedding, &image.positions, 0, total.min(TILE));
        });
    }
    for k in 0..n_tiles {
        prof::timed!("mm_h2d_px", upload_into(device, px_host, px_dev).await)?;
        prof::timed!("mm_h2d_pos", upload_into(device, pos_host, pos_dev).await)?;
        let mut out = output_tile(device, &mm.pool).await?;
        let next = (k + 1) * TILE;
        if next < total {
            let rows = (total - next).min(TILE);
            // The launch is submitted at its first poll and awaited asynchronously: the host buffers of the next tile are filled while the device works.
            let (launched, ()) = tokio::join!(
                launch_image_tile(device, weights, &*px_dev, &*pos_dev, &mut *xn1, &mut *y, &mut *xn2, &mut out),
                async {
                    prof::timed!("mm_prep", {
                        fill_px(px_host, image, next, rows);
                        fill_pos(pos_host, pos_embedding, &image.positions, next, rows);
                    });
                }
            );
            launched?;
        } else {
            launch_image_tile(device, weights, &*px_dev, &*pos_dev, &mut *xn1, &mut *y, &mut *xn2, &mut out).await?;
        }
        // SAFETY: same row-major layout (H = 16 x 240).
        tiles.push(unsafe { out.reshape::<Chip, m![Tm64, H]>() });
    }
    prof::add("mm_image_encode_opt", started.elapsed());
    Ok(EncodedRows { tiles, rows: total, pool: Some(mm.pool.clone()) })
}

/// Encodes all frames of an audio clip on the device (optimised pipeline): one fused launch per 64-row tile (rmsnorm -> projection).
pub async fn encode_audio(device: &mut Device, weights: &Weights, audio: &AudioFrames) -> Result<EncodedRows, Error> {
    let started = Instant::now();
    let mm = &weights.mm;
    let total = audio.frames.len();
    let mut scratch = scratch_of(&mut *device, mm).await?;
    let MmScratch { xna, frames_dev, frames_host, .. } = &mut *scratch;
    let n_tiles = total.div_ceil(TILE);
    let mut tiles = Vec::with_capacity(n_tiles);
    if n_tiles > 0 {
        prof::timed!("mma_prep", fill_frames(frames_host, audio, 0, total.min(TILE)));
    }
    for k in 0..n_tiles {
        prof::timed!("mma_h2d", upload_into(device, frames_host, frames_dev).await)?;
        let mut out = output_tile(device, &mm.pool).await?;
        let next = (k + 1) * TILE;
        if next < total {
            let rows = (total - next).min(TILE);
            let (launched, ()) = tokio::join!(launch_audio_tile(device, weights, &*frames_dev, &mut *xna, &mut out), async {
                prof::timed!("mma_prep", fill_frames(frames_host, audio, next, rows));
            });
            launched?;
        } else {
            launch_audio_tile(device, weights, &*frames_dev, &mut *xna, &mut out).await?;
        }
        // SAFETY: same row-major layout (H = 16 x 240).
        tiles.push(unsafe { out.reshape::<Chip, m![Tm64, H]>() });
    }
    prof::add("mm_audio_encode_opt", started.elapsed());
    Ok(EncodedRows { tiles, rows: total, pool: Some(mm.pool.clone()) })
}

