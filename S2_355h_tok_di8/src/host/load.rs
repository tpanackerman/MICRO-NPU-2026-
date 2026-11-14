use std::fs::File;
use std::path::{Path, PathBuf};

use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Df, Ds, Dummy2, H, L, Mv, Ov, Pf, Ps, Pv, Qf, Qs, Rv, W};
use crate::{Chip, LAYERS};

pub(crate) struct MlpWeights {
    pub up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]>,
    pub up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    pub up_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub gate_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub down_global_scale: HbmTensor<f32, Chip, m![1]>,
}

pub(crate) struct SlidingLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    pub k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    pub v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    pub q_weight_scale: HbmTensor<bf16, Chip, m![Qs]>,
    pub k_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    pub v_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
    pub rep: SlRep,
}

/// rep2: per-consumer-slice copies of the constants the fused / native T = 256 sliding launches read (`ffnc::NormRep` / `ScalarRep`), same values.
pub(crate) struct SlRep {
    pub post_attention_norm: crate::device::ffnc::NormRep,
    pub pre_feedforward_norm: crate::device::ffnc::NormRep,
    pub post_feedforward_norm: crate::device::ffnc::NormRep,
    pub layer_scalar: crate::device::ffnc::ScalarRep,
    pub input_norm: crate::qkv8::InNormRep,
    pub ffc: FfcRep,
}

/// rep4: constants of the non-fused T = 256 bf16 layer (`ops_fl::layer_{sliding,full}_t256r2`): checkpoint up / gate global scales, layer scalar.
pub(crate) struct FfcRep {
    pub up_gs: crate::device::ffnc::GsRep,
    pub gate_gs: crate::device::ffnc::GsRep,
    pub layer_scalar: crate::device::ffnc::ScalarRep256,
}

async fn ffc_rep(device: &mut Device, mlp: &MlpWeights, layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>) -> Result<FfcRep, Box<dyn std::error::Error>> {
    Ok(FfcRep {
        up_gs: rep_f32(device, &mlp.up_global_scale, 8 * 512).await?,
        gate_gs: rep_f32(device, &mlp.gate_global_scale, 8 * 512).await?,
        // ScalarRep256 = [Tk256 / 128, Tk256 % 128, H / 1920] = 512 groups of `1 # 8`
        layer_scalar: rep_bf16(device, layer_scalar, 512).await?,
    })
}

/// rep4: a scalar f32 (`m![1]`) copied `n` times (every `1 # 8` group of `R`).
async fn rep_f32<R: M>(device: &mut Device, t: &HbmTensor<f32, Chip, m![1]>, n: usize) -> Result<HbmTensor<f32, Chip, R>, Box<dyn std::error::Error>> {
    let host: HostTensor<f32, m![1]> = t.to_host(&mut device.pdma).await?;
    let v = host.into_vec()[0];
    Ok(HostTensor::<f32, R>::from_vec(vec![v; n]).to_hbm(&mut device.pdma).await?)
}

/// rep2: `n` contiguous copies of a device constant (read back once at load).
async fn rep_bf16<S: M, R: M>(device: &mut Device, t: &HbmTensor<bf16, Chip, S>, n: usize) -> Result<HbmTensor<bf16, Chip, R>, Box<dyn std::error::Error>> {
    let host: HostTensor<bf16, S> = t.to_host(&mut device.pdma).await?;
    Ok(HostTensor::<bf16, R>::from_vec(host.into_vec().repeat(n)).to_hbm(&mut device.pdma).await?)
}

pub(crate) struct FullLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub q_weight: HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    pub k_weight: HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    pub q_weight_scale: HbmTensor<bf16, Chip, m![Qf]>,
    pub k_weight_scale: HbmTensor<bf16, Chip, m![Pf]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
    /// rep2: the input RMSNorm weight per token slice of the full quad QKV (`qkv8::InNormRep`)
    pub input_norm_rep: crate::qkv8::InNormRep,
    pub ffc: FfcRep,
}

pub(crate) enum Layer {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

pub(crate) struct VisionWeights {
    pub patch_ln1_weight: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_ln1_bias: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_dense_weight: HbmTensor<bf16, Chip, m![Mv, Rv]>,
    pub patch_dense_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Ov]>,
    pub pos_embedding: Vec<bf16>,
}

pub(crate) struct AudioWeights {
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Aa]>,
}

/// The text embedding table `[W, H]` bf16 as the checkpoint stores it, memory-mapped on the host: the chunk path builds the embedding
/// rows of a chunk on the host (the rows are scaled like `ops::embed_token` and uploaded together with the chunk).
pub(crate) struct EmbedHost {
    mmap: memmap2::Mmap,
    offset: usize,
}

impl EmbedHost {
    /// The little-endian bf16 bytes of the embedding row of `token` (`H` elements).
    pub(crate) fn row(&self, token: usize) -> &[u8] {
        let width = H::SIZE * 2;
        let start = self.offset + token * width;
        &self.mmap[start..start + width]
    }
}

pub struct Weights {
    pub(crate) model_dir: PathBuf,
    pub(crate) embedding_table: crate::ops_lmhead::LmTable,
    pub(crate) final_norm: HbmTensor<bf16, Chip, m![H]>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) vision: VisionWeights,
    pub(crate) audio: AudioWeights,
    pub(crate) mm: crate::host::mm::MmWeights,
    /// Packed copies for the chunk kernels (empty when `S2_CHUNKPATH=0`).
    pub(crate) chunk: Vec<crate::host::chunkw::ChunkLayer>,
    pub(crate) embed_host: EmbedHost,
}

async fn load_bf16<E: M>(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<bf16, E> = HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    Ok(host.to_hbm(&mut device.pdma).await?)
}

/// The checkpoint's `[W, H]` embedding table (little-endian bf16 bytes) permuted into LM-head STREAM ORDER (`shared::lm_head::LmTableMap`): per band of 16384
/// rows the 256 B chunks (128 columns) are written `[W % 32, H / 128, cluster, slice, H % 128]`. A pure byte permutation, so the logits are unchanged;
/// the 16 bands are independent and are permuted by a few threads (the source rows of one `(cluster, slice)` group are 32 consecutive rows).
pub(crate) fn lm_table_bytes(src: &[u8]) -> Vec<u8> {
    let band_bytes = LM_BAND_ROWS * H::SIZE * 2;
    assert_eq!(src.len(), W::SIZE * H::SIZE * 2);
    let mut out = vec![0u8; src.len()];
    {
        let work = std::sync::Mutex::new(out.chunks_mut(band_bytes).enumerate().collect::<Vec<_>>());
        let workers = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 8);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let Some((band, dst)) = work.lock().expect("lm table work queue").pop() else { break };
                        lm_band_bytes(&src[band * band_bytes..(band + 1) * band_bytes], dst);
                    }
                });
            }
        });
    }
    out
}

const LM_BAND_ROWS: usize = 16384;

/// One band of `lm_table_bytes`: `src` = 16384 natural rows, `dst` = the same bytes in `[W % 32, H / 128, cluster, slice, H % 128]` order.
/// CPU loop order: blocks of `LM_BLOCK` (cluster, slice) groups = `LM_BLOCK * 32` consecutive source rows (read front to back, cache resident across the 32 `r32`
/// passes); every source row is read whole and its 30 chunks go to 30 destination streams 128 KB apart that advance by 256 B per row.
fn lm_band_bytes(src: &[u8], dst: &mut [u8]) {
    const CHUNK: usize = 256;
    const LM_BLOCK: usize = 16;
    let row_bytes = H::SIZE * 2;
    let chunks = row_bytes / CHUNK;
    assert_eq!(row_bytes % CHUNK, 0);
    assert_eq!((src.len(), dst.len()), (LM_BAND_ROWS * row_bytes, LM_BAND_ROWS * row_bytes));
    for cs0 in (0..512).step_by(LM_BLOCK) {
        for r32 in 0..32 {
            for cs in cs0..cs0 + LM_BLOCK {
                // cs = cluster * 256 + slice, and the band row is cluster * 8192 + slice * 32 + r32 = cs * 32 + r32
                let row = &src[(cs * 32 + r32) * row_bytes..(cs * 32 + r32 + 1) * row_bytes];
                for j in 0..chunks {
                    let d = ((r32 * chunks + j) * 512 + cs) * CHUNK;
                    dst[d..d + CHUNK].copy_from_slice(&row[j * CHUNK..(j + 1) * CHUNK]);
                }
            }
        }
    }
}

#[cfg(test)]
mod lm_table_tests {
    use super::*;

    /// Every sampled `(w, h)` of a band lands at the element offset of the mapping `m![W / 16384, W % 32, H / 128, W / 8192 % 2, W / 32 % 256, H % 128]`.
    #[test]
    fn band_matches_mapping() {
        let row_bytes = H::SIZE * 2;
        let src: Vec<u8> = (0..LM_BAND_ROWS * row_bytes).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        lm_band_bytes(&src, &mut dst);
        let mut state = 12345u64;
        for _ in 0..200_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let w = (state >> 33) as usize % LM_BAND_ROWS;
            let h = (state >> 11) as usize % H::SIZE;
            let (r32, cluster, slice) = (w % 32, w / 8192 % 2, w / 32 % 256);
            let (j, h128) = (h / 128, h % 128);
            let element = ((((r32 * (H::SIZE / 128) + j) * 2 + cluster) * 256 + slice) * 128) + h128;
            assert_eq!(&dst[element * 2..element * 2 + 2], &src[(w * H::SIZE + h) * 2..(w * H::SIZE + h) * 2 + 2]);
        }
        let mut sorted_src = src.clone();
        let mut sorted_dst = dst.clone();
        sorted_src.sort_unstable();
        sorted_dst.sort_unstable();
        assert!(sorted_src == sorted_dst, "not a permutation");
    }
}

/// The LM-head table on the device, in stream order (see `lm_table_bytes`); the token embeddings come from the host copy (`Weights::embed_host`).
async fn load_lm_table(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<crate::ops_lmhead::LmTable, Box<dyn std::error::Error>> {
    let name = "model.language_model.embed_tokens.weight";
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::BF16 || view.shape() != [W::SIZE, H::SIZE] {
        return Err(format!("{name}: unexpected embedding table layout").into());
    }
    let host: HostTensor<bf16, crate::device::shared::lm_head::LmTableMap> = HostTensor::from_buf(lm_table_bytes(view.data()));
    Ok(host.to_hbm(&mut device.pdma).await?)
}

async fn load_scalar_bf16(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, m![1 # 8]>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<bf16, m![1]> =
        HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    let value = host.into_vec()[0];
    Ok(HostTensor::<bf16, m![1 # 8]>::from_vec(vec![value; 8])
        .to_hbm(&mut device.pdma)
        .await?)
}

async fn load_bf16_column<E: M>(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::BF16 || view.shape() != [E::SIZE, 1] {
        return Err(format!(
            "{name} is {:?} {:?}, expected BF16 [{}, 1]",
            view.dtype(),
            view.shape(),
            E::SIZE
        )
        .into());
    }
    let host: HostTensor<bf16, E> = HostTensor::from_buf(view.data().to_vec());
    Ok(host.to_hbm(&mut device.pdma).await?)
}

async fn load_f8<E: M>(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<f8e4m3, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<f8e4m3, E> =
        HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    Ok(host.to_hbm(&mut device.pdma).await?)
}

async fn load_f4<E: M>(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    shape: [usize; 2],
) -> Result<HbmTensor<f4e2m1, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let packed = [shape[0], shape[1] / 2];
    if view.dtype() != safetensors::Dtype::U8 || view.shape() != packed {
        return Err(format!(
            "{name} is {:?} {:?}, expected U8 {packed:?}",
            view.dtype(),
            view.shape()
        )
        .into());
    }
    let host: HostTensor<f4e2m1, E> = HostTensor::from_buf(view.data().to_vec());
    Ok(host.to_hbm(&mut device.pdma).await?)
}

fn read_f32(tensors: &safetensors::SafeTensors<'_>, name: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::F32 || view.data().len() != 4 {
        return Err(format!("{name} is {:?}, not a scalar f32 tensor", view.dtype()).into());
    }
    Ok(f32::from_le_bytes(view.data().try_into()?))
}

fn reciprocal_global_scale(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let value = read_f32(tensors, name)?;
    if value == 0.0 || !value.is_finite() {
        return Err(format!("invalid NVFP4 global scale {name}: {value}").into());
    }
    Ok(1.0 / value)
}

async fn load_global_scale(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<f32, Chip, m![1]>, Box<dyn std::error::Error>> {
    let value = reciprocal_global_scale(tensors, name)?;
    Ok(HostTensor::<f32, m![1]>::from_vec(vec![value])
        .to_hbm(&mut device.pdma)
        .await?)
}

pub fn model_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = std::env::var_os("RNGD_MODEL_DIR")
        .ok_or("RNGD_MODEL_DIR is not set; point it at the model directory or its parent")?;
    let root = PathBuf::from(root);
    let candidates = [
        root.clone(),
        root.join("Gemma4-12B-it"),
        root.join("gemma-4-12b-it-NVFP4"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("model.safetensors").is_file())
        .ok_or_else(|| format!("no model.safetensors found under {}", root.display()).into())
}

async fn load_mlp_projection<E: M, Es: M>(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    name: &str,
    shape: [usize; 2],
) -> Result<
    (
        HbmTensor<f4e2m1, Chip, E>,
        HbmTensor<f8e4m3, Chip, Es>,
        HbmTensor<f32, Chip, m![1]>,
    ),
    Box<dyn std::error::Error>,
> {
    Ok((
        load_f4(device, tensors, &format!("{prefix}.{name}.weight_packed"), shape).await?,
        load_f8(device, tensors, &format!("{prefix}.{name}.weight_scale")).await?,
        load_global_scale(device, tensors, &format!("{prefix}.{name}.weight_global_scale")).await?,
    ))
}

async fn load_mlp(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
) -> Result<MlpWeights, Box<dyn std::error::Error>> {
    let (up_weight_packed, up_weight_scale, up_global_scale) =
        load_mlp_projection::<m![L, H], m![L, H / 16]>(device, tensors, prefix, "up_proj", [L::SIZE, H::SIZE]).await?;
    let (gate_weight_packed, gate_weight_scale, gate_global_scale) =
        load_mlp_projection::<m![L, H], m![L, H / 16]>(device, tensors, prefix, "gate_proj", [L::SIZE, H::SIZE])
            .await?;
    let (down_weight_packed, down_weight_scale, down_global_scale) =
        load_mlp_projection::<m![H, L], m![H, L / 16]>(device, tensors, prefix, "down_proj", [H::SIZE, L::SIZE])
            .await?;
    Ok(MlpWeights {
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    })
}

async fn load_common(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
) -> Result<
    (
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        MlpWeights,
    ),
    Box<dyn std::error::Error>,
> {
    Ok((
        load_bf16(device, tensors, &format!("{prefix}.pre_feedforward_layernorm.weight")).await?,
        load_bf16(device, tensors, &format!("{prefix}.post_feedforward_layernorm.weight")).await?,
        load_bf16(device, tensors, &format!("{prefix}.input_layernorm.weight")).await?,
        load_mlp(device, tensors, &format!("{prefix}.mlp")).await?,
    ))
}

async fn load_sliding(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<SlidingLayer, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(device, tensors, &prefix).await?;
    let post_attention_norm = load_bf16(device, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?;
    let layer_scalar = load_scalar_bf16(device, tensors, &format!("{prefix}.layer_scalar")).await?;
    let rep = SlRep {
        post_attention_norm: rep_bf16(device, &post_attention_norm, 128).await?,
        pre_feedforward_norm: rep_bf16(device, &pre_feedforward_norm, 128).await?,
        post_feedforward_norm: rep_bf16(device, &post_feedforward_norm, 128).await?,
        layer_scalar: rep_bf16(device, &layer_scalar, 256).await?,
        input_norm: rep_bf16(device, &input_norm, 256).await?,
        ffc: ffc_rep(device, &mlp, &layer_scalar).await?,
    };
    Ok(SlidingLayer {
        input_norm,
        q_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.q_proj.weight")).await?,
        k_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.k_proj.weight")).await?,
        v_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.v_proj.weight")).await?,
        q_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.q_proj.weight_scale")).await?,
        k_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.k_proj.weight_scale")).await?,
        v_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.v_proj.weight_scale")).await?,
        q_norm: load_bf16(device, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load_bf16(device, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm,
        o_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar,
        mlp,
        rep,
    })
}

async fn load_full(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<FullLayer, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(device, tensors, &prefix).await?;
    let input_norm_rep = rep_bf16(device, &input_norm, 256).await?;
    let layer_scalar = load_scalar_bf16(device, tensors, &format!("{prefix}.layer_scalar")).await?;
    let ffc = ffc_rep(device, &mlp, &layer_scalar).await?;
    Ok(FullLayer {
        input_norm,
        q_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.q_proj.weight")).await?,
        k_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.k_proj.weight")).await?,
        q_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.q_proj.weight_scale")).await?,
        k_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.k_proj.weight_scale")).await?,
        q_norm: load_bf16(device, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load_bf16(device, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm: load_bf16(device, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?,
        o_weight: load_f8(device, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(device, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar,
        mlp,
        input_norm_rep,
        ffc,
    })
}

async fn load_vision(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<VisionWeights, Box<dyn std::error::Error>> {
    let pos_embedding_view = tensors.tensor("model.vision_embedder.pos_embedding")?;
    let pos_embedding_host: HostTensor<bf16, m![Pv, Dummy2, Mv]> =
        HostTensor::from_safetensors(&pos_embedding_view).map_err(|error| format!("pos_embedding: {error}"))?;

    Ok(VisionWeights {
        patch_ln1_weight: load_bf16(device, tensors, "model.vision_embedder.patch_ln1.weight").await?,
        patch_ln1_bias: load_bf16(device, tensors, "model.vision_embedder.patch_ln1.bias").await?,
        patch_dense_weight: load_bf16(device, tensors, "model.vision_embedder.patch_dense.weight").await?,
        patch_dense_bias: load_bf16(device, tensors, "model.vision_embedder.patch_dense.bias").await?,
        patch_ln2_weight: load_bf16(device, tensors, "model.vision_embedder.patch_ln2.weight").await?,
        patch_ln2_bias: load_bf16(device, tensors, "model.vision_embedder.patch_ln2.bias").await?,
        pos_norm_weight: load_bf16(device, tensors, "model.vision_embedder.pos_norm.weight").await?,
        pos_norm_bias: load_bf16(device, tensors, "model.vision_embedder.pos_norm.bias").await?,
        embedding_projection_weight: load_bf16(device, tensors, "model.embed_vision.embedding_projection.weight")
            .await?,
        pos_embedding: pos_embedding_host.into_vec(),
    })
}

async fn load_audio(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<AudioWeights, Box<dyn std::error::Error>> {
    Ok(AudioWeights {
        embedding_projection_weight: load_bf16(device, tensors, "model.embed_audio.embedding_projection.weight")
            .await?,
    })
}

pub async fn load_weights(device: &mut Device) -> Result<Weights, Box<dyn std::error::Error>> {
    let model_dir = model_dir()?;
    let model_path = model_dir.join("model.safetensors");
    let file = File::open(&model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let embedding_table = load_lm_table(device, &tensors).await?;
    let final_norm = load_bf16(device, &tensors, "model.language_model.norm.weight").await?;
    let vision = load_vision(device, &tensors).await?;
    let audio = load_audio(device, &tensors).await?;
    let mm = crate::host::mm::load_mm(device, &tensors).await?;

    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        if layer % 6 == 5 {
            layers.push(Layer::Full(load_full(device, &tensors, layer).await?));
        } else {
            layers.push(Layer::Sliding(load_sliding(device, &tensors, layer).await?));
        }
    }

    let chunk_started = std::time::Instant::now();
    let chunk = if crate::host::chunkw::chunk_path_enabled() {
        crate::host::chunkw::load_chunk_layers(device, &tensors).await?
    } else {
        Vec::new()
    };
    eprintln!("S2 load: chunk weights {:.0} ms", chunk_started.elapsed().as_secs_f64() * 1e3);
    let embed_host = {
        let view = tensors.tensor("model.language_model.embed_tokens.weight")?;
        if view.dtype() != safetensors::Dtype::BF16 || view.shape() != [W::SIZE, H::SIZE] {
            return Err("unexpected embedding table layout".into());
        }
        let offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
        let second = unsafe { memmap2::Mmap::map(&File::open(&model_path)?)? };
        EmbedHost { mmap: second, offset }
    };

    Ok(Weights {
        model_dir,
        embedding_table,
        final_norm,
        layers,
        vision,
        audio,
        mm,
        chunk,
        embed_host,
    })
}

/// Only the LM head (the embedding table doubles as the output projection) and the final norm: for kernel tests that
/// do not need the 48 layers.
pub struct LmHeadWeights {
    pub embedding_table: crate::ops_lmhead::LmTable,
    pub final_norm: HbmTensor<bf16, Chip, m![H]>,
}

pub async fn load_lm_head(device: &mut Device) -> Result<LmHeadWeights, Box<dyn std::error::Error>> {
    let model_dir = model_dir()?;
    let model_path = model_dir.join("model.safetensors");
    let file = File::open(&model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    Ok(LmHeadWeights {
        embedding_table: load_lm_table(device, &tensors).await?,
        final_norm: load_bf16(device, &tensors, "model.language_model.norm.weight").await?,
    })
}

pub fn tokenizer_path(weights: &Weights) -> PathBuf {
    tokenizer_path_in(&weights.model_dir)
}

pub fn tokenizer_path_in(model_dir: &Path) -> PathBuf {
    model_dir.join("tokenizer.json")
}
