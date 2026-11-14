//! Packed copies of the weights the chunk kernels stream (qkvc QKV tiles, ffnc AO and FFN tiles), uploaded once at startup.
//! The natural-layout weights stay loaded: the token-serial fallback (positions >= 2048, soft rows without a host copy, `S2_CHUNKPATH=0`) uses them.
//!
//! `S2_CHUNKPATH=0` skips this module (no packed copies, no chunk path): the pre-integration behaviour, for A/B runs.

use furiosa_opt_std::prelude::*;

use crate::axes::Ke;
use crate::device::ffnc::{C32, C64, E256, E32, Fi4, Fi5, G32, Gd8, Rt96, Rt240, Rt256};
use crate::host::fp8;
use crate::host::packed::{self, FoldFfn, PackedAo, PackedFfnM};
use crate::host::qkvc::{self, FullChunkWeights, SlidingChunkWeights};
use crate::{Chip, LAYERS, ops_f8};

type BoxError = Box<dyn std::error::Error>;

pub(crate) enum ChunkLayer {
    Sliding { qkv: SlidingChunkWeights, ao: PackedAo<Ke>, ffn: PackedFfnM },
    Full { qkv: FullChunkWeights, ao: PackedAo<E256>, ffn: PackedFfnM },
}

/// Whether the chunk path (packed weights + chunk kernels) is enabled: default yes, `S2_CHUNKPATH=0` disables it.
pub fn chunk_path_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_CHUNKPATH").map(|value| value != "0").unwrap_or(true))
}

async fn up<D: MaterializableScalar, E: M>(device: &mut Device, bytes: Vec<u8>) -> Result<HbmTensor<D, Chip, E>, BoxError> {
    Ok(HostTensor::<D, E>::from_buf(bytes).to_hbm(&mut device.pdma).await?)
}

/// FFN tensors of one layer in the tile-merged layout of `ffn_chunk`; the six byte permutations run in parallel.
async fn load_ffn(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    layer: usize,
) -> Result<PackedFfnM, BoxError> {
    let get = |name: &str| -> Result<&[u8], BoxError> { Ok(tensors.tensor(&format!("{prefix}.{name}"))?.data()) };
    let (up_w4, up_s) = (get("mlp.up_proj.weight_packed")?, get("mlp.up_proj.weight_scale")?);
    let (gate_w4, gate_s) = (get("mlp.gate_proj.weight_packed")?, get("mlp.gate_proj.weight_scale")?);
    let (down_w4, down_s) = (get("mlp.down_proj.weight_packed")?, get("mlp.down_proj.weight_scale")?);
    let pre_norm = get("pre_feedforward_layernorm.weight")?;
    let (a, b, c, d, e, f) = std::thread::scope(|scope| {
        let a = scope.spawn(|| packed::pack_up_w_merged(up_w4));
        let b = scope.spawn(|| packed::pack_up_scale_merged(up_s));
        let c = scope.spawn(|| packed::pack_up_w_merged(gate_w4));
        let d = scope.spawn(|| packed::pack_up_scale_merged(gate_s));
        let e = scope.spawn(|| packed::pack_down_w_merged(down_w4));
        let f = scope.spawn(|| packed::pack_down_scale_merged(down_s));
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap(), d.join().unwrap(), e.join().unwrap(), f.join().unwrap())
    });
    // S2_FP8 (default ON): scale tensors of the lossy fold, uploaded only when the fold is on; the provided weight bytes are not touched (the map is elementwise on the packed scales)
    let fold = if fp8::fold_on() {
        let gs = |name: &str| -> Result<f32, BoxError> {
            let v = f32::from_le_bytes(get(name)?.try_into()?);
            if v == 0.0 || !v.is_finite() {
                return Err(format!("invalid NVFP4 global scale {name}: {v}").into());
            }
            Ok(8.0 / v)
        };
        let (ugs, ggs, dgs) = (gs("mlp.up_proj.weight_global_scale")?, gs("mlp.gate_proj.weight_global_scale")?, gs("mlp.down_proj.weight_global_scale")?);
        Some(FoldFfn {
            up_s: up(device, fp8::div8_scales(&b)).await?,
            gate_s: up(device, fp8::div8_scales(&d)).await?,
            down_s: up(device, fp8::div8_scales(&f)).await?,
            up_gs: up(device, (ugs / crate::device::f8::ffn::FOLD_ALPHA).to_le_bytes().to_vec()).await?,
            gate_gs: up(device, (ggs / crate::device::f8::ffn::FOLD_ALPHA).to_le_bytes().to_vec()).await?,
            down_gs: up(device, (dgs / crate::device::f8::ffn::FOLD_ALPHA).to_le_bytes().to_vec()).await?,
            w8: None,
        })
    } else {
        None
    };
    let (n_up, n_dn) = (a.len() * 2, e.len() * 2);
    let mut m = PackedFfnM {
        up_w: up(device, a).await?,
        up_s: up(device, b).await?,
        gate_w: up(device, c).await?,
        gate_s: up(device, d).await?,
        down_w: up(device, e).await?,
        down_s: up(device, f).await?,
        pre_norm: up(device, packed::pad_norm(pre_norm)).await?,
        fold,
    };
    // R14 fixedcost PREBAKE: the folded layers' e4m3 FFN weights, decoded ONCE here by the run-time decode pass itself (`ops_f8::prebake_{up,down}8`, bit-identical);
    // the fused / native T = 256 fold launches read them instead of decoding every pass
    if fp8::ffn_fold(layer) {
        if let Some(f) = m.fold.as_mut() {
            let mut w_up: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]> = up(device, vec![0u8; n_up]).await?;
            let mut w_gate: HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]> = up(device, vec![0u8; n_up]).await?;
            let mut w_down: HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]> = up(device, vec![0u8; n_dn]).await?;
            launch(ops_f8::prebake_up8, (&mut *device, &m.up_w, &f.up_s, &mut w_up)).await?;
            launch(ops_f8::prebake_up8, (&mut *device, &m.gate_w, &f.gate_s, &mut w_gate)).await?;
            launch(ops_f8::prebake_down8, (&mut *device, &m.down_w, &f.down_s, &mut w_down)).await?;
            f.w8 = Some(packed::FoldW8 { up: w_up, gate: w_gate, down: w_down });
        }
    }
    Ok(m)
}

pub(crate) async fn load_chunk_layers(
    device: &mut Device,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<Vec<ChunkLayer>, BoxError> {
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        let o_w = tensors.tensor(&format!("{prefix}.self_attn.o_proj.weight"))?.data();
        let o_s = tensors.tensor(&format!("{prefix}.self_attn.o_proj.weight_scale"))?.data();
        let ffn = load_ffn(device, tensors, &prefix, layer).await?;
        if layer % 6 == 5 {
            layers.push(ChunkLayer::Full {
                qkv: qkvc::load_full_chunk_weights(device, tensors, layer).await?,
                ao: packed::upload_ao::<E256>(device, o_w, 8192, 256, o_s).await?,
                ffn,
            });
        } else {
            layers.push(ChunkLayer::Sliding {
                qkv: qkvc::load_sliding_chunk_weights(device, tensors, layer).await?,
                ao: packed::upload_ao::<Ke>(device, o_w, 4096, 128, o_s).await?,
                ffn,
            });
        }
    }
    Ok(layers)
}
