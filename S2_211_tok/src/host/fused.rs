//! Host side of the fused sliding layer (`ops_fused::sliding_layer_fused_b16`, `device/fused`): the grouped weights, the small-weight blob and the global
//! scales of every sliding layer, built once at weight load from the device tensors the per-kernel path uses (download, concatenate, upload: the same
//! bytes; the per-kernel tensors then become views of the grouped ones, so HBM holds one copy), and the fixed layout of the per-pass table region.
//!
//! The fused launch runs the 256-row passes with 16 decode riders on the KVD-256 bf16-FFN path; default ON, `S2_FUSED=0` keeps the per-kernel layer.

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::fused::{Gs3, Sw, Tb, Wkv, Wug, sw, tb};
use crate::host::chunkw::ChunkLayer;
use crate::host::load::{Layer, SlidingLayer};
use crate::host::packed::{PackedAo, PackedFfnM};
use crate::host::qkvc::SlidingChunkWeights;
use crate::host::views::piece_at;
use crate::ops_ffnc::{C32, E32, Fb8, Fi4, G32, Hi3, Rt80, Rt240};

type BoxError = Box<dyn std::error::Error>;

/// Whether the fused sliding layer is used (default yes, `S2_FUSED=0` disables it).
pub fn enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_FUSED").map(|value| value != "0").unwrap_or(true))
}

/// The fused-kernel operands of one sliding layer that the per-kernel path does not already hold.
pub(crate) struct FusedSliding {
    pub small: HbmTensor<bf16, Chip, m![Sw]>,
    pub wkv: HbmTensor<f8e4m3, Chip, m![Wkv, Rq2, Gg, Kc, Rt, Ke]>,
    pub ao_w: HbmTensor<f8e4m3, Chip, m![Hi3, G32, C32, Rt80, E32]>,
    pub ug_w: HbmTensor<f4e2m1, Chip, m![Wug, G32, C32, Fi4, Rt240, E32]>,
    pub ug_s: HbmTensor<f8e4m3, Chip, m![Wug, G32, C32, Fi4, Rt240, Fb8]>,
    pub gs3: HbmTensor<f32, Chip, m![Gs3, Dummy8]>,
}

async fn host_bytes<D: MaterializableScalar, E: M>(device: &mut Device, t: &HbmTensor<D, Chip, E>) -> Result<Vec<u8>, BoxError> {
    let host: HostTensor<D, E> = t.to_host(&mut device.pdma).await?;
    Ok(host.into_inner().into_buf())
}

async fn up<D: MaterializableScalar, E: M>(device: &mut Device, bytes: Vec<u8>) -> Result<HbmTensor<D, Chip, E>, BoxError> {
    assert_eq!(bytes.len(), HbmTensor::<D, Chip, E>::size(), "fused operand size");
    Ok(HostTensor::<D, E>::from_buf(bytes).to_hbm(&mut device.pdma).await?)
}

/// Builds the fused operands of one sliding layer from its per-kernel tensors.
pub(crate) async fn build(device: &mut Device, l: &SlidingLayer, qkv: &SlidingChunkWeights, ao: &PackedAo<Ke>, ffn: &PackedFfnM) -> Result<FusedSliding, BoxError> {
    let mut wkv = host_bytes(device, &qkv.wk).await?;
    wkv.extend(host_bytes(device, &qkv.wv).await?);
    let mut ao_w = host_bytes(device, &ao.w[0]).await?;
    ao_w.extend(host_bytes(device, &ao.w[1]).await?);
    ao_w.extend(host_bytes(device, &ao.w[2]).await?);
    let mut ug_w = host_bytes(device, &ffn.up_w).await?;
    ug_w.extend(host_bytes(device, &ffn.gate_w).await?);
    let mut ug_s = host_bytes(device, &ffn.up_s).await?;
    ug_s.extend(host_bytes(device, &ffn.gate_s).await?);

    // the small-weight blob, every piece at its slot offset (`device::fused::sw::OFFSETS`, in this order)
    let pieces: [Vec<u8>; 13] = [
        host_bytes(device, &l.input_norm).await?,
        host_bytes(device, &l.post_attention_norm).await?,
        host_bytes(device, &l.post_feedforward_norm).await?,
        host_bytes(device, &qkv.sq).await?,
        host_bytes(device, &ffn.pre_norm).await?,
        host_bytes(device, &qkv.sk).await?,
        host_bytes(device, &qkv.sv).await?,
        host_bytes(device, &ao.s[0]).await?,
        host_bytes(device, &ao.s[1]).await?,
        host_bytes(device, &ao.s[2]).await?,
        host_bytes(device, &l.q_norm).await?,
        host_bytes(device, &l.k_norm).await?,
        host_bytes(device, &l.layer_scalar).await?,
    ];
    let mut small = vec![0u8; Sw::SIZE * 2];
    for ((offset, len), bytes) in sw::OFFSETS.iter().zip(&pieces) {
        assert_eq!(bytes.len(), len * 2, "small-weight piece at {offset}");
        small[offset * 2..(offset + len) * 2].copy_from_slice(bytes);
    }

    let mut gs3 = vec![0u8; 3 * 8 * 4];
    for (k, t) in [&l.mlp.up_global_scale, &l.mlp.gate_global_scale, &l.mlp.down_global_scale].into_iter().enumerate() {
        let b = host_bytes(device, t).await?;
        gs3[k * 32..k * 32 + 4].copy_from_slice(&b[..4]);
    }

    Ok(FusedSliding {
        small: up(device, small).await?,
        wkv: up(device, wkv).await?,
        ao_w: up(device, ao_w).await?,
        ug_w: up(device, ug_w).await?,
        ug_s: up(device, ug_s).await?,
        gs3: up(device, gs3).await?,
    })
}

/// Points the per-kernel K / V, o_proj and up / gate tensors of a layer at their bytes inside the grouped operands; the separate copies are dropped.
fn share(fo: &FusedSliding, qkv: &mut SlidingChunkWeights, ao: &mut PackedAo<Ke>, ffn: &mut PackedFfnM) -> Result<(), BoxError> {
    let kv = HbmTensor::<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>::size();
    let o = HbmTensor::<f8e4m3, Chip, m![G32, C32, Rt80, Ke]>::size();
    let uw = HbmTensor::<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>::size();
    let us = HbmTensor::<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>::size();
    assert!([kv, o, uw, us].iter().all(|bytes| bytes % 8192 == 0), "grouped operand pieces on 8 KB boundaries");
    qkv.wk = piece_at(&fo.wkv, 0)?;
    qkv.wv = piece_at(&fo.wkv, kv)?;
    for (i, w) in ao.w.iter_mut().enumerate() {
        *w = piece_at(&fo.ao_w, i * o)?;
    }
    ffn.up_w = piece_at(&fo.ug_w, 0)?;
    ffn.gate_w = piece_at(&fo.ug_w, uw)?;
    ffn.up_s = piece_at(&fo.ug_s, 0)?;
    ffn.gate_s = piece_at(&fo.ug_s, us)?;
    Ok(())
}

/// The fused operands of every sliding layer (`None` for full layers), built at weight load; the per-kernel tensors become views of them.
pub(crate) async fn build_all(device: &mut Device, layers: &[Layer], chunk: &mut [ChunkLayer]) -> Result<Vec<Option<FusedSliding>>, BoxError> {
    let started = std::time::Instant::now();
    let mut all = Vec::with_capacity(layers.len());
    for (layer, c) in layers.iter().zip(chunk.iter_mut()) {
        all.push(match (layer, c) {
            (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                let fo = build(device, l, qkv, ao, ffn).await?;
                share(&fo, qkv, ao, ffn)?;
                Some(fo)
            }
            _ => None,
        });
    }
    eprintln!("S2 load: fused sliding operands {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
    Ok(all)
}

/// Byte offsets inside the table region (`[Tb]` bf16) of its pieces, at the slots of `device::fused::tb` (piece sizes 2^18 / 2^14 / 2^12 elements; every
/// offset a multiple of the piece size, so of 8 KB): the per-kernel sliding tables of the same pass are views at these offsets (one copy in the upload).
pub mod at {
    use crate::device::fused::tb;
    pub fn mask(w: usize) -> usize {
        ((tb::MASK + w) << 18) * 2
    }
    pub fn pen(w: usize) -> usize {
        ((tb::PEN + w) << 12) * 2
    }
    pub fn cos(u: usize) -> usize {
        ((tb::COS + u) << 14) * 2
    }
    pub fn sin(u: usize) -> usize {
        ((tb::SIN + u) << 14) * 2
    }
    pub fn mask_d() -> usize {
        (tb::MASK_D << 14) * 2
    }
}

/// The per-pass table region (`[Tb]` bf16): masks / pens of the two waves, cos / sin of the four units, the 16 decode rows' mask, at `at::*`.
pub fn table_region(mask: [&[u8]; 2], pen: [&[u8]; 2], cos: &[Vec<u8>], sin: &[Vec<u8>], mask_d: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; Tb::SIZE * 2];
    let mut put = |at: usize, slot_elems: usize, bytes: &[u8]| {
        assert!(bytes.len() <= slot_elems * 2, "table piece larger than its slot");
        out[at..at + bytes.len()].copy_from_slice(bytes);
    };
    for w in 0..2 {
        put(at::mask(w), 1 << 18, mask[w]);
        put(at::pen(w), 1 << 12, pen[w]);
    }
    for u in 0..4 {
        put(at::cos(u), 1 << 14, &cos[u]);
        put(at::sin(u), 1 << 14, &sin[u]);
    }
    put(at::mask_d(), 1 << 14, mask_d);
    out
}
