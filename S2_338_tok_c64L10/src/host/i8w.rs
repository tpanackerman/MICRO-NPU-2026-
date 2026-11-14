//! `S2_I8_LAYERS` (R14 w8a8, LOSSY; final4: default ON for 2..23,29,35,41,47): int8 W8A8 up / gate projections of the sliding layers in the fused K launch (`ops_i8`).
//!
//! What it does to the provided weights: the NVFP4 up / gate tensors stay in HBM exactly as provided (the bf16 kernels keep using them); a SECOND copy is
//! built at load time, int8 per (output row x 128-element K chunk) with an f32 scale: `q = round_ties_even(f4 x block_scale / s)`, `s = amax / 127`
//! (amax = max |f4 x block_scale| over the chunk), the f32 scale uploaded is `s x (1 / global_scale)`. Activations are quantised per (token x chunk) on
//! the device (`device/int8/ffn.rs`). Re-encodes the provided weights (lossy, ~0.7 % rms of the weight) like the FP8 fold: needs the user's decision to ship.
//!
//! Per-layer ALLOW-LIST (big8): `S2_I8_LAYERS` = comma list of layer indices and `a..b` / `a-b` ranges (e.g. `2..23,29,35,41,47`); unset / empty / `none` / `off`
//! = int8 OFF everywhere (nothing is built or allocated, the K launches are big's). Only SLIDING layers that are not folded by `S2_FP8` get int8 tensors
//! (full-attention indices in the list are ignored), and the host launches the int8 K only when the NEXT layer is fusable too (else the bf16 path runs).
//! Legacy `S2_I8=A` / `a` / `1` / `on` (when `S2_I8_LAYERS` is unset) = every sliding layer; any other `S2_I8` value is OFF. The int8 K REPLACES the bf16 K of
//! its layer (no extra device image beyond the int8 K itself; there is no int8 K').
use std::sync::OnceLock;

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::device::ffnc::{C32, C64, E32, Fi4, Fi5, G32, Gd8, Rt96, Rt240, Rt256};
use crate::host::fp8;
use crate::device::int8::ffn::Ho32;

type BoxError = Box<dyn std::error::Error>;

/// int8 up / gate tensors of one sliding layer.
pub struct I8Ffn {
    /// Once-decoded original down weights, without folding the global scale.
    pub down_bf16: HbmTensor<bf16, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    pub up_q: HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub up_ws: HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    pub gate_q: HbmTensor<i8, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    pub gate_ws: HbmTensor<f32, Chip, m![G32, C32, Fi4, Rt240]>,
    /// `S2_I8_SMOOTH`: the pre-FF norm weight divided by the per-input-channel smoothing scale (gamma_j / s_j, s_j = 2^e exact), else None
    pub pre_w: Option<HbmTensor<bf16, Chip, m![C32, E32]>>,
    /// all8b: +-1 H32 for the device activation rotation
    pub had: crate::device::int8::ffn::HadT,
}

/// `S2_I8_SMOOTH=<scales.tsv>` (LAB, default unset): per-input-channel power-of-two smoothing of the int8 up / gate activations (lowbit
/// smooth_scales, rows `<layer> preff 3840 <src> <one char per channel: l..p = 2^-2..2^2>`): W[:, j] * s_j, gamma_j / s_j; the product is unchanged.
pub fn smooth_exps(layer: usize) -> Option<Vec<f32>> {
    static TAB: OnceLock<Vec<Option<Vec<f32>>>> = OnceLock::new();
    let tab = TAB.get_or_init(|| {
        let mut t: Vec<Option<Vec<f32>>> = vec![None; 64];
        let Ok(path) = std::env::var("S2_I8_SMOOTH") else { return t };
        // `1` / `on` = the table compiled in (lowbit smooth_scales, preff rows), else a file path
        let text = if matches!(path.as_str(), "1" | "on") {
            include_str!("i8_smooth_preff.tsv").to_string()
        } else if let Ok(text) = std::fs::read_to_string(&path) {
            text
        } else {
            eprintln!("S2_I8_SMOOTH: cannot read {path}");
            return t;
        };
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() >= 5 && f[1] == "preff" {
                if let Ok(l) = f[0].parse::<usize>() {
                    let e: Vec<i32> = f[4].trim().bytes().map(|b| b as i32 - b'n' as i32).collect();
                    if l < 64 && e.len() == N_H && e.iter().all(|v| v.abs() <= 8) {
                        t[l] = Some(e.iter().map(|&v| (v as f32).exp2()).collect());
                    }
                }
            }
        }
        eprintln!("S2_I8_SMOOTH: {} layers from {path}", t.iter().filter(|x| x.is_some()).count());
        t
    });
    tab.get(layer).cloned().flatten()
}


fn parse_layers(spec: &str) -> u64 {
    let mut mask = 0u64;
    for part in spec.split(',') {
        let part = part.trim();
        let (a, z) = match part.split_once("..").or_else(|| part.split_once('-')) {
            Some((a, z)) => (a.trim().parse::<usize>(), z.trim().parse::<usize>()),
            None => (part.parse::<usize>(), part.parse::<usize>()),
        };
        match (a, z) {
            (Ok(a), Ok(z)) => (a..=z.min(63)).for_each(|l| mask |= 1 << l),
            _ if matches!(part.to_ascii_lowercase().as_str(), "" | "none" | "off") => {}
            _ => eprintln!("S2_I8_LAYERS: ignoring {part:?}"),
        }
    }
    mask
}

/// final4 default allow-list (big8p validated set).
const DEFAULT_LAYERS: &str = "2..23,29,35,41,47";

/// Bit mask of the allow-listed layers (bit i = layer i); 0 = int8 OFF.
fn mask() -> u64 {
    static MASK: OnceLock<u64> = OnceLock::new();
    *MASK.get_or_init(|| {
        if let Ok(v) = std::env::var("S2_I8_LAYERS") {
            return parse_layers(&v);
        }
        match std::env::var("S2_I8").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            // final4: int8 DEFAULT ON for the validated allow-list (big8p); `S2_I8=off` (or `S2_I8_LAYERS=none`) gives the final2 path
            "" => parse_layers(DEFAULT_LAYERS),
            "0" | "off" | "none" => 0,
            "a" | "1" | "on" => u64::MAX,
            other => {
                eprintln!("S2_I8: unknown value {other:?}, int8 path OFF");
                0
            }
        }
    })
}

/// all8b S3: full-attention layers of the allow-list get int8 too (`S2_I8_FULL` = 1 / on; default FULL_DEFAULT).
pub const FULL_DEFAULT: bool = false;
pub fn full_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("S2_I8_FULL").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "1" | "on" => true,
        "0" | "off" => false,
        _ => FULL_DEFAULT,
    })
}

/// all8b S2: int8 layers also run int8 in the K' slot and in unfused passes (`S2_I8_NATIVE` = 0 / off disables; default on).
pub fn native_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("S2_I8_NATIVE").unwrap_or_default().trim().to_ascii_lowercase().as_str(), "0" | "off"))
}

/// Whether the int8 path is switched on at all (non-empty allow-list).
pub fn on() -> bool {
    mask() != 0
}

/// Whether the (sliding) layer `layer` is in the allow-list and gets the int8 up / gate tensors.
pub fn layer_on(layer: usize) -> bool {
    layer < 64 && mask() & (1 << layer) != 0
}

/// e2m1 magnitudes by the low 3 bits of a nibble (bit 3 = sign).
const F4: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Bytes per (row, chunk) group in the packed up / gate layouts: 128 f4 values, 8 block scales.
const GW: usize = 64;
const GS: usize = 8;

/// Checkpoint byte offsets (f4 bytes, scale bytes) of output group `og` of the tile-merged layout `[G32, C32, Fi4, Rt240]` (= `packed::pack_up_w_merged`:
/// row n = g * 960 + i * 240 + rr, K chunk c of 128), or None for the zero pad chunks 30 / 31.
fn src_of(og: usize) -> Option<(usize, usize)> {
    let rr = og % 240;
    let i = (og / 240) % 4;
    let c = (og / 960) % 32;
    let g = og / (960 * 32);
    if c >= 30 {
        return None;
    }
    let n = g * 960 + i * 240 + rr;
    Some((n * (N_H / 2) + c * GW, n * (N_H / 16) + c * GS))
}

const N_H: usize = 3840;

/// all8b: Hadamard block of the int8 up / gate route (i8free recipe "had32"): the device rotates every 32-element block of the activation chunk by the
/// unnormalised +-1 Sylvester matrix H (`device::int8::ffn::rot32`), the host stores W' = W H / 32 (1/32 exact). W' x' = W H H x / 32 = W x.
pub const HAD: usize = 32;

/// v <- (H v) / HAD on each HAD-block (in-place fast Walsh-Hadamard transform, Sylvester order: (H v)_o = sum_i (-1)^popcount(o & i) v_i).
fn had_rows(v: &mut [f32]) {
    for blk in v.chunks_mut(HAD) {
        let mut h = 1;
        while h < HAD {
            for i in (0..HAD).step_by(2 * h) {
                for j in i..i + h {
                    let (a, c) = (blk[j], blk[j + h]);
                    blk[j] = a + c;
                    blk[j + h] = a - c;
                }
            }
            h *= 2;
        }
        for x in blk.iter_mut() {
            *x /= HAD as f32;
        }
    }
}

/// bf16 bytes of the +-1 H32 matrix [Ho32, Ki32] for the device rotation (row o, column i: (-1)^popcount(o & i)).
pub fn had32_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(32 * 32 * 2);
    for o in 0..32u32 {
        for i in 0..32u32 {
            let v: u16 = if (o & i).count_ones() % 2 == 0 { 0x3F80 } else { 0xBF80 };
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// int8 re-quantisation of the output groups `[og0, og0 + ws.len())` of one up / gate matrix, read from the CHECKPOINT tensors (f4 `[15360, 1920]` bytes,
/// e4m3 block scales `[15360, 240]`) and written in the tile-merged layout: `q` gets 128 bytes per group (element order = the f4 element order: low nibble
/// first), `ws` one f32 per group.
fn quant_groups(w4: &[u8], s8: &[u8], rgs: f32, og0: usize, q: &mut [u8], ws: &mut [f32], sm: Option<&[f32]>) {
    for (k, wsg) in ws.iter_mut().enumerate() {
        let out = &mut q[k * 128..(k + 1) * 128];
        let Some((wo, so)) = src_of(og0 + k) else {
            out.fill(0);
            *wsg = 0.0;
            continue;
        };
        let wb = &w4[wo..wo + GW];
        let sb = &s8[so..so + GS];
        if sm.is_some() || HAD > 0 {
            // smoothed: w'_j = w_j * 2^e_j (element j of chunk c: block b, byte k, nibble lo/hi = element b*16 + 2k (+1))
            let c = (og0 + k) / 960 % 32;
            let mut v = [0f32; 128];
            for b in 0..8 {
                let bsv = fp8::e4m3_to_f32(sb[b]);
                for kk in 0..8 {
                    let byte = wb[b * 8 + kk];
                    for (h, nib) in [(0usize, byte & 15), (1, byte >> 4)] {
                        let j = b * 16 + 2 * kk + h;
                        let mag = F4[(nib & 7) as usize] * bsv * sm.map_or(1.0, |sm| sm[c * 128 + j]);
                        v[j] = if nib & 8 != 0 { -mag } else { mag };
                    }
                }
            }
            if HAD > 0 {
                had_rows(&mut v);
            }
            let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            if amax == 0.0 || !amax.is_finite() {
                out.fill(0);
                *wsg = 0.0;
                continue;
            }
            let sc = amax / 127.0;
            for j in 0..128 {
                out[j] = ((v[j] / sc).round_ties_even().clamp(-127.0, 127.0) as i32) as i8 as u8;
            }
            *wsg = sc * rgs;
            continue;
        }
        let mut bs = [0f32; 8];
        let mut amax = 0f32;
        for b in 0..8 {
            bs[b] = fp8::e4m3_to_f32(sb[b]);
            let mut mc = 0u8;
            for &byte in &wb[b * 8..b * 8 + 8] {
                mc = mc.max(byte & 7).max((byte >> 4) & 7);
            }
            amax = amax.max(F4[mc as usize] * bs[b]);
        }
        if amax == 0.0 || !amax.is_finite() {
            out.fill(0);
            *wsg = 0.0;
            continue;
        }
        let s = amax / 127.0;
        for b in 0..8 {
            let mut t = [0u8; 16];
            for (n, tv) in t.iter_mut().enumerate() {
                let r = ((F4[n & 7] * bs[b]) / s).round_ties_even().clamp(-127.0, 127.0) as i32;
                *tv = (if n & 8 != 0 { -r } else { r }) as i8 as u8;
            }
            for k in 0..8 {
                let byte = wb[b * 8 + k];
                out[b * 16 + 2 * k] = t[(byte & 15) as usize];
                out[b * 16 + 2 * k + 1] = t[(byte >> 4) as usize];
            }
        }
        *wsg = s * rgs;
    }
}

/// (int8 bytes, f32 scale bytes) of one matrix in the tile-merged layout from its checkpoint tensors, split over `parts` threads.
fn quant_matrix(w4: &[u8], s8: &[u8], rgs: f32, parts: usize, sm: Option<&[f32]>) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w4.len(), 15360 * N_H / 2, "up/gate f4 checkpoint size");
    assert_eq!(s8.len(), 15360 * N_H / 16, "up/gate scale checkpoint size");
    let groups = G32::SIZE * C32::SIZE * Fi4::SIZE * Rt240::SIZE;
    let mut q = vec![0u8; groups * 128];
    let mut ws = vec![0f32; groups];
    let per = groups.div_ceil(parts.max(1));
    std::thread::scope(|scope| {
        for (i, (qc, wc)) in q.chunks_mut(per * 128).zip(ws.chunks_mut(per)).enumerate() {
            scope.spawn(move || quant_groups(w4, s8, rgs, i * per, qc, wc, sm));
        }
    });
    let ws_bytes = ws.iter().flat_map(|v| v.to_le_bytes()).collect();
    (q, ws_bytes)
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).clamp(1, 8)
}

/// Host bytes of the int8 up / gate tensors of one layer (int8 up, f32 scales up, int8 gate, f32 scales gate).
pub struct I8Bytes(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>);

/// Re-quantises one layer's up / gate checkpoint tensors (f4 bytes, block-scale bytes, reciprocal global scales `*_rgs`) on host threads.
pub fn quantise(up_w4: &[u8], up_s8: &[u8], up_rgs: f32, gate_w4: &[u8], gate_s8: &[u8], gate_rgs: f32, pre_norm: &[u8], sm: Option<&[f32]>) -> I8Bytes {
    let half = (threads() / 2).max(1);
    let ((uq, uws), (gq, gws)) = std::thread::scope(|scope| {
        let u = scope.spawn(|| quant_matrix(up_w4, up_s8, up_rgs, half, sm));
        let g = scope.spawn(|| quant_matrix(gate_w4, gate_s8, gate_rgs, half, sm));
        (u.join().unwrap(), g.join().unwrap())
    });
    // gamma_j / 2^e_j in bf16 (exact: a power-of-two scale of a bf16 value), padded to 4096 like `packed::pad_norm`
    let pre = sm.map(|e| {
        assert_eq!(pre_norm.len(), N_H * 2);
        let mut out = vec![0u8; 4096 * 2];
        for j in 0..N_H {
            let g = f32::from_bits((u16::from_le_bytes([pre_norm[2 * j], pre_norm[2 * j + 1]]) as u32) << 16) / e[j];
            let b = ((g.to_bits() + 0x7FFF + ((g.to_bits() >> 16) & 1)) >> 16) as u16;
            out[2 * j..2 * j + 2].copy_from_slice(&b.to_le_bytes());
        }
        out
    });
    I8Bytes(uq, uws, gq, gws, pre)
}

/// Uploads the int8 tensors of one layer (called AFTER the layer's provided tensors are uploaded, so the bf16 weights keep their addresses).
pub async fn upload(device: &mut Device, bytes: I8Bytes, down_bf16: Vec<u8>) -> Result<I8Ffn, BoxError> {
    let I8Bytes(uq, uws, gq, gws, pre) = bytes;
    let pre_w = match pre {
        Some(b) => Some(HostTensor::<bf16, m![C32, E32]>::from_buf(b).to_hbm(&mut device.pdma).await?),
        None => None,
    };
    Ok(I8Ffn {
        down_bf16: HostTensor::<bf16, m![Gd8, C64, Fi5, Rt96, Rt256]>::from_buf(down_bf16).to_hbm(&mut device.pdma).await?,
        pre_w,
        had: crate::device::int8::ffn::HadHost::from_buf(had32_bytes().repeat(512)).to_hbm(&mut device.pdma).await?,
        up_q: HostTensor::<i8, m![G32, C32, Fi4, Rt240, E32]>::from_buf(uq).to_hbm(&mut device.pdma).await?,
        up_ws: HostTensor::<f32, m![G32, C32, Fi4, Rt240]>::from_buf(uws).to_hbm(&mut device.pdma).await?,
        gate_q: HostTensor::<i8, m![G32, C32, Fi4, Rt240, E32]>::from_buf(gq).to_hbm(&mut device.pdma).await?,
        gate_ws: HostTensor::<f32, m![G32, C32, Fi4, Rt240]>::from_buf(gws).to_hbm(&mut device.pdma).await?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_group_roundtrip() {
        // one group: block b holds nibbles n (positive codes 0..7, then negative), scale byte 0x38 (= 1.0)
        let mut w4 = vec![0u8; GW];
        for (k, byte) in w4.iter_mut().enumerate() {
            let lo = (k % 8) as u8;
            let hi = ((k + 3) % 8) as u8 | 8;
            *byte = lo | (hi << 4);
        }
        let s8 = vec![0x38u8; GS];
        let mut q = vec![0u8; 128];
        let mut ws = vec![0f32; 1];
        quant_groups(&w4, &s8, 1.0, 0, &mut q, &mut ws, None);
        assert!((ws[0] - 6.0 / 127.0).abs() < 1e-7);
        for j in 0..128 {
            let byte = w4[j / 2];
            let nib = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
            let v = F4[(nib & 7) as usize] * if nib & 8 != 0 { -1.0 } else { 1.0 };
            let want = (v / (6.0 / 127.0)).round_ties_even() as i32;
            assert_eq!(q[j] as i8 as i32, want, "element {j}");
        }
    }
}
