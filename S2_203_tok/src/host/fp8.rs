//! `S2_FP8`: the LOSSY FP8 weight fold of the FFN (fold1 build: DEFAULT ON when `S2_FP8` is unset; `S2_FP8=off` restores the bf16 FFN of fin6).
//!
//! What it does to the provided weights (`device/f8/ffn.rs`): the NVFP4 FFN weights stay in HBM exactly as provided; at launch time the up / gate / down tiles are decoded on chip to e4m3
//! with the block scale folded in, `e4m3(f4 value x block_scale / 8)`, and contracted e4m3 x e4m3 against dynamically quantised activations. The host only uploads, per layer, the block
//! scales divided by 8 (a power-of-two change, exact in e4m3) and 8 x the reciprocal global scales; the provided weight tensors are not modified.
//!
//! `S2_FP8`: `fold` / `on` / `1` = fold ON (opt-in since build 7nf), unset / `off` / `0` / `none` = bf16 everywhere (the fin6 FFN numerics; DEFAULT).  `S2_FP8_BF16_LAYERS` (comma list, default `0,1,5,11,17,23,29,35,41,46,47` = the first two layers, layer 46 and ALL eight full-attention layers (fold2 rung RG1), `none` = no reserve) names layers that
//! keep the bf16 FFN even when the fold is on.
use std::sync::OnceLock;

fn parse(value: &str) -> bool {
    // build 7nf: the fold is OFF unless `S2_FP8=fold` (or `on` / `1`) is set (the fold flips vqav2.117333 at C >= 15 and failed a private vqav2 case in submission 5111c2a6)
    let mut fold = false;
    for part in value.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "" => {}
            "off" | "0" | "none" => fold = false,
            "fold" | "on" | "1" => fold = true,
            other => eprintln!("S2_FP8: unknown component {other:?} ignored"),
        }
    }
    fold
}

/// Whether the fold is switched on (default; `S2_FP8=off` disables).
pub fn fold_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse(&std::env::var("S2_FP8").unwrap_or_default()))
}

fn reserve_mask() -> u64 {
    static MASK: OnceLock<u64> = OnceLock::new();
    *MASK.get_or_init(|| {
        let value = std::env::var("S2_FP8_BF16_LAYERS").unwrap_or_else(|_| "0,1,5,11,17,23,29,35,41,46,47".into());
        let v = value.trim();
        if v.eq_ignore_ascii_case("none") {
            return 0;
        }
        v.split(',').filter_map(|s| s.trim().parse::<usize>().ok()).filter(|&l| l < 64).fold(0u64, |m, l| m | (1 << l))
    })
}

/// The FFN of layer `index` runs the fold (the fold is on and the layer is not in the bf16 reserve).
pub fn ffn_fold(index: usize) -> bool {
    fold_on() && (index >= 64 || reserve_mask() & (1 << index) == 0)
}

fn e4m3_slow(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0 { sign * (m / 8.0) * 2f32.powi(-6) } else { sign * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// e4m3 (bias 7, no infinity, 0x7F / 0xFF = NaN) -> f32.
pub fn e4m3_to_f32(b: u8) -> f32 {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = e4m3_slow(i as u8);
        }
        t
    })[b as usize]
}

/// f32 -> e4m3, round to nearest even, saturating at +-448.
pub fn f32_to_e4m3(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a.is_nan() {
        return 0x7F;
    }
    if a >= 464.0 {
        return sign | 0x7E;
    }
    let (mut lo, mut hi) = (0usize, 0x7Eusize);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if e4m3_to_f32(mid as u8) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = lo;
    if below == 0x7E {
        return sign | 0x7E;
    }
    let (vl, vh) = (e4m3_to_f32(below as u8), e4m3_to_f32(below as u8 + 1));
    let code = if a - vl < vh - a {
        below
    } else if a - vl > vh - a {
        below + 1
    } else if below % 2 == 0 {
        below
    } else {
        below + 1
    };
    sign | code as u8
}

/// NVFP4 block scales (e4m3 bytes) divided by 8, exact in e4m3 for every scale of the checkpoint (min 1.375 -> 0.17 stays normal), so that `f4 x scale / 8` stays below 448.
/// Zero stays zero (the packed layouts pad with zero scales). Works on packed bytes: the map is elementwise.
pub fn div8_scales(scales: &[u8]) -> Vec<u8> {
    let mut table = [0u8; 256];
    for (i, t) in table.iter_mut().enumerate() {
        *t = f32_to_e4m3(e4m3_to_f32(i as u8) / 8.0);
    }
    scales.iter().map(|&b| table[b as usize]).collect()
}
