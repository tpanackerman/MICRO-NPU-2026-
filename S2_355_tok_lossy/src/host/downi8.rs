//! i8d: int8 W8A8 down projection of the int8 (`S2_I8_LAYERS`) layers (replaces the once-decoded bf16 copy of `downbf16.rs` for the int8 body).
//!
//! Weights: int8 per (output row x K chunk of 240 real / 256 padded columns), symmetric, `s = amax / 127`, after a Hadamard-32 rotation along K
//! inside each padded chunk (8 blocks of 32; block 7 = 16 real columns + the 16 zero pad columns). Same recipe as the int8 up / gate (`i8w::quant_groups`):
//! the stored weight is W' = W H / 32 (unnormalised +-1 Sylvester H, the 1/32 exact) so that W' (H x) = W x, the device rotates the activation chunk by
//! the same +-1 H (`device::int8::ffn`, `had32_bytes` matrix), and the uploaded f32 scale is `s x (1 / global_scale)`: the NVFP4 global down scale is
//! FOLDED into the per-(row, chunk) scale (the device no longer multiplies the reduced sum by `down_global_scale`).
//! Layout: int8 `[Gd8, C64, Fi5, Rt96, Rt256]` (the bf16 `down_bf16` order: output row = gd * 480 + tile * 96 + rr, K = chunk * 240 + j) and f32
//! scales `[Gd8, C64, Fi5, Rt96]` (one per (row, chunk) group). Pure host code (std only), testable without the device crate.
use super::fp8::e4m3_to_f32;

/// e2m1 magnitudes by the low 3 bits of a nibble (bit 3 = sign).
const F4: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const ROWS: usize = 3840;
const K: usize = 15360;
/// (row, chunk) groups of `[Gd8, C64, Fi5, Rt96]`
pub(crate) const GROUPS: usize = 8 * 64 * 5 * 96;
/// padded / real columns of one K chunk
pub(crate) const CW: usize = 256;
const CR: usize = 240;
/// Hadamard block (= `i8w::HAD`)
pub(crate) const HAD: usize = 32;

/// v <- (H v) / HAD on each HAD-block (in-place fast Walsh-Hadamard transform, Sylvester order: (H v)_o = sum_i (-1)^popcount(o & i) v_i) = `i8w::had_rows`.
pub(crate) fn had_rows(v: &mut [f32]) {
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

/// Checkpoint row and K chunk of output group `og` of `[Gd8, C64, Fi5, Rt96]` (= `downbf16::decode_range`).
fn src_of(og: usize) -> (usize, usize) {
    let rr = og % 96;
    let tile = og / 96 % 5;
    let chunk = og / (96 * 5) % 64;
    let gd = og / (96 * 5 * 64);
    (gd * 480 + tile * 96 + rr, chunk)
}

/// The 256 padded chunk values `f4 x block_scale` of group `og` (240 real, 16 zero pads), read from the checkpoint tensors (f4 `[3840, 7680]` bytes,
/// e4m3 block scales `[3840, 960]`).
pub(crate) fn decode_group(w4: &[u8], s8: &[u8], og: usize) -> [f32; CW] {
    let (row, chunk) = src_of(og);
    let mut v = [0f32; CW];
    for (j, vj) in v.iter_mut().enumerate().take(CR) {
        let k = chunk * CR + j;
        let b = w4[row * (K / 2) + k / 2];
        let n = if k % 2 == 0 { b & 15 } else { b >> 4 };
        let magnitude = F4[(n & 7) as usize] * e4m3_to_f32(s8[row * (K / 16) + k / 16]);
        *vj = if n & 8 != 0 { -magnitude } else { magnitude };
    }
    v
}

/// int8 re-quantisation of the groups `[first, first + ws.len())`: `q` gets 256 bytes per group (rotated chunk order), `ws` one f32 per group
/// (`s x rgs`, `rgs` = 1 / global scale). An all-zero group gives q = 0, ws = 0.
pub(crate) fn quant_range(w4: &[u8], s8: &[u8], rgs: f32, first: usize, q: &mut [u8], ws: &mut [f32]) {
    for (k, wsg) in ws.iter_mut().enumerate() {
        let out = &mut q[k * CW..(k + 1) * CW];
        let mut v = decode_group(w4, s8, first + k);
        had_rows(&mut v);
        let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        if amax == 0.0 || !amax.is_finite() {
            out.fill(0);
            *wsg = 0.0;
            continue;
        }
        let sc = amax / 127.0;
        for j in 0..CW {
            out[j] = ((v[j] / sc).round_ties_even().clamp(-127.0, 127.0) as i32) as i8 as u8;
        }
        *wsg = sc * rgs;
    }
}

/// (int8 bytes `[Gd8, C64, Fi5, Rt96, Rt256]`, f32 scale bytes `[Gd8, C64, Fi5, Rt96]`) of one down matrix from its checkpoint tensors, on host threads.
pub(crate) fn quantise(w4: &[u8], s8: &[u8], rgs: f32) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w4.len(), ROWS * K / 2, "down f4 checkpoint size");
    assert_eq!(s8.len(), ROWS * K / 16, "down scale checkpoint size");
    let mut q = vec![0u8; GROUPS * CW];
    let mut ws = vec![0f32; GROUPS];
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).clamp(1, 8);
    let per = GROUPS.div_ceil(workers);
    std::thread::scope(|scope| {
        for (i, (qc, wc)) in q.chunks_mut(per * CW).zip(ws.chunks_mut(per)).enumerate() {
            scope.spawn(move || quant_range(w4, s8, rgs, i * per, qc, wc));
        }
    });
    let ws_bytes = ws.iter().flat_map(|v| v.to_le_bytes()).collect();
    (q, ws_bytes)
}

/// Device model of the activation side (`device::int8::ffn`, per (token, chunk)): x' = H x per 32-block (unnormalised, f32 sums, committed in bf16),
/// s = sqrt(max x'^2 + 1e-30) / 127, q = clip(round(x' / s), +-127). Returns (q, s). Test helper.
#[cfg(test)]
pub(crate) fn act_quant(x: &[f32; CW]) -> ([i32; CW], f32) {
    let bf = |v: f32| -> f32 {
        let bits = v.to_bits();
        f32::from_bits(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16)
    };
    let mut r = [0f32; CW];
    for b in 0..CW / HAD {
        for o in 0..HAD {
            let mut acc = 0f32;
            for i in 0..HAD {
                let h = if (o & i).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                acc += h * bf(x[b * HAD + i]);
            }
            r[b * HAD + o] = bf(acc);
        }
    }
    let am2 = r.iter().fold(0f32, |m, v| m.max(v * v));
    let s = (am2 + 1e-30).sqrt() * (1.0 / 127.0);
    let mut q = [0i32; CW];
    for j in 0..CW {
        q[j] = ((r[j] / s).round_ties_even() as i32).clamp(-127, 127);
    }
    (q, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* (deterministic, no dependency)
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            let (a, b) = (self.unit().max(1e-7), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    #[test]
    fn had_rows_matches_device_matrix() {
        // the FWHT is H v / 32 with H[o][i] = (-1)^popcount(o & i): the +-1 matrix the device gets from `i8w::had32_bytes`
        let mut rng = Rng(7);
        let x: Vec<f32> = (0..HAD).map(|_| rng.normal()).collect();
        let mut y = x.clone();
        had_rows(&mut y);
        for o in 0..HAD {
            let want: f32 = (0..HAD).map(|i| if (o & i).count_ones() % 2 == 0 { x[i] } else { -x[i] }).sum::<f32>() / HAD as f32;
            assert!((y[o] - want).abs() < 1e-5, "row {o}: {} vs {want}", y[o]);
        }
    }

    /// dequant(quant(rot(w))) . (s_x q_x) ~ w . x on random checkpoint-like data (f4 x e4m3 weights, heavy-tailed activations with zero pads),
    /// group order / scale fold / pad handling through the real `quant_range` on a full-size synthetic checkpoint slice.
    #[test]
    fn down_quant_roundtrip() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        // a full-size checkpoint (the group -> row / chunk addressing is exercised for real); random nibbles, block scales 2^-3 .. 2^3 (e4m3 0x20..0x50)
        let w4: Vec<u8> = (0..ROWS * K / 2).map(|_| rng.next() as u8).collect();
        let s8: Vec<u8> = (0..ROWS * K / 16).map(|_| 0x20 + (rng.next() % 0x31) as u8).collect();
        let rgs = 1.0 / 20864.0;
        let mut worst = 0f64;
        let mut sum_rel2 = 0f64;
        let mut n = 0usize;
        for &og in &[0usize, 1, 95, 96, 479, 480, 12345, 77777, 122_879, 122_880, 200_000, GROUPS - 1] {
            let mut q = vec![0u8; CW];
            let mut ws = vec![0f32; 1];
            quant_range(&w4, &s8, rgs, og, &mut q, &mut ws);
            let w = decode_group(&w4, &s8, og);
            assert!(w[CR..].iter().all(|&v| v == 0.0), "pad columns must be zero");
            // the group's row / chunk: compare against the checkpoint directly
            let (row, chunk) = src_of(og);
            let k0 = chunk * CR + 7;
            let b = w4[row * (K / 2) + k0 / 2];
            let nib = if k0 % 2 == 0 { b & 15 } else { b >> 4 };
            let direct = F4[(nib & 7) as usize] * e4m3_to_f32(s8[row * (K / 16) + k0 / 16]) * if nib & 8 != 0 { -1.0 } else { 1.0 };
            assert_eq!(w[7], direct, "group {og}: decode addressing");
            for trial in 0..4 {
                // activations: gelu-like (mostly small, a few outliers), real columns only (pads stay 0 as on the device)
                let mut x = [0f32; CW];
                for (j, xj) in x.iter_mut().enumerate().take(CR) {
                    let g = rng.normal();
                    *xj = if rng.unit() < 0.02 { g * 20.0 } else { g * 0.3 } + if trial == 3 && j == 5 { 300.0 } else { 0.0 };
                }
                let want: f64 = (0..CW).map(|j| w[j] as f64 * x[j] as f64).sum::<f64>() * rgs as f64;
                let (qx, sx) = act_quant(&x);
                let got: f64 = (0..CW).map(|j| (q[j] as i8 as i64) * qx[j] as i64).sum::<i64>() as f64 * ws[0] as f64 * sx as f64;
                // scale of the error: |w| |x| (Cauchy-Schwarz bound of the dot product)
                let wn: f64 = (0..CW).map(|j| (w[j] as f64).powi(2)).sum::<f64>().sqrt() * rgs as f64;
                let xn: f64 = (0..CW).map(|j| (x[j] as f64).powi(2)).sum::<f64>().sqrt();
                let rel = (got - want).abs() / (wn * xn).max(1e-30);
                worst = worst.max(rel);
                sum_rel2 += rel * rel;
                n += 1;
            }
            // all-zero group -> q = 0, ws = 0
        }
        let rms = (sum_rel2 / n as f64).sqrt();
        eprintln!("down_quant_roundtrip: {n} dots, error / (|w||x|): rms {rms:.2e}, worst {worst:.2e}");
        assert!(worst < 2e-2, "worst relative error {worst}");
        assert!(rms < 8e-3, "rms relative error {rms}");
    }

    #[test]
    fn zero_group() {
        let w4 = vec![0u8; ROWS * K / 2];
        let s8 = vec![0x38u8; ROWS * K / 16];
        let mut q = vec![1u8; CW];
        let mut ws = vec![1f32; 1];
        quant_range(&w4, &s8, 1.0, 5, &mut q, &mut ws);
        assert!(q.iter().all(|&b| b == 0) && ws[0] == 0.0);
    }
}
