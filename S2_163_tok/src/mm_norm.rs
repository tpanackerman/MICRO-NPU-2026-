//! Host-side row-wise normalisations of the batched encoders (pure f32 math, bf16 rounding at the stage boundary like the device
//! kernels of `device/vision/layernorm.rs` and `device/audio/projection.rs`).  No device dependencies.

/// Epsilon of every normalisation (same as `crate::EPS`).
pub const NORM_EPS: f32 = 1e-6;

#[inline(always)]
pub fn bf_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 -> bf16 bits, round to nearest even (NaN stays a quiet NaN).
#[inline(always)]
pub fn f32_to_bf(v: f32) -> u16 {
    let b = v.to_bits();
    let rounded = (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16;
    if v.is_nan() { 0x7FC0 } else { rounded }
}

#[inline]
pub fn sum8(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        for l in 0..8 {
            acc[l] += c[l];
        }
    }
    let tail: f32 = chunks.remainder().iter().sum();
    acc.iter().sum::<f32>() + tail
}

#[inline]
pub fn sumsq8(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        for l in 0..8 {
            acc[l] += c[l] * c[l];
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|v| v * v).sum();
    acc.iter().sum::<f32>() + tail
}

/// `out = (((x - mean) * w) / sqrt(var + eps)) + b` rounded to bf16 (the operation order of `device/vision/layernorm.rs`).
/// `centred` is scratch.
pub fn layer_norm_row(x: &[f32], w: &[f32], b: &[f32], out: &mut [u16], centred: &mut Vec<f32>) {
    let n = x.len() as f32;
    let mean = sum8(x) / n;
    centred.clear();
    centred.extend(x.iter().map(|v| v - mean));
    let std = (sumsq8(centred) / n + NORM_EPS).sqrt();
    for (((o, c), w), b) in out.iter_mut().zip(centred.iter()).zip(w).zip(b) {
        *o = f32_to_bf((c * w) / std + b);
    }
}

/// `out = x / sqrt(mean(x^2) + eps)` rounded to bf16 (rmsnorm without weight).
pub fn rms_norm_row(x: &[f32], out: &mut [u16]) {
    let rms = (sumsq8(x) / x.len() as f32 + NORM_EPS).sqrt();
    for (o, v) in out.iter_mut().zip(x) {
        *o = f32_to_bf(v / rms);
    }
}

pub fn bits_to_f32(src: &[u16], dst: &mut Vec<f32>) {
    dst.clear();
    dst.extend(src.iter().map(|&b| bf_to_f32(b)));
}

/// Runs `f(row_index, out_row, scratch)` over the `row_len`-wide rows of `out`, spread over a few threads.
pub fn par_rows<F: Fn(usize, &mut [u16], &mut Vec<f32>) + Sync>(out: &mut [u16], row_len: usize, f: F) {
    let rows = out.len() / row_len;
    // Spawning threads costs more than a 64-row tile of normalisations (measured: 0.3 ms serial vs 0.6-1.7 ms threaded for 64 x 6912).
    let threads = if out.len() < (8 << 20) {
        1
    } else {
        crate::host::cpuplace::parallelism().min(8).min(rows.max(1))
    };
    if threads <= 1 {
        let mut scratch = Vec::new();
        for (i, row) in out.chunks_mut(row_len).enumerate() {
            f(i, row, &mut scratch);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                crate::host::cpuplace::leave_launch_cpu();
                let mut scratch = Vec::new();
                for (i, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * per + i, row, &mut scratch);
                }
            });
        }
    });
}
