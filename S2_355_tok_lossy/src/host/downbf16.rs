//! Exact BF16 predecode of the original checkpoint down weights.
//! No global scale is folded. The device's original post-reduction multiply remains intact.
const F4: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const ROWS: usize = 3840;
const K: usize = 15360;
const GROUPS: usize = 8 * 64 * 5 * 96;

/// Each chunk has 240 real elements followed by the original 16-element zero pad.
/// Output group layout is [Gd8, C64, Fi5, Rt96]; element layout is [Rt256].
pub(crate) fn decode_range(w4: &[u8], s8: &[u8], first: usize, out: &mut [u8]) {
    assert_eq!(out.len() % (256 * 2), 0);
    for (local, group) in out.chunks_exact_mut(256 * 2).enumerate() {
        let og = first + local;
        let rr = og % 96;
        let tile = og / 96 % 5;
        let chunk = og / (96 * 5) % 64;
        let gd = og / (96 * 5 * 64);
        let row = gd * 480 + tile * 96 + rr;
        for j in 0..240 {
            let k = chunk * 240 + j;
            let b = w4[row * (K / 2) + k / 2];
            let n = if k % 2 == 0 { b & 15 } else { b >> 4 };
            let magnitude = F4[(n & 7) as usize];
            let v = if n & 8 != 0 { -magnitude } else { magnitude };
            let scale = super::fp8::e4m3_to_f32(s8[row * (K / 16) + k / 16]);
            let product = v * scale;
            // Finite e2m1 * e4m3 has at most five fraction bits, fitting BF16 exactly.
            let bits = product.to_bits();
            let bf = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
            group[2*j..2*j+2].copy_from_slice(&bf.to_le_bytes());
        }
        group[480..].fill(0);
    }
}

pub(crate) fn decode(w4: &[u8], s8: &[u8]) -> Vec<u8> {
    assert_eq!(w4.len(), ROWS * K / 2);
    assert_eq!(s8.len(), ROWS * K / 16);
    let mut out = vec![0u8; GROUPS * 256 * 2];
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).clamp(1, 8);
    let groups_per_worker = GROUPS.div_ceil(workers);
    std::thread::scope(|scope| {
        for (part, bytes) in out.chunks_mut(groups_per_worker * 256 * 2).enumerate() {
            scope.spawn(move || decode_range(w4, s8, part * groups_per_worker, bytes));
        }
    });
    out
}
