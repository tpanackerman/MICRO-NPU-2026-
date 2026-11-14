use rand::Rng;

use furiosa_opt_std::prelude::bf16;

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingConfig {
    pub const DEFAULT: Self = Self {
        temperature: 0.0,
        top_k: 64,
        top_p: 0.95,
    };

    /// Greedy requests can take their token from an argmax (on the host, or later on the device).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Samples one token id from f32 logits. `banned` ids are excluded unless every id is banned.
pub fn sample(logits: &[f32], config: &SamplingConfig, banned: &[usize], rng: &mut impl Rng) -> usize {
    assert!(!logits.is_empty(), "sample requires at least one logit");
    if config.temperature <= 0.0 {
        return argmax(logits, banned);
    }
    let temperature = config.temperature;

    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    if config.top_k > 0 {
        let keep = config.top_k.saturating_add(banned.len()).min(ranked.len());
        ranked.select_nth_unstable_by(keep - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        ranked.truncate(keep);
    }
    ranked.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    if !banned.is_empty() {
        let filtered: Vec<usize> = ranked.iter().copied().filter(|id| !banned.contains(id)).collect();
        if !filtered.is_empty() {
            ranked = filtered;
        }
    }
    if config.top_k > 0 {
        ranked.truncate(config.top_k.min(ranked.len()));
    }

    let max_logit = logits[ranked[0]];
    let mut probs: Vec<f32> = ranked
        .iter()
        .map(|&i| ((logits[i] - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    let mut nucleus_len = probs.len();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= config.top_p {
            nucleus_len = i + 1;
            break;
        }
    }
    let nucleus_sum: f32 = probs[..nucleus_len].iter().sum();

    let threshold = rng.random::<f32>() * nucleus_sum;
    let mut acc = 0.0;
    for (i, &p) in probs[..nucleus_len].iter().enumerate() {
        acc += p;
        if acc >= threshold {
            return ranked[i];
        }
    }
    ranked[nucleus_len - 1]
}

/// `f32::total_cmp` as an integer key: `key(a) < key(b)` exactly when `a.total_cmp(&b)` is `Less`.
#[inline(always)]
fn total_key(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    bits ^ (((bits >> 31) as u32) >> 1) as i32
}

/// Elements per block of the single-pass scan. The block maximum is a plain reduction that the
/// compiler vectorizes; the index is only looked up once, inside the winning block.
const BLOCK: usize = 512;

/// Index of the highest key outside `banned` (ties go to the lowest index). `banned` need not be
/// sorted or in range. If every index is banned, the highest key overall is chosen.
///
/// The ban is applied by splitting the scan at the banned indices, so no element is tested
/// against a set.
fn argmax_by_key<T: Copy>(values: &[T], banned: &[usize], key: impl Fn(T) -> i32 + Copy) -> usize {
    assert!(!values.is_empty(), "argmax requires at least one value");
    let mut cuts: Vec<usize> = banned.iter().copied().filter(|&i| i < values.len()).collect();
    cuts.sort_unstable();
    cuts.dedup();

    let mut best: Option<(i32, usize)> = None; // (max key, start of the first block that reaches it)
    let mut scan = |from: usize, to: usize| {
        let mut start = from;
        while start < to {
            let end = (start + BLOCK).min(to);
            let block_max = values[start..end].iter().map(|&v| key(v)).max().unwrap();
            if best.is_none_or(|(top, _)| block_max > top) {
                best = Some((block_max, start));
            }
            start = end;
        }
    };
    let mut from = 0;
    for &cut in &cuts {
        scan(from, cut);
        from = cut + 1;
    }
    scan(from, values.len());

    match best {
        Some((top, start)) => {
            let end = (start + BLOCK).min(values.len());
            start + values[start..end].iter().position(|&v| key(v) == top).unwrap()
        }
        None => argmax_by_key(values, &[], key),
    }
}

/// Highest logit outside `banned`; falls back to the overall highest if every token is banned.
/// Ties go to the lowest index and values compare with `f32::total_cmp`.
pub fn argmax(logits: &[f32], banned: &[usize]) -> usize {
    argmax_by_key(logits, banned, total_key)
}

/// `argmax` over bf16 logits, without converting them to a Vec<f32> first.
pub fn argmax_bf16(logits: &[bf16], banned: &[usize]) -> usize {
    argmax_by_key(logits, banned, |value| total_key(value.to_f32()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original implementation: a set lookup per element and `max_by` with `total_cmp`.
    fn reference(logits: &[f32], banned: &[usize]) -> usize {
        let set: std::collections::HashSet<usize> = banned.iter().copied().collect();
        let best = |allowed: &dyn Fn(usize) -> bool| {
            (0..logits.len())
                .filter(|&i| allowed(i))
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))
        };
        best(&|i| !set.contains(&i)).or_else(|| best(&|_| true)).unwrap()
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn matches_reference() {
        let mut state = 7u64;
        let specials = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, 1.0, -1.0];
        for round in 0..400 {
            let len = 1 + (lcg(&mut state) as usize % 3000);
            let logits: Vec<f32> = (0..len)
                .map(|_| match lcg(&mut state) % 8 {
                    0 => specials[(lcg(&mut state) % 8) as usize],
                    1 | 2 => (lcg(&mut state) % 5) as f32, // many ties
                    _ => (lcg(&mut state) % 20000) as f32 / 100.0 - 100.0,
                })
                .collect();
            let nbanned = (lcg(&mut state) % 9) as usize;
            let banned: Vec<usize> = (0..nbanned).map(|_| lcg(&mut state) as usize % (len + 3)).collect();
            assert_eq!(argmax(&logits, &banned), reference(&logits, &banned), "round {round}");
            let as_bf16: Vec<bf16> = logits.iter().map(|&v| bf16::from_f32(v)).collect();
            let widened: Vec<f32> = as_bf16.iter().map(|v| v.to_f32()).collect();
            assert_eq!(argmax_bf16(&as_bf16, &banned), reference(&widened, &banned), "bf16 round {round}");
        }
    }

    /// Timing of the production path (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_vocab_argmax() {
        let mut state = 99u64;
        let logits: Vec<bf16> = (0..262_144)
            .map(|_| bf16::from_f32((lcg(&mut state) % 40000) as f32 / 1000.0 - 20.0))
            .collect();
        let banned = [1usize, 50, 100, 101, 106, 258_882, 258_883];
        let started = std::time::Instant::now();
        let mut sink = 0usize;
        for _ in 0..200 {
            sink = sink.wrapping_add(argmax_bf16(&logits, &banned));
        }
        let per = started.elapsed().as_secs_f64() / 200.0 * 1e6;
        let widened: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
        let started = std::time::Instant::now();
        for _ in 0..200 {
            sink = sink.wrapping_add(reference(&widened, &banned));
        }
        let old = started.elapsed().as_secs_f64() / 200.0 * 1e6;
        println!("argmax_bf16 {per:.0} us, old f32+HashSet {old:.0} us (sink {sink})");
    }

    #[test]
    fn everything_banned_falls_back() {
        let logits = [1.0, 3.0, 3.0, 2.0];
        assert_eq!(argmax(&logits, &[0, 1, 2, 3]), 1);
        assert_eq!(argmax(&logits, &[1]), 2);
        assert_eq!(argmax(&logits, &[]), 1);
    }
}
