//! Planner of the 256-row hybrid pass (`S2_PASS_ROWS=256`, engine `runtime/hybrid256.rs`): up to four 64-row prefill units plus 8 / 16 decode rows that sit in the pad
//! rows of the last unit.
//!
//! Rules (the policy simulator, `research/policy.md` 5.2, shows that packers that reorder completion lose 10-17 %):
//! - FIFO: the oldest request is served first and takes as many rows as the units allow; leftover units go to the NEXT requests only;
//! - the exception that keeps FIFO completion: when the oldest request's remainder fits beside the decode rows (`<= 64 - Bd` rows) it runs in the LAST unit and the
//!   following requests take the whole units before it (the same swap as the 128-row `fills` packer);
//! - runs start at unit boundaries (`rpa_host::prefill_tables`), mid-prompt cuts are multiples of 8 rows (8-aligned positions for the full layers);
//! - a plan below `S2_P256_MIN` rows is not worth a 256-row pass: the 128-row plan (`worker::plan_hybrid`) is used instead.

use crate::host::runtime::{self, hybrid_decode_bucket};
use crate::host::worker::{Chunk, plan_hybrid};

/// Rows of one prefill unit.
pub const UNIT: usize = 64;
/// Units of the 256-row pass.
pub const UNITS: usize = 4;
/// Decode rows a 256-row pass carries (the 16-row bucket).
pub const MAX_RIDERS: usize = 16;

/// Smallest prefill row count for which a 256-row plan replaces the 128-row plan (env `S2_P256_MIN`, default 190).
pub fn plan_min_rows() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("S2_P256_MIN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(190))
}

/// mid-prompt runs are multiples of 8 rows
fn mid8(n: usize, rest: usize) -> usize {
    if n >= rest { n } else { n - n % 8 }
}

/// Sequential FIFO fill of whole units: `limit` = unit index the fill may not pass (4, or 3 when the last unit is kept for the oldest request's tail), `reserve` = decode rows
/// the last unit must leave free (0 when the last unit is not used by this fill). Returns the planned chunks and the units used.
fn fill(cands: &[(usize, usize, usize)], limit: usize, reserve: usize) -> (Vec<Chunk>, usize) {
    let mut chunks = Vec::new();
    let mut u = 0usize;
    for &(session, pos0, rest) in cands {
        if u >= limit {
            break;
        }
        let cap = UNIT * (limit - u) - reserve;
        let n = mid8(rest.min(cap), rest);
        if n == 0 {
            break;
        }
        chunks.push(Chunk { session, pos0, n, last: n == rest });
        u += n.div_ceil(UNIT);
        if n < rest {
            break;
        }
    }
    (chunks, u)
}

/// The prefill runs of a 256-row pass. `candidates` = `(session, next position, prompt tokens left)` oldest first; `d` = decode rows riding the pass (at most 16).
pub fn plan_hybrid256(candidates: &[(usize, usize, usize)], d: usize) -> Vec<Chunk> {
    let Some(&(s1, p1, r1)) = candidates.first() else { return Vec::new() };
    let bd = hybrid_decode_bucket(d);
    // the oldest request's remainder fits beside the decode rows and others are waiting: the others take the units before it
    if r1 + bd <= UNIT && candidates.len() > 1 {
        let (mut chunks, units) = fill(&candidates[1..], UNITS - 1, 0);
        if units > 0 {
            chunks.push(Chunk { session: s1, pos0: p1, n: r1, last: true });
            return chunks;
        }
    }
    fill(candidates, UNITS, bd).0
}

/// `S2_RIDER_TRIM=0` turns the rider trim off (default ON).
fn rider_trim() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_RIDER_TRIM").map(|v| v != "0").unwrap_or(true))
}

/// The plan of one iteration and how many decode rows ride it. `candidates` = the fast prefilling requests (oldest first), `decoders` = fast decode rows waiting, `cap128` =
/// the decode-rider cap of the 128-row pass (`S2_DEC_CAP`, 0 = unlimited). With `S2_PASS_ROWS=256`, a 256-row plan with 16 riders is tried first and used when it carries at
/// least `S2_P256_MIN` prefill rows; otherwise the 128-row plan with `cap128` riders.
pub fn plan_pass(candidates: &[(usize, usize, usize)], decoders: usize, cap128: usize) -> (Vec<Chunk>, usize) {
    if runtime::pass_rows() >= 256 && !candidates.is_empty() {
        let mut riders = decoders.min(MAX_RIDERS);
        let plan = plan_hybrid256(candidates, riders);
        let rows: usize = plan.iter().map(|c| c.n).sum();
        if rows >= plan_min_rows() {
            debug_assert!(runtime::hybrid_fits_any(&plan.iter().map(|c| c.n).collect::<Vec<_>>(), riders));
            // rider trim (`S2_RIDER_TRIM=0` off): the LM head serves the riders and the rows that finish a prompt in groups of 16 rows per launch; one finishing row on top of 16 riders costs a second
            // launch, so ride 16 - finishing and keep the 16-row bucket (>= 9 riders, the unit layout does not change).
            if rider_trim() && riders > 8 {
                let finishing = plan.iter().filter(|c| c.last).count();
                if riders + finishing > MAX_RIDERS && MAX_RIDERS - finishing >= 9 {
                    riders = MAX_RIDERS - finishing;
                }
            }
            return (plan, riders);
        }
    }
    let riders = if cap128 > 0 && !candidates.is_empty() { decoders.min(cap128) } else { decoders };
    (plan_hybrid(candidates, riders), riders)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(plan: &[Chunk]) -> Vec<(usize, usize)> {
        plan.iter().map(|c| (c.session, c.n)).collect()
    }

    #[test]
    fn oldest_takes_all_units_with_decode_rows_in_the_last() {
        assert_eq!(rows(&plan_hybrid256(&[(0, 0, 1000)], 8)), vec![(0, 248)]);
        assert_eq!(rows(&plan_hybrid256(&[(0, 0, 1000)], 16)), vec![(0, 240)]);
        assert_eq!(rows(&plan_hybrid256(&[(0, 0, 1000)], 0)), vec![(0, 256)]);
    }

    #[test]
    fn leftover_units_go_to_the_next_request() {
        assert_eq!(rows(&plan_hybrid256(&[(0, 0, 100), (1, 0, 900)], 8)), vec![(0, 100), (1, 120)]);
        assert_eq!(rows(&plan_hybrid256(&[(0, 0, 256), (1, 0, 900)], 8)), vec![(0, 248)]);
    }

    #[test]
    fn short_tail_rides_in_the_last_unit() {
        let plan = plan_hybrid256(&[(0, 0, 40), (1, 0, 900), (2, 0, 500)], 8);
        assert_eq!(rows(&plan), vec![(1, 192), (0, 40)]);
        assert!(plan.last().unwrap().last);
        let plan = plan_hybrid256(&[(0, 0, 40), (1, 0, 100), (2, 0, 500)], 8);
        assert_eq!(rows(&plan), vec![(1, 100), (2, 64), (0, 40)]);
    }

    #[test]
    fn mid_cuts_are_multiples_of_eight() {
        for d in [0, 8, 16] {
            for rest in [300, 777, 1030] {
                let plan = plan_hybrid256(&[(0, 5 * 8, rest)], d);
                assert!(plan.iter().all(|c| c.last || c.n % 8 == 0));
            }
        }
    }
}
