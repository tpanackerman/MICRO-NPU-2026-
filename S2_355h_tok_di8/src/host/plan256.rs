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
    *MIN.get_or_init(|| std::env::var("S2_P256_MIN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

fn nosplit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("S2_NOSPLIT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(128))
}

/// mid-prompt runs are multiples of 8 rows
fn mid8(n: usize, rest: usize) -> usize {
    if n >= rest { n } else { n - n % 8 }
}

/// Sequential FIFO fill of whole units: `limit` = unit index the fill may not pass (4, or 3 when the last unit is kept for the oldest request's tail), `reserve` = decode rows
/// the last unit must leave free (0 when the last unit is not used by this fill). Returns the planned chunks and the units used.
fn fill(cands: &[(usize, usize, usize)], limit: usize, reserve: usize) -> (Vec<Chunk>, usize) {
    // ns1: with S2_NOSPLIT a fill that would plan nothing falls back to the splitting fill (a pass never ends up without its prefill run)
    let r = fill_inner(cands, limit, reserve, nosplit());
    if r.0.is_empty() && nosplit() > 0 { fill_inner(cands, limit, reserve, 0) } else { r }
}

fn fill_inner(cands: &[(usize, usize, usize)], limit: usize, reserve: usize, nosplit_rows: usize) -> (Vec<Chunk>, usize) {
    let mut chunks = Vec::new();
    let mut u = 0usize;
    for &(session, pos0, rest) in cands {
        if u >= limit {
            break;
        }
        let cap = UNIT * (limit - u) - reserve;
        // ns1 probe (`S2_NOSPLIT=<rows>`, default 0 = off): a prompt remainder of at most that many rows is never split across passes; when it does not fit
        // the rest of this pass it waits (the fill tries the next candidate), so its prefill attention always runs as ONE chunk
        if rest <= nosplit_rows && rest > cap {
            continue;
        }
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
    let plan = plan_hybrid256_base(candidates, d);
    if packcut_on() { packcut_refine(candidates, d, plan) } else { plan }
}

fn plan_hybrid256_base(candidates: &[(usize, usize, usize)], d: usize) -> Vec<Chunk> {
    let Some(&(s1, p1, r1)) = candidates.first() else { return Vec::new() };
    let bd = runtime::hybrid_decode_bucket256(d);
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

/// Tail-aware cut (`S2_PACKCUT=1`, default OFF): the run that ends a pass without finishing its prompt (a "cut") leaves a tail; a tail that is not a multiple of a unit wastes the rest of the
/// unit it later occupies (a run starts at a unit boundary, and beside the decode rows only `64 - Bd` rows of the last unit are usable). When the default plan ends with such a cut the planner
/// tries shorter cuts (8-row steps, up to `S2_PACKCUT_SPAN` rows shorter, default 120): the freed units are refilled by the next candidates, and the cut that maximises
/// `rows * 10 - S2_PACKCUT_GAMMA10 * tail_waste(rest - n)` (default gamma 10 = 1.0) is kept. Pure function of the candidate list and `d`, so `predict_next` reproduces the real plan.
fn packcut_on() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("S2_PACKCUT").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1) != 0)
}

fn packcut_params() -> (usize, i64) {
    static P: std::sync::OnceLock<(usize, i64)> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(d);
        (env("S2_PACKCUT_SPAN", 120), env("S2_PACKCUT_GAMMA10", 10) as i64)
    })
}

/// Unit capacity lost when the last chunk of a request has `t` rows.
fn tail_waste(t: usize, bd: usize) -> usize {
    if t == 0 {
        return 0;
    }
    let cap_pass = UNITS * UNIT - bd;
    let tt = (t - 1) % cap_pass + 1;
    let mut w = UNIT * tt.div_ceil(UNIT) - tt;
    if tt <= UNIT - bd {
        w = w.min(UNIT - bd - tt);
    }
    w
}

/// `fill_inner` with the candidate `cut_s` limited to `cut_n` rows (and the fill going on with the next candidates).
fn fill_cut(cands: &[(usize, usize, usize)], limit: usize, reserve: usize, nosplit_rows: usize, cut_s: usize, cut_n: usize) -> (Vec<Chunk>, usize) {
    let mut chunks = Vec::new();
    let mut u = 0usize;
    for &(session, pos0, rest) in cands {
        if u >= limit {
            break;
        }
        let cap = UNIT * (limit - u) - reserve;
        if session == cut_s {
            let n = mid8(cut_n.min(rest).min(cap), rest);
            if n == 0 {
                continue;
            }
            chunks.push(Chunk { session, pos0, n, last: n == rest });
            u += n.div_ceil(UNIT);
            continue;
        }
        if rest <= nosplit_rows && rest > cap {
            continue;
        }
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

fn packcut_refine(cands: &[(usize, usize, usize)], d: usize, plan: Vec<Chunk>) -> Vec<Chunk> {
    let Some(last) = plan.last() else { return plan };
    if last.last {
        return plan;
    }
    let bd = runtime::hybrid_decode_bucket256(d);
    let (cut_s, cut_n) = (last.session, last.n);
    let Some(&(_, _, rest)) = cands.iter().find(|c| c.0 == cut_s) else { return plan };
    let (span, gamma10) = packcut_params();
    let ns = nosplit();
    let score = |p: &[Chunk]| -> i64 {
        let rows: usize = p.iter().map(|c| c.n).sum();
        let tw = p.iter().find(|c| c.session == cut_s).filter(|c| !c.last).map_or(0, |c| tail_waste(rest - c.n, bd));
        rows as i64 * 10 - gamma10 * tw as i64
    };
    let mut best = plan;
    let mut best_score = score(&best);
    let (s1, p1, r1) = cands[0];
    let mut dn = 8;
    while dn <= span {
        if dn + 8 > cut_n {
            break;
        }
        let n = cut_n - dn;
        dn += 8;
        let alt = if r1 + bd <= UNIT && cands.len() > 1 {
            let (mut ch, un) = fill_cut(&cands[1..], UNITS - 1, 0, ns, cut_s, n);
            if un > 0 {
                ch.push(Chunk { session: s1, pos0: p1, n: r1, last: true });
                ch
            } else {
                fill_cut(cands, UNITS, bd, ns, cut_s, n).0
            }
        } else {
            fill_cut(cands, UNITS, bd, ns, cut_s, n).0
        };
        if alt.is_empty() {
            continue;
        }
        let sc = score(&alt);
        if sc > best_score {
            best = alt;
            best_score = sc;
        }
    }
    best
}

/// Bounded ramp policy (`S2_SJF_THR`, default 0 = off; candidate default 17): while fewer than `thr` requests decode (hysteresis: SJF mode switches ON below `thr` and OFF at or above
/// `S2_SJF_OFF`, default `thr`), the rider slots of the pass are not all used and the first need is riders: the prefilling requests that have waited at most `S2_SJF_MAXWAIT` iterations
/// (default 60) are planned shortest remaining prompt first; requests that waited longer keep FIFO priority in front, so no request starves. `cands` = `(session, position, rows left)` oldest
/// first, `ages[i]` = iterations since the admission of `cands[i]`. `commit` = false (the prediction of the next pass) leaves the mode untouched.
pub fn sjf_order(cands: &mut Vec<(usize, usize, usize)>, ages: &[usize], decoders: usize, commit: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ON: AtomicBool = AtomicBool::new(false);
    static THR: std::sync::OnceLock<(usize, usize, usize, usize)> = std::sync::OnceLock::new();
    let &(thr, off, kmin, maxwait) = THR.get_or_init(|| {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d);
        let thr = env("S2_SJF_THR", 17);
        (thr, env("S2_SJF_OFF", thr), env("S2_SJF_KMIN", 3), env("S2_SJF_MAXWAIT", 90))
    });
    if thr == 0 || cands.len() < kmin.max(2) || ages.len() != cands.len() {
        return;
    }
    let mut on = ON.load(Ordering::Relaxed);
    if decoders < thr {
        on = true;
    } else if decoders >= off {
        on = false;
    }
    if commit {
        ON.store(on, Ordering::Relaxed);
    }
    if !on {
        return;
    }
    let mut keyed: Vec<((usize, usize, usize), usize)> = cands.iter().copied().zip(ages.iter().copied()).collect();
    let (old, mut fresh): (Vec<_>, Vec<_>) = keyed.drain(..).partition(|&(_, age)| age > maxwait);
    fresh.sort_by_key(|&((session, _, rest), _)| (rest, session));
    *cands = old.into_iter().chain(fresh).map(|(c, _)| c).collect();
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
        let max_riders = if runtime::hybrid32_enabled() { 32 } else { MAX_RIDERS };
        let mut riders = decoders.min(max_riders);
        let plan = plan_hybrid256(candidates, riders);
        let rows: usize = plan.iter().map(|c| c.n).sum();
        if rows >= plan_min_rows() {
            debug_assert!(runtime::hybrid_fits_any(&plan.iter().map(|c| c.n).collect::<Vec<_>>(), riders));
            // rider trim (`S2_RIDER_TRIM=0` off): the LM head serves the riders and the rows that finish a prompt in groups of 16 rows per launch; one finishing row on top of 16 riders costs a second
            // launch, so ride 16 - finishing and keep the 16-row bucket (>= 9 riders, the unit layout does not change).
            if rider_trim() && riders > 8 {
                let finishing = plan.iter().filter(|c| c.last).count();
                let bucket_limit = if riders > 16 { 32 } else { MAX_RIDERS };
                if riders + finishing > bucket_limit && bucket_limit - finishing >= 9 {
                    riders = bucket_limit - finishing;
                }
            }
            return (plan, riders);
        }
    }
    let riders = if !candidates.is_empty() { decoders.min(if cap128 > 0 { cap128.min(16) } else { 16 }) } else { decoders };
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
