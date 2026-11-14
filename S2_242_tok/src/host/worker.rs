//! Request scheduling and KV ownership for the fixed HTTP frontend: a Sarathi-style loop.
//!
//! Every iteration runs ALL decode rows (one pending token per decoding request, as one
//! `decode_batch`) plus a prefill token budget, made of chunks of up to `chunk` prompt tokens taken
//! from the oldest prefilling requests first. A chunk never crosses a multiple of `chunk`, so it
//! never wraps the 1024 sliding ring or crosses a 512-position full-attention page.
//!
//! Environment overrides (read once at startup):
//! - `S2_CHUNK`: chunk size, a divisor of 512 (default 64);
//! - `S2_CAPACITY`: requests computed together, R (default `CAPACITY`);
//! - `S2_PREFILL_BUDGET`: prompt tokens per iteration (default `PREFILL_BUDGET`);
//! - `S2_DEC_CAP`: most decode rows that ride a hybrid pass next to prefill rows (default `DEC_CAP` = 8; 0 = unlimited). Decoding sessions are never refused: the ones over the cap skip
//!   the pass and the least recently served go first (round robin), so the pass keeps the 8-row decode bucket (120 prefill rows) instead of the 16-row bucket (112 rows, +3.5 ms);
//! - `S2_PACK_SWAP`: swap packing in `plan_hybrid` (default on, 0 = off).

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use crate::api::worker::{Job, StreamEvent};
use crate::axes::Tf;
use crate::host::generate::{Delta, Generation, Next, aligned_chunk};
use crate::host::load;
use crate::host::model::{HybDecode, HybPrefill, Model};
use crate::host::prof;
use crate::host::runtime::{Blocks, DecDesc, PassDesc, RunDesc, Workspace};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;

/// Default number of requests computed together (R); more wait in the queue. Per request, the
/// sliding KV is about 335 MB and the full-attention pool takes `CONTEXT / 512` blocks.
pub const CAPACITY: usize = 16;
/// Default admitted sessions with `S2_PASS_ROWS=256` (the flat caches are built for `Rn` = 18 slots; 16 riders per pass need a deeper admitted pool, R11 t256c).
pub const CAPACITY_256: usize = 18;
/// Prompt plus generated positions allowed per request.
pub const CONTEXT: usize = 8192;
/// Default prefill chunk size (tokens).
pub const CHUNK: usize = 128;
/// Default prompt tokens computed per iteration, next to every decode row.
pub const PREFILL_BUDGET: usize = 128;
/// Profiling names for the riders-per-hybrid-pass histogram (`rd00` .. `rd16`: count of hybrid passes with that many decode riders).
const RIDE_NAMES: [&str; 17] = [
    "rd00", "rd01", "rd02", "rd03", "rd04", "rd05", "rd06", "rd07", "rd08", "rd09", "rd10", "rd11", "rd12", "rd13", "rd14", "rd15", "rd16",
];

/// Cap of the decode rows of a decode-only iteration (`S2_DECODE_ONLY_CAP`): 16 with `S2_PASS_ROWS=256` (one `decode_batch` group), 0 (no cap) otherwise.
fn decode_only_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let default = if crate::host::runtime::pass_rows() >= 256 { 16 } else { 0 };
        std::env::var("S2_DECODE_ONLY_CAP").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default)
    })
}

/// Default cap of the decode rows that ride a hybrid pass (the Bd = 8 bucket); 0 = no cap.
pub const DEC_CAP: usize = 8;

/// Scheduler parameters.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub chunk: usize,
    pub capacity: usize,
    pub budget: usize,
    /// Decode rows per hybrid pass (0 = every decoding session rides).
    pub dec_cap: usize,
    /// Hybrid pass (prefill runs + decode rows in one 128-row batch) for requests that fit the flat KV caches; `S2_HYBRID=0` sends every request through the legacy per-slot path.
    pub hybrid: bool,
}

impl Config {
    pub fn from_env() -> Self {
        let read = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(default)
        };
        let mut chunk = read("S2_CHUNK", CHUNK);
        if Tf::SIZE % chunk != 0 {
            eprintln!("S2 warning: S2_CHUNK={chunk} does not divide {}; using {CHUNK}", Tf::SIZE);
            chunk = CHUNK;
        }
        Self {
            chunk,
            capacity: read("S2_CAPACITY", if crate::host::runtime::pass_rows() >= 256 { CAPACITY_256 } else { CAPACITY }),
            budget: read("S2_PREFILL_BUDGET", PREFILL_BUDGET),
            dec_cap: std::env::var("S2_DEC_CAP").ok().and_then(|value| value.trim().parse::<usize>().ok()).unwrap_or(DEC_CAP),
            hybrid: std::env::var("S2_HYBRID").map(|value| value != "0").unwrap_or(true),
        }
    }
}

pub struct Worker {
    runtime: tokio::runtime::Runtime,
    device: Device,
    model: Model,
    workspace: Workspace,
    tokenizer: Tokenizer,
    config: Config,
}

struct Session {
    generation: Generation,
    blocks: Blocks,
    events: mpsc::Sender<StreamEvent>,
    stream: bool,
    /// The worker saw the job (it waits in `pending` until a slot is free).
    received: Instant,
    /// Prompt computation starts (after the multimodal precompute).
    prefill_start: Instant,
    /// Milliseconds spent waiting for a slot, and encoding images/audio at admission.
    wait_ms: f64,
    encode_ms: f64,
    first_token: Option<Instant>,
    prompt: usize,
    /// Fast request: its KV lives in the flat caches (`Blocks::flat`) and it runs through the hybrid / batched-decode passes. Others use the legacy per-slot path.
    fast: bool,
    /// Iteration number of the last hybrid / batched pass this session rode with a decode row (0 = never; set when its prompt finishes): the decode-rider cap serves the least recent first.
    last_ride: usize,
}

/// One prefill chunk of the iteration plan.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub session: usize,
    pub pos0: usize,
    pub n: usize,
    /// Ends with the last prompt token.
    pub last: bool,
}

/// The prefill chunks of one iteration. `requests` yields `(session, next position, prompt tokens
/// left)` oldest request first; at most `budget` tokens are planned, in chunks of at most `chunk`
/// tokens that never cross a multiple of `chunk`. Several chunks of one request may follow each
/// other when the budget exceeds one chunk.
///
/// `whole`: do not plan a chunk that the remaining budget would cut short (and that is not the tail of its prompt). Off by default: the chunk
/// kernels take any start position inside a 512-page, so a cut chunk is fine.
fn plan_prefill(
    requests: impl IntoIterator<Item = (usize, usize, usize)>,
    chunk: usize,
    budget: usize,
    whole: bool,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut budget = budget;
    for (session, mut pos, mut remaining) in requests {
        while remaining > 0 && budget > 0 {
            let n = aligned_chunk(pos, remaining, chunk, budget);
            if whole && n < aligned_chunk(pos, remaining, chunk, chunk) {
                break;
            }
            chunks.push(Chunk { session, pos0: pos, n, last: n == remaining });
            (pos, remaining, budget) = (pos + n, remaining - n, budget - n);
        }
    }
    chunks
}

/// Swap packing on (`S2_PACK_SWAP`, default on; 0 = off), read once.
fn pack_swap() -> bool {
    static SWAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SWAP.get_or_init(|| std::env::var("S2_PACK_SWAP").map(|value| value.trim() != "0").unwrap_or(true))
}

/// The prefill runs of a hybrid pass. `candidates` = `(session, next position, prompt tokens left)` of the fast prefilling requests, oldest first; `d` = fast decode rows.
/// The pass has 128 rows: the decode rows take the last `Bd` (8 or 16), the oldest request takes up to `128 - Bd` rows (any length; mid-prompt runs are multiples of 8,
/// so the positions stay 8-aligned for the full layers); when its remainder fits unit 0 (<= 64 rows), the next request takes up to `64 - Bd` rows in unit 1.
///
/// Swap packing (`S2_PACK_SWAP`): when the oldest request's remainder also fits next to the decode rows (`r1 + Bd <= 64`) and the next request needs more than the `64 - Bd` rows that
/// unit 1 would leave it, the next request takes the WHOLE of unit 0 and the oldest request's tail rides unit 1 beside the decode rows (`Bd` more prompt rows per pass). The oldest
/// request still completes first (its tail is in this pass), so the completion order stays FIFO.
pub fn plan_hybrid(candidates: &[(usize, usize, usize)], d: usize) -> Vec<Chunk> {
    plan_hybrid_with(candidates, d, pack_swap())
}

pub fn plan_hybrid_with(candidates: &[(usize, usize, usize)], d: usize, swap: bool) -> Vec<Chunk> {
    let Some(&(s1, p1, r1)) = candidates.first() else { return Vec::new() };
    let bd = crate::host::runtime::hybrid_decode_bucket(d);
    let cap = crate::host::runtime::HYBRID_ROWS - bd;
    let n1 = r1.min(cap);
    let mut chunks = vec![Chunk { session: s1, pos0: p1, n: n1, last: n1 == r1 }];
    if n1 <= 64 {
        if let Some(&(s2, p2, r2)) = candidates.get(1) {
            if swap && bd > 0 && r1 + bd <= 64 && r2 > 64 - bd {
                // unit 0 = next request (64 rows, or its whole remainder when shorter), unit 1 = the oldest request's tail + the decode rows. Mid-prompt runs are multiples of 8.
                let n2 = r2.min(64);
                chunks = vec![Chunk { session: s2, pos0: p2, n: n2, last: n2 == r2 }, Chunk { session: s1, pos0: p1, n: r1, last: true }];
            } else {
                let n2 = r2.min(64 - bd);
                if n2 > 0 {
                    chunks.push(Chunk { session: s2, pos0: p2, n: n2, last: n2 == r2 });
                }
            }
        }
    }
    debug_assert!(crate::host::runtime::hybrid_fits(&chunks.iter().map(|c| c.n).collect::<Vec<_>>(), d));
    chunks
}

/// The next hybrid pass as far as its tables go (`PassDesc`), predicted while the current pass is planned: every decode row rider advances one position (a request that reaches its
/// `max_new_tokens` leaves; a stop token cannot be foreseen), every request whose prompt ends in this pass becomes a decode row at position `prompt_len`, then the same decode-rider
/// cap and `plan_hybrid` run on that shadow state. New arrivals are not foreseen. The engine builds the tables of the prediction on a thread while the device runs the current pass and
/// uses them only if the real pass has an equal description, so a wrong prediction costs nothing but the thread.
fn predict_next(sessions: &[Session], hybrid_chunks: &[Chunk], fast_decode: &[usize], iterations: usize, config: &Config) -> Option<(PassDesc, Vec<Chunk>)> {
    struct Shadow {
        slot: usize,
        table: Vec<usize>,
        prompt_len: usize,
        max_new: usize,
        pos: usize,
        remaining: usize,
        decode: bool,
        last_ride: usize,
    }
    let mut shadow: Vec<Option<Shadow>> = sessions
        .iter()
        .map(|session| {
            let g = &session.generation;
            if !session.fast || !(g.is_decode() || g.is_prefill()) {
                return None;
            }
            Some(Shadow {
                slot: session.blocks.sliding,
                table: session.blocks.flat.clone(),
                prompt_len: g.prompt_len(),
                max_new: g.max_new_tokens(),
                pos: g.position(),
                remaining: g.prompt_remaining(),
                decode: g.is_decode(),
                last_ride: session.last_ride,
            })
        })
        .collect();
    for &i in fast_decode {
        let Some(x) = shadow[i].as_mut() else { continue };
        if x.pos + 2 - x.prompt_len >= x.max_new {
            shadow[i] = None;
        } else {
            x.pos += 1;
            x.last_ride = iterations + 1;
        }
    }
    for chunk in hybrid_chunks {
        let Some(x) = shadow[chunk.session].as_mut() else { continue };
        if chunk.last {
            if x.max_new <= 1 {
                shadow[chunk.session] = None;
            } else {
                x.decode = true;
                x.pos = x.prompt_len;
                x.remaining = 0;
                x.last_ride = iterations + 1;
            }
        } else {
            x.pos += chunk.n;
            x.remaining -= chunk.n;
        }
    }
    let candidates: Vec<(usize, usize, usize)> = shadow
        .iter()
        .enumerate()
        .filter_map(|(i, x)| x.as_ref().filter(|x| !x.decode && x.remaining > 0).map(|x| (i, x.pos, x.remaining)))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let mut decode: Vec<usize> = shadow.iter().enumerate().filter_map(|(i, x)| x.as_ref().filter(|x| x.decode).map(|_| i)).collect();
    let (chunks, riders) = crate::host::plan256::plan_pass(&candidates, decode.len(), config.dec_cap);
    if decode.len() > riders {
        decode.sort_by_key(|&i| (shadow[i].as_ref().map_or(0, |x| x.last_ride), i));
        decode.truncate(riders);
        decode.sort_unstable();
    }
    if chunks.is_empty() {
        return None;
    }
    let runs = chunks
        .iter()
        .map(|c| {
            let x = shadow[c.session].as_ref().expect("planned from a live shadow");
            RunDesc { slot: x.slot, table: x.table.clone(), pos0: c.pos0, n: c.n, last: c.last }
        })
        .collect();
    let decs = decode
        .iter()
        .map(|&i| {
            let x = shadow[i].as_ref().expect("live decode shadow");
            DecDesc { slot: x.slot, table: x.table.clone(), pos: x.pos }
        })
        .collect();
    Some((PassDesc { runs, decs }, chunks))
}

/// The prefill rows of the predicted chunks (R13 pre-upload): request id and the embedding source of every row, in chunk order. `None` when a chunk cannot be reproduced exactly
/// (the pre-upload is then skipped for that pass; the tables prebuild is unaffected).
fn hint_rows(sessions: &[Session], chunks: &[Chunk]) -> Option<Vec<crate::host::runtime::HintRun>> {
    use crate::host::runtime::{HintRow, HintRun};
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let prompt = sessions[chunk.session].generation.prompt_chunk_from(chunk.pos0, chunk.n)?;
        let inputs = prompt.inputs();
        if inputs.len() != chunk.n {
            return None;
        }
        let request = inputs.first()?.request;
        let mut rows = Vec::with_capacity(inputs.len());
        for input in &inputs {
            rows.push(match (input.soft, input.embeddings) {
                (Some(token), Some(embeddings)) => HintRow::Soft(embeddings.host_row(token)?.to_vec()),
                (Some(_), None) => return None,
                (None, _) => HintRow::Text(input.token),
            });
        }
        out.push(HintRun { request, rows });
    }
    Some(out)
}

/// R14 d13e P1: the lanes of the transition from the pass being planned (`chunks` = its hybrid runs, `fast_decode` = its riders, in pass order) into the predicted next pass `next`.
/// Lane j = bucket row 240 + j of this pass: its riders, then (P0) the rows that finish a prompt. `None` when the transition cannot be fed (not two 16-bucket 256-row passes, a sampled
/// row, a next-pass row whose token neither the device nor the host has).
fn d13e_feed_plan(sessions: &[Session], chunks: &[Chunk], fast_decode: &[usize], next: &PassDesc) -> Option<crate::host::runtime::d13e::FeedPlan> {
    use crate::host::runtime::d13e;
    let d = fast_decode.len();
    let counts: Vec<usize> = chunks.iter().map(|c| c.n).collect();
    if chunks.is_empty() || !crate::host::runtime::hybrid_is_256(&counts, d) || crate::host::runtime::hybrid_decode_bucket(d) != 16 {
        return None;
    }
    let ncounts: Vec<usize> = next.runs.iter().map(|r| r.n).collect();
    if next.runs.is_empty() || !crate::host::runtime::hybrid_is_256(&ncounts, next.decs.len()) || crate::host::runtime::hybrid_decode_bucket(next.decs.len()) != 16 {
        return None;
    }
    let finishing: Vec<usize> = chunks.iter().filter(|c| c.last).map(|c| c.session).collect();
    let mut lane_sessions: Vec<usize> = fast_decode.to_vec();
    if d + finishing.len() <= d13e::LANES && finishing.len() <= 8 {
        lane_sessions.extend(&finishing);
    }
    let mut k_lanes = [None; d13e::LANES];
    for (j, &i) in lane_sessions.iter().enumerate() {
        let g = &sessions[i].generation;
        if !g.is_greedy() {
            return None;
        }
        k_lanes[j] = Some((sessions[i].blocks.sliding, g.ban().index() as u8));
    }
    let mut next_rows = [None; d13e::LANES];
    for (b, dec) in next.decs.iter().enumerate() {
        next_rows[b] = Some(dec.slot);
    }
    d13e::plan_lanes(&k_lanes, &next_rows, |slot| {
        let session = sessions.iter().find(|s| s.fast && s.blocks.sliding == slot)?;
        let g = &session.generation;
        (g.is_decode() && g.is_greedy()).then(|| g.input().0.token)
    })
}

fn fail(sessions: &[Session], extra: Option<&mpsc::Sender<StreamEvent>>, message: &str) {
    for events in sessions.iter().map(|session| &session.events).chain(extra) {
        let _ = events.send(StreamEvent::Error(message.to_owned()));
    }
}

impl Worker {
    pub fn new(tokenizer: Tokenizer) -> Result<Self, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let config = Config::from_env();
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut device = Device::new(crate::ops::embed_token.topology())?;
        let (model, workspace) = runtime.block_on(async {
            let weights = load::load_weights(&mut device).await?;
            let mut workspace = Workspace::new(&mut device, &weights, CONTEXT, config.capacity).await?;
            let mut model = Model::new(&mut device, weights, &mut workspace).await?;
            // R14 d13e P1 (S2_D13E=2 only): the 2 GiB row table, allocated after the weights and the workspace
            model.d13e_init(&mut device).await;
            Ok::<_, Box<dyn std::error::Error>>((model, workspace))
        })?;
        crate::host::cpuplace::place();
        eprintln!(
            "S2 startup_ms={:.0} capacity={} chunk={} budget={}",
            prof::since(started),
            config.capacity,
            config.chunk,
            config.budget
        );
        Ok(Self {
            runtime,
            device,
            model,
            workspace,
            tokenizer,
            config,
        })
    }

    pub fn run(mut self, jobs: mpsc::Receiver<Job>) {
        let config = self.config;
        let mut sessions = Vec::<Session>::new(); // oldest first
        let mut pending = VecDeque::<(Instant, Job)>::new();
        let (mut iterations, mut rows_total, mut decode_total, mut prefill_total) = (0usize, 0usize, 0usize, 0usize);
        let mut rings: Vec<_> = (0..config.capacity).rev().collect();
        // Pages of the flat full-layer pool of the batched decode path (`Blocks::flat`); a request that fits `decode::LIMIT` positions takes up to 5.
        let mut flat_free: Vec<usize> = if self.workspace.has_flat() { (0..crate::axes::Pn::SIZE).rev().collect() } else { Vec::new() };
        // Every admitted request can reach its limit without reclaiming another request's KV.
        let mut free: Vec<_> = (0..config.capacity * Workspace::blocks_for(CONTEXT)).rev().collect();
        loop {
            if sessions.is_empty() && pending.is_empty() {
                if rows_total > 0 {
                    prof::flush(&format!(
                        "S2ITER it={iterations} rows={rows_total} dec={decode_total} pre={prefill_total}"
                    ));
                    (rows_total, decode_total, prefill_total) = (0, 0, 0);
                }
                let Ok(job) = jobs.recv() else { break };
                pending.push_back((Instant::now(), job));
            }
            while let Ok(job) = jobs.try_recv() {
                pending.push_back((Instant::now(), job));
            }

            // Admission: a free slot, then every soft-token embedding of the request, once.
            while sessions.len() < config.capacity {
                let Some((received, job)) = pending.pop_front() else { break };
                let admitted = Instant::now();
                let empty = job.request.max_new_tokens == 0;
                let mut generation = match Generation::new(&self.tokenizer, job.request, CONTEXT) {
                    Ok(generation) => generation,
                    Err(error) => {
                        let _ = job.events.send(StreamEvent::Error(error.to_string()));
                        continue;
                    }
                };
                if empty {
                    let output = generation.advance(Next::None, |_| true).unwrap().unwrap();
                    let _ = job.events.send(StreamEvent::Done(output));
                    continue;
                }
                if generation.image().is_some() || generation.audio().is_some() {
                    let embedded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.runtime.block_on(self.model.embed_media(
                            &mut self.device,
                            generation.image(),
                            generation.audio(),
                        ))
                    }));
                    match embedded {
                        Ok(Ok(embeddings)) => generation.attach_embeddings(embeddings),
                        failed => {
                            let message = match failed {
                                Ok(Err(error)) => error.to_string(),
                                _ => "media encoding panicked; worker is offline".to_owned(),
                            };
                            fail(&sessions, Some(&job.events), &message);
                            return;
                        }
                    }
                }
                let prefill_start = Instant::now();
                // Batched decode path: the request must fit the flat caches (positions < LIMIT) and its pages must be available.
                let positions = generation.prompt_len() + generation.max_new_tokens();
                let flat_pages = Workspace::blocks_for(positions);
                let fast = config.hybrid
                    && self.workspace.has_flat()
                    && positions <= crate::host::runtime::decode_limit()
                    && generation.prompt_len() <= crate::host::runtime::HYBRID_PREFILL_LIMIT
                    && flat_free.len() >= flat_pages;
                let flat: Vec<usize> = if fast { (0..flat_pages).map(|_| flat_free.pop().expect("checked")).collect() } else { Vec::new() };
                sessions.push(Session {
                    generation,
                    blocks: Blocks {
                        full: Vec::new(),
                        sliding: rings.pop().expect("a slot is free below capacity"),
                        flat,
                    },
                    events: job.events,
                    stream: job.stream,
                    received,
                    prefill_start,
                    wait_ms: (admitted - received).as_secs_f64() * 1e3,
                    encode_ms: (prefill_start - admitted).as_secs_f64() * 1e3,
                    first_token: None,
                    prompt: 0,
                    fast,
                    last_ride: 0,
                });
            }
            if sessions.is_empty() {
                continue;
            }

            let t_plan = Instant::now();
            // Plan. Fast requests (flat KV caches): every fast decode row plus up to two prefill runs share ONE hybrid pass (`plan_hybrid`); with no prefill pending, the fast
            // decode rows run as the batched decode pass. Legacy requests (no flat KV): their decode rows, then up to `budget` prompt tokens in chunks (the per-slot path).
            let mut decode: Vec<usize> = (0..sessions.len()).filter(|&i| sessions[i].generation.is_decode()).collect();
            let mut fast_decode: Vec<usize> = decode.iter().copied().filter(|&i| sessions[i].fast).collect();
            let legacy_decode: Vec<usize> = decode.iter().copied().filter(|&i| !sessions[i].fast).collect();
            let fast_prefill: Vec<(usize, usize, usize)> = sessions
                .iter()
                .enumerate()
                .filter(|(_, session)| session.fast && session.generation.is_prefill())
                .map(|(index, session)| (index, session.generation.position(), session.generation.prompt_remaining()))
                .collect();
            // Decode-rider cap: next to prefill rows only `dec_cap` decode rows ride the pass (the 8-row bucket: 120 prefill rows, no 16-row bucket). The sessions left out skip this pass
            // untouched (no position, KV or ring change); the least recently served go first, ties by age. Decode-only passes keep every row.
            // (`S2_PASS_ROWS=256`: a 256-row plan carries up to 16 riders, see `plan256::plan_pass`.)
            let (planned, riders) = crate::host::plan256::plan_pass(&fast_prefill, fast_decode.len(), config.dec_cap);
            let cap_bound = !fast_prefill.is_empty() && fast_decode.len() > riders;
            if cap_bound {
                fast_decode.sort_by_key(|&i| (sessions[i].last_ride, i));
                fast_decode.truncate(riders);
                fast_decode.sort_unstable();
                decode = legacy_decode.iter().chain(&fast_decode).copied().collect();
                decode.sort_unstable();
            }
            // Decode-only iterations (no fast prefill pending) run `decode_batch` in 16-row groups of ~30 ms each; with the 256-row pool of up to 18 sessions a group of 2 more rows costs a whole extra
            // launch (~25 ms), so `S2_DECODE_ONLY_CAP` (default 16 with S2_PASS_ROWS=256, 0 = no cap) lets the 16 least recently served ride and the others skip, like the hybrid rider cap.
            let dec_only_cap = decode_only_cap();
            if fast_prefill.is_empty() && dec_only_cap > 0 && fast_decode.len() > dec_only_cap {
                fast_decode.sort_by_key(|&i| (sessions[i].last_ride, i));
                fast_decode.truncate(dec_only_cap);
                fast_decode.sort_unstable();
                decode = legacy_decode.iter().chain(&fast_decode).copied().collect();
                decode.sort_unstable();
                prof::add("cap_only", std::time::Duration::ZERO);
            }
            let mut chunks = planned;
            let n_hybrid = chunks.len();
            chunks.extend(plan_prefill(
                sessions
                    .iter()
                    .enumerate()
                    .filter(|(_, session)| !session.fast)
                    .map(|(index, session)| (index, session.generation.position(), session.generation.prompt_remaining())),
                config.chunk,
                config.budget,
                false,
            ));
            // Full-attention blocks for every position this iteration writes.
            for &index in &decode {
                let session = &mut sessions[index];
                let need = Workspace::blocks_for(session.generation.position() + 1);
                while session.blocks.full.len() < need {
                    session.blocks.full.push(free.pop().expect("admission bounds KV usage"));
                }
            }
            for chunk in &chunks {
                let session = &mut sessions[chunk.session];
                let need = Workspace::blocks_for(chunk.pos0 + chunk.n);
                while session.blocks.full.len() < need {
                    session.blocks.full.push(free.pop().expect("admission bounds KV usage"));
                }
            }

            if n_hybrid > 0 && crate::host::runtime::hybrid_prebuild_enabled() {
                let predicted = predict_next(&sessions, &chunks[..n_hybrid], &fast_decode, iterations, &config);
                let rows = if crate::host::runtime::preup_enabled() { predicted.as_ref().and_then(|(_, next)| hint_rows(&sessions, next)) } else { None };
                let feed = if crate::host::runtime::d13e::level() >= 2 && self.model.d13e_feed_on() {
                    predicted.as_ref().and_then(|(desc, _)| d13e_feed_plan(&sessions, &chunks[..n_hybrid], &fast_decode, desc))
                } else {
                    None
                };
                self.model.set_hybrid_hint(predicted.map(|(desc, _)| desc));
                self.model.set_hybrid_hint_rows(rows);
                self.model.set_hybrid_hint_feed(feed);
            }
            prof::add("w_plan", t_plan.elapsed());
            let iteration = Instant::now();
            let prefill_rows: usize = chunks.iter().map(|chunk| chunk.n).sum();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.runtime.block_on(async {
                    // Decode rows that do not ride the hybrid pass: all of them when no prefill run is hybrid, else the legacy ones.
                    let batched: &[usize] = if n_hybrid == 0 { &decode } else { &legacy_decode };
                    if !batched.is_empty() {
                        let (inputs, positions): (Vec<_>, Vec<_>) = batched.iter().map(|&i| sessions[i].generation.input()).unzip();
                        let blocks: Vec<_> = batched.iter().map(|&i| &sessions[i].blocks).collect();
                        prof::timed!("decode_batch", self
                            .model
                            .decode_batch(&mut self.device, &inputs, &positions, &blocks, &mut self.workspace)
                            .await)?;
                    }
                    if n_hybrid > 0 {
                        let prompts: Vec<_> = chunks[..n_hybrid]
                            .iter()
                            .map(|chunk| {
                                sessions[chunk.session]
                                    .generation
                                    .prompt_chunk_from(chunk.pos0, chunk.n)
                                    .expect("planned chunk lies inside the prompt")
                            })
                            .collect();
                        let prefills: Vec<HybPrefill<'_>> = chunks[..n_hybrid]
                            .iter()
                            .zip(&prompts)
                            .map(|(chunk, prompt)| HybPrefill {
                                blocks: &sessions[chunk.session].blocks,
                                pos0: chunk.pos0,
                                inputs: prompt.inputs(),
                                want_next: chunk.last,
                            })
                            .collect();
                        let decodes: Vec<HybDecode<'_>> = fast_decode
                            .iter()
                            .map(|&i| {
                                let (input, pos) = sessions[i].generation.input();
                                HybDecode { blocks: &sessions[i].blocks, input, pos }
                            })
                            .collect();
                        let t_hs = Instant::now();
                        let stepped = self.model.hybrid_step(&mut self.device, &prefills, &decodes, &mut self.workspace).await;
                        let hs = t_hs.elapsed();
                        prof::add("hybrid_step", hs);
                        // instrumentation: pass time by engine (hs256_*: the 256-row engine) and decode bucket, riders per pass histogram, how often the cap left decoders out
                        let counts: Vec<usize> = prefills.iter().map(|p| p.inputs.len()).collect();
                        let e256 = crate::host::runtime::hybrid_is_256(&counts, decodes.len());
                        let bucket = match (e256, decodes.len()) {
                            (false, 0) => "hs_b0",
                            (false, 1..=8) => "hs_b8",
                            (false, _) => "hs_b16",
                            (true, 0) => "hs256_b0",
                            (true, 1..=8) => "hs256_b8",
                            (true, _) => "hs256_b16",
                        };
                        prof::add(bucket, hs);
                        prof::add(RIDE_NAMES[decodes.len().min(16)], std::time::Duration::ZERO);
                        if cap_bound {
                            prof::add("cap_bound", std::time::Duration::ZERO);
                        }
                        stepped?;
                    }
                    let mut wanted = decode
                        .iter()
                        .map(|&i| sessions[i].generation.sample_request(sessions[i].blocks.sliding))
                        .collect::<Vec<_>>();
                    for (k, chunk) in chunks.iter().enumerate() {
                        let session = &sessions[chunk.session];
                        if k >= n_hybrid {
                            let prompt = session
                                .generation
                                .prompt_chunk_from(chunk.pos0, chunk.n)
                                .expect("planned chunk lies inside the prompt");
                            let inputs = prompt.inputs();
                            prof::timed!("prefill_chunk", self
                                .model
                                .prefill_chunk(
                                    &mut self.device,
                                    &inputs,
                                    chunk.pos0,
                                    &session.blocks,
                                    &mut self.workspace,
                                    chunk.last,
                                )
                                .await)?;
                        }
                        if chunk.last {
                            wanted.push(session.generation.sample_request(session.blocks.sliding));
                        }
                    }
                    prof::timed!("next_tokens", self.model.next_tokens(&mut self.device, &wanted).await)
                })
            }));
            let next = match result {
                Ok(Ok(next)) => next,
                failed => {
                    let message = match failed {
                        Ok(Err(error)) => error.to_string(),
                        _ => "generation panicked; worker is offline".to_owned(),
                    };
                    fail(&sessions, None, &message);
                    return;
                }
            };
            prof::add("iteration", iteration.elapsed());
            iterations += 1;
            for &i in &decode {
                sessions[i].last_ride = iterations;
            }
            rows_total += decode.len() + prefill_rows;
            decode_total += decode.len();
            prefill_total += prefill_rows;
            if iterations % 64 == 0 {
                prof::flush(&format!(
                    "S2ITER it={iterations} rows={rows_total} dec={decode_total} pre={prefill_total}"
                ));
                (rows_total, decode_total, prefill_total) = (0, 0, 0);
            }

            // Apply the results: decode rows first, then the chunks that finished a prompt.
            let t_apply = Instant::now();
            let mut next = next.into_iter();
            let mut finished: Vec<Option<StreamEvent>> = (0..sessions.len()).map(|_| None).collect();
            let mut apply = |index: usize, session: &mut Session, step: Step, next: Next| {
                let result = {
                    let events = &session.events;
                    let stream = session.stream;
                    match step {
                        Step::Decode => session
                            .generation
                            .advance(next, |delta| forward_delta(events, stream, delta)),
                        Step::Chunk(n) => {
                            session.generation.consume_prompt(n);
                            Ok(None)
                        }
                        Step::LastChunk(n) => session
                            .generation
                            .finish_chunk(n, next, |delta| forward_delta(events, stream, delta)),
                    }
                };
                if matches!(step, Step::LastChunk(_)) && session.first_token.is_none() {
                    session.first_token = Some(Instant::now());
                    session.prompt = session.generation.prompt_len();
                    session.last_ride = iterations;
                }
                finished[index] = match result {
                    Ok(None) => None,
                    Ok(Some(output)) => {
                        session.first_token.get_or_insert_with(Instant::now);
                        report(session, output.completion_tokens);
                        Some(StreamEvent::Done(output))
                    }
                    Err(error) => Some(StreamEvent::Error(error.to_string())),
                };
            };
            for &index in &decode {
                apply(index, &mut sessions[index], Step::Decode, next.next().unwrap_or(Next::None));
            }
            for chunk in &chunks {
                if chunk.last {
                    apply(chunk.session, &mut sessions[chunk.session], Step::LastChunk(chunk.n), next.next().unwrap_or(Next::None));
                } else {
                    apply(chunk.session, &mut sessions[chunk.session], Step::Chunk(chunk.n), Next::None);
                }
            }

            let mut index = 0;
            sessions.retain_mut(|session| {
                let event = finished[index].take();
                index += 1;
                let Some(event) = event else { return true };
                let _ = session.events.send(event);
                free.append(&mut session.blocks.full);
                flat_free.append(&mut session.blocks.flat);
                rings.push(session.blocks.sliding);
                false
            });
            prof::add("w_apply", t_apply.elapsed());
        }
    }
}

/// Send one streamed delta; false once the client is gone.
fn forward_delta(events: &mpsc::Sender<StreamEvent>, stream: bool, delta: Delta) -> bool {
    !stream
        || events
            .send(match delta {
                Delta::Content(text) => StreamEvent::Delta(text.to_owned()),
                Delta::Reasoning(text) => StreamEvent::ReasoningDelta(text.to_owned()),
            })
            .is_ok()
}

#[derive(Clone, Copy)]
enum Step {
    /// One decode row.
    Decode,
    /// A prompt chunk of `n` tokens that does not end the prompt.
    Chunk(usize),
    /// The final prompt chunk (`n` tokens): its last row yields the first output token.
    LastChunk(usize),
}

/// The S2REQ line: `ttft_ms` counts from the worker receiving the job; `wait_ms` is the queue wait,
/// `enc_ms` the multimodal precompute, `prefill_ms` the prompt computation (interleaved with other
/// requests' work) up to the first token, `tpot_ms` the mean time per generated token after it.
fn report(session: &Session, completion_tokens: usize) {
    let now = Instant::now();
    let first = session.first_token.unwrap_or(now);
    let ttft = (first - session.received).as_secs_f64() * 1e3;
    let prefill = (first - session.prefill_start).as_secs_f64() * 1e3;
    let decode = (now - first).as_secs_f64() * 1e3;
    let tpot = if completion_tokens > 1 { decode / (completion_tokens - 1) as f64 } else { 0.0 };
    eprintln!(
        "S2REQ prompt={} ttft_ms={:.0} total_ms={:.0} wait_ms={:.0} enc_ms={:.0} prefill_ms={:.0} decode_ms={:.0} tpot_ms={:.1} toks={}",
        session.prompt,
        ttft,
        (now - session.received).as_secs_f64() * 1e3,
        session.wait_ms,
        session.encode_ms,
        prefill,
        decode,
        tpot,
        completion_tokens,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_aligned_and_budgeted() {
        // (session, position, remaining): a fresh 150-token prompt, a decoding-free 40-token tail at 100.
        let plan = plan_prefill([(0, 0, 150), (3, 100, 40)], 64, 200, false);
        let shape: Vec<_> = plan.iter().map(|c| (c.session, c.pos0, c.n, c.last)).collect();
        assert_eq!(shape, vec![(0, 0, 64, false), (0, 64, 64, false), (0, 128, 22, true), (3, 100, 28, false), (3, 128, 12, true)]);
        for c in &plan {
            assert_eq!(c.pos0 / 64, (c.pos0 + c.n - 1) / 64, "chunk crosses a multiple of 64");
        }
        assert_eq!(plan.iter().map(|c| c.n).sum::<usize>(), 190);
    }

    #[test]
    fn whole_chunks_never_split_by_the_budget() {
        // budget 100: a full chunk of A, then only 36 left: A next chunk (64 tokens) is not planned, B tail (20 <= 36) is.
        let plan = plan_prefill([(0, 0, 200), (1, 0, 20)], 64, 100, true);
        let shape: Vec<_> = plan.iter().map(|c| (c.session, c.pos0, c.n, c.last)).collect();
        assert_eq!(shape, vec![(0, 0, 64, false), (1, 0, 20, true)]);
    }

    #[test]
    fn small_budget_and_chunk_one() {
        let plan = plan_prefill([(0, 63, 5), (1, 0, 2)], 64, 3, false);
        let shape: Vec<_> = plan.iter().map(|c| (c.session, c.pos0, c.n, c.last)).collect();
        assert_eq!(shape, vec![(0, 63, 1, false), (0, 64, 2, false)]);
        let plan = plan_prefill([(0, 5, 2), (1, 0, 3)], 1, 4, false);
        assert_eq!(plan.iter().map(|c| (c.session, c.pos0, c.n)).collect::<Vec<_>>(), vec![(0, 5, 1), (0, 6, 1), (1, 0, 1), (1, 1, 1)]);
        assert!(plan[1].last && !plan[2].last);
        assert!(plan_prefill([(0, 0, 10)], 64, 0, false).is_empty());
    }
}
