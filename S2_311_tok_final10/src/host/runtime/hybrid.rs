//! The HYBRID pass: prefill rows and decode rows of one iteration in ONE 128-row batch through all 48 layers, so every weight matrix streams once per iteration.
//!
//! Row layout of the batch (128 rows = two 64-row units of the ragged attention kernels):
//! - rows `0 .. 128 - Bd`: prefill runs. A run = consecutive prompt rows of ONE request (`pos0 .. pos0 + n`); it starts at a unit boundary (row 0 or row 64) and occupies
//!   `ceil(n / 64)` units. A request's prompt chunk may be any length (p0 multiple of 8 for the full layers);
//! - rows `128 - Bd .. 128`: decode rows (`Bd` = 8 or 16 bucket; real rows first, then padding clones). They sit in the padding rows of the last prefill unit (rows >= `n` of a unit are
//!   never appended: sliding -> scratch row, full -> trash-page blocks).
//!
//! Per layer (all tensors of the batch are allocated once as 128-row tensors; every kernel argument is a typed VIEW of a byte range of them, `host::views::piece`, so no copy
//! kernels and no second kernel image are needed): the plain 64-row QKV kernel on each half of the residual (outputs straight into the halves of the 128-row stage tensors) ->
//! ragged prefill attention (the pair of units, KV append fused) -> the existing fused decode attention on the decode rows (q / k / v are views of the last rows of the stage tensors,
//! the output is a view of the last rows of the attention tensor; separate launches, so the write order is the launch order) -> ONE `layer_*_t128` launch (AO + FFN of all 128 rows).
//! Prefill writes the flat KV caches directly (no per-slot caches, no migration).

use super::*;
use crate::axes::{Kw, Rg8, Rh, Rw8, Tpk, Bd, Bd16, Bk, Cx, Df, Gf, Gs, Hg, Kc, Ke, Kf, Kf64, Kx5, Ns, P128, Pb2, Pq, Pr, Pw, R8, Seg, Seg4, Tc, Tf, Ts, Tx};
use crate::device::ffnc::{E256, Tk8, Tk128};
use crate::host::chunkw::ChunkLayer;
use crate::host::fp8;
use crate::host::qkvc as qk;
use crate::host::sleepwait::{Slot, sleepy};
use crate::host::views::{piece, piece_at};
use crate::{dattn_host, ops_dattn, ops_f8, ops_ffnc, ops_qkvc, ops_rpa, ops_rpa_full, rpa_host};

type Res = HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]>;
type Xn = HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>;
/// joint scratch of the pair QKV: both 64-row blocks of the normalised activations (one 1 MiB tensor = the two 512 KiB scratches of the previous layout)
type XnJ = HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>;

/// Rows of the hybrid batch.
pub(super) const ROWS: usize = 128;
const UNIT: usize = 64;
/// Positions the tables cover (the 5-page decode kernel's limit).
const POSITIONS: usize = super::decode::LIMIT;
/// Prefill chunks must end below this position (the full-layer chunk kernels cover key pages 0..3).
pub(super) const PREFILL_LIMIT: usize = 2048;
const WIDTH: usize = 4096;

/// One prefill run of the batch: `rows` are the prompt rows at positions `pos0 ..` of the request in `slot` (flat KV pages `table`).
pub struct HybRun<'a> {
    /// Unique id of the request (generation) the run belongs to (`Input::request`): with `PassDesc` it identifies the embedding rows of the run (R13 pre-upload).
    pub request: u64,
    pub slot: usize,
    pub table: Vec<usize>,
    pub pos0: usize,
    pub rows: Vec<Row<'a>>,
    /// Leave the final hidden row of the run's last row in `hidden[slot]`.
    pub want_hidden: bool,
}

/// Owned embedding source of one PREDICTED prefill row (R13 lmoverlap pre-upload): a text token or the host copy of a soft row.
pub enum HintRow {
    Text(usize),
    Soft(Vec<u8>),
}

/// The predicted prefill runs of the next pass (same order as `PassDesc::runs`): request id and the rows of each run.
pub struct HintRun {
    pub request: u64,
    pub rows: Vec<HintRow>,
}

/// One decode row of the batch: the pending token `row` at position `pos` of the request in `slot`.
pub struct HybDec<'a> {
    pub slot: usize,
    pub table: Vec<usize>,
    pub pos: usize,
    pub row: Row<'a>,
    pub want_hidden: bool,
}

/// Rows a decode batch of `d` rows reserves at the end of the batch (0, 8 or 16).
pub fn decode_bucket(d: usize) -> usize {
    match d {
        0 => 0,
        1..=8 => 8,
        9..=16 => 16,
        _ => panic!("at most 16 decode rows per hybrid pass"),
    }
}

/// Whether `runs` (row counts) and `d` decode rows fit the batch: runs fill units from unit 0 on; the last unit must leave `Bd` rows free for the decode rows.
pub fn fits(run_rows: &[usize], d: usize) -> bool {
    let bd = decode_bucket(d);
    let mut units = 0;
    let mut last = 0;
    for &n in run_rows {
        if n == 0 || n > 2 * UNIT {
            return false;
        }
        units += n.div_ceil(UNIT);
        last = n - UNIT * (n.div_ceil(UNIT) - 1);
    }
    if units > 2 {
        return false;
    }
    if bd == 0 {
        return true;
    }
    // decode rows live in unit 1: free rows of unit 1 = 64 - (rows of the last run if it ends in unit 1)
    match units {
        0 | 1 => true,
        _ => last + bd <= UNIT,
    }
}

struct SlidingTables {
    ring: HbmTensor<i32, Chip, m![Pw, Tx]>,
    stage: HbmTensor<i32, Chip, m![Pw, Cx]>,
    mask: HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>,
    pen: HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>,
    append: HbmTensor<i32, Chip, m![P128]>,
    dec8: Option<(HbmTensor<i32, Chip, m![Bd]>, HbmTensor<i32, Chip, m![Bd, Tx]>, HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>)>,
    dec16: Option<(HbmTensor<i32, Chip, m![Bd16]>, HbmTensor<i32, Chip, m![Bd16, Tx]>, HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>)>,
    /// KVD tables (8-decode-row passes of the packed path only): window gather index of the two units, cache row offsets of the two 64-row QKV blocks.
    kvd: Option<(HbmTensor<i32, Chip, m![Pw, Kw]>, HbmTensor<i32, Chip, m![Tc]>, HbmTensor<i32, Chip, m![Tc]>)>,
}

enum FullPrefill {
    P2 { index: HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>, mask: HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>, pen: HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]> },
    P4 { index: HbmTensor<i32, Chip, m![Pw, Hg, Seg4, Bk]>, mask: HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf64]>, pen: HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf64]> },
}

struct FullTables {
    pre: FullPrefill,
    append: HbmTensor<i32, Chip, m![Pb2]>,
    /// one entry per 8-row decode group
    dec: Vec<(HbmTensor<i32, Chip, m![Bd, Kx5]>, HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>, HbmTensor<i32, Chip, m![Bd]>)>,
}

struct Tables {
    cos_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    sin_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    cos_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    sin_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    s: SlidingTables,
    f: FullTables,
}

/// The decode rows' final hidden rows of the last pass, still ON THE DEVICE (`S2_N2`): `views[k]` = 8 consecutive rows of the final residual (the 128-row engine and the 8-row bucket of the
/// 256-row engine: one view; the 16 riders of the 256-row engine: two), `slots[b]` = request slot of row `b` (row `b % 8` of view `b / 8`).
/// `Model::next_tokens` TAKES them (they are valid for exactly the next `next_tokens` call: a later pass of any kind must never see them) and feeds `view` straight to
/// `final_norm_and_argmax_rows8_res` (no D2H of the hidden rows, no re-upload); `flush` materialises the host copies (`hidden[slot]`) when the rows are needed elsewhere (sampled
/// requests, requests that are not all riders of the pass, the host argmax).
pub struct DevRows {
    pub slots: Vec<usize>,
    pub views: Vec<HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>>,
}

impl DevRows {
    /// Copies the rows to the host (`hidden[slot]`).
    pub async fn flush(&self, device: &mut Device, hidden: &mut [Hidden]) -> Result<(), Error> {
        // R15 LQ: the rows are written by launches that may still be queued
        super::lq::drain()?;
        prof::add("n2_flush", std::time::Duration::ZERO);
        let width = H::SIZE * 2;
        for (v, view) in self.views.iter().enumerate() {
            let h: HostTensor<bf16, m![Rw8, Kc, Ke]> = prof::timed!("h_d2h", view.to_host(&mut device.pdma).await)?;
            let host = h.into_inner().into_buf();
            for b in 0..8 {
                if let Some(&slot) = self.slots.get(8 * v + b) {
                    hidden[slot] = host[b * WIDTH * 2..b * WIDTH * 2 + width].to_vec();
                }
            }
        }
        Ok(())
    }
}

/// Device-resident LM-head input, default ON (token-identical to the host round trip, jobs 148665 / 148668; paired C=12 A/B w14: iteration -0.3 ms on fast boards, -1.1 ms on slow
/// ones; 56 of 64 passes take the device path, the 8 passes that finish a prompt keep the host path). `S2_N2=0` = the previous behaviour (hidden rows come back to the host and are
/// re-uploaded).
pub fn n2_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_N2").map(|value| value != "0").unwrap_or(true))
}

pub(super) struct HybridEngine {
    /// Decode rows of the last pass whose hidden rows were NOT copied to the host (`n2_enabled`), else `None`.
    pub(super) dev_rows: Option<DevRows>,
    /// The persistent packed-upload tensor (`S2_UPLOAD_REUSE`).
    big: Option<HbmTensor<bf16, Chip, m![Tpk]>>,
    xnj: XnJ,
    /// ping-pong residual `[128, 4096]` (the layer kernels read one and write the other; QKV halves are views of the input one)
    res: [Res; 2],
    // sliding stage tensors: q / attention output as the two units of the ragged kernel, k / v as the 128 staged rows
    q_s: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_s: HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    v_s: HbmTensor<bf16, Chip, m![P128, Ns, Ds]>,
    out_s: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    // full stage tensors (k / v as 8-row blocks, the type of the ragged full kernel)
    q_f: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    v_f: HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    out_f: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    rope: std::sync::Arc<Rope>,
    /// The tables of the NEXT pass, being built on a worker thread (`PassDesc` = the pass it was built for; used only when the real pass is identical).
    prebuilt: Option<(PassDesc, std::thread::JoinHandle<Prebuilt>)>,
}

/// The RoPE rows of every position (host copies; shared with the prebuild thread).
pub(super) struct Rope {
    pub(super) cos_s: Vec<u8>,
    pub(super) sin_s: Vec<u8>,
    pub(super) cos_f: Vec<u8>,
    pub(super) sin_f: Vec<u8>,
}

/// One prefill run of a pass, without its tokens: everything the attention tables depend on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunDesc {
    pub slot: usize,
    pub table: Vec<usize>,
    pub pos0: usize,
    pub n: usize,
    /// The run ends its prompt (its last row's token is wanted; R14 d13e: the P0 copy tables of the pass depend on it).
    pub last: bool,
}

/// One decode row of a pass, without its token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecDesc {
    pub slot: usize,
    pub table: Vec<usize>,
    pub pos: usize,
}

/// A hybrid pass as far as the tables are concerned (slots, page tables, positions; no token ids). Two passes with equal descriptions use byte-identical tables, so the worker
/// can PREDICT the next pass, build its tables while the device runs this one, and the engine uses them only if the real description turns out equal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassDesc {
    pub runs: Vec<RunDesc>,
    pub decs: Vec<DecDesc>,
}

pub(super) fn desc_of(runs: &[HybRun<'_>], decs: &[HybDec<'_>]) -> PassDesc {
    PassDesc {
        runs: runs.iter().map(|r| RunDesc { slot: r.slot, table: r.table.clone(), pos0: r.pos0, n: r.rows.len(), last: r.want_hidden }).collect(),
        decs: decs.iter().map(|d| DecDesc { slot: d.slot, table: d.table.clone(), pos: d.pos }).collect(),
    }
}

/// Position of every row of the batch (padding rows repeat the nearest real row before them).
fn row_positions(desc: &PassDesc) -> Vec<usize> {
    let bd = decode_bucket(desc.decs.len());
    let mut row_pos = vec![0usize; ROWS];
    let mut filled = vec![false; ROWS];
    let mut unit0 = 0;
    for run in &desc.runs {
        let base = UNIT * unit0;
        for c in 0..run.n {
            row_pos[base + c] = run.pos0 + c;
            filled[base + c] = true;
        }
        unit0 += run.n.div_ceil(UNIT);
    }
    for (b, dec) in desc.decs.iter().enumerate() {
        row_pos[ROWS - bd + b] = dec.pos;
        filled[ROWS - bd + b] = true;
    }
    let mut src = (0..ROWS).find(|&r| filled[r]).expect("a real row");
    for r in 0..ROWS {
        if filled[r] {
            src = r;
        } else {
            row_pos[r] = row_pos[src];
        }
    }
    row_pos
}

/// ONE pair-QKV launch per sliding layer (`ops_qkvc::sliding_qkv_pair`: the weight tiles are streamed once for the two 64-row blocks) instead of two `sliding_qkv_chunk` launches.
/// Default ON (bit-identical to the two-launch path, jobs 148492 / 148493; paired C=12 A/B w11: h_qkv_s -1.2 ms, hybrid_step -1.0 ms per pass); `S2_QKV2=0` = old two-launch path.
pub fn qkv2_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_QKV2").map(|value| value != "0").unwrap_or(true))
}

/// ONE attention launch per sliding layer in the 8-decode-row bucket (`ops_rpa::sliding_ragged_s2_b8_h`: prefill pair + KV appends + decode attention; the decode rows' output is stored
/// into the last 8 rows of the prefill output tensor THROUGH THE SAME HANDLE, after the prefill store). Default ON: bit-identical to the two launches (rider-hidden fingerprint over
/// 12 x 30 passes x 40 layers, jobs hh_a / 149017), paired C=12 A/B w17: attention part -1.2 ms, hybrid_step -1.4 ms per pass. `S2_N3=0` = the two separate launches.
pub fn n3_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_N3").map(|value| value != "0").unwrap_or(true))
}

/// KVD (`S2_KVD=0` = off, default ON): in an 8-decode-row pass with N3 + the pair QKV, the QKV launch scatters the K / V rows (prefill rows and decode rows) straight into the flat ring caches
/// and the attention launch reads the unit's chunk keys back from the ring in ONE window gather of 1024 + 128 rows: no stage tensors, no append DMAs, no separate chunk gather.
pub fn kvd_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_KVD").map(|value| value != "0").unwrap_or(true))
}

/// `S2_UPLOAD_REUSE=0`: allocate a fresh device tensor for the packed upload every pass (the previous behaviour); default ON: one persistent tensor, `.output(&mut big)`.
pub fn upload_reuse_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_UPLOAD_REUSE").map(|value| value != "0").unwrap_or(true))
}

/// `S2_PREBUILD=0` turns the prebuild off. Default ON: the worker predicts the next pass (`Model::set_hybrid_hint`) and the engine builds its packed tables (and packs them into ONE
/// upload) on a thread while the device runs the current pass. Arena (jobs w7_C, w8_C): 240 of 240 predictions hit; the host-serial time of a pass drops by ~1.9 ms (tables + upload
/// 3.1 -> 1.0 ms on the same kind of node, `scripts/hostgap.py`). A wrong prediction is detected by comparing the pass descriptions and costs one synchronous build (the old path).
pub fn prebuild_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_PREBUILD").map(|value| value != "0").unwrap_or(true))
}

/// `S2_PACK_UPLOADS=1` packs the embedding rows and every table of a pass into ONE upload (views of one buffer); default OFF = the 22-25 separate uploads per pass.
/// Measured on Arena (jobs 148111 / 148114, C = 12, 96 requests): token-identical, but no faster (h_tables + h_h2d 3.99 ms packed vs 3.70 ms separate per pass, normalised
/// tok/s 122.8 vs 125.5): the 2.9 MB transfer costs 1.2 ms and the host-side packing / views 2.3 ms, so the default stays on the proven separate uploads.
fn packed_uploads() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && (prebuild_enabled() || std::env::var("S2_PACK_UPLOADS").map(|value| value == "1").unwrap_or(false)))
}

/// The packed per-pass upload: the embedding rows first, then every table at an 8 KB boundary.
struct Packer {
    bytes: Vec<u8>,
}

pub(super) const PACK_ALIGN: usize = 8192;

impl Packer {
    fn new(embed: Vec<u8>) -> Self {
        assert_eq!(embed.len(), HbmTensor::<bf16, Chip, m![Tk128, Kc, Ke]>::size(), "embedding rows");
        Self { bytes: embed }
    }

    /// Append the bytes of one table of type `HbmTensor<D, Chip, E>`; returns its offset.
    fn put<D: Scalar, E: M>(&mut self, data: &[u8]) -> usize {
        assert_eq!(data.len(), HbmTensor::<D, Chip, E>::size(), "table size");
        let off = self.bytes.len().next_multiple_of(PACK_ALIGN);
        self.bytes.resize(off, 0);
        self.bytes.extend_from_slice(data);
        off
    }

    /// The upload buffer, zero padded to the size of `[Tpk]`.
    fn finish(mut self) -> Vec<u8> {
        static MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if self.bytes.len() > MAX.fetch_max(self.bytes.len(), std::sync::atomic::Ordering::Relaxed) {
            eprintln!("S2 hybrid packed upload: {} bytes used of {}", self.bytes.len(), HbmTensor::<bf16, Chip, m![Tpk]>::size());
        }
        assert!(self.bytes.len() <= HbmTensor::<bf16, Chip, m![Tpk]>::size(), "packed upload does not fit [Tpk]");
        self.bytes.resize(HbmTensor::<bf16, Chip, m![Tpk]>::size(), 0);
        self.bytes
    }
}

/// The bytes of a slice of plain scalars as they sit in memory (little endian host = the device byte order), without a copy. (The per-element `flat_map` conversion this
/// replaces cost 1.0 ms per 262144 elements on the Arena worker, `test_int --upbench`, job 148253, and made the packed upload slower than 25 separate ones.)
pub(super) fn scalar_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a plain scalar (`i32` / `bf16`) without padding; the byte view covers exactly the slice and is only read.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub(super) fn i32_bytes(v: &[i32]) -> &[u8] {
    scalar_bytes(v)
}

pub(super) fn bf_bytes(v: &[bf16]) -> &[u8] {
    assert_eq!(std::mem::size_of::<bf16>(), 2);
    assert!(cfg!(target_endian = "little"));
    scalar_bytes(v)
}

pub(super) async fn zero<D: MaterializableScalar + num_traits::Zero, E: M>(device: &mut Device) -> Result<HbmTensor<D, Chip, E>, Error> {
    HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await
}

pub(super) fn rows_of(table: &[u8], dim: usize, positions: &[usize]) -> Vec<u8> {
    let width = dim * 2;
    let mut out = Vec::with_capacity(positions.len() * width);
    for &p in positions {
        out.extend_from_slice(&table[p * width..(p + 1) * width]);
    }
    out
}

/// Where every table sits in the packed upload buffer (byte offsets, multiples of `PACK_ALIGN`).
struct Layout {
    bd: usize,
    npages: usize,
    cos_s: Vec<usize>,
    sin_s: Vec<usize>,
    cos_f: Vec<usize>,
    sin_f: Vec<usize>,
    ring: usize,
    stage: usize,
    mask: usize,
    pen: usize,
    append: usize,
    sdec: Option<(usize, usize, usize)>,
    kvd: Option<(usize, usize, usize)>,
    fidx: usize,
    fmask: usize,
    fpen: usize,
    fdec: Vec<(usize, usize, usize)>,
    fappend: usize,
}

/// The packed upload buffer of a pass (`[Tpk]` bytes: embedding rows zeroed at offset 0 for the caller to fill, then every table) and its layout. Pure host work (no device).
struct Prebuilt {
    buf: Vec<u8>,
    layout: Layout,
}

/// Build every attention / rope table of the pass `desc` and pack them (see `Prebuilt`). Depends on `desc` only, so it can run ahead of the pass on another thread.
fn build_packed(rope: &Rope, desc: &PassDesc) -> Prebuilt {
    let bd = decode_bucket(desc.decs.len());
    let row_pos = row_positions(desc);
    let dslots: Vec<usize> = desc.decs.iter().map(|d| d.slot).collect();
    let dpos: Vec<usize> = desc.decs.iter().map(|d| d.pos).collect();
    let dtabs: Vec<Vec<usize>> = desc.decs.iter().map(|d| d.table.clone()).collect();
    let rpa_runs: Vec<rpa_host::Run> = desc.runs.iter().map(|r| rpa_host::Run { slot: r.slot, p0: r.pos0, n: r.n }).collect();
    let pt = rpa_host::prefill_tables(2, &rpa_runs);
    let mut units = Vec::new();
    let mut last_pos = 0;
    for run in &desc.runs {
        let n = run.n;
        for j in 0..n.div_ceil(UNIT) {
            units.push(rpa_host::FullUnit { table: run.table.clone(), p0: run.pos0 + UNIT * j, n: (n - UNIT * j).min(UNIT) });
        }
        last_pos = last_pos.max(run.pos0 + n - 1);
    }
    let npages = if last_pos < 1024 { 2 } else { 4 };
    let ft = rpa_host::full_prefill_tables(2, npages, &units);
    let sdec: Option<(Vec<i32>, Vec<i32>, Vec<bf16>)> = (bd > 0).then(|| {
        (
            rpa_host::decode_append_index(&dslots, &dpos, bd),
            rpa_host::decode_gather_index(&dslots, &dpos, bd),
            rpa_host::decode_mask(&dpos, bd),
        )
    });
    let fdec: Vec<(Vec<i32>, Vec<bf16>, Vec<i32>)> = (0..bd / 8)
        .map(|g| {
            let r = 8 * g..(8 * g + 8).min(desc.decs.len());
            let (po, ta): (&[usize], &[Vec<usize>]) = if r.start < desc.decs.len() { (&dpos[r.clone()], &dtabs[r.clone()]) } else { (&[], &[]) };
            (
                dattn_host::full_gather_index(ta, 8, Kx5::SIZE),
                dattn_host::full_mask(po, 8, Kx5::SIZE, 64),
                rpa_host::full_decode_append_index(ta, po, 8),
            )
        })
        .collect();
    let kvd_t = (kvd_enabled() && bd == 8).then(|| rpa_host::kvd_tables(&rpa_runs, &dslots, &dpos, bd));
    let mut pk = Packer::new(vec![0u8; ROWS * WIDTH * 2]);
    let (mut cos_s, mut sin_s, mut cos_f, mut sin_f) = (vec![], vec![], vec![], vec![]);
    for h in 0..2 {
        let pos = &row_pos[h * UNIT..(h + 1) * UNIT];
        cos_s.push(pk.put::<bf16, m![Tc, Ds]>(&rows_of(&rope.cos_s, Ds::SIZE, pos)));
        sin_s.push(pk.put::<bf16, m![Tc, Ds]>(&rows_of(&rope.sin_s, Ds::SIZE, pos)));
        cos_f.push(pk.put::<bf16, m![Tc, Df]>(&rows_of(&rope.cos_f, Df::SIZE, pos)));
        sin_f.push(pk.put::<bf16, m![Tc, Df]>(&rows_of(&rope.sin_f, Df::SIZE, pos)));
    }
    let ring = pk.put::<i32, m![Pw, Tx]>(&i32_bytes(&pt.ring_index[0]));
    let stage = pk.put::<i32, m![Pw, Cx]>(&i32_bytes(&pt.stage_index[0]));
    let mask = pk.put::<bf16, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>(&bf_bytes(&pt.mask[0]));
    let pen = pk.put::<bf16, m![Pw, Ts % 32, Seg, Ts / 32]>(&bf_bytes(&pt.pen[0]));
    let append = pk.put::<i32, m![P128]>(&i32_bytes(&pt.append_index));
    let sdec_o = match (&sdec, bd) {
        (Some((app, idx, mask)), 8) => Some((pk.put::<i32, m![Bd]>(&i32_bytes(app)), pk.put::<i32, m![Bd, Tx]>(&i32_bytes(idx)), pk.put::<bf16, m![Bd, Tx % 32, Tx / 32]>(&bf_bytes(mask)))),
        (Some((app, idx, mask)), 16) => Some((pk.put::<i32, m![Bd16]>(&i32_bytes(app)), pk.put::<i32, m![Bd16, Tx]>(&i32_bytes(idx)), pk.put::<bf16, m![Bd16, Tx % 32, Tx / 32]>(&bf_bytes(mask)))),
        _ => None,
    };
    let kvd_o = kvd_t.as_ref().map(|(win, i0, i1)| (pk.put::<i32, m![Pw, Kw]>(&i32_bytes(win)), pk.put::<i32, m![Tc]>(&i32_bytes(i0)), pk.put::<i32, m![Tc]>(&i32_bytes(i1))));
    let (fidx, fmask, fpen) = if npages == 2 {
        (
            pk.put::<i32, m![Pw, Hg, Seg, Bk]>(&i32_bytes(&ft.index[0])),
            pk.put::<bf16, m![Pw, Tf % 32, Tc, Kf]>(&bf_bytes(&ft.mask[0])),
            pk.put::<bf16, m![Pw, Tf % 32, Kf]>(&bf_bytes(&ft.pen[0])),
        )
    } else {
        (
            pk.put::<i32, m![Pw, Hg, Seg4, Bk]>(&i32_bytes(&ft.index[0])),
            pk.put::<bf16, m![Pw, Tf % 32, Tc, Kf64]>(&bf_bytes(&ft.mask[0])),
            pk.put::<bf16, m![Pw, Tf % 32, Kf64]>(&bf_bytes(&ft.pen[0])),
        )
    };
    let fdec_o: Vec<(usize, usize, usize)> = fdec
        .iter()
        .map(|(gidx, mask, app)| {
            (
                pk.put::<i32, m![Bd, Kx5]>(&i32_bytes(gidx)),
                pk.put::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>(&bf_bytes(mask)),
                pk.put::<i32, m![Bd]>(&i32_bytes(app)),
            )
        })
        .collect();
    let fappend = pk.put::<i32, m![Pb2]>(&i32_bytes(&ft.append_index));
    let buf = pk.finish();
    Prebuilt { buf, layout: Layout { bd, npages, cos_s, sin_s, cos_f, sin_f, ring, stage, mask, pen, append, sdec: sdec_o, kvd: kvd_o, fidx, fmask, fpen, fdec: fdec_o, fappend } }
}

impl HybridEngine {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        let (cos_s, sin_s) = qk::rope_rows_bytes(Ds::SIZE, 10_000.0, Ds::SIZE, 0, POSITIONS);
        let (cos_f, sin_f) = qk::rope_rows_bytes(Df::SIZE, 1_000_000.0, 128, 0, POSITIONS);
        Ok(Self {
            xnj: zero(device).await?,
            res: [zero(device).await?, zero(device).await?],
            q_s: zero(device).await?,
            k_s: zero(device).await?,
            v_s: zero(device).await?,
            out_s: zero(device).await?,
            q_f: zero(device).await?,
            k_f: zero(device).await?,
            v_f: zero(device).await?,
            out_f: zero(device).await?,
            rope: std::sync::Arc::new(Rope { cos_s, sin_s, cos_f, sin_f }),
            prebuilt: None,
            dev_rows: None,
            big: None,
        })
    }

    /// Start building the packed tables of the predicted pass `next` on a worker thread (used by the pass that runs before it, on whichever engine runs it).
    pub(super) fn spawn_prebuild(&mut self, next: PassDesc) {
        let rope = self.rope.clone();
        let job = next.clone();
        if let Ok(handle) = std::thread::Builder::new().name("hyb-prebuild".into()).spawn(move || { crate::host::cpuplace::leave_launch_cpu(); build_packed(&rope, &job) }) {
            self.prebuilt = Some((next, handle));
        }
    }

    async fn tables_split(&self, device: &mut Device, runs: &[HybRun<'_>], decs: &[HybDec<'_>], row_pos: &[usize]) -> Result<Tables, Error> {
        let bd = decode_bucket(decs.len());
        // phase 1: every table on the host
        let dslots: Vec<usize> = decs.iter().map(|d| d.slot).collect();
        let dpos: Vec<usize> = decs.iter().map(|d| d.pos).collect();
        let dtabs: Vec<Vec<usize>> = decs.iter().map(|d| d.table.clone()).collect();
        let cpu = prof::timed!("h_tab_cpu", {
            let rpa_runs: Vec<rpa_host::Run> = runs.iter().map(|r| rpa_host::Run { slot: r.slot, p0: r.pos0, n: r.rows.len() }).collect();
            let pt = rpa_host::prefill_tables(2, &rpa_runs);
            let mut units = Vec::new();
            let mut last_pos = 0;
            for run in runs {
                let n = run.rows.len();
                for j in 0..n.div_ceil(UNIT) {
                    units.push(rpa_host::FullUnit { table: run.table.clone(), p0: run.pos0 + UNIT * j, n: (n - UNIT * j).min(UNIT) });
                }
                last_pos = last_pos.max(run.pos0 + n - 1);
            }
            let npages = if last_pos < 1024 { 2 } else { 4 };
            let ft = rpa_host::full_prefill_tables(2, npages, &units);
            let sdec: Option<(Vec<i32>, Vec<i32>, Vec<bf16>)> = (bd > 0).then(|| {
                (
                    rpa_host::decode_append_index(&dslots, &dpos, bd),
                    rpa_host::decode_gather_index(&dslots, &dpos, bd),
                    rpa_host::decode_mask(&dpos, bd),
                )
            });
            let fdec: Vec<(Vec<i32>, Vec<bf16>, Vec<i32>)> = (0..bd / 8)
                .map(|g| {
                    let r = 8 * g..(8 * g + 8).min(decs.len());
                    let (po, ta): (&[usize], &[Vec<usize>]) = if r.start < decs.len() { (&dpos[r.clone()], &dtabs[r.clone()]) } else { (&[], &[]) };
                    (
                        dattn_host::full_gather_index(ta, 8, Kx5::SIZE),
                        dattn_host::full_mask(po, 8, Kx5::SIZE, 64),
                        rpa_host::full_decode_append_index(ta, po, 8),
                    )
                })
                .collect();
            (pt, ft, npages, sdec, fdec)
        });
        let (pt, ft, npages, sdec, fdec) = cpu;
        // phase 2: uploads
        let (mut cos_s, mut sin_s, mut cos_f, mut sin_f) = (vec![], vec![], vec![], vec![]);
        for h in 0..2 {
            let pos = &row_pos[h * UNIT..(h + 1) * UNIT];
            cos_s.push(HostTensor::<bf16, m![Tc, Ds]>::from_buf(rows_of(&self.rope.cos_s, Ds::SIZE, pos)).to_hbm(&mut device.pdma).await?);
            sin_s.push(HostTensor::<bf16, m![Tc, Ds]>::from_buf(rows_of(&self.rope.sin_s, Ds::SIZE, pos)).to_hbm(&mut device.pdma).await?);
            cos_f.push(HostTensor::<bf16, m![Tc, Df]>::from_buf(rows_of(&self.rope.cos_f, Df::SIZE, pos)).to_hbm(&mut device.pdma).await?);
            sin_f.push(HostTensor::<bf16, m![Tc, Df]>::from_buf(rows_of(&self.rope.sin_f, Df::SIZE, pos)).to_hbm(&mut device.pdma).await?);
        }
        let s = SlidingTables {
            ring: HostTensor::<i32, m![Pw, Tx]>::from_vec(pt.ring_index[0].clone()).to_hbm(&mut device.pdma).await?,
            stage: HostTensor::<i32, m![Pw, Cx]>::from_vec(pt.stage_index[0].clone()).to_hbm(&mut device.pdma).await?,
            mask: HostTensor::<bf16, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>::from_vec(pt.mask[0].clone()).to_hbm(&mut device.pdma).await?,
            pen: HostTensor::<bf16, m![Pw, Ts % 32, Seg, Ts / 32]>::from_vec(pt.pen[0].clone()).to_hbm(&mut device.pdma).await?,
            append: HostTensor::<i32, m![P128]>::from_vec(pt.append_index.clone()).to_hbm(&mut device.pdma).await?,
            dec8: match (&sdec, bd) {
                (Some((app, idx, mask)), 8) => Some((
                    HostTensor::<i32, m![Bd]>::from_vec(app.clone()).to_hbm(&mut device.pdma).await?,
                    HostTensor::<i32, m![Bd, Tx]>::from_vec(idx.clone()).to_hbm(&mut device.pdma).await?,
                    HostTensor::<bf16, m![Bd, Tx % 32, Tx / 32]>::from_vec(mask.clone()).to_hbm(&mut device.pdma).await?,
                )),
                _ => None,
            },
            dec16: match (&sdec, bd) {
                (Some((app, idx, mask)), 16) => Some((
                    HostTensor::<i32, m![Bd16]>::from_vec(app.clone()).to_hbm(&mut device.pdma).await?,
                    HostTensor::<i32, m![Bd16, Tx]>::from_vec(idx.clone()).to_hbm(&mut device.pdma).await?,
                    HostTensor::<bf16, m![Bd16, Tx % 32, Tx / 32]>::from_vec(mask.clone()).to_hbm(&mut device.pdma).await?,
                )),
                _ => None,
            },
            kvd: None,
        };
        let pre = if npages == 2 {
            FullPrefill::P2 {
                index: HostTensor::<i32, m![Pw, Hg, Seg, Bk]>::from_vec(ft.index[0].clone()).to_hbm(&mut device.pdma).await?,
                mask: HostTensor::<bf16, m![Pw, Tf % 32, Tc, Kf]>::from_vec(ft.mask[0].clone()).to_hbm(&mut device.pdma).await?,
                pen: HostTensor::<bf16, m![Pw, Tf % 32, Kf]>::from_vec(ft.pen[0].clone()).to_hbm(&mut device.pdma).await?,
            }
        } else {
            FullPrefill::P4 {
                index: HostTensor::<i32, m![Pw, Hg, Seg4, Bk]>::from_vec(ft.index[0].clone()).to_hbm(&mut device.pdma).await?,
                mask: HostTensor::<bf16, m![Pw, Tf % 32, Tc, Kf64]>::from_vec(ft.mask[0].clone()).to_hbm(&mut device.pdma).await?,
                pen: HostTensor::<bf16, m![Pw, Tf % 32, Kf64]>::from_vec(ft.pen[0].clone()).to_hbm(&mut device.pdma).await?,
            }
        };
        let mut dec = Vec::new();
        for (gidx, mask, app) in &fdec {
            dec.push((
                HostTensor::<i32, m![Bd, Kx5]>::from_vec(gidx.clone()).to_hbm(&mut device.pdma).await?,
                HostTensor::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>::from_vec(mask.clone()).to_hbm(&mut device.pdma).await?,
                HostTensor::<i32, m![Bd]>::from_vec(app.clone()).to_hbm(&mut device.pdma).await?,
            ));
        }
        let f = FullTables {
            pre,
            append: HostTensor::<i32, m![Pb2]>::from_vec(ft.append_index.clone()).to_hbm(&mut device.pdma).await?,
            dec,
        };
        Ok(Tables { cos_s, sin_s, cos_f, sin_f, s, f })
    }

    /// Upload the packed buffer of a pass in ONE host-to-device transfer: the buffer holds the embedding rows (offset 0, = the first residual) and every table (each at an
    /// 8 KB boundary); the device tensors are views of it (`views::piece_at`).
    async fn upload_packed(&mut self, device: &mut Device, pre: Prebuilt) -> Result<(Res, Tables), Error> {
        let Prebuilt { buf, layout: lo } = pre;
        // `S2_UPLOAD_REUSE` (default ON, 0 = a fresh allocation every pass): the packed buffer is written into ONE persistent device tensor (upbench: 2.9 MB 423 us fresh vs 230 us persistent).
        // Safe because every view of the previous pass (tables, residuals, the rows `next_tokens` took) is dead when the next pass uploads.
        if upload_reuse_enabled() {
            if let Some(persistent) = self.big.as_mut() {
                prof::timed!("h_h2d", HostTensor::<bf16, m![Tpk]>::from_buf(buf).to_hbm(&mut device.pdma).output(persistent).await)?;
            } else {
                self.big = Some(prof::timed!("h_h2d", HostTensor::<bf16, m![Tpk]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
            }
        } else {
            self.big = Some(prof::timed!("h_h2d", HostTensor::<bf16, m![Tpk]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
        }
        let big: &HbmTensor<bf16, Chip, m![Tpk]> = self.big.as_ref().expect("uploaded above");
        let res0: Res = piece_at::<bf16, m![Tpk], bf16, m![Tk128, Kc, Ke]>(&big, 0)?;
        let cos_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.cos_s.iter().map(|&o| piece_at(&big, o)).collect::<Result<_, _>>()?;
        let sin_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.sin_s.iter().map(|&o| piece_at(&big, o)).collect::<Result<_, _>>()?;
        let cos_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.cos_f.iter().map(|&o| piece_at(&big, o)).collect::<Result<_, _>>()?;
        let sin_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.sin_f.iter().map(|&o| piece_at(&big, o)).collect::<Result<_, _>>()?;
        let bd = lo.bd;
        let s = SlidingTables {
            ring: piece_at(&big, lo.ring)?,
            stage: piece_at(&big, lo.stage)?,
            mask: piece_at(&big, lo.mask)?,
            pen: piece_at(&big, lo.pen)?,
            append: piece_at(&big, lo.append)?,
            dec8: match (&lo.sdec, bd) {
                (Some((a, i, k)), 8) => Some((piece_at(&big, *a)?, piece_at(&big, *i)?, piece_at(&big, *k)?)),
                _ => None,
            },
            dec16: match (&lo.sdec, bd) {
                (Some((a, i, k)), 16) => Some((piece_at(&big, *a)?, piece_at(&big, *i)?, piece_at(&big, *k)?)),
                _ => None,
            },
            kvd: match &lo.kvd {
                Some((w, i0, i1)) => Some((piece_at(&big, *w)?, piece_at(&big, *i0)?, piece_at(&big, *i1)?)),
                None => None,
            },
        };
        let pre = if lo.npages == 2 {
            FullPrefill::P2 { index: piece_at(&big, lo.fidx)?, mask: piece_at(&big, lo.fmask)?, pen: piece_at(&big, lo.fpen)? }
        } else {
            FullPrefill::P4 { index: piece_at(&big, lo.fidx)?, mask: piece_at(&big, lo.fmask)?, pen: piece_at(&big, lo.fpen)? }
        };
        let mut dec = Vec::new();
        for &(g, k, a) in &lo.fdec {
            dec.push((piece_at(&big, g)?, piece_at(&big, k)?, piece_at(&big, a)?));
        }
        let f = FullTables { pre, append: piece_at(&big, lo.fappend)?, dec };
        Ok((res0, Tables { cos_s: cos_s_t, sin_s: sin_s_t, cos_f: cos_f_t, sin_f: sin_f_t, s, f }))
    }

    /// One hybrid pass. `runs` (>= 1) and `decs` (<= 16) must satisfy `fits`. Leaves the final hidden rows of the runs with `want_hidden` and of the decode rows with
    /// `want_hidden` in `hidden[slot]`.
    pub(super) async fn run(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        runs: &[HybRun<'_>],
        decs: &[HybDec<'_>],
        hidden: &mut [Hidden],
        hint: Option<PassDesc>,
    ) -> Result<(), Error> {
        let d = decs.len();
        let bd = decode_bucket(d);
        assert!(!runs.is_empty() && fits(&runs.iter().map(|r| r.rows.len()).collect::<Vec<_>>(), d), "runs / decode rows do not fit one hybrid batch");
        for run in runs {
            assert!(run.pos0 + run.rows.len() <= PREFILL_LIMIT, "hybrid prefill rows must lie below {PREFILL_LIMIT}");
        }

        // ---- the row map ----
        let t_embed = std::time::Instant::now();
        let packed = packed_uploads();
        let desc = if packed { Some(desc_of(runs, decs)) } else { None };
        // packed uploads: the tables of this pass were (maybe) built ahead on a thread; otherwise build them now. The embedding rows go straight into the buffer.
        let mut pre: Option<Prebuilt> = None;
        if let Some(desc) = &desc {
            pre = Some(match self.prebuilt.take() {
                Some((d0, handle)) if d0 == *desc => match prof::timed!("h_tab_wait", handle.join()) {
                    Ok(pre) => {
                        prof::add("h_prebuilt_hit", std::time::Duration::ZERO);
                        pre
                    }
                    Err(_) => prof::timed!("h_tab_cpu", build_packed(&self.rope, desc)),
                },
                other => {
                    if other.is_some() {
                        prof::add("h_prebuilt_miss", std::time::Duration::ZERO);
                    }
                    prof::timed!("h_tab_cpu", build_packed(&self.rope, desc))
                }
            });
        }
        let mut row_pos = vec![0usize; ROWS];
        let mut local: Vec<u8> = if pre.is_none() { vec![0u8; ROWS * WIDTH * 2] } else { Vec::new() };
        let embed: &mut [u8] = match pre.as_mut() {
            Some(p) => &mut p.buf[..ROWS * WIDTH * 2],
            None => &mut local[..],
        };
        let mut filled = vec![false; ROWS];
        let width = H::SIZE * 2;
        let mut want: Vec<(usize, usize)> = Vec::new(); // (row, slot) of every hidden row to return
        let mut unit0 = 0;
        for run in runs {
            let base = UNIT * unit0;
            for (c, row) in run.rows.iter().enumerate() {
                chunk::embed_row_into(weights, row, &mut embed[(base + c) * WIDTH * 2..(base + c) * WIDTH * 2 + width]);
                row_pos[base + c] = run.pos0 + c;
                filled[base + c] = true;
            }
            if run.want_hidden {
                want.push((base + run.rows.len() - 1, run.slot));
            }
            unit0 += run.rows.len().div_ceil(UNIT);
        }
        for (b, dec) in decs.iter().enumerate() {
            let r = ROWS - bd + b;
            chunk::embed_row_into(weights, &dec.row, &mut embed[r * WIDTH * 2..r * WIDTH * 2 + width]);
            row_pos[r] = dec.pos;
            filled[r] = true;
            if dec.want_hidden {
                want.push((r, dec.slot));
            }
        }
        // padding rows: clones of the nearest real row before them (any finite row will do; their outputs are never used)
        let mut src = (0..ROWS).find(|&r| filled[r]).expect("a real row");
        for r in 0..ROWS {
            if filled[r] {
                src = r;
            } else {
                embed.copy_within(src * WIDTH * 2..src * WIDTH * 2 + width, r * WIDTH * 2);
                row_pos[r] = row_pos[src];
            }
        }

        prof::add("h_embed", t_embed.elapsed());
        let (res0, t) = if let Some(pre) = pre {
            prof::timed!("h_tables", self.upload_packed(device, pre).await)?
        } else {
            let res0 = prof::timed!("h_h2d", HostTensor::<bf16, m![Tk128, Kc, Ke]>::from_buf(local).to_hbm(&mut device.pdma).await)?;
            (res0, prof::timed!("h_tables", self.tables_split(device, runs, decs, &row_pos).await)?)
        };
        // the tables of the predicted next pass: built on a thread while the device runs this pass
        if packed {
            if let Some(next) = hint {
                self.spawn_prebuild(next);
            }
        }
        self.res[0] = res0;
        // N2 predicate (also K03): every row that needs a token is one of the (<= 8) decode rows, so only the last 8 rows of the final residual are ever read
        let on_device = n2_enabled() && bd == 8 && decs.iter().all(|dec| dec.want_hidden) && want.iter().all(|&(row, _)| row >= ROWS - bd);
        let mut cur = 0usize;
        let qkv2 = qkv2_enabled();
        let n3 = n3_enabled() && bd == 8;
        // KVD predicate: pair QKV + N3 + the packed tables of this pass carry the KVD tables
        let kvd = n3 && qkv2 && t.s.kvd.is_some();
        prof::add(if kvd { "kvd_on" } else { "kvd_off" }, std::time::Duration::ZERO);
        // sliding decode views: piece index of the first decode piece (8-row pieces: 15; 16-row pieces: 7)
        let dsg = if bd == 8 { 15usize } else { 7usize };

        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            let flat = ws.flat.as_mut().expect("flat caches");
            let (res_a, res_b) = self.res.split_at_mut(1);
            let (res_in, res_out) = if cur == 0 { (&res_a[0], &mut res_b[0]) } else { (&res_b[0], &mut res_a[0]) };
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.sliding[index].as_mut().expect("sliding flat cache");
                    if kvd {
                        // KVD: QKV scatters K / V (prefill + decode rows) into the ring caches, then ONE attention launch (prefill pair on the ring window + decode attention)
                        let (win, i0, i1) = t.s.kvd.as_ref().expect("kvd tables");
                        let (_, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                        let x: HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]> = piece(res_in, 0)?;
                        let mut q0: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                        let mut q1: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 1)?;
                        prof::timed!("h_qkv_s", launch(
                            ops_qkvc::sliding_qkv_pair_direct_j,
                            (
                                &mut *device, &x, &l.input_norm, &mut self.xnj, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                &t.cos_s[0], &t.sin_s[0], &t.cos_s[1], &t.sin_s[1], &mut q0, &mut q1, i0, i1, &mut *fk, &mut *fv,
                            ),
                        ).await)?;
                        let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dsg)?;
                        let mut o8: HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]> = piece(&self.out_s, 0)?;
                        prof::timed!("h_attn_s", launch(
                            ops_rpa::sliding_ragged_s2_b8_kvd,
                            (&mut *device, &self.q_s, win, &t.s.mask, &t.s.pen, &mut o8, &q, idx, mask, &*fk, &*fv),
                        ).await)?;
                    } else {
                    if qkv2 {
                        let x: HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]> = piece(res_in, 0)?;
                        let mut q0: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                        let mut k0: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, 0)?;
                        let mut v0: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, 0)?;
                        let mut q1: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 1)?;
                        let mut k1: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, 1)?;
                        let mut v1: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, 1)?;
                        prof::timed!("h_qkv_s", launch(
                            ops_qkvc::sliding_qkv_pair_j,
                            (
                                &mut *device, &x, &l.input_norm, &mut self.xnj, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                &t.cos_s[0], &t.sin_s[0], &t.cos_s[1], &t.sin_s[1], &mut q0, &mut k0, &mut v0, &mut q1, &mut k1, &mut v1,
                            ),
                        ).await)?;
                    } else {
                    for h in 0..2 {
                        let x: HbmTensor<bf16, Chip, m![Tc, Kc, Ke]> = piece(res_in, h)?;
                        let mut q: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, h)?;
                        let mut k: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, h)?;
                        let mut v: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, h)?;
                        let mut xn: Xn = piece(&self.xnj, 0)?;
                        prof::timed!("h_qkv_s", launch(
                            ops_qkvc::sliding_qkv_chunk,
                            (
                                &mut *device, &x, &l.input_norm, &mut xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                                &l.q_norm, &l.k_norm, &t.cos_s[h], &t.sin_s[h], &mut q, &mut k, &mut v,
                            ),
                        ).await)?;
                    }
                    }
                    if n3 {
                        // ONE launch: prefill pair + appends + decode attention (its output is stored into the last 8 rows of `out_s` through the same handle, after the prefill store)
                        let (app, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                        let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dsg)?;
                        let k: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.k_s, dsg)?;
                        let v: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.v_s, dsg)?;
                        let mut o8: HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]> = piece(&self.out_s, 0)?;
                        prof::timed!("h_attn_s", launch(
                            ops_rpa::sliding_ragged_s2_b8_h,
                            (
                                &mut *device, &self.q_s, &t.s.ring, &t.s.stage, &t.s.mask, &t.s.pen, &self.k_s, &self.v_s, &t.s.append, &mut o8,
                                &q, &k, &v, app, idx, mask, &mut *fk, &mut *fv,
                            ),
                        ).await)?;
                    } else {
                    prof::timed!("h_attn_s", launch(
                        ops_rpa::sliding_ragged_s2,
                        (&mut *device, &self.q_s, &t.s.ring, &t.s.stage, &t.s.mask, &t.s.pen, &self.k_s, &self.v_s, &t.s.append, &mut self.out_s, &mut *fk, &mut *fv),
                    ).await)?;
                    if bd == 8 {
                        let (app, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                        let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dsg)?;
                        let k: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.k_s, dsg)?;
                        let v: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.v_s, dsg)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.out_s, dsg)?;
                        prof::timed!("h_dattn_s", launch(
                            ops_dattn::sliding_decode_attention_fused,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, idx, mask, &mut o),
                        ).await)?;
                    } else if bd == 16 {
                        let (app, idx, mask) = t.s.dec16.as_ref().expect("decode tables");
                        let q: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.q_s, dsg)?;
                        let k: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.k_s, dsg)?;
                        let v: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.v_s, dsg)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.out_s, dsg)?;
                        prof::timed!("h_dattn_s", launch(
                            ops_dattn::sliding_decode_attention_fused_16,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, idx, mask, &mut o),
                        ).await)?;
                    }
                    }
                    }
                    let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                    if let (true, Some(f)) = (fp8::ffn_fold(index), ffn.fold.as_ref()) {
                        // S2_FP8 fold (default ON, lossy): e4m3 FFN with the block scale folded into the weights on chip
                        prof::timed!("h_layer_s", sleepy(Slot::LayerS, launch(
                            ops_f8::layer_sliding_t128_f8fb,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, res_in,
                                &l.pre_feedforward_norm, &ffn.up_w, &f.up_s, &f.up_gs, &ffn.gate_w, &f.gate_s, &f.gate_gs,
                                &ffn.down_w, &f.down_s, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, res_out,
                            ),
                        )).await)?;
                    } else {
                        prof::timed!("h_layer_s", sleepy(Slot::LayerS, launch(
                            ops_ffnc::layer_sliding_t128,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, res_in,
                                &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, res_out,
                            ),
                        )).await)?;
                    }
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.full[index].as_mut().expect("full flat pool");
                    for h in 0..2 {
                        let x: HbmTensor<bf16, Chip, m![Tc, Kc, Ke]> = piece(res_in, h)?;
                        let mut q: HbmTensor<bf16, Chip, m![Tc, Gf, Df]> = piece(&self.q_f, h)?;
                        let mut k: HbmTensor<bf16, Chip, m![Tc, Df]> = piece(&self.k_f, h)?;
                        let mut v: HbmTensor<bf16, Chip, m![Tc, Df]> = piece(&self.v_f, h)?;
                        let mut xn: Xn = piece(&self.xnj, 0)?;
                        prof::timed!("h_qkv_f", launch(
                            ops_qkvc::full_qkv_chunk,
                            (
                                &mut *device, &x, &l.input_norm, &mut xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                                &l.q_norm, &l.k_norm, &t.cos_f[h], &t.sin_f[h], &mut q, &mut k, &mut v,
                            ),
                        ).await)?;
                    }
                    {
                        let mut kb: HbmTensor<bf16, Chip, m![Pq, R8, Df]> = piece(&*fk, 0)?;
                        let mut vb: HbmTensor<bf16, Chip, m![Pq, R8, Df]> = piece(&*fv, 0)?;
                        match &t.f.pre {
                            FullPrefill::P2 { index: idx, mask, pen } => {
                                prof::timed!("h_attn_f", launch(
                                    ops_rpa_full::full_ragged_prefill_s2_p2,
                                    (&mut *device, &self.q_f, &self.k_f, &self.v_f, &t.f.append, idx, mask, pen, &mut self.out_f, &mut kb, &mut vb),
                                ).await)?;
                            }
                            FullPrefill::P4 { index: idx, mask, pen } => {
                                prof::timed!("h_attn_f", launch(
                                    ops_rpa_full::full_ragged_prefill_s2_p4,
                                    (&mut *device, &self.q_f, &self.k_f, &self.v_f, &t.f.append, idx, mask, pen, &mut self.out_f, &mut kb, &mut vb),
                                ).await)?;
                            }
                        }
                    }
                    if bd == 8 {
                        let (gidx, mask, app) = &t.f.dec[0];
                        let q: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 15)?;
                        let k: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 15)?;
                        let v: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 15)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 15)?;
                        prof::timed!("h_dattn_f", launch(
                            ops_dattn::full_decode_attention_5_fused,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, gidx, mask, &mut o),
                        ).await)?;
                    } else if bd == 16 {
                        let (ga, ma, aa) = &t.f.dec[0];
                        let (gb, mb, ab) = &t.f.dec[1];
                        let qa: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 14)?;
                        let qb: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 15)?;
                        let ka: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 14)?;
                        let kb: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 15)?;
                        let va: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 14)?;
                        let vb: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 15)?;
                        let mut oa: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 14)?;
                        let mut ob: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 15)?;
                        prof::timed!("h_dattn_f", launch(
                            ops_dattn::full_decode_attention_5_fused_x2,
                            (&mut *device, &qa, &qb, &ka, &va, &kb, &vb, aa, ab, &mut *fk, &mut *fv, ga, gb, ma, mb, &mut oa, &mut ob),
                        ).await)?;
                    }
                    if on_device && index + 1 == weights.layers.len() {
                        // K03: the last layer feeds only the final norm and the LM head, which read the 8 decode rows alone (the N2 view of the last 8 rows), and its K / V (written by
                        // the QKV / append launches above, from the previous layer output) are already in place: AO + FFN of those 8 rows, not of all 128
                        let attn: HbmTensor<bf16, Chip, m![Tk8, Kc, E256]> = piece(&self.out_f, 15)?;
                        let rin: HbmTensor<bf16, Chip, m![Tk8, Kc, Ke]> = piece(res_in, 15)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Tk8, Kc, Ke]> = piece(&*res_out, 15)?;
                        prof::timed!("h_layer_f", launch(
                            ops_ffnc::layer_full_t8,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                            ),
                        ).await)?;
                    } else {
                        let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, E256]> = piece(&self.out_f, 0)?;
                        // b13s (slim build): full-attention layers never fold (the full-layer fold kernel was removed)
                        {
                            prof::timed!("h_layer_f", sleepy(Slot::LayerF, launch(
                                ops_ffnc::layer_full_t128,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, res_in,
                                    &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                    &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, res_out,
                                ),
                            )).await)?;
                        }
                    }
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
            cur = 1 - cur;
        }

        // N2: when every row that needs a token is one of the (<= 8) decode rows, leave their hidden rows on the device for the argmax kernel
        self.dev_rows = None;
        // A pass that also ENDS a prompt keeps the riders on the device when the rows to serve exceed 8 (riders + finishing rows): `next_tokens` then serves the riders and the finishing rows
        // (copied to the host below) in ONE 16-row argmax launch (`final_norm_and_argmax_res1_h1`) instead of D2H + two launches. With <= 8 rows it keeps the host path.
        let tail_only = want.iter().all(|&(row, _)| row >= ROWS - bd);
        let finishing = want.iter().filter(|&&(row, _)| row < ROWS - bd).count();
        let keep = !tail_only && finishing + decs.len() > 8;
        // `on_device` (K03, hoisted above the layer loop) is the STRICT predicate (tail_only): it prunes layer 47 to the last 8 rows, which is only valid when no consumed row lies outside them.
        // The wider predicate (keep passes read the finishing-prompt rows from the full 128-row residual, so those passes ran the full layer 47) is `serve_on_device`, used from here on only.
        let serve_on_device = on_device || (n2_enabled() && bd == 8 && decs.iter().all(|dec| dec.want_hidden) && keep);
        if n2_enabled() {
            prof::add(if serve_on_device { "n2_on" } else { "n2_off" }, std::time::Duration::ZERO);
        }
        if serve_on_device {
            let view: HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]> = piece(&self.res[cur], 15)?;
            self.dev_rows = Some(DevRows { slots: decs.iter().map(|dec| dec.slot).collect(), views: vec![view] });
            for dec in decs {
                hidden[dec.slot].clear();
            }
            if keep {
                prof::add("n2_keep", std::time::Duration::ZERO);
                for (row, slot) in want.into_iter().filter(|&(row, _)| row < ROWS - bd) {
                    let one: HbmTensor<bf16, Chip, m![Kc, Ke]> = piece(&self.res[cur], row)?;
                    let h: HostTensor<bf16, m![Kc, Ke]> = prof::timed!("h_d2h", one.to_host(&mut device.pdma).await)?;
                    let host = h.into_inner().into_buf();
                    hidden[slot] = host[..width].to_vec();
                }
            }
            return Ok(());
        }

        // D2H of only the rows that are needed: the decode rows (one view of the last 8 / 16 rows) and the last row of every run that ends its prompt (one 8 KB view each)
        if decs.iter().any(|dec| dec.want_hidden) {
            let host: Vec<u8> = if bd == 8 {
                let rows: HbmTensor<bf16, Chip, m![Bd, Kc, Ke]> = piece(&self.res[cur], 15)?;
                let h: HostTensor<bf16, m![Bd, Kc, Ke]> = prof::timed!("h_d2h", rows.to_host(&mut device.pdma).await)?;
                h.into_inner().into_buf()
            } else {
                let rows: HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]> = piece(&self.res[cur], 7)?;
                let h: HostTensor<bf16, m![Bd16, Kc, Ke]> = prof::timed!("h_d2h", rows.to_host(&mut device.pdma).await)?;
                h.into_inner().into_buf()
            };
            for (b, dec) in decs.iter().enumerate() {
                if dec.want_hidden {
                    hidden[dec.slot] = host[b * WIDTH * 2..b * WIDTH * 2 + width].to_vec();
                }
            }
        }
        for (row, slot) in want.into_iter().filter(|&(row, _)| row < ROWS - bd) {
            let one: HbmTensor<bf16, Chip, m![Kc, Ke]> = piece(&self.res[cur], row)?;
            let h: HostTensor<bf16, m![Kc, Ke]> = prof::timed!("h_d2h", one.to_host(&mut device.pdma).await)?;
            let host = h.into_inner().into_buf();
            hidden[slot] = host[..width].to_vec();
        }
        Ok(())
    }
}
