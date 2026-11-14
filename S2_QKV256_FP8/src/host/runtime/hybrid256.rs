//! The 256-ROW HYBRID PASS: up to four 64-row prefill units plus 8 / 16 decode rows in ONE 256-row batch through all 48 layers, so every weight matrix streams once per 256
//! rows of the batch instead of once per 128. Sibling of `hybrid.rs` (128 rows), whose helpers (rope rows, packing, flags) it shares.
//!
//! Row layout: rows `0 .. 256 - Bd` = prefill runs (a run starts at a unit boundary and occupies `ceil(n / 64)` units, `n <= 256`), rows `256 - Bd .. 256` = decode rows (they sit in
//! the pad rows of unit 3).
//!
//! Per layer (all tensors are allocated once as 256-row tensors; every kernel argument is a typed VIEW of a byte range, `host::views::piece`, so no copies and no extra images):
//! - QKV: sliding = TWO `sliding_qkv_pair` launches (units 0+1, 2+3; weights streamed once per pair); full = the plain 64-row kernel on each of the four units;
//! - attention: sliding = ONE `sliding_ragged_s4` (two waves of two units + KV append) + the fused decode attention on the decode rows (a separate launch, so the write order is
//!   the launch order); full = `full_ragged_prefill_s4_p{2,4}` + the fused decode attention;
//! - AO + FFN: `layer_*_t128` (the FP8 fold kernels `layer_*_t128_f8fb` on the folded layers) TWICE on rows 0..128 and 128..256 (the same image back to back: no reload; the weights
//!   stream twice). Port of the t256b engine onto fin8 (R11 t256c): stage-based attention (the KVD ring path belongs to the 128-row engine).
//! The tables of the pass are packed into ONE upload (`Tpk256`) and built ahead on a thread when the worker predicts the pass (the same machinery as `hybrid.rs`).

use super::hybrid::{DevRows, PACK_ALIGN, PassDesc, Rope, bf_bytes, desc_of, i32_bytes, rows_of, zero};
use super::hybrid::{HybDec, HybRun, decode_bucket};
use super::*;
use crate::axes::{Bd, Bd16, Bk, Cx, Df, Gf, Gs, Hg, Kc, Ke, Kf, Kf64, Kx5, Ns, P256, Pb4, Pq, Pw, R8, Rg4, Rg8, Rh, Rw, Rw8, Seg, Seg4, Tc, Tf, Tpk256, Ts, Tx, Kw5};
use crate::device::ffnc::{E256, Tk8, Tk16, Tk128, Tk256};
use crate::host::chunkw::ChunkLayer;
use crate::host::fp8;
use crate::host::qkvc as qk;
use crate::host::sleepwait::{Slot, sleepy};
use crate::host::views::{piece, piece_at};
use crate::{dattn_host, ops_dattn, ops_f8, ops_ffnc, ops_qkvc, ops_rpa, ops_rpa_full, rpa_host};

type Res = HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>;
type Xn = HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>;
/// joint scratch of the pair QKV (qb: both 64-row blocks of the normalised activations; the single-block kernels use its first block)
type XnJ = HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>;
/// zero-initialised scratch of the native T = 256 QKV (`ops_qkvc::{sliding_qkv_quad_direct, full_qkv_quad}`): the normalised activations of all 256 rows
type XnQ = HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>;
/// the residual of the pass viewed as two 128-row waves of two 64-row units
type ResQ = HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Kc, Ke]>;

/// Rows of the batch.
pub(super) const ROWS: usize = 256;
const UNIT: usize = 64;
const UNITS: usize = 4;
/// Positions the tables cover (the 5-page decode kernel's limit).
const POSITIONS: usize = super::decode::LIMIT;
/// Prefill chunks must end below this position (the full-layer chunk kernels cover key pages 0..3).
pub(super) const PREFILL_LIMIT: usize = 2048;
const WIDTH: usize = 4096;

/// Whether `run_rows` and `d` decode rows fit one 256-row batch (runs fill units from unit 0; the last unit must leave `Bd` rows free for the decode rows).
pub fn fits(run_rows: &[usize], d: usize) -> bool {
    let bd = decode_bucket(d);
    let mut units = 0;
    let mut last = 0;
    for &n in run_rows {
        if n == 0 || n > ROWS {
            return false;
        }
        units += n.div_ceil(UNIT);
        last = n - UNIT * (n.div_ceil(UNIT) - 1);
    }
    if units > UNITS {
        return false;
    }
    if bd == 0 || units < UNITS {
        return true;
    }
    last + bd <= UNIT
}

/// `S2_N3_256=0` keeps the separate sliding-layer decode attention launch; default ON: the decode rows ride the same launch as the prefill attention (`ops_rpa::sliding_ragged_s4_b{8,16}_h`,
/// the 256-row form of N3: the decode rows are the last `Bd` rows of unit 3 and are stored through the SAME handle as the wave-1 prefill store).
fn n3_256() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_N3_256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_NATIVE256=0` keeps two `layer_sliding_t128_f8fb` launches per folded sliding layer; default ON: ONE native T = 256 launch (`ops_f8::layer_sliding_t256_f8fb`).
fn native256() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_NATIVE256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_NATIVE256BF=0` keeps two `layer_{sliding,full}_t128` launches per bf16-FFN layer (every layer when the fold is off); default ON: ONE native T = 256 launch
/// (`ops_ffnc::layer_{sliding,full}_t256p`: the same per-token arithmetic, every up / gate / down weight tile decoded and loaded into the TRF once for both 128-token halves).
fn native256bf() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_NATIVE256BF").map(|value| value != "0").unwrap_or(true))
}

/// `S2_KVD256=0` keeps the stage-based S = 4 attention (QKV stores K / V to the stage tensors, the attention launch appends them to the ring); default ON in the 8 / 16-decode-row buckets:
/// the QKV launches scatter K / V straight into the ring caches and the attention launch reads each unit's keys back in ONE window gather of 1280 rows (no stage tensors, no append DMAs).
fn kvd256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_KVD256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_QKV256=0` keeps the two `sliding_qkv_pair_direct_j` launches per sliding layer in the KVD-256 passes; default ON: ONE native T = 256 launch
/// (`ops_qkvc::sliding_qkv_quad_direct`: every Q / K / V weight tile decoded once for the 256 rows, one input norm, the shared k / v norm chain).
fn qkv256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_QKV256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_QKVF256=0` keeps four 64-row `full_qkv_chunk` launches per full layer; default ON: ONE native T = 256 launch (`ops_qkvc::full_qkv_quad`).
fn qkvf256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_QKVF256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_K03_256=0` keeps the full 256-row AO + FFN launch of the last layer; default ON: when every row read after the last layer is a decode row (no prompt finishes in the
/// pass), the last layer's AO + FFN runs on the 8 / 16 decode rows only (`layer_full_t8` / `layer_full_t16`, the K03 of the 128-row engine). The last layer's K / V are written by
/// its QKV / attention launches before, and its AO + FFN output only feeds the final norm + LM head, which read the decode rows alone.
fn k03_256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_K03_256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_RES2=0` keeps the 8 + 8 split of the 16 riders (first 8 riders through the host); default ON: both groups are device views of the residual (`final_norm_and_argmax_res2`).
fn res2_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_RES2").map(|value| value != "0").unwrap_or(true))
}

struct SlidingTables {
    ring: [HbmTensor<i32, Chip, m![Pw, Tx]>; 2],
    stage: [HbmTensor<i32, Chip, m![Pw, Cx]>; 2],
    mask: [HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>; 2],
    pen: [HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>; 2],
    append: HbmTensor<i32, Chip, m![P256]>,
    dec8: Option<(HbmTensor<i32, Chip, m![Bd]>, HbmTensor<i32, Chip, m![Bd, Tx]>, HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>)>,
    dec16: Option<(HbmTensor<i32, Chip, m![Bd16]>, HbmTensor<i32, Chip, m![Bd16, Tx]>, HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>)>,
    /// KVD-256 tables (8 / 16-decode-row passes): window gather index of each wave, cache row offsets of the four 64-row QKV blocks.
    kvd: Option<([HbmTensor<i32, Chip, m![Pw, Kw5]>; 2], [HbmTensor<i32, Chip, m![Tc]>; 4])>,
}

enum FullPrefill {
    P2 {
        index: [HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>; 2],
        mask: [HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>; 2],
        pen: [HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]>; 2],
    },
    P4 {
        index: [HbmTensor<i32, Chip, m![Pw, Hg, Seg4, Bk]>; 2],
        mask: [HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf64]>; 2],
        pen: [HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf64]>; 2],
    },
}

struct FullTables {
    pre: FullPrefill,
    append: HbmTensor<i32, Chip, m![Pb4]>,
    /// one entry per 8-row decode group
    dec: Vec<(HbmTensor<i32, Chip, m![Bd, Kx5]>, HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>, HbmTensor<i32, Chip, m![Bd]>)>,
}

struct Tables {
    /// the rope rows of the 4 units as one tensor each (the per-unit tensors below are views of the same bytes)
    cos_s4: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    sin_s4: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ds]>,
    cos_f4: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    sin_f4: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]>,
    cos_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    sin_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    cos_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    sin_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    s: SlidingTables,
    f: FullTables,
}

pub(super) struct HybridEngine256 {
    /// Decode rows of the last pass whose hidden rows were NOT copied to the host (`n2_enabled`, 8-row bucket only), else `None`.
    pub(super) dev_rows: Option<DevRows>,
    /// The persistent packed-upload tensor (`S2_UPLOAD_REUSE`).
    big: Option<HbmTensor<bf16, Chip, m![Tpk256]>>,
    xnj: XnJ,
    xnq: XnQ,
    /// ping-pong residual `[256, 4096]`
    res: [Res; 2],
    // sliding stage tensors: q / attention output as two waves of two units (views `[Pw, Tc, ..]` of wave w), k / v as the 256 staged rows
    q_s: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]>,
    k_s: HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    v_s: HbmTensor<bf16, Chip, m![P256, Ns, Ds]>,
    out_s: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]>,
    // full stage tensors (k / v as 8-row blocks, the type of the ragged full kernel)
    q_f: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    v_f: HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    out_f: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Gf, Df]>,
    rope: std::sync::Arc<Rope>,
    /// The tables of the NEXT pass, being built on a worker thread (`PassDesc` = the pass it was built for; used only when the real pass is identical).
    prebuilt: Option<(PassDesc, std::thread::JoinHandle<Prebuilt>)>,
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

/// The packed per-pass upload: the embedding rows first, then every table at an 8 KB boundary.
struct Packer {
    bytes: Vec<u8>,
}

impl Packer {
    fn new(embed: Vec<u8>) -> Self {
        assert_eq!(embed.len(), HbmTensor::<bf16, Chip, m![Tk256, Kc, Ke]>::size(), "embedding rows");
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

    /// The upload buffer, zero padded to the size of `[Tpk256]`.
    fn finish(mut self) -> Vec<u8> {
        static MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if self.bytes.len() > MAX.fetch_max(self.bytes.len(), std::sync::atomic::Ordering::Relaxed) {
            eprintln!("S2 hybrid256 packed upload: {} bytes used of {}", self.bytes.len(), HbmTensor::<bf16, Chip, m![Tpk256]>::size());
        }
        assert!(self.bytes.len() <= HbmTensor::<bf16, Chip, m![Tpk256]>::size(), "packed upload does not fit [Tpk256]");
        self.bytes.resize(HbmTensor::<bf16, Chip, m![Tpk256]>::size(), 0);
        self.bytes
    }
}

/// Where every table sits in the packed upload buffer (byte offsets, multiples of `PACK_ALIGN`).
struct Layout {
    bd: usize,
    npages: usize,
    rope4: [usize; 4],
    cos_s: Vec<usize>,
    sin_s: Vec<usize>,
    cos_f: Vec<usize>,
    sin_f: Vec<usize>,
    ring: [usize; 2],
    stage: [usize; 2],
    mask: [usize; 2],
    pen: [usize; 2],
    append: usize,
    sdec: Option<(usize, usize, usize)>,
    kvd: Option<([usize; 2], [usize; 4])>,
    fidx: [usize; 2],
    fmask: [usize; 2],
    fpen: [usize; 2],
    fdec: Vec<(usize, usize, usize)>,
    fappend: usize,
}

/// The packed upload buffer of a pass (`[Tpk256]` bytes: embedding rows zeroed at offset 0 for the caller to fill, then every table) and its layout. Pure host work.
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
    let pt = rpa_host::prefill_tables(4, &rpa_runs);
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
    let ft = rpa_host::full_prefill_tables(4, npages, &units);
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
    let mut pk = Packer::new(vec![0u8; ROWS * WIDTH * 2]);
    // the rope rows of the 4 units back to back (one `[Rw, Rh, Tc, D]` table each for the native T = 256 QKV); unit u = the u-th [Tc, D] block (every block is a multiple of 8 KB,
    // so the per-unit views keep the packing alignment)
    let rope4 = [
        pk.put::<bf16, m![Rw, Rh, Tc, Ds]>(&rows_of(&rope.cos_s, Ds::SIZE, &row_pos)),
        pk.put::<bf16, m![Rw, Rh, Tc, Ds]>(&rows_of(&rope.sin_s, Ds::SIZE, &row_pos)),
        pk.put::<bf16, m![Rw, Rh, Tc, Df]>(&rows_of(&rope.cos_f, Df::SIZE, &row_pos)),
        pk.put::<bf16, m![Rw, Rh, Tc, Df]>(&rows_of(&rope.sin_f, Df::SIZE, &row_pos)),
    ];
    let (bs, bf) = (HbmTensor::<bf16, Chip, m![Tc, Ds]>::size(), HbmTensor::<bf16, Chip, m![Tc, Df]>::size());
    let cos_s: Vec<usize> = (0..UNITS).map(|u| rope4[0] + u * bs).collect();
    let sin_s: Vec<usize> = (0..UNITS).map(|u| rope4[1] + u * bs).collect();
    let cos_f: Vec<usize> = (0..UNITS).map(|u| rope4[2] + u * bf).collect();
    let sin_f: Vec<usize> = (0..UNITS).map(|u| rope4[3] + u * bf).collect();
    let mut ring = [0usize; 2];
    let mut stage = [0usize; 2];
    let mut mask = [0usize; 2];
    let mut pen = [0usize; 2];
    for w in 0..2 {
        ring[w] = pk.put::<i32, m![Pw, Tx]>(i32_bytes(&pt.ring_index[w]));
        stage[w] = pk.put::<i32, m![Pw, Cx]>(i32_bytes(&pt.stage_index[w]));
        mask[w] = pk.put::<bf16, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>(bf_bytes(&pt.mask[w]));
        pen[w] = pk.put::<bf16, m![Pw, Ts % 32, Seg, Ts / 32]>(bf_bytes(&pt.pen[w]));
    }
    let append = pk.put::<i32, m![P256]>(i32_bytes(&pt.append_index));
    let sdec_o = match (&sdec, bd) {
        (Some((app, idx, mask)), 8) => Some((pk.put::<i32, m![Bd]>(i32_bytes(app)), pk.put::<i32, m![Bd, Tx]>(i32_bytes(idx)), pk.put::<bf16, m![Bd, Tx % 32, Tx / 32]>(bf_bytes(mask)))),
        (Some((app, idx, mask)), 16) => Some((pk.put::<i32, m![Bd16]>(i32_bytes(app)), pk.put::<i32, m![Bd16, Tx]>(i32_bytes(idx)), pk.put::<bf16, m![Bd16, Tx % 32, Tx / 32]>(bf_bytes(mask)))),
        _ => None,
    };
    let kvd_o = (kvd256_enabled() && (bd == 8 || bd == 16)).then(|| {
        let (win, idx) = rpa_host::kvd256_tables(&rpa_runs, &dslots, &dpos, bd);
        let w = [pk.put::<i32, m![Pw, Kw5]>(i32_bytes(&win[0])), pk.put::<i32, m![Pw, Kw5]>(i32_bytes(&win[1]))];
        let i = [0, 1, 2, 3].map(|u| pk.put::<i32, m![Tc]>(i32_bytes(&idx[u])));
        (w, i)
    });
    let mut fidx = [0usize; 2];
    let mut fmask = [0usize; 2];
    let mut fpen = [0usize; 2];
    for w in 0..2 {
        if npages == 2 {
            fidx[w] = pk.put::<i32, m![Pw, Hg, Seg, Bk]>(i32_bytes(&ft.index[w]));
            fmask[w] = pk.put::<bf16, m![Pw, Tf % 32, Tc, Kf]>(bf_bytes(&ft.mask[w]));
            fpen[w] = pk.put::<bf16, m![Pw, Tf % 32, Kf]>(bf_bytes(&ft.pen[w]));
        } else {
            fidx[w] = pk.put::<i32, m![Pw, Hg, Seg4, Bk]>(i32_bytes(&ft.index[w]));
            fmask[w] = pk.put::<bf16, m![Pw, Tf % 32, Tc, Kf64]>(bf_bytes(&ft.mask[w]));
            fpen[w] = pk.put::<bf16, m![Pw, Tf % 32, Kf64]>(bf_bytes(&ft.pen[w]));
        }
    }
    let fdec_o: Vec<(usize, usize, usize)> = fdec
        .iter()
        .map(|(gidx, mask, app)| {
            (
                pk.put::<i32, m![Bd, Kx5]>(i32_bytes(gidx)),
                pk.put::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>(bf_bytes(mask)),
                pk.put::<i32, m![Bd]>(i32_bytes(app)),
            )
        })
        .collect();
    let fappend = pk.put::<i32, m![Pb4]>(i32_bytes(&ft.append_index));
    let buf = pk.finish();
    Prebuilt { buf, layout: Layout { bd, npages, rope4, cos_s, sin_s, cos_f, sin_f, ring, stage, mask, pen, append, sdec: sdec_o, kvd: kvd_o, fidx, fmask, fpen, fdec: fdec_o, fappend } }
}

impl HybridEngine256 {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        let (cos_s, sin_s) = qk::rope_rows_bytes(Ds::SIZE, 10_000.0, Ds::SIZE, 0, POSITIONS);
        let (cos_f, sin_f) = qk::rope_rows_bytes(Df::SIZE, 1_000_000.0, 128, 0, POSITIONS);
        Ok(Self {
            xnj: zero(device).await?,
            xnq: zero(device).await?,
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

    /// Start building the packed tables of the predicted pass `next` on a worker thread.
    pub(super) fn spawn_prebuild(&mut self, next: PassDesc) {
        let rope = self.rope.clone();
        let job = next.clone();
        if let Ok(handle) = std::thread::Builder::new().name("hyb256-prebuild".into()).spawn(move || { crate::host::cpuplace::leave_launch_cpu(); build_packed(&rope, &job) }) {
            self.prebuilt = Some((next, handle));
        }
    }

    /// Upload the packed buffer of a pass in ONE host-to-device transfer: the buffer holds the embedding rows (offset 0, = the first residual) and every table (each at an
    /// 8 KB boundary); the device tensors are views of it (`views::piece_at`).
    async fn upload_packed(&mut self, device: &mut Device, pre: Prebuilt) -> Result<(Res, Tables), Error> {
        let Prebuilt { buf, layout: lo } = pre;
        if super::hybrid::upload_reuse_enabled() {
            if let Some(persistent) = self.big.as_mut() {
                prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).output(persistent).await)?;
            } else {
                self.big = Some(prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
            }
        } else {
            self.big = Some(prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
        }
        let big: &HbmTensor<bf16, Chip, m![Tpk256]> = self.big.as_ref().expect("uploaded above");
        let res0: Res = piece_at::<bf16, m![Tpk256], bf16, m![Tk256, Kc, Ke]>(big, 0)?;
        let cos_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.cos_s.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let sin_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.sin_s.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let cos_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.cos_f.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let sin_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.sin_f.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let bd = lo.bd;
        let s = SlidingTables {
            ring: [piece_at(big, lo.ring[0])?, piece_at(big, lo.ring[1])?],
            stage: [piece_at(big, lo.stage[0])?, piece_at(big, lo.stage[1])?],
            mask: [piece_at(big, lo.mask[0])?, piece_at(big, lo.mask[1])?],
            pen: [piece_at(big, lo.pen[0])?, piece_at(big, lo.pen[1])?],
            append: piece_at(big, lo.append)?,
            dec8: match (&lo.sdec, bd) {
                (Some((a, i, k)), 8) => Some((piece_at(big, *a)?, piece_at(big, *i)?, piece_at(big, *k)?)),
                _ => None,
            },
            dec16: match (&lo.sdec, bd) {
                (Some((a, i, k)), 16) => Some((piece_at(big, *a)?, piece_at(big, *i)?, piece_at(big, *k)?)),
                _ => None,
            },
            kvd: match &lo.kvd {
                Some((w, i)) => Some(([piece_at(big, w[0])?, piece_at(big, w[1])?], [piece_at(big, i[0])?, piece_at(big, i[1])?, piece_at(big, i[2])?, piece_at(big, i[3])?])),
                None => None,
            },
        };
        let pre = if lo.npages == 2 {
            FullPrefill::P2 {
                index: [piece_at(big, lo.fidx[0])?, piece_at(big, lo.fidx[1])?],
                mask: [piece_at(big, lo.fmask[0])?, piece_at(big, lo.fmask[1])?],
                pen: [piece_at(big, lo.fpen[0])?, piece_at(big, lo.fpen[1])?],
            }
        } else {
            FullPrefill::P4 {
                index: [piece_at(big, lo.fidx[0])?, piece_at(big, lo.fidx[1])?],
                mask: [piece_at(big, lo.fmask[0])?, piece_at(big, lo.fmask[1])?],
                pen: [piece_at(big, lo.fpen[0])?, piece_at(big, lo.fpen[1])?],
            }
        };
        let mut dec = Vec::new();
        for &(g, k, a) in &lo.fdec {
            dec.push((piece_at(big, g)?, piece_at(big, k)?, piece_at(big, a)?));
        }
        let f = FullTables { pre, append: piece_at(big, lo.fappend)?, dec };
        let cos_s4 = piece_at(big, lo.rope4[0])?;
        let sin_s4 = piece_at(big, lo.rope4[1])?;
        let cos_f4 = piece_at(big, lo.rope4[2])?;
        let sin_f4 = piece_at(big, lo.rope4[3])?;
        Ok((res0, Tables { cos_s4, sin_s4, cos_f4, sin_f4, cos_s: cos_s_t, sin_s: sin_s_t, cos_f: cos_f_t, sin_f: sin_f_t, s, f }))
    }

    /// One 256-row hybrid pass. `runs` (>= 1) and `decs` (<= 16) must satisfy `fits`. Leaves the final hidden rows of the runs with `want_hidden` and of the decode rows with
    /// `want_hidden` in `hidden[slot]`. `hint` = the predicted NEXT pass when it also runs on this engine (its tables are built on a thread while the device works).
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
        assert!(!runs.is_empty() && fits(&runs.iter().map(|r| r.rows.len()).collect::<Vec<_>>(), d), "runs / decode rows do not fit one 256-row hybrid batch");
        for run in runs {
            assert!(run.pos0 + run.rows.len() <= PREFILL_LIMIT, "hybrid prefill rows must lie below {PREFILL_LIMIT}");
        }

        // ---- the row map and the packed tables ----
        let t_embed = std::time::Instant::now();
        let desc = desc_of(runs, decs);
        let mut pre: Prebuilt = match self.prebuilt.take() {
            Some((d0, handle)) if d0 == desc => match prof::timed!("t_tab_wait", handle.join()) {
                Ok(pre) => {
                    prof::add("t_prebuilt_hit", std::time::Duration::ZERO);
                    pre
                }
                Err(_) => prof::timed!("t_tab_cpu", build_packed(&self.rope, &desc)),
            },
            other => {
                if other.is_some() {
                    prof::add("t_prebuilt_miss", std::time::Duration::ZERO);
                }
                prof::timed!("t_tab_cpu", build_packed(&self.rope, &desc))
            }
        };
        let embed: &mut [u8] = &mut pre.buf[..ROWS * WIDTH * 2];
        let mut filled = vec![false; ROWS];
        let width = H::SIZE * 2;
        let mut want: Vec<(usize, usize)> = Vec::new(); // (row, slot) of every hidden row to return
        let mut unit0 = 0;
        for run in runs {
            let base = UNIT * unit0;
            for (c, row) in run.rows.iter().enumerate() {
                chunk::embed_row_into(weights, row, &mut embed[(base + c) * WIDTH * 2..(base + c) * WIDTH * 2 + width]);
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
            }
        }
        prof::add("t_embed", t_embed.elapsed());
        let (res0, t) = prof::timed!("t_tables", self.upload_packed(device, pre).await)?;
        // the tables of the predicted next pass: built on a thread while the device runs this pass
        if let Some(next) = hint {
            self.spawn_prebuild(next);
        }
        self.res[0] = res0;
        let mut cur = 0usize;
        let qkv2 = super::hybrid::qkv2_enabled();
        let n3 = n3_256();
        // piece index of the first decode piece in the 256-row tensors (8-row pieces: 31; 16-row pieces: 15)
        let dall = if bd == 8 { 31usize } else { 15usize };
        // K03-256: every row read after the last layer is a decode row (`tail_only` below) -> the last layer's AO + FFN on those rows only
        let k03 = k03_256_enabled() && bd >= 8 && want.iter().all(|&(row, _)| row >= ROWS - bd);
        prof::add(if k03 { "k03_on" } else { "k03_off" }, std::time::Duration::ZERO);

        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            let flat = ws.flat.as_mut().expect("flat caches");
            let (res_a, res_b) = self.res.split_at_mut(1);
            let (res_in, res_out) = if cur == 0 { (&res_a[0], &mut res_b[0]) } else { (&res_b[0], &mut res_a[0]) };
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.sliding[index].as_mut().expect("sliding flat cache");
                    if let (true, Some((win, kidx))) = (qkv2 && n3 && kvd256_enabled(), t.s.kvd.as_ref()) {
                        // KVD-256: QKV scatters K / V (prefill rows + decode rows) into the ring caches, then ONE attention launch (4 prefill units on the ring windows + decode attention)
                        if qkv256_enabled() {
                            // native T = 256: ONE launch for the 4 units (every weight tile decoded once for the 256 rows)
                            let x: ResQ = piece(res_in, 0)?;
                            let mut q_all: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                            prof::timed!("t_qkv_s", launch(
                                ops_qkvc::sliding_qkv_quad_direct,
                                (
                                    &mut *device, &x, &l.input_norm, &mut self.xnq, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                    &t.cos_s4, &t.sin_s4, &mut q_all, &kidx[0], &kidx[1], &kidx[2], &kidx[3], &mut *fk, &mut *fv,
                                ),
                            ).await)?;
                        } else {
                        for w in 0..2 {
                            let x: HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]> = piece(res_in, w)?;
                            let mut q0: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 2 * w)?;
                            let mut q1: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 2 * w + 1)?;
                            prof::timed!("t_qkv_s", launch(
                                ops_qkvc::sliding_qkv_pair_direct_j,
                                (
                                    &mut *device, &x, &l.input_norm, &mut self.xnj, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                    &t.cos_s[2 * w], &t.sin_s[2 * w], &t.cos_s[2 * w + 1], &t.sin_s[2 * w + 1], &mut q0, &mut q1, &kidx[2 * w], &kidx[2 * w + 1], &mut *fk, &mut *fv,
                                ),
                            ).await)?;
                        }
                        }
                        let q0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                        let q1: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 1)?;
                        let mut o0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.out_s, 0)?;
                        if bd == 8 {
                            let (_, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                            let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b8_kvd,
                                (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, idx, mask, &*fk, &*fv),
                            ).await)?;
                        } else {
                            let (_, idx, mask) = t.s.dec16.as_ref().expect("decode tables");
                            let q: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b16_kvd,
                                (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, idx, mask, &*fk, &*fv),
                            ).await)?;
                        }
                    } else {
                    if qkv2 {
                        for w in 0..2 {
                            let x: HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]> = piece(res_in, w)?;
                            let mut q0: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 2 * w)?;
                            let mut k0: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, 2 * w)?;
                            let mut v0: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, 2 * w)?;
                            let mut q1: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, 2 * w + 1)?;
                            let mut k1: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, 2 * w + 1)?;
                            let mut v1: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, 2 * w + 1)?;
                            prof::timed!("t_qkv_s", launch(
                                ops_qkvc::sliding_qkv_pair_j,
                                (
                                    &mut *device, &x, &l.input_norm, &mut self.xnj, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                    &t.cos_s[2 * w], &t.sin_s[2 * w], &t.cos_s[2 * w + 1], &t.sin_s[2 * w + 1], &mut q0, &mut k0, &mut v0, &mut q1, &mut k1, &mut v1,
                                ),
                            ).await)?;
                        }
                    } else {
                        for u in 0..UNITS {
                            let x: HbmTensor<bf16, Chip, m![Tc, Kc, Ke]> = piece(res_in, u)?;
                            let mut q: HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]> = piece(&self.q_s, u)?;
                            let mut k: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.k_s, u)?;
                            let mut v: HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]> = piece(&self.v_s, u)?;
                            let mut xn: Xn = piece(&self.xnj, 0)?;
                            prof::timed!("t_qkv_s", launch(
                                ops_qkvc::sliding_qkv_chunk,
                                (
                                    &mut *device, &x, &l.input_norm, &mut xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                                    &l.q_norm, &l.k_norm, &t.cos_s[u], &t.sin_s[u], &mut q, &mut k, &mut v,
                                ),
                            ).await)?;
                        }
                    }
                    {
                        let q0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                        let q1: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 1)?;
                        let mut o0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.out_s, 0)?;
                        if n3 && bd == 8 {
                            let (app, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                            let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                            let k: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.k_s, dall)?;
                            let v: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.v_s, dall)?;
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b8_h,
                                (
                                    &mut *device, &q0, &q1, &t.s.ring[0], &t.s.ring[1], &t.s.stage[0], &t.s.stage[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1],
                                    &self.k_s, &self.v_s, &t.s.append, &mut o0, &mut o1, &q, &k, &v, app, idx, mask, &mut *fk, &mut *fv,
                                ),
                            ).await)?;
                        } else if n3 && bd == 16 {
                            let (app, idx, mask) = t.s.dec16.as_ref().expect("decode tables");
                            let q: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                            let k: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.k_s, dall)?;
                            let v: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.v_s, dall)?;
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b16_h,
                                (
                                    &mut *device, &q0, &q1, &t.s.ring[0], &t.s.ring[1], &t.s.stage[0], &t.s.stage[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1],
                                    &self.k_s, &self.v_s, &t.s.append, &mut o0, &mut o1, &q, &k, &v, app, idx, mask, &mut *fk, &mut *fv,
                                ),
                            ).await)?;
                        } else {
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4,
                                (
                                    &mut *device, &q0, &q1, &t.s.ring[0], &t.s.ring[1], &t.s.stage[0], &t.s.stage[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1],
                                    &self.k_s, &self.v_s, &t.s.append, &mut o0, &mut o1, &mut *fk, &mut *fv,
                                ),
                            ).await)?;
                        }
                    }
                    if n3 {
                        // the decode rows were handled by the fused launch above
                    } else if bd == 8 {
                        let (app, idx, mask) = t.s.dec8.as_ref().expect("decode tables");
                        let q: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                        let k: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.k_s, dall)?;
                        let v: HbmTensor<bf16, Chip, m![Bd, Ns, Ds]> = piece(&self.v_s, dall)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd, Ns, Gs, Ds]> = piece(&self.out_s, dall)?;
                        prof::timed!("t_dattn_s", launch(
                            ops_dattn::sliding_decode_attention_fused,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, idx, mask, &mut o),
                        ).await)?;
                    } else if bd == 16 {
                        let (app, idx, mask) = t.s.dec16.as_ref().expect("decode tables");
                        let q: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                        let k: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.k_s, dall)?;
                        let v: HbmTensor<bf16, Chip, m![Bd16, Ns, Ds]> = piece(&self.v_s, dall)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.out_s, dall)?;
                        prof::timed!("t_dattn_s", launch(
                            ops_dattn::sliding_decode_attention_fused_16,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, idx, mask, &mut o),
                        ).await)?;
                    }
                    }
                    if let (true, true, Some(f)) = (native256(), fp8::ffn_fold(index), ffn.fold.as_ref()) {
                        // NATIVE T = 256: ONE launch for the 256 rows, every weight tile decoded once (`ops_f8::layer_sliding_t256_f8fb`, bit-identical per token to the two T = 128 launches)
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        prof::timed!("t_layer_s", sleepy(Slot::LayerS, launch(
                            ops_f8::layer_sliding_t256_f8fb,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                &l.pre_feedforward_norm, &ffn.up_w, &f.up_s, &f.up_gs, &ffn.gate_w, &f.gate_s, &f.gate_gs,
                                &ffn.down_w, &f.down_s, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                            ),
                        )).await)?;
                    } else if native256bf() && !fp8::ffn_fold(index) {
                        // NATIVE T = 256, bf16 FFN: ONE launch for the 256 rows, every up / gate / down weight tile decoded and loaded into the TRF once (`ops_ffnc::layer_sliding_t256p`,
                        // the same per-token arithmetic as the two `layer_sliding_t128` launches)
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        prof::timed!("t_layer_s", sleepy(Slot::LayerS, launch(
                            ops_ffnc::layer_sliding_t256p,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                            ),
                        )).await)?;
                    } else {
                    for h in 0..2 {
                        let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(&self.out_s, h)?;
                        let rin: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(res_in, h)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(&*res_out, h)?;
                        if let (true, Some(f)) = (fp8::ffn_fold(index), ffn.fold.as_ref()) {
                            prof::timed!("t_layer_s", sleepy(Slot::LayerS, launch(
                                ops_f8::layer_sliding_t128_f8fb,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &l.pre_feedforward_norm, &ffn.up_w, &f.up_s, &f.up_gs, &ffn.gate_w, &f.gate_s, &f.gate_gs,
                                    &ffn.down_w, &f.down_s, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            )).await)?;
                        } else {
                            prof::timed!("t_layer_s", sleepy(Slot::LayerS, launch(
                                ops_ffnc::layer_sliding_t128,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                    &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            )).await)?;
                        }
                    }
                    }
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.full[index].as_mut().expect("full flat pool");
                    if qkvf256_enabled() {
                        // native T = 256: ONE launch for the 4 units (every Q / K weight tile decoded once for the 256 rows)
                        let x: ResQ = piece(res_in, 0)?;
                        let mut q_all: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Gf, Df]> = piece(&self.q_f, 0)?;
                        let mut k_all: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]> = piece(&self.k_f, 0)?;
                        let mut v_all: HbmTensor<bf16, Chip, m![Rw, Rh, Tc, Df]> = piece(&self.v_f, 0)?;
                        prof::timed!("t_qkv_f", launch(
                            ops_qkvc::full_qkv_quad,
                            (
                                &mut *device, &x, &l.input_norm, &mut self.xnq, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &l.q_norm, &l.k_norm, &t.cos_f4, &t.sin_f4,
                                &mut q_all, &mut k_all, &mut v_all,
                            ),
                        ).await)?;
                    } else {
                    for u in 0..UNITS {
                        let x: HbmTensor<bf16, Chip, m![Tc, Kc, Ke]> = piece(res_in, u)?;
                        let mut q: HbmTensor<bf16, Chip, m![Tc, Gf, Df]> = piece(&self.q_f, u)?;
                        let mut k: HbmTensor<bf16, Chip, m![Tc, Df]> = piece(&self.k_f, u)?;
                        let mut v: HbmTensor<bf16, Chip, m![Tc, Df]> = piece(&self.v_f, u)?;
                        let mut xn: Xn = piece(&self.xnj, 0)?;
                        prof::timed!("t_qkv_f", launch(
                            ops_qkvc::full_qkv_chunk,
                            (
                                &mut *device, &x, &l.input_norm, &mut xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                                &l.q_norm, &l.k_norm, &t.cos_f[u], &t.sin_f[u], &mut q, &mut k, &mut v,
                            ),
                        ).await)?;
                    }
                    }
                    {
                        let mut kb: HbmTensor<bf16, Chip, m![Pq, R8, Df]> = piece(&*fk, 0)?;
                        let mut vb: HbmTensor<bf16, Chip, m![Pq, R8, Df]> = piece(&*fv, 0)?;
                        let q0: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]> = piece(&self.q_f, 0)?;
                        let q1: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]> = piece(&self.q_f, 1)?;
                        let mut o0: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]> = piece(&self.out_f, 0)?;
                        let mut o1: HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]> = piece(&self.out_f, 1)?;
                        match &t.f.pre {
                            FullPrefill::P2 { index: idx, mask, pen } => {
                                prof::timed!("t_attn_f", launch(
                                    ops_rpa_full::full_ragged_prefill_s4_p2,
                                    (
                                        &mut *device, &q0, &q1, &self.k_f, &self.v_f, &t.f.append, &idx[0], &idx[1], &mask[0], &pen[0], &mask[1], &pen[1], &mut o0, &mut o1,
                                        &mut kb, &mut vb,
                                    ),
                                ).await)?;
                            }
                            FullPrefill::P4 { index: idx, mask, pen } => {
                                prof::timed!("t_attn_f", launch(
                                    ops_rpa_full::full_ragged_prefill_s4_p4,
                                    (
                                        &mut *device, &q0, &q1, &self.k_f, &self.v_f, &t.f.append, &idx[0], &idx[1], &mask[0], &pen[0], &mask[1], &pen[1], &mut o0, &mut o1,
                                        &mut kb, &mut vb,
                                    ),
                                ).await)?;
                            }
                        }
                    }
                    if bd == 8 {
                        let (gidx, mask, app) = &t.f.dec[0];
                        let q: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 31)?;
                        let k: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 31)?;
                        let v: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 31)?;
                        let mut o: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 31)?;
                        prof::timed!("t_dattn_f", launch(
                            ops_dattn::full_decode_attention_5_fused,
                            (&mut *device, &q, &k, &v, app, &mut *fk, &mut *fv, gidx, mask, &mut o),
                        ).await)?;
                    } else if bd == 16 {
                        let (ga, ma, aa) = &t.f.dec[0];
                        let (gb, mb, ab) = &t.f.dec[1];
                        let qa: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 30)?;
                        let qb: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.q_f, 31)?;
                        let ka: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 30)?;
                        let kb: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.k_f, 31)?;
                        let va: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 30)?;
                        let vb: HbmTensor<bf16, Chip, m![Bd, Df]> = piece(&self.v_f, 31)?;
                        let mut oa: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 30)?;
                        let mut ob: HbmTensor<bf16, Chip, m![Bd, Gf, Df]> = piece(&self.out_f, 31)?;
                        prof::timed!("t_dattn_f", launch(
                            ops_dattn::full_decode_attention_5_fused_x2,
                            (&mut *device, &qa, &qb, &ka, &va, &kb, &vb, aa, ab, &mut *fk, &mut *fv, ga, gb, ma, mb, &mut oa, &mut ob),
                        ).await)?;
                    }
                    if k03 && index + 1 == weights.layers.len() && !fp8::ffn_fold(index) {
                        // K03-256: AO + FFN of the decode rows only (rows 256 - bd .. 256)
                        if bd == 16 {
                            let attn: HbmTensor<bf16, Chip, m![Tk16, Kc, E256]> = piece(&self.out_f, 15)?;
                            let rin: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = piece(res_in, 15)?;
                            let mut rout: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = piece(&*res_out, 15)?;
                            prof::timed!("t_layer_f", launch(
                                ops_ffnc::layer_full_t16,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                    &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            ).await)?;
                        } else {
                            let attn: HbmTensor<bf16, Chip, m![Tk8, Kc, E256]> = piece(&self.out_f, 31)?;
                            let rin: HbmTensor<bf16, Chip, m![Tk8, Kc, Ke]> = piece(res_in, 31)?;
                            let mut rout: HbmTensor<bf16, Chip, m![Tk8, Kc, Ke]> = piece(&*res_out, 31)?;
                            prof::timed!("t_layer_f", launch(
                                ops_ffnc::layer_full_t8,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                    &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            ).await)?;
                        }
                    } else if native256bf() && !fp8::ffn_fold(index) {
                        // NATIVE T = 256, bf16 FFN (`ops_ffnc::layer_full_t256p`): one launch, every weight tile decoded once for both 128-token halves
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, E256]> = piece(&self.out_f, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        prof::timed!("t_layer_f", sleepy(Slot::LayerF, launch(
                            ops_ffnc::layer_full_t256p,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                            ),
                        )).await)?;
                    } else {
                    for h in 0..2 {
                        let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, E256]> = piece(&self.out_f, h)?;
                        let rin: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(res_in, h)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(&*res_out, h)?;
                        if let (true, Some(f)) = (fp8::ffn_fold(index), ffn.fold.as_ref()) {
                            prof::timed!("t_layer_f", sleepy(Slot::LayerF, launch(
                                ops_f8::layer_full_t128_f8fb,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &l.pre_feedforward_norm, &ffn.up_w, &f.up_s, &f.up_gs, &ffn.gate_w, &f.gate_s, &f.gate_gs,
                                    &ffn.down_w, &f.down_s, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            )).await)?;
                        } else {
                            prof::timed!("t_layer_f", sleepy(Slot::LayerF, launch(
                                ops_ffnc::layer_full_t128,
                                (
                                    &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                    &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                    &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                ),
                            )).await)?;
                        }
                    }
                    }
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
            cur = 1 - cur;
        }

        // N2 (fin8 form): the LAST 8 rows of the final residual stay on the device for the argmax kernel (`DevRows`: ONE 8-row view). With 16 riders the first 8 riders (rows 240..247) and the rows
        // that finish a prompt are copied to the host (`next_tokens` serves them as host rows in the same 16-row launch `final_norm_and_argmax_res1_h1`).
        self.dev_rows = None;
        let tail_only = want.iter().all(|&(row, _)| row >= ROWS - bd);
        let finishing = want.iter().filter(|&&(row, _)| row < ROWS - bd).count();
        let keep = !tail_only && finishing + decs.len() > 8;
        let on_device = super::hybrid::n2_enabled() && bd >= 8 && decs.iter().all(|dec| dec.want_hidden) && (tail_only || keep);
        if super::hybrid::n2_enabled() {
            prof::add(if on_device { "n2_on" } else { "n2_off" }, std::time::Duration::ZERO);
        }
        if on_device && bd == 16 && tail_only && res2_enabled() {
            // 16 riders, no prompt finishes in this pass: BOTH 8-row views of the riders stay on the device (`final_norm_and_argmax_res2`: no D2H, no re-upload)
            let views: Vec<HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>> = vec![piece(&self.res[cur], 30)?, piece(&self.res[cur], 31)?];
            self.dev_rows = Some(DevRows { slots: decs.iter().map(|dec| dec.slot).collect(), views });
            for dec in decs {
                hidden[dec.slot].clear();
            }
            prof::add("res2_on", std::time::Duration::ZERO);
            return Ok(());
        }
        if on_device {
            let view: HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]> = piece(&self.res[cur], 31)?;
            // slots[b] = request slot of row b of the view (the rows 248..255 hold riders 8..15 when bd == 16, riders 0..7 when bd == 8)
            let first = if bd == 16 { 8 } else { 0 };
            self.dev_rows = Some(DevRows { slots: decs.iter().skip(first).map(|dec| dec.slot).collect(), views: vec![view] });
            for dec in decs.iter().skip(first) {
                hidden[dec.slot].clear();
            }
            if bd == 16 {
                let rows: HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]> = piece(&self.res[cur], 30)?;
                let h: HostTensor<bf16, m![Rw8, Kc, Ke]> = prof::timed!("t_d2h", rows.to_host(&mut device.pdma).await)?;
                let host = h.into_inner().into_buf();
                for (b, dec) in decs.iter().take(8).enumerate() {
                    hidden[dec.slot] = host[b * WIDTH * 2..b * WIDTH * 2 + width].to_vec();
                }
            }
            for (row, slot) in want.into_iter().filter(|&(row, _)| row < ROWS - bd) {
                let one: HbmTensor<bf16, Chip, m![Kc, Ke]> = piece(&self.res[cur], row)?;
                let h: HostTensor<bf16, m![Kc, Ke]> = prof::timed!("t_d2h", one.to_host(&mut device.pdma).await)?;
                let host = h.into_inner().into_buf();
                hidden[slot] = host[..width].to_vec();
            }
            return Ok(());
        }

        // D2H of only the rows that are needed: the decode rows (one view of the last 8 / 16 rows) and the last row of every run that ends its prompt (one 8 KB view each)
        if decs.iter().any(|dec| dec.want_hidden) {
            let host: Vec<u8> = if bd == 8 {
                let rows: HbmTensor<bf16, Chip, m![Bd, Kc, Ke]> = piece(&self.res[cur], 31)?;
                let h: HostTensor<bf16, m![Bd, Kc, Ke]> = prof::timed!("t_d2h", rows.to_host(&mut device.pdma).await)?;
                h.into_inner().into_buf()
            } else {
                let rows: HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]> = piece(&self.res[cur], 15)?;
                let h: HostTensor<bf16, m![Bd16, Kc, Ke]> = prof::timed!("t_d2h", rows.to_host(&mut device.pdma).await)?;
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
            let h: HostTensor<bf16, m![Kc, Ke]> = prof::timed!("t_d2h", one.to_host(&mut device.pdma).await)?;
            let host = h.into_inner().into_buf();
            hidden[slot] = host[..width].to_vec();
        }
        Ok(())
    }
}
