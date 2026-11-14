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
use super::hybrid::{HintRow, HintRun, HybDec, HybRun, decode_bucket};
use super::d13e;
use super::*;
use crate::axes::{Sb18, Sb26, Sp24, Sp32};
use crate::axes::{Bn2, Gb4, W, Rb32, Rw32, Sb, Sb40, Bd, Bd16, Bk, Cx, Df, Gf, Gs, Hg, Kc, Ke, Kf, Kf64, Kx5, Ns, P256, Pb4, Pq, Pw, R8, Rg4, Rg8, Rh, Rw8, Seg, Seg4, Tc, Tf, Tpk256, Ts, Tx, Kw5, Ub, Hg4, Kq32};
use crate::device::ffnc::{E256, Tk8, Tk16, Tk128, Tk256};
use crate::host::chunkw::ChunkLayer;
use crate::host::fp8;
use crate::host::i8w;
use crate::host::qkvc as qk;
use crate::host::sleepwait::{Slot, sleepy, sleepy_with};
use crate::host::views::{piece, piece_at};

/// The sliding flat cache viewed as 32-row blocks (race-free block-granular decode gather).
type CacheB = HbmTensor<bf16, Chip, m![Rb32, Rw32, Ns, Ds]>;
use crate::{dattn_host, ops_dattn, ops_f8, ops_ffnc, ops_fl, ops_i8, ops_qkvc, ops_rpa, ops_rpa_full, rpa_host};

type Res = HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>;
type Xn = HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>;
/// joint scratch of the pair QKV (qb: both 64-row blocks of the normalised activations; the single-block kernels use its first block)
type XnJ = HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>;

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

/// `S2_QKV256=0` keeps the two `sliding_qkv_pair_direct_j` launches per sliding layer of the KVD-256 path; default ON: ONE `ops_qkvc::sliding_qkv_quad_direct` launch for the 256 rows (every
/// QKV weight tile decoded and loaded into the TRF once for 256 tokens; bit-identical per token). NPU builds only (it needs views).
fn qkv256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_QKV256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_QKV256F=0` keeps the four 64-row `full_qkv_chunk` launches per full layer; default ON (with S2_QKV256): ONE `ops_qkvc::full_qkv_quad` launch for the 256 rows.
fn qkv256f_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| qkv256_enabled() && std::env::var("S2_QKV256F").map(|value| value != "0").unwrap_or(true))
}

/// `S2_PREUP=0` keeps the b8 behaviour (the packed buffer of a pass is built, filled and uploaded at the start of that pass); default ON (NPU builds): the worker's predicted next pass is
/// filled and uploaded into the OTHER packed buffer WHILE the current pass runs (R13 lmoverlap), and only the decode rows are written when the pass starts.
pub fn preup_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && super::hybrid::upload_reuse_enabled() && std::env::var("S2_PREUP").map(|value| value != "0").unwrap_or(true))
}

/// `S2_HEAD_PACK=0` keeps the b11 grouping of a 16-rider pass that finishes a prompt (riders 8.. on the device, riders 0..7 + finishing rows on the host); default ON: riders 0..7 on the device (R14 s211host, grouping from S2_211_tok).
fn head_pack_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_HEAD_PACK").map(|value| value != "0").unwrap_or(true))
}

/// `S2_K03_256=0` keeps the full 256-row AO + FFN launch of the last layer; default ON: when every row read after the last layer is a decode row (no prompt finishes in the
/// pass), the last layer's AO + FFN runs on the 8 / 16 decode rows only (`layer_full_t8` / `layer_full_t16`, the K03 of the 128-row engine). The last layer's K / V are written by
/// its QKV / attention launches before, and its AO + FFN output only feeds the final norm + LM head, which read the decode rows alone. (R14 s211host; selection from S2_QKV256_FP8.)
fn k03_256_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_K03_256").map(|value| value != "0").unwrap_or(true))
}

/// `S2_RES2=0` keeps the 8 + 8 split of the 16 riders (first 8 riders through the host); default ON: both groups are device views of the residual (`final_norm_and_argmax_res2`).
fn res2_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_RES2").map(|value| value != "0").unwrap_or(true))
}

/// R14 d13b: the q rows of the three decode tiers inside the q stage (rows 240..248 / 248..252 / 252..256 of the 256-row stage).
fn tier_q<E: M>(
    qa: &HbmTensor<bf16, Chip, E>,
) -> Result<(HbmTensor<bf16, Chip, m![Bd16 = 8, Ns, Gs, Ds]>, HbmTensor<bf16, Chip, m![Bd16 = 4, Ns, Gs, Ds]>, HbmTensor<bf16, Chip, m![Bd16 = 4, Ns, Gs, Ds]>), Error> {
    Ok((piece(qa, 30)?, piece(qa, 62)?, piece(qa, 63)?))
}

/// `S2_FUSED256` (R13 fusedlayer; default 2): 0 = the b8 chain (scaled decode gather); 3 = the same chain with the BLOCK decode gather (reference arm of the fused kernels); 1 = K' (N3 + AO + FFN in ONE launch,
/// QKV pair launches as before); 2 = S2p: K_l = N3(l) + AO + FFN(l) + QKV(l+1), so five consecutive sliding layers launch the same image. Only the 16-decode-row bucket without a folded FFN is fused.
fn fused_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("S2_FUSED256").ok().and_then(|v| v.parse::<u8>().ok()).filter(|&v| v <= 3).unwrap_or(2))
}

/// `S2_WINDOWONCE` (R14 d13a; b12 default ON, 0 = off): when both prefill waves of a 16-rider fused pass gather the SAME window on both clusters (one run fills the four
/// units, so the two `[Pw, Kw5]` window index tables are equal) AND the riders fit the S2_DECTIER tiers, the K' launch (last sliding layer of a fused group) uses the
/// `_w1t` twin `ops_fl::sliding_n3_layer_b16_w1t`: wave 0 gathers the window once per cluster, wave 1 reuses its key / value slot tensors (bit-identical outputs, half the
/// prefill window gather bytes) and the decode rows run on the live-row tiers. Without S2_DECTIER (or without a tier fit) the K' image is the b11 one.
fn windowonce() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // b13s (slim build): the shared-window K' image was removed; always the b11 K'
    *FLAG.get_or_init(|| false)
}

/// R14 fp8pass `S2_FP8_FUSED` (default ON; only matters with `S2_FP8=fold`): a FOLDED sliding layer of a 16-decode-row pass also runs inside the fused chain
/// (`ops_fl::sliding_n3_layer{_qkv,_qkv_b16t,_b16_w1t}..._f8`: the same N3 variant + the fold's AO / FFN body), instead of leaving the S2p chain (`0` = unfused fold layer).
fn fold_fused() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("S2_FP8_FUSED").map(|value| value != "0").unwrap_or(true))
}

/// The sliding decode rows use the block-granular unscaled gather (modes 1 - 3).
fn racefree() -> bool {
    cfg!(backend = "npu") && fused_mode() >= 1
}

struct TierTables {
    idx_a: HbmTensor<i32, Chip, m![Bd16 = 8, Sb]>,
    mask_a: HbmTensor<bf16, Chip, m![Bd16 = 8, Rw32, Sb40]>,
    idx_b: HbmTensor<i32, Chip, m![Bd16 = 4, Sb26]>,
    mask_b: HbmTensor<bf16, Chip, m![Bd16 = 4, Rw32, Sp32]>,
    idx_c: HbmTensor<i32, Chip, m![Bd16 = 4, Sb18]>,
    mask_c: HbmTensor<bf16, Chip, m![Bd16 = 4, Rw32, Sp24]>,
}

struct SlidingTables {
    ring: [HbmTensor<i32, Chip, m![Pw, Tx]>; 2],
    stage: [HbmTensor<i32, Chip, m![Pw, Cx]>; 2],
    mask: [HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Seg, Ts / 32]>; 2],
    pen: [HbmTensor<bf16, Chip, m![Pw, Ts % 32, Seg, Ts / 32]>; 2],
    append: HbmTensor<i32, Chip, m![P256]>,
    dec8: Option<(HbmTensor<i32, Chip, m![Bd]>, HbmTensor<i32, Chip, m![Bd, Tx]>, HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]>)>,
    dec16: Option<(HbmTensor<i32, Chip, m![Bd16]>, HbmTensor<i32, Chip, m![Bd16, Tx]>, HbmTensor<bf16, Chip, m![Bd16, Tx % 32, Tx / 32]>)>,
    /// Race-free decode tables (`S2_RACEFREE`): block numbers `[Bd, Sb]` and key penalty `[Bd, Rw32, Sb40]` of the 8- / 16-row bucket.
    blk8: Option<(HbmTensor<i32, Chip, m![Bd, Sb]>, HbmTensor<bf16, Chip, m![Bd, Rw32, Sb40]>)>,
    blk16: Option<(HbmTensor<i32, Chip, m![Bd16, Sb]>, HbmTensor<bf16, Chip, m![Bd16, Rw32, Sb40]>)>,
    /// R14 d13b: tiered decode tables of a 16-row pass (S2_DECTIER, riders sorted and fitting the tiers).
    tier: Option<TierTables>,
    /// KVD-256 tables (8 / 16-decode-row passes): window gather index of each wave, cache row offsets of the four 64-row QKV blocks.
    kvd: Option<([HbmTensor<i32, Chip, m![Pw, Kw5]>; 2], [HbmTensor<i32, Chip, m![Tc]>; 4])>,
    /// R14 qkvre QP4: the four ring row-offset tables of `kvd` as ONE [Ub, Tc] table (the merged-post quads of K / the prologue read it)
    kidx4: Option<HbmTensor<i32, Chip, m![Ub, Tc]>>,
    /// R14 d13a: the two KVD-256 window index tables are equal (both waves gather the same 1280 rows on each cluster)
    kvd_share: bool,
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
    /// R14 attnre item E (S2_FPH4): the 4-page bucket in the hg4 layout (`ops_rpa_full::full_ragged_prefill_s4_p4h`)
    P4h {
        index: [HbmTensor<i32, Chip, m![Pw, Hg4, Seg4, Bk]>; 2],
        mask: [HbmTensor<bf16, Chip, m![Pw, Bk, Tc, Kq32]>; 2],
        pen: [HbmTensor<bf16, Chip, m![Pw, Bk, Kq32]>; 2],
    },
}

struct FullTables {
    pre: FullPrefill,
    append: HbmTensor<i32, Chip, m![Pb4]>,
    /// one entry per 8-row decode group
    dec: Vec<(HbmTensor<i32, Chip, m![Bd, Kx5]>, HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]>, HbmTensor<i32, Chip, m![Bd]>)>,
}

struct Tables {
    cos_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    sin_s: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>>,
    /// the four sliding rope blocks as one `[Ub, Tc, Ds]` table each (S2_QKV256 / fused modes: the four blocks are contiguous in the upload)
    cos_s4: Option<HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>>,
    sin_s4: Option<HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>>,
    /// the same for the full layers (`[Ub, Tc, Df]`, S2_QKV256F)
    cos_f4: Option<HbmTensor<bf16, Chip, m![Ub, Tc, Df]>>,
    sin_f4: Option<HbmTensor<bf16, Chip, m![Ub, Tc, Df]>>,
    cos_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    sin_f: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>>,
    s: SlidingTables,
    f: FullTables,
    d13e_copy: Option<(HbmTensor<i32, Chip, m![Rw8]>, HbmTensor<i32, Chip, m![Rw8]>)>,
    d13e_feed: Option<FeedTabs>,
}

/// R14 d13e P1: the reduce / feed tables of the transition into a pass (views into its packed buffer).
pub(super) struct FeedTabs {
    pub(super) pen: [HbmTensor<i32, Chip, m![Bn2, Rw8]>; 2],
    pub(super) lo: [HbmTensor<i32, Chip, m![Rw8]>; 2],
    pub(super) hi: [HbmTensor<i32, Chip, m![Rw8]>; 2],
    pub(super) dst: [HbmTensor<i32, Chip, m![Rw8]>; 2],
}

pub(super) struct HybridEngine256 {
    /// Decode rows of the last pass whose hidden rows were NOT copied to the host (`n2_enabled`, 8-row bucket only), else `None`.
    pub(super) dev_rows: Option<DevRows>,
    /// The persistent packed-upload tensors (`S2_UPLOAD_REUSE`); with `S2_PREUP` two (the pass uses `big[big_idx]`, the next pass is pre-uploaded into the other), else only `big[0]`.
    big: [Option<HbmTensor<bf16, Chip, m![Tpk256]>>; 2],
    big_idx: usize,
    /// The pre-upload of the pass that follows the last one (finished at the end of every run).
    preup: Option<PreUp>,
    /// R14 d13e P2: the late token read of the last device-fed boundary (tokens read during the next pass).
    late: Option<LateRead>,
    xnj: XnJ,
    /// zero-initialised scratch of the quad QKV (S2_QKV256 / fused modes) + the second q stage (q parity A / B: K_l reads one while it writes the other)
    xnq: Option<HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>>,
    /// R14 w8qkv: the quad scratch of the `ops_fl` quad entries (K / K_f8 / prologue) and of `ops_qkvc::full_qkv_quad`, typed by the `crate::qkv8` switch (e4m3 when ON)
    xnq8: Option<crate::qkv8::XnScratch>,
    q_s2: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]>,
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
    blk: Option<(usize, usize)>,
    /// R14 d13b: tiered decode tables (index, mask) x 3 tiers, when S2_DECTIER and the (sorted) riders of a 16-row pass fit `rpa_host::BLK_TIERS`.
    tier: Option<[(usize, usize); 3]>,
    kvd: Option<([usize; 2], [usize; 4])>,
    kidx4: Option<usize>,
    kvd_share: bool,
    fidx: [usize; 2],
    fmask: [usize; 2],
    fpen: [usize; 2],
    /// the 4-page full prefill tables are in the hg4 layout (S2_FPH4)
    fh4: bool,
    fdec: Vec<(usize, usize, usize)>,
    fappend: usize,
    /// R14 d13e P0: (sidx, didx) of the finishing-row copy into the decode bucket
    d13e_copy: Option<(usize, usize)>,
    /// R14 d13e P1: tables of the transition INTO this pass: pen[2], lo[2], hi[2], dst[2]
    d13e_feed: Option<[usize; 8]>,
}

/// The packed upload buffer of a pass (`[Tpk256]` bytes: embedding rows zeroed at offset 0 for the caller to fill, then every table) and its layout. Pure host work.
struct Prebuilt {
    buf: Vec<u8>,
    layout: Layout,
}

/// Build every attention / rope table of the pass `desc` and pack them (see `Prebuilt`). Depends on `desc` only, so it can run ahead of the pass on another thread.
fn build_packed(rope: &Rope, desc: &PassDesc, feed: Option<&d13e::FeedPlan>) -> Prebuilt {
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
    let (mut cos_s, mut sin_s, mut cos_f, mut sin_f) = (vec![], vec![], vec![], vec![]);
    let joint_rope = (qkv256_enabled() || fused_mode() >= 1) && kvd256_enabled() && (bd == 8 || bd == 16);
    if joint_rope {
        // S2_QKV256 / fused modes: the four sliding rope blocks contiguous (one `[Ub, Tc, Ds]` table each; the per-block tables are views of it at multiples of 32 KB)
        let base = pk.put::<bf16, m![Ub, Tc, Ds]>(&rows_of(&rope.cos_s, Ds::SIZE, &row_pos));
        cos_s.extend((0..UNITS).map(|h| base + h * UNIT * Ds::SIZE * 2));
        let base = pk.put::<bf16, m![Ub, Tc, Ds]>(&rows_of(&rope.sin_s, Ds::SIZE, &row_pos));
        sin_s.extend((0..UNITS).map(|h| base + h * UNIT * Ds::SIZE * 2));
    }
    let joint_rope_f = qkv256f_enabled();
    if joint_rope_f {
        let base = pk.put::<bf16, m![Ub, Tc, Df]>(&rows_of(&rope.cos_f, Df::SIZE, &row_pos));
        cos_f.extend((0..UNITS).map(|h| base + h * UNIT * Df::SIZE * 2));
        let base = pk.put::<bf16, m![Ub, Tc, Df]>(&rows_of(&rope.sin_f, Df::SIZE, &row_pos));
        sin_f.extend((0..UNITS).map(|h| base + h * UNIT * Df::SIZE * 2));
    }
    for h in 0..UNITS {
        let pos = &row_pos[h * UNIT..(h + 1) * UNIT];
        if !joint_rope {
            cos_s.push(pk.put::<bf16, m![Tc, Ds]>(&rows_of(&rope.cos_s, Ds::SIZE, pos)));
            sin_s.push(pk.put::<bf16, m![Tc, Ds]>(&rows_of(&rope.sin_s, Ds::SIZE, pos)));
        }
        if !joint_rope_f {
            cos_f.push(pk.put::<bf16, m![Tc, Df]>(&rows_of(&rope.cos_f, Df::SIZE, pos)));
            sin_f.push(pk.put::<bf16, m![Tc, Df]>(&rows_of(&rope.sin_f, Df::SIZE, pos)));
        }
    }
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
    let blk_o = (racefree() && (bd == 8 || bd == 16)).then(|| {
        let (bi, bm) = rpa_host::blk_decode_tables(&dslots, &dpos, bd);
        if bd == 8 {
            (pk.put::<i32, m![Bd, Sb]>(i32_bytes(&bi)), pk.put::<bf16, m![Bd, Rw32, Sb40]>(bf_bytes(&bm)))
        } else {
            (pk.put::<i32, m![Bd16, Sb]>(i32_bytes(&bi)), pk.put::<bf16, m![Bd16, Rw32, Sb40]>(bf_bytes(&bm)))
        }
    });
    let mut kvd_share = false;
    let tier_o = if rpa_host::dectier() && racefree() && bd == 16 {
        rpa_host::blk_decode_tables_tier(&dslots, &dpos).map(|t| {
            [
                (pk.put::<i32, m![Bd16 = 8, Sb]>(i32_bytes(&t[0].0)), pk.put::<bf16, m![Bd16 = 8, Rw32, Sb40]>(bf_bytes(&t[0].1))),
                (pk.put::<i32, m![Bd16 = 4, Sb26]>(i32_bytes(&t[1].0)), pk.put::<bf16, m![Bd16 = 4, Rw32, Sp32]>(bf_bytes(&t[1].1))),
                (pk.put::<i32, m![Bd16 = 4, Sb18]>(i32_bytes(&t[2].0)), pk.put::<bf16, m![Bd16 = 4, Rw32, Sp24]>(bf_bytes(&t[2].1))),
            ]
        })
    } else {
        None
    };
    let kvd_o = (kvd256_enabled() && (bd == 8 || bd == 16)).then(|| {
        let (win, idx) = rpa_host::kvd256_tables(&rpa_runs, &dslots, &dpos, bd);
        kvd_share = win[0] == win[1];
        let w = [pk.put::<i32, m![Pw, Kw5]>(i32_bytes(&win[0])), pk.put::<i32, m![Pw, Kw5]>(i32_bytes(&win[1]))];
        let i = [0, 1, 2, 3].map(|u| pk.put::<i32, m![Tc]>(i32_bytes(&idx[u])));
        let joint: Vec<i32> = (0..4).flat_map(|u| idx[u].iter().copied()).collect();
        let j = pk.put::<i32, m![Ub, Tc]>(i32_bytes(&joint));
        (w, i, j)
    });
    let kidx4_o = kvd_o.as_ref().map(|(_, _, j)| *j);
    let kvd_o = kvd_o.map(|(w, i, _)| (w, i));
    let mut fidx = [0usize; 2];
    let mut fmask = [0usize; 2];
    let mut fpen = [0usize; 2];
    let fh4 = npages == 4; // R14 attnre item E: the hg4 layout is the only 4-page prefill body (the b11 s4_p4 entry was deleted: net 0 device entries)
    for w in 0..2 {
        if npages == 2 {
            fidx[w] = pk.put::<i32, m![Pw, Hg, Seg, Bk]>(i32_bytes(&ft.index[w]));
            fmask[w] = pk.put::<bf16, m![Pw, Tf % 32, Tc, Kf]>(bf_bytes(&ft.mask[w]));
            fpen[w] = pk.put::<bf16, m![Pw, Tf % 32, Kf]>(bf_bytes(&ft.pen[w]));
        } else if fh4 {
            fidx[w] = pk.put::<i32, m![Pw, Hg4, Seg4, Bk]>(i32_bytes(&rpa_host::h4_index(&ft.index[w], 4)));
            fmask[w] = pk.put::<bf16, m![Pw, Bk, Tc, Kq32]>(bf_bytes(&rpa_host::h4_mask(&ft.mask[w], 4, Tc::SIZE)));
            fpen[w] = pk.put::<bf16, m![Pw, Bk, Kq32]>(bf_bytes(&rpa_host::h4_mask(&ft.pen[w], 4, 1)));
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
    // R14 d13e (only when S2_D13E >= 1, so the b11 buffer is byte-identical otherwise): P0 copy of the finishing rows into the bucket lanes d .., P1 feed tables of the transition into this pass
    let d13e_copy = if d13e::level() >= 1 {
        let mut unit0 = 0;
        let mut fin = Vec::new();
        for run in &desc.runs {
            if run.last {
                fin.push(UNIT * unit0 + run.n - 1);
            }
            unit0 += run.n.div_ceil(UNIT);
        }
        d13e::copy_rows(&fin, desc.decs.len(), bd).map(|(s, t)| (pk.put::<i32, m![Rw8]>(&d13e::i32_le(&s)), pk.put::<i32, m![Rw8]>(&d13e::i32_le(&t))))
    } else {
        None
    };
    let d13e_feed = match (d13e::level() >= 2 && bd == 16, feed) {
        (true, Some(plan)) => {
            let mut o = [0usize; 8];
            for g in 0..2 {
                o[g] = pk.put::<i32, m![Bn2, Rw8]>(&d13e::i32_le(&plan.pen(g)));
                o[2 + g] = pk.put::<i32, m![Rw8]>(&d13e::i32_le(&plan.lo(g)));
                o[4 + g] = pk.put::<i32, m![Rw8]>(&d13e::i32_le(&plan.hi(g)));
                o[6 + g] = pk.put::<i32, m![Rw8]>(&d13e::i32_le(&plan.dst(g)));
            }
            Some(o)
        }
        _ => None,
    };
    let buf = pk.finish();
    Prebuilt { buf, layout: Layout { bd, npages, cos_s, sin_s, cos_f, sin_f, ring, stage, mask, pen, append, sdec: sdec_o, blk: blk_o, tier: tier_o, kvd: kvd_o, kidx4: kidx4_o, kvd_share, fidx, fmask, fpen, fh4, fdec: fdec_o, fappend, d13e_copy, d13e_feed } }
}

type UpFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>>>>;

/// A host-to-device transfer in flight. The future (dropped first) reads the host buffer and writes the target buffer through raw pointers, so both must stay put until it completes.
struct Uploading {
    fut: UpFuture,
    _host: Box<HostTensor<bf16, m![Tpk256]>>,
    /// set by the runtime's reply channel when the DMA thread answers (the runtime's `Transfer::poll` spins 200 us on EVERY poll that finds the reply missing, so a naive per-tick poll burns
    /// 0.2 ms of CPU each time; after the first poll only a wake-up justifies the next one)
    flag: std::sync::Arc<WakeFlag>,
    waker: std::task::Waker,
    started: bool,
}

struct WakeFlag(std::sync::atomic::AtomicBool);

impl std::task::Wake for WakeFlag {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

type RawT = HbmTensor<i32, Chip, m![Bn2, Rw8 # 2048]>;
type RawFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<HostTensor<i32, m![Bn2, Rw8 # 2048]>, Error>>>>;

/// R14 d13e P2: the token read of a device-fed boundary, done during the NEXT pass (same wake-flag polling as the pre-upload).
pub(super) enum LateStage {
    Armed,
    Up { fut: RawFuture, flag: std::sync::Arc<WakeFlag>, waker: std::task::Waker },
    Done(Vec<usize>),
    Delivered,
}

pub(super) struct LateRead {
    raw: *const RawT,
    pub(super) stage: LateStage,
}

// SAFETY: like `PreUp`: used only by the worker thread that owns the engine and the device.
unsafe impl Send for LateRead {}

fn lane_tokens(raw: &[i32]) -> Vec<usize> {
    (0..d13e::LANES).map(|j| (raw[(j / 8) * 2048 + j % 8] as u32 >> 13) as usize).collect()
}

impl LateRead {
    fn start(&mut self, dev: *mut Device) {
        // SAFETY: the raw tensor lives in the model for the process; the transfer is the only pdma user while outstanding (the pre-upload transfer has finished, see `tick`).
        let fut: RawFuture = unsafe {
            let device = &mut *dev;
            let raw = &*self.raw;
            Box::pin(std::future::IntoFuture::into_future(raw.to_host::<m![Bn2, Rw8 # 2048]>(&mut device.pdma)))
        };
        let flag = std::sync::Arc::new(WakeFlag(std::sync::atomic::AtomicBool::new(true)));
        let waker = std::task::Waker::from(flag.clone());
        self.stage = LateStage::Up { fut, flag, waker };
    }

    /// One bounded step (in the nap of a long launch): start the read once `free` (no other pdma transfer outstanding), then poll on wake-ups. Returns the tokens when they land.
    fn tick(&mut self, dev: *mut Device, free: bool) -> Option<Vec<usize>> {
        if matches!(self.stage, LateStage::Armed) {
            if !free {
                return None;
            }
            self.start(dev);
        }
        if let LateStage::Up { fut, flag, waker } = &mut self.stage {
            if !flag.0.swap(false, std::sync::atomic::Ordering::AcqRel) {
                return None;
            }
            let mut cx = std::task::Context::from_waker(waker);
            match fut.as_mut().poll(&mut cx) {
                std::task::Poll::Ready(Ok(host)) => return Some(lane_tokens(&host.into_vec())),
                std::task::Poll::Ready(Err(error)) => {
                    eprintln!("S2 d13e late read failed: {error}");
                    self.stage = LateStage::Armed;
                }
                std::task::Poll::Pending => {}
            }
        }
        None
    }

    /// Completes the read (blocking) unless the tokens were already delivered to the callback.
    async fn finish(&mut self, dev: *mut Device) -> Result<Option<Vec<usize>>, Error> {
        loop {
            match std::mem::replace(&mut self.stage, LateStage::Delivered) {
                LateStage::Delivered => return Ok(None),
                LateStage::Done(tokens) => return Ok(Some(tokens)),
                LateStage::Armed => self.start(dev),
                LateStage::Up { fut, .. } => return Ok(Some(lane_tokens(&fut.await?.into_vec()))),
            }
        }
    }
}

enum PreStage {
    /// the tables of the predicted pass are being built on a thread
    Build(std::thread::JoinHandle<Prebuilt>),
    /// the buffer with the prefill-row embeddings filled in, not yet submitted
    Filled(Vec<u8>),
    Up(Uploading),
    Done,
    Failed,
}

/// R13 lmoverlap: the pre-upload of the PREDICTED next pass into the other packed buffer. All steps run on the launch thread inside the naps of the long layer launches
/// (`PreUp::tick`, one bounded step per call: fill the prefill rows, submit the transfer, poll it), `drain` finishes it at the end of the pass. The transfer itself is executed by the
/// runtime's DMA thread; no helper thread of ours exists. Raw pointers to the device and the target buffer are used because the launch futures hold `&mut Device` while the transfer
/// uses `device.pdma` (the same disclosed reborrow as the runtime's own cpuplace helper, R11 asyncloop).
pub(super) struct PreUp {
    desc: PassDesc,
    /// R14 d13e: the feed plan of the transition into this pass (its tables are in the buffer)
    feed: Option<d13e::FeedPlan>,
    /// R14 d13e: the device gathered this pass's decode rows into the buffer (set by `d13e_boundary`; read only by the hit of THIS pre-upload)
    fed: bool,
    requests: Vec<u64>,
    target: usize,
    rows: Vec<HintRun>,
    layout: Option<Layout>,
    /// the device views of the tables inside the target buffer, made as soon as the transfer completed (inside a nap, off the critical path)
    views: Option<(Res, Tables)>,
    stage: PreStage,
    dev: *mut Device,
    big: *mut HbmTensor<bf16, Chip, m![Tpk256]>,
}

/// The embedding region of the packed buffer of a pass (b8 code): prefill rows, decode rows, padding rows = clones of the nearest real row before them.
fn fill_embed_rows(weights: &Weights, runs: &[HybRun<'_>], decs: &[HybDec<'_>], bd: usize, embed: &mut [u8]) {
    let width = H::SIZE * 2;
    let mut filled = vec![false; ROWS];
    let mut unit0 = 0;
    for run in runs {
        let base = UNIT * unit0;
        for (c, row) in run.rows.iter().enumerate() {
            chunk::embed_row_into(weights, row, &mut embed[(base + c) * WIDTH * 2..(base + c) * WIDTH * 2 + width]);
            filled[base + c] = true;
        }
        unit0 += run.rows.len().div_ceil(UNIT);
    }
    for (b, dec) in decs.iter().enumerate() {
        let r = ROWS - bd + b;
        chunk::embed_row_into(weights, &dec.row, &mut embed[r * WIDTH * 2..r * WIDTH * 2 + width]);
        filled[r] = true;
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
}

// SAFETY: a `PreUp` is only used by the worker thread that owns the engine and the device; the raw pointers and the future never cross threads while a transfer is outstanding.
unsafe impl Send for PreUp {}

fn fill_hint_rows(weights: &Weights, runs: &[HintRun], embed: &mut [u8]) {
    let width = H::SIZE * 2;
    let mut filled = vec![false; ROWS];
    let mut unit0 = 0;
    for run in runs {
        let base = UNIT * unit0;
        for (c, row) in run.rows.iter().enumerate() {
            let dst = &mut embed[(base + c) * WIDTH * 2..(base + c) * WIDTH * 2 + width];
            match row {
                HintRow::Text(token) => chunk::embed_text_into(weights, *token, dst),
                HintRow::Soft(bytes) => dst.copy_from_slice(bytes),
            }
            filled[base + c] = true;
        }
        unit0 += run.rows.len().div_ceil(UNIT);
    }
    // padding rows (the decode rows are padding here: they are written when the pass starts): clones of the nearest real row before them, like `run_pass`
    let mut src = (0..ROWS).find(|&r| filled[r]).expect("a real row");
    for r in 0..ROWS {
        if filled[r] {
            src = r;
        } else {
            embed.copy_within(src * WIDTH * 2..src * WIDTH * 2 + width, r * WIDTH * 2);
        }
    }
}

impl PreUp {
    /// Polls the transfer: the first time always, then only after a wake-up (or when `force`d).
    fn poll_up(up: &mut Uploading, force: bool) -> Option<Result<(), Error>> {
        if up.started && !force && !up.flag.0.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return None;
        }
        up.started = true;
        let mut cx = std::task::Context::from_waker(&up.waker);
        match up.fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(result) => Some(result),
            std::task::Poll::Pending => None,
        }
    }

    fn settle(result: Option<Result<(), Error>>, up: Uploading) -> PreStage {
        match result {
            Some(Ok(())) => PreStage::Done,
            Some(Err(error)) => {
                eprintln!("S2 preup transfer failed: {error}");
                PreStage::Failed
            }
            None => PreStage::Up(up),
        }
    }

    fn make_views(&mut self) {
        if !matches!(self.stage, PreStage::Done) || self.views.is_some() {
            return;
        }
        let t = std::time::Instant::now();
        // SAFETY: `big` points into the engine's `big` array, which outlives this object (see `tick`).
        let made = self.layout.as_ref().map(|lo| tables_of(unsafe { &*self.big }, lo));
        match made {
            Some(Ok(views)) => self.views = Some(views),
            _ => self.stage = PreStage::Failed,
        }
        prof::add("pu_views", t.elapsed());
    }

    /// One bounded step (called from the nap of a long launch).
    fn tick(&mut self, weights: &Weights) {
        self.tick_step(weights);
        self.make_views();
    }

    fn tick_step(&mut self, weights: &Weights) {
        let t = std::time::Instant::now();
        match std::mem::replace(&mut self.stage, PreStage::Failed) {
            PreStage::Build(handle) => {
                if !handle.is_finished() {
                    self.stage = PreStage::Build(handle);
                    return;
                }
                if let Ok(Prebuilt { mut buf, layout }) = handle.join() {
                    fill_hint_rows(weights, &self.rows, &mut buf[..ROWS * WIDTH * 2]);
                    self.layout = Some(layout);
                    self.stage = PreStage::Filled(buf);
                }
                prof::add("pu_fill", t.elapsed());
            }
            PreStage::Filled(buf) => {
                let host = Box::new(HostTensor::<bf16, m![Tpk256]>::from_buf(buf));
                // SAFETY: `dev` and `big` outlive the transfer (the engine owns `big`, the worker owns the device; `drain` / `Drop` wait for completion); the launch thread does not touch
                // `device.pdma` or this buffer while the transfer is outstanding (the next use of either is after `drain`).
                let fut: UpFuture = unsafe {
                    let device = &mut *self.dev;
                    let big = &mut *self.big;
                    let host_ref: &HostTensor<bf16, m![Tpk256]> = &*(&*host as *const HostTensor<bf16, m![Tpk256]>);
                    Box::pin(host_ref.to_hbm(&mut device.pdma).output(big))
                };
                let flag = std::sync::Arc::new(WakeFlag(std::sync::atomic::AtomicBool::new(false)));
                let waker = std::task::Waker::from(flag.clone());
                let mut up = Uploading { fut, _host: host, flag, waker, started: false };
                let first = Self::poll_up(&mut up, true);
                self.stage = Self::settle(first, up);
                prof::add("pu_start", t.elapsed());
            }
            PreStage::Up(mut up) => {
                let polled = Self::poll_up(&mut up, false);
                self.stage = Self::settle(polled, up);
                prof::add("pu_poll", t.elapsed());
            }
            other => self.stage = other,
        }
    }

    /// Runs the remaining steps to completion (blocking for the transfer).
    async fn drain(&mut self, weights: &Weights) {
        loop {
            match std::mem::replace(&mut self.stage, PreStage::Failed) {
                PreStage::Done => {
                    self.stage = PreStage::Done;
                    self.make_views();
                    return;
                }
                PreStage::Failed => return,
                PreStage::Up(mut up) => {
                    let result = (&mut up.fut).await;
                    self.stage = Self::settle(Some(result), up);
                }
                PreStage::Build(handle) => {
                    self.stage = PreStage::Build(handle);
                    std::thread::sleep(std::time::Duration::from_micros(100));
                    self.tick(weights);
                }
                other => {
                    self.stage = other;
                    self.tick(weights);
                }
            }
        }
    }
}

impl Drop for PreUp {
    fn drop(&mut self) {
        // never free the host buffer / drop the target while the DMA thread may still be copying
        if let PreStage::Up(up) = &mut self.stage {
            let started = std::time::Instant::now();
            while Self::poll_up(up, true).is_none() && started.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    }
}

impl HybridEngine256 {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        let (cos_s, sin_s) = qk::rope_rows_bytes(Ds::SIZE, 10_000.0, Ds::SIZE, 0, POSITIONS);
        let (cos_f, sin_f) = qk::rope_rows_bytes(Df::SIZE, 1_000_000.0, 128, 0, POSITIONS);
        Ok(Self {
            xnj: zero(device).await?,
            xnq: if qkv256_enabled() || fused_mode() >= 1 { Some(zero(device).await?) } else { None },
            xnq8: if qkv256_enabled() || fused_mode() >= 1 {
                let n = Ub::SIZE * Tc::SIZE * Kc::SIZE * Ke::SIZE * std::mem::size_of::<crate::qkv8::XnQ>();
                Some(crate::qkv8::XnHost::from_buf(vec![0u8; n]).to_hbm(&mut device.pdma).await?)
            } else {
                None
            },
            q_s2: zero(device).await?,
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
            big: [None, None],
            big_idx: 0,
            preup: None,
            late: None,
        })
    }

    /// Start building the packed tables of the predicted pass `next` on a worker thread.
    pub(super) fn spawn_prebuild(&mut self, next: PassDesc) {
        let rope = self.rope.clone();
        let job = next.clone();
        if let Ok(handle) = std::thread::Builder::new().name("hyb256-prebuild".into()).spawn(move || { crate::host::cpuplace::leave_launch_cpu(); build_packed(&rope, &job, None) }) {
            self.prebuilt = Some((next, handle));
        }
    }

    /// Upload the packed buffer of a pass in ONE host-to-device transfer: the buffer holds the embedding rows (offset 0, = the first residual) and every table (each at an
    /// 8 KB boundary); the device tensors are views of it (`views::piece_at`).
    async fn upload_packed(&mut self, device: &mut Device, pre: Prebuilt, idx: usize) -> Result<(Res, Tables), Error> {
        let Prebuilt { buf, layout: lo } = pre;
        if super::hybrid::upload_reuse_enabled() {
            if let Some(persistent) = self.big[idx].as_mut() {
                prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).output(persistent).await)?;
            } else {
                self.big[idx] = Some(prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
            }
        } else {
            self.big[idx] = Some(prof::timed!("t_h2d", HostTensor::<bf16, m![Tpk256]>::from_buf(buf).to_hbm(&mut device.pdma).await)?);
        }
        let big: &HbmTensor<bf16, Chip, m![Tpk256]> = self.big[idx].as_ref().expect("uploaded above");
        tables_of(big, &lo)
    }

    /// One 256-row hybrid pass. `runs` (>= 1) and `decs` (<= 16) must satisfy `fits`. Leaves the final hidden rows of the runs with `want_hidden` and of the decode rows with
    /// `want_hidden` in `hidden[slot]`. `hint` = the predicted NEXT pass when it also runs on this engine (its tables are built on a thread while the device works); `hint_rows` = its
    /// prefill rows (R13 pre-upload).
    pub(super) async fn run(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        runs: &[HybRun<'_>],
        decs: &[HybDec<'_>],
        hidden: &mut [Hidden],
        hint: Option<PassDesc>,
        hint_rows: Option<Vec<HintRun>>,
        hint_feed: Option<d13e::FeedPlan>,
        blind: bool,
        late_cb: Option<&mut dyn FnMut(&[usize])>,
    ) -> Result<(), Error> {
        let result = self.run_pass(device, ws, weights, runs, decs, hidden, hint, hint_rows, hint_feed, blind, late_cb).await;
        // the pre-upload of the next pass finishes here (normally long done: the pass takes 70 ms and the transfer ~2 ms)
        if let Some(pu) = self.preup.as_mut() {
            prof::timed!("pu_drain", pu.drain(weights).await);
        }
        result
    }
}

impl HybridEngine256 {
    /// R14 d13e P1: whether the transition into the pre-uploaded next pass can be fed from the device: the pre-upload is complete with its feed tables, and its feed plan was made for
    /// exactly the lanes the last pass left on the device (`lane_slots` = `DevRows::slots`).
    pub(super) fn d13e_feed_ready(&self, lane_slots: &[usize]) -> bool {
        self.preup.as_ref().is_some_and(|pu| {
            matches!(pu.stage, PreStage::Done)
                && pu.views.as_ref().is_some_and(|(_, t)| t.d13e_feed.is_some())
                && pu.feed.as_ref().is_some_and(|f| f.k_slots == lane_slots)
        })
    }

    /// R14 d13e P2: the pass `desc` (prefill request ids `requests`) is the pre-uploaded, device-fed one, so it can launch before its decode rows' tokens are on the host.
    pub(super) fn d13e_blind_ready(&self, desc: &PassDesc, requests: &[u64]) -> bool {
        let idx = if preup_enabled() { 1 - self.big_idx } else { 0 };
        self.preup.as_ref().is_some_and(|pu| {
            pu.fed && matches!(pu.stage, PreStage::Done) && pu.views.is_some() && pu.target == idx && pu.desc == *desc && pu.requests.iter().eq(requests.iter())
        })
    }

    /// R14 d13e P2: the lane tokens of the last late read if the callback has not consumed them (blocking until the read lands); `None` if delivered or nothing armed.
    pub(super) async fn d13e_late_take(&mut self, device: &mut Device) -> Result<Option<Vec<usize>>, Error> {
        let Some(mut lr) = self.late.take() else { return Ok(None) };
        let dev: *mut Device = &mut *device;
        let t = std::time::Instant::now();
        let out = lr.finish(dev).await;
        prof::add("d13e_late_wait", t.elapsed());
        out
    }

    /// R14 d13e P1, after the res2 LM-head launch of the last pass (keys in `keys16`): reduce both 8-row groups to their tokens (the host rule, the variant of each lane's session),
    /// gather the next pass's decode rows into its pre-uploaded buffer, read the 2 x 8 tokens back. Returns the token of every lane (lane = row 240 + j of the last pass).
    pub(super) async fn d13e_boundary(&mut self, device: &mut Device, fd: &d13e::FeedDev, keys16: &crate::ops_lmhead::Keys16x2, noread: bool) -> Result<Vec<usize>, Error> {
        let pu = self.preup.as_ref().expect("d13e_feed_ready checked");
        // the keys16 of the res2 launch are reduced HERE, before any other LM-head launch of this boundary can rewrite them (host groups run after this call)
        let plan = pu.feed.as_ref().expect("checked");
        let ft = pu.views.as_ref().and_then(|(_, t)| t.d13e_feed.as_ref()).expect("checked");
        let big = self.big[pu.target].as_ref().expect("pre-upload target");
        let mut bucket: HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]> = piece_at::<bf16, m![Tpk256], bf16, m![Bd16, Kc, Ke]>(big, (ROWS - 16) * WIDTH * 2)?;
        let mut idx: Vec<HbmTensor<i32, Chip, m![Rw8]>> = Vec::with_capacity(2);
        for g in 0..2 {
            let keys: crate::ops_d13e::Keys8x2 = piece(keys16, g)?;
            let mut raw = d13e::FeedDev::piece(&fd.raw, g)?;
            prof::timed!("d13e_reduce", launch(crate::ops_d13e::d13e_reduce, (&mut *device, &keys, &fd.g3, &fd.g6, &ft.pen[g], &fd.lo_free, &fd.hi_free, &mut raw)).await)?;
            if plan.overridden(g) {
                let mut off = d13e::FeedDev::piece(&fd.off, g)?;
                prof::timed!("d13e_reduce", launch(crate::ops_d13e::d13e_reduce, (&mut *device, &keys, &fd.g3, &fd.g6, &ft.pen[g], &ft.lo[g], &ft.hi[g], &mut off)).await)?;
                idx.push(off);
            } else {
                idx.push(raw);
            }
        }
        prof::timed!("d13e_feed", launch(crate::ops_d13e::d13e_feed16, (&mut *device, &fd.table, &idx[0], &ft.dst[0], &idx[1], &ft.dst[1], &mut bucket)).await)?;
        if noread {
            // P2: the tokens are read during the next pass (`LateRead`)
            self.late = Some(LateRead { raw: &fd.raw, stage: LateStage::Armed });
            if let Some(pu) = self.preup.as_mut() {
                pu.fed = true;
            }
            prof::add("d13e_p2", std::time::Duration::ZERO);
            return Ok(Vec::new());
        }
        let raw = prof::timed!("d13e_d2h", fd.raw.to_host::<m![Bn2, Rw8 # 2048]>(&mut device.pdma).await)?.into_vec();
        let tokens: Vec<usize> = (0..d13e::LANES).map(|j| (raw[(j / 8) * 2048 + j % 8] as u32 >> 13) as usize).collect();
        if let Some(pu) = self.preup.as_mut() {
            pu.fed = true;
        }
        Ok(tokens)
    }
}

/// The device views of the pass tables inside the packed buffer `big`.
fn tables_of(big: &HbmTensor<bf16, Chip, m![Tpk256]>, lo: &Layout) -> Result<(Res, Tables), Error> {
    {
        let res0: Res = piece_at::<bf16, m![Tpk256], bf16, m![Tk256, Kc, Ke]>(big, 0)?;
        let cos_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.cos_s.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let sin_s_t: Vec<HbmTensor<bf16, Chip, m![Tc, Ds]>> = lo.sin_s.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let cos_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.cos_f.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let sin_f_t: Vec<HbmTensor<bf16, Chip, m![Tc, Df]>> = lo.sin_f.iter().map(|&o| piece_at(big, o)).collect::<Result<_, _>>()?;
        let bd = lo.bd;
        let joint = (qkv256_enabled() || fused_mode() >= 1) && kvd256_enabled() && (bd == 8 || bd == 16);
        let (cos_s4, sin_s4) = if joint { (Some(piece_at(big, lo.cos_s[0])?), Some(piece_at(big, lo.sin_s[0])?)) } else { (None, None) };
        let (cos_f4, sin_f4) = if qkv256f_enabled() { (Some(piece_at(big, lo.cos_f[0])?), Some(piece_at(big, lo.sin_f[0])?)) } else { (None, None) };
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
            blk8: match (&lo.blk, bd) {
                (Some((i, k)), 8) => Some((piece_at(big, *i)?, piece_at(big, *k)?)),
                _ => None,
            },
            blk16: match (&lo.blk, bd) {
                (Some((i, k)), 16) => Some((piece_at(big, *i)?, piece_at(big, *k)?)),
                _ => None,
            },
            tier: match &lo.tier {
                Some(t) => Some(TierTables {
                    idx_a: piece_at(big, t[0].0)?,
                    mask_a: piece_at(big, t[0].1)?,
                    idx_b: piece_at(big, t[1].0)?,
                    mask_b: piece_at(big, t[1].1)?,
                    idx_c: piece_at(big, t[2].0)?,
                    mask_c: piece_at(big, t[2].1)?,
                }),
                None => None,
            },
            kvd: match &lo.kvd {
                Some((w, i)) => Some(([piece_at(big, w[0])?, piece_at(big, w[1])?], [piece_at(big, i[0])?, piece_at(big, i[1])?, piece_at(big, i[2])?, piece_at(big, i[3])?])),
                None => None,
            },
            kidx4: match lo.kidx4 {
                Some(j) => Some(piece_at(big, j)?),
                None => None,
            },
            kvd_share: lo.kvd_share,
        };
        let pre = if lo.npages == 2 {
            FullPrefill::P2 {
                index: [piece_at(big, lo.fidx[0])?, piece_at(big, lo.fidx[1])?],
                mask: [piece_at(big, lo.fmask[0])?, piece_at(big, lo.fmask[1])?],
                pen: [piece_at(big, lo.fpen[0])?, piece_at(big, lo.fpen[1])?],
            }
        } else if lo.fh4 {
            FullPrefill::P4h {
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
        let d13e_copy = match lo.d13e_copy {
            Some((a, b)) => Some((piece_at(big, a)?, piece_at(big, b)?)),
            None => None,
        };
        let d13e_feed = match lo.d13e_feed {
            Some(o) => Some(FeedTabs {
                pen: [piece_at(big, o[0])?, piece_at(big, o[1])?],
                lo: [piece_at(big, o[2])?, piece_at(big, o[3])?],
                hi: [piece_at(big, o[4])?, piece_at(big, o[5])?],
                dst: [piece_at(big, o[6])?, piece_at(big, o[7])?],
            }),
            None => None,
        };
        Ok((res0, Tables { cos_s: cos_s_t, sin_s: sin_s_t, cos_s4, sin_s4, cos_f4, sin_f4, cos_f: cos_f_t, sin_f: sin_f_t, s, f, d13e_copy, d13e_feed }))
    }
}

impl HybridEngine256 {
    /// The body of `run` (see there).
    async fn run_pass(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        runs: &[HybRun<'_>],
        decs: &[HybDec<'_>],
        hidden: &mut [Hidden],
        hint: Option<PassDesc>,
        hint_rows: Option<Vec<HintRun>>,
        hint_feed: Option<d13e::FeedPlan>,
        blind: bool,
        mut late_cb: Option<&mut dyn FnMut(&[usize])>,
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
        let width = H::SIZE * 2;
        let use_preup = preup_enabled();
        let idx = if use_preup { 1 - self.big_idx } else { 0 };
        let dev_ptr: *mut Device = &mut *device;
        // R13: was this pass pre-uploaded while the previous one ran? (`run` drained the transfer; a stale or mismatching pre-upload is simply dropped)
        let mut hit: Option<(Res, Tables)> = None;
        let mut fed = false;
        if let Some(mut pu) = self.preup.take() {
            pu.drain(weights).await;
            let ok = matches!(pu.stage, PreStage::Done) && pu.views.is_some() && pu.target == idx && pu.desc == desc && pu.requests.iter().eq(runs.iter().map(|r| &r.request));
            prof::add(if ok { "pu_hit" } else { "pu_miss" }, std::time::Duration::ZERO);
            if ok {
                hit = pu.views.take();
                fed = pu.fed;
            }
        }
        if blind && !(hit.is_some() && fed) {
            // the worker checked `d13e_blind_ready` with the same comparisons: unreachable unless the pre-upload failed in between
            return Err(Error::Chips("d13e blind pass without its pre-uploaded, device-fed buffer".into()));
        }
        let mut want: Vec<(usize, usize)> = Vec::new(); // (row, slot) of every hidden row to return
        let mut unit0 = 0;
        for run in runs {
            if run.want_hidden {
                want.push((UNIT * unit0 + run.rows.len() - 1, run.slot));
            }
            unit0 += run.rows.len().div_ceil(UNIT);
        }
        for (b, dec) in decs.iter().enumerate() {
            if dec.want_hidden {
                want.push((ROWS - bd + b, dec.slot));
            }
        }
        let (res0, t) = if let Some(views) = hit {
            // the prefill rows and every table are already on the device (big[idx]); only the decode rows (tokens known now) are written
            let t_patch = std::time::Instant::now();
            if fed {
                // R14 d13e P1: the device already gathered these decode rows into big[idx] (`feed`), from the same tokens
                prof::add("d13e_fed", std::time::Duration::ZERO);
            } else if bd > 0 {
                let mut rows = vec![0u8; bd * WIDTH * 2];
                for (b, dec) in decs.iter().enumerate() {
                    chunk::embed_row_into(weights, &dec.row, &mut rows[b * WIDTH * 2..b * WIDTH * 2 + width]);
                }
                for b in decs.len()..bd {
                    rows.copy_within((decs.len() - 1) * WIDTH * 2..(decs.len() - 1) * WIDTH * 2 + width, b * WIDTH * 2);
                }
                let big = self.big[idx].as_ref().expect("pre-uploaded buffer");
                let start = (ROWS - bd) * WIDTH * 2;
                if bd == 8 {
                    let mut view: HbmTensor<bf16, Chip, m![Bd, Kc, Ke]> = piece_at::<bf16, m![Tpk256], bf16, m![Bd, Kc, Ke]>(big, start)?;
                    let host = HostTensor::<bf16, m![Bd, Kc, Ke]>::from_buf(rows);
                    prof::timed!("t_h2d", host.to_hbm(&mut device.pdma).output(&mut view).await)?;
                } else {
                    let mut view: HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]> = piece_at::<bf16, m![Tpk256], bf16, m![Bd16, Kc, Ke]>(big, start)?;
                    let host = HostTensor::<bf16, m![Bd16, Kc, Ke]>::from_buf(rows);
                    prof::timed!("t_h2d", host.to_hbm(&mut device.pdma).output(&mut view).await)?;
                }
            }
            prof::add("t_embed", t_patch.elapsed());
            views
        } else {
            let mut pre: Prebuilt = match self.prebuilt.take() {
                Some((d0, handle)) if d0 == desc => match prof::timed!("t_tab_wait", handle.join()) {
                    Ok(pre) => {
                        prof::add("t_prebuilt_hit", std::time::Duration::ZERO);
                        pre
                    }
                    Err(_) => prof::timed!("t_tab_cpu", build_packed(&self.rope, &desc, None)),
                },
                other => {
                    if other.is_some() {
                        prof::add("t_prebuilt_miss", std::time::Duration::ZERO);
                    }
                    prof::timed!("t_tab_cpu", build_packed(&self.rope, &desc, None))
                }
            };
            fill_embed_rows(weights, runs, decs, bd, &mut pre.buf[..ROWS * WIDTH * 2]);
            prof::add("t_embed", t_embed.elapsed());
            prof::timed!("t_tables", self.upload_packed(device, pre, idx).await)?
        };
        self.big_idx = idx;
        // the NEXT pass: with `S2_PREUP` its rows are filled and its buffer uploaded into the other packed buffer while this pass runs; otherwise (and when the worker gave no rows) only
        // its tables are built on a thread (b8 behaviour)
        let mut pre_next: Option<PreUp> = None;
        if let Some(next) = hint {
            let rows_ok = hint_rows.as_ref().is_some_and(|rows| rows.len() == next.runs.len() && rows.iter().zip(&next.runs).all(|(h, r)| h.rows.len() == r.n));
            if use_preup && rows_ok && !next.decs.is_empty() {
                let target = 1 - idx;
                if self.big[target].is_none() {
                    self.big[target] = Some(prof::timed!("pu_alloc", HostTensor::<bf16, m![Tpk256]>::from_buf(vec![0u8; HbmTensor::<bf16, Chip, m![Tpk256]>::size()]).to_hbm(&mut device.pdma).await)?);
                }
                let big_ptr: *mut HbmTensor<bf16, Chip, m![Tpk256]> = self.big[target].as_mut().expect("allocated above");
                let rope = self.rope.clone();
                let job = next.clone();
                let feed = if bd == 16 { hint_feed.clone() } else { None };
                let feed_keep = feed.clone();
                if let Ok(handle) = std::thread::Builder::new().name("hyb256-preup".into()).spawn(move || {
                    crate::host::cpuplace::leave_launch_cpu();
                    build_packed(&rope, &job, feed.as_ref())
                }) {
                    let rows = hint_rows.expect("checked");
                    pre_next = Some(PreUp {
                        requests: rows.iter().map(|r| r.request).collect(),
                        desc: next,
                        feed: feed_keep,
                        fed: false,
                        target,
                        rows,
                        layout: None,
                        views: None,
                        stage: PreStage::Build(handle),
                        dev: dev_ptr,
                        big: big_ptr,
                    });
                }
            } else {
                self.spawn_prebuild(next);
            }
        }
        // shadow work of the long launches (called once per launch, after it was submitted)
        self.preup = pre_next;
        let preup_slot = &mut self.preup;
        let late_slot = &mut self.late;
        let mut tick = || {
            // R14 d13e P2 first: the late token read of the previous boundary (16 KB; its tokens are waited for by the host), then the pre-upload of the next pass
            // (6 MB, needed only at the end of this pass); one pdma transfer at a time
            let late_busy = late_slot.as_ref().is_some_and(|lr| matches!(lr.stage, LateStage::Armed | LateStage::Up { .. }));
            if let Some(lr) = late_slot.as_mut() {
                let free = preup_slot.as_ref().is_none_or(|pu| !matches!(pu.stage, PreStage::Up(_)));
                if let Some(tokens) = lr.tick(dev_ptr, free) {
                    match late_cb.as_mut() {
                        Some(cb) => {
                            prof::add("d13e_late_cb", std::time::Duration::ZERO);
                            cb(&tokens);
                            lr.stage = LateStage::Delivered;
                        }
                        None => lr.stage = LateStage::Done(tokens),
                    }
                }
            }
            if let Some(pu) = preup_slot.as_mut() {
                // while the late read is outstanding the pre-upload only builds / fills (its transfer starts after the read)
                if !late_busy || !matches!(pu.stage, PreStage::Filled(_)) {
                    pu.tick(weights);
                }
            }
        };
        self.res[0] = res0;
        let mut cur = 0usize;
        // fused mode 2: the QKV of the layer about to run was already computed by the previous launch (K_{l-1})
        let mut qkv_done = false;
        let qkv2 = super::hybrid::qkv2_enabled();
        let n3 = n3_256();
        // piece index of the first decode piece in the 256-row tensors (8-row pieces: 31; 16-row pieces: 15)
        let dall = if bd == 8 { 31usize } else { 15usize };
        // K03-256: every row read after the last layer is a decode row (`tail_only` below) -> the last layer's AO + FFN on those rows only
        let k03 = k03_256_enabled() && bd >= 8 && want.iter().all(|&(row, _)| row >= ROWS - bd);
        prof::add(if k03 { "k03_on" } else { "k03_off" }, std::time::Duration::ZERO);

        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            let flat = ws.flat.as_mut().expect("flat caches");
            let done_in = std::mem::replace(&mut qkv_done, false);
            let (res_a, res_b) = self.res.split_at_mut(1);
            let (res_in, res_out) = if cur == 0 { (&res_a[0], &mut res_b[0]) } else { (&res_b[0], &mut res_a[0]) };
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let fm = fused_mode();
                    // fusable(i): the sliding layer i of a 16-decode-row KVD pass with a bf16 FFN runs as K / K' (S2_FUSED256 = 1, 2)
                    let fusable = |i: usize| -> bool {
                        (fm == 1 || fm == 2)
                            && bd == 16
                            && qkv2
                            && n3
                            && kvd256_enabled()
                            && native256bf()
                            && i < weights.layers.len()
                            && i < super::debug_layers()
                            && matches!((&weights.layers[i], &weights.chunk[i]), (Layer::Sliding(_), ChunkLayer::Sliding { .. }))
                            && (!fp8::ffn_fold(i) || fold_fused())
                    };
                    if let (true, Some((win, kidx)), Some((bidx, bmask)), Some(cos4), Some(sin4)) =
                        (fusable(index), t.s.kvd.as_ref(), t.s.blk16.as_ref(), t.cos_s4.as_ref(), t.sin_s4.as_ref())
                    {
                        let next_ok = fm == 2 && fusable(index + 1);
                        // q stage parity: layer l reads the q stage `index % 2` (written by its prologue or by K_{l-1}); K_l writes the stage of the other parity
                        let (qa, qb) = if index % 2 == 0 { (&self.q_s, &self.q_s2) } else { (&self.q_s2, &self.q_s) };
                        let (sl_a, sl_b) = flat.sliding.split_at_mut(index + 1);
                        let (fk, fv) = sl_a[index].as_mut().expect("sliding flat cache");
                        let xnq = self.xnq8.as_mut().expect("quad scratch");
                        if !done_in {
                            // prologue: the quad QKV of this layer as its own launch (first sliding layer after a full layer / the pass start)
                            let x: HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]> = piece(res_in, 0)?;
                            let mut q: HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]> = piece(qa, 0)?;
                            let qn = ops_fl::QkvNext {
                                input_norm: &l.input_norm, wq: &qkv.wq, sq: &qkv.sq, wk: &qkv.wk, sk: &qkv.sk, wv: &qkv.wv, sv: &qkv.sv, q_norm: &l.q_norm, k_norm: &l.k_norm,
                                cos: cos4, sin: sin4, kidx0: &kidx[0], kidx1: &kidx[1], kidx2: &kidx[2], kidx3: &kidx[3],
                            };
                            prof::timed!("t_qkv_s", launch(
                                ops_fl::sliding_qkv_quad,
                                (&mut *device, &x, &qn, t.s.kidx4.as_ref().expect("kidx4 (built with the KVD tables)"), &mut *xnq, &mut q, &mut *fk, &mut *fv),
                            ).await)?;
                        }
                        let q0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(qa, 0)?;
                        let q1: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(qa, 1)?;
                        let q_d: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(qa, dall)?;
                        let mut o0: HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]> = piece(&self.out_s, 0)?;
                        let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                        let kb: CacheB = piece(&*fk, 0)?;
                        let vb: CacheB = piece(&*fv, 0)?;
                        let n3in = ops_fl::N3In16 {
                            q_p0: &q0, q_p1: &q1, win0: &win[0], win1: &win[1], mask_p0: &t.s.mask[0], pen_p0: &t.s.pen[0], mask_p1: &t.s.mask[1], pen_p1: &t.s.pen[1],
                            q_d: &q_d, index_d: bidx, mask_d: bmask, k_cache: &*fk, v_cache: &*fv, k_blk: &kb, v_blk: &vb,
                        };
                        let lw = ops_fl::LayerW {
                            w0: &ao.w[0], w1: &ao.w[1], w2: &ao.w[2], s0: &ao.s[0], s1: &ao.s[1], s2: &ao.s[2], post_attn_norm: &l.post_attention_norm, pre_w: &ffn.pre_norm,
                            up_w: &ffn.up_w, up_s: &ffn.up_s, ugs: &l.mlp.up_global_scale, gate_w: &ffn.gate_w, gate_s: &ffn.gate_s, ggs: &l.mlp.gate_global_scale,
                            down_w: &ffn.down_w, down_s: &ffn.down_s, dgs: &l.mlp.down_global_scale, post_ff: &l.post_feedforward_norm,
                        };
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        // R14 d13a: both waves read the same window on both clusters => K' (the last layer of a group) gathers it once per cluster. K keeps its two
                        // window pairs: its wave-0 pair is issued under the TU-bound AO/FFN stream (already hidden, measured +2.1k HW cycles for a K twin), so only K' gains.
                        let w1 = windowonce() && t.s.kvd_share;
                        // R14 fp8pass: a folded layer keeps its place in the fused chain (fold AO / FFN body inside the same K / K' variant)
                        // R14 fixedcost PREBAKE: the fold body reads the pre-decoded e4m3 weights (`FoldFfn::w8`, built at load for every `ffn_fold` layer)
                        let lw8 = match (fp8::ffn_fold(index), ffn.fold.as_ref().and_then(|f| f.w8.as_ref().map(|w8| (f, w8)))) {
                            (true, Some((f, w8))) => Some(ops_fl::LayerW8 {
                                w0: &ao.w[0], w1: &ao.w[1], w2: &ao.w[2], s0: &ao.s[0], s1: &ao.s[1], s2: &ao.s[2], post_attn_norm: &l.post_attention_norm,
                                pre_w: &l.pre_feedforward_norm, up_w8: &w8.up, ugs: &f.up_gs, gate_w8: &w8.gate, ggs: &f.gate_gs,
                                down_w8: &w8.down, dgs: &f.down_gs, post_ff: &l.post_feedforward_norm,
                            }),
                            _ => None,
                        };
                        // S2_I8_LAYERS (default OFF; int8 tensors exist only for allow-listed, non-folded sliding layers): the int8 K REPLACES the bf16 K of this layer;
                        // not next-fusable => the bf16 path below (there is no int8 K')
                        let lwi = match (lw8.is_none() && next_ok, ffn.i8.as_ref()) {
                            (true, Some(q)) => Some(ops_i8::LayerW8 {
                                w0: &ao.w[0], w1: &ao.w[1], w2: &ao.w[2], s0: &ao.s[0], s1: &ao.s[1], s2: &ao.s[2], post_attn_norm: &l.post_attention_norm, pre_w: q.pre_w.as_ref().unwrap_or(&ffn.pre_norm),
                                up_q: &q.up_q, up_ws: &q.up_ws, gate_q: &q.gate_q, gate_ws: &q.gate_ws,
                                down_w: &ffn.down_w, down_s: &ffn.down_s, dgs: &l.mlp.down_global_scale, post_ff: &l.post_feedforward_norm, had: &q.had,
                            }),
                            _ => None,
                        };
                        // big8 image budget: the bf16 K image is NOT built (the int8 K replaces it); a fusable layer without int8 or fold tensors runs the
                        // unfused chain below and the next layer runs its own quad prologue
                        let fused_k = next_ok && (lwi.is_some() || lw8.is_some());
                        if fused_k {
                            let (Layer::Sliding(l2), ChunkLayer::Sliding { qkv: qkvn, .. }) = (&weights.layers[index + 1], &weights.chunk[index + 1]) else { unreachable!("fusable") };
                            let (nk, nv) = sl_b[0].as_mut().expect("sliding flat cache");
                            let qn = ops_fl::QkvNext {
                                input_norm: &l2.input_norm, wq: &qkvn.wq, sq: &qkvn.sq, wk: &qkvn.wk, sk: &qkvn.sk, wv: &qkvn.wv, sv: &qkvn.sv, q_norm: &l2.q_norm, k_norm: &l2.k_norm,
                                cos: cos4, sin: sin4, kidx0: &kidx[0], kidx1: &kidx[1], kidx2: &kidx[2], kidx3: &kidx[3],
                            };
                            let mut qo: HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]> = piece(qb, 0)?;
                            let k4 = t.s.kidx4.as_ref().expect("kidx4 (built with the KVD tables)");
                            if let Some(lwi) = lwi.as_ref() {
                                prof::timed!("t_fused_i8_s", launch(
                                    ops_i8::sliding_n3_layer_qkv_b16_i8,
                                    (&mut *device, &n3in, lwi, &qn, k4, &l.layer_scalar, &rin, &mut o0, &mut o1, &mut rout, &mut *xnq, &mut qo, &mut *nk, &mut *nv),
                                ).await)?;
                            } else if let Some(lw8) = lw8.as_ref() {
                                prof::timed!("t_fused8_s", launch(
                                    ops_fl::sliding_n3_layer_qkv_b16_f8,
                                    (&mut *device, &n3in, lw8, &qn, k4, &l.layer_scalar, &rin, &mut o0, &mut o1, &mut rout, &mut *xnq, &mut qo, &mut *nk, &mut *nv),
                                ).await)?;
                            }
                        } else {
                            // R14 b12: K' uses the combined shared-window + tiered-decode image (`_w1t`) only when BOTH apply to this pass (one run fills the four units AND the riders fit
                            // the tiers); every other K' is the b11 image. K: tiered image when the riders fit, else b11's.
                            // b13s2 (slim build): no K' image. The K'-slot layer (last sliding layer of a fused group, its QKV already done by K_{l-1}) runs the
                            // unfused chain: the N3 attention launch on the same tables (`sliding_ragged_s4_b16_kvd_blk`) + the native T=256 layer (bf16 or fold),
                            // bit-identical per token to K' (the S2_FUSED256=0 chain reproduces the fused hashes).
                            let _ = (w1, &n3in, &lw);
                            prof::timed!("t_attn_s1", launch(
                                ops_rpa::sliding_ragged_s4_b16_kvd_blk,
                                (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q_d, bidx, bmask, &*fk, &*fv, &kb, &vb),
                            ).await)?;
                            let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                            // all8b S2: an int8 layer in the K' slot runs the int8 native layer
                            let lw_i8 = if fp8::ffn_fold(index) || !i8w::native_on() { None } else { ffn.i8.as_ref().map(|q| ops_i8::LayerW8 {
                                w0: &ao.w[0], w1: &ao.w[1], w2: &ao.w[2], s0: &ao.s[0], s1: &ao.s[1], s2: &ao.s[2], post_attn_norm: &l.post_attention_norm, pre_w: q.pre_w.as_ref().unwrap_or(&ffn.pre_norm),
                                up_q: &q.up_q, up_ws: &q.up_ws, gate_q: &q.gate_q, gate_ws: &q.gate_ws,
                                down_w: &ffn.down_w, down_s: &ffn.down_s, dgs: &l.mlp.down_global_scale, post_ff: &l.post_feedforward_norm, had: &q.had,
                            }) };
                            if let Some(lw_i8) = lw_i8.as_ref() {
                                prof::timed!("t_layer_i8_s1", launch(ops_i8::layer_sliding_t256_i8, (&mut *device, &attn, lw_i8, &l.layer_scalar, &rin, &mut rout)).await)?;
                            } else if let (true, Some(f), Some(w8)) = (lw8.is_some(), ffn.fold.as_ref(), ffn.fold.as_ref().and_then(|f| f.w8.as_ref())) {
                                prof::timed!("t_layer8_s1", launch(
                                    ops_f8::layer_sliding_t256_f8pb,
                                    (
                                        &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                        &l.pre_feedforward_norm, &w8.up, &f.up_gs, &w8.gate, &f.gate_gs,
                                        &w8.down, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                    ),
                                ).await)?;
                            } else {
                                prof::timed!("t_layer_s1", launch(
                                    ops_fl::layer_sliding_t256r2,
                                    (
                                        &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                        &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                        &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                    ),
                                ).await)?;
                            }
                        }
                        qkv_done = fused_k;
                        cur = 1 - cur;
                        continue;
                    }
                    let (fk, fv) = flat.sliding[index].as_mut().expect("sliding flat cache");
                    if let (true, Some((win, kidx))) = (qkv2 && n3 && kvd256_enabled(), t.s.kvd.as_ref()) {
                        // KVD-256: QKV scatters K / V (prefill rows + decode rows) into the ring caches, then ONE attention launch (4 prefill units on the ring windows + decode attention)
                        if let (true, Some(xnq), Some(cos4), Some(sin4)) = (fused_mode() >= 1, self.xnq8.as_mut(), t.cos_s4.as_ref(), t.sin_s4.as_ref()) {
                            // modes 1 - 3: the quad QKV (one launch for the 256 rows), the same kernel the fused launches use
                            let x: HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]> = piece(res_in, 0)?;
                            let mut q: HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                            let qn = ops_fl::QkvNext {
                                input_norm: &l.input_norm, wq: &qkv.wq, sq: &qkv.sq, wk: &qkv.wk, sk: &qkv.sk, wv: &qkv.wv, sv: &qkv.sv, q_norm: &l.q_norm, k_norm: &l.k_norm,
                                cos: cos4, sin: sin4, kidx0: &kidx[0], kidx1: &kidx[1], kidx2: &kidx[2], kidx3: &kidx[3],
                            };
                            prof::timed!("t_qkv_s", launch(
                                ops_fl::sliding_qkv_quad,
                                (&mut *device, &x, &qn, t.s.kidx4.as_ref().expect("kidx4 (built with the KVD tables)"), &mut *xnq, &mut q, &mut *fk, &mut *fv),
                            ).await)?;
                        } else if let (Some(xnq), Some(cos4), Some(sin4)) = (self.xnq.as_mut(), t.cos_s4.as_ref(), t.sin_s4.as_ref()) {
                            // R13 qkv256: ONE launch for the four 64-row blocks (every QKV weight tile decoded once for 256 tokens)
                            let x: HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]> = piece(res_in, 0)?;
                            let mut q: HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]> = piece(&self.q_s, 0)?;
                            prof::timed!("t_qkv_s", launch(
                                ops_qkvc::sliding_qkv_quad_direct,
                                (
                                    &mut *device, &x, &l.input_norm, &mut *xnq, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv, &l.q_norm, &l.k_norm,
                                    cos4, sin4, &mut q, &kidx[0], &kidx[1], &kidx[2], &kidx[3], &mut *fk, &mut *fv,
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
                            if let (true, Some((bidx, bmask))) = (racefree(), t.s.blk8.as_ref()) {
                                // race-free decode part: block-granular unscaled gather over the block view of the caches (no table-load / gather race)
                                let kb: CacheB = piece(&*fk, 0)?;
                                let vb: CacheB = piece(&*fv, 0)?;
                                prof::timed!("t_attn_s", launch(
                                    ops_rpa::sliding_ragged_s4_b8_kvd_blk,
                                    (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, bidx, bmask, &*fk, &*fv, &kb, &vb),
                                ).await)?;
                            } else {
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b8_kvd,
                                (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, idx, mask, &*fk, &*fv),
                            ).await)?;
                            }
                        } else {
                            let (_, idx, mask) = t.s.dec16.as_ref().expect("decode tables");
                            let q: HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]> = piece(&self.q_s, dall)?;
                            let mut o1: HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]> = piece(&self.out_s, 1)?;
                            if let (true, Some((bidx, bmask))) = (racefree(), t.s.blk16.as_ref()) {
                                let kb: CacheB = piece(&*fk, 0)?;
                                let vb: CacheB = piece(&*fv, 0)?;
                                prof::timed!("t_attn_s", launch(
                                    ops_rpa::sliding_ragged_s4_b16_kvd_blk,
                                    (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, bidx, bmask, &*fk, &*fv, &kb, &vb),
                                ).await)?;
                            } else {
                            prof::timed!("t_attn_s", launch(
                                ops_rpa::sliding_ragged_s4_b16_kvd,
                                (&mut *device, &q0, &q1, &win[0], &win[1], &t.s.mask[0], &t.s.pen[0], &t.s.mask[1], &t.s.pen[1], &mut o0, &mut o1, &q, idx, mask, &*fk, &*fv),
                            ).await)?;
                            }
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
                    if let (true, true, Some(f), Some(w8)) = (native256(), fp8::ffn_fold(index), ffn.fold.as_ref(), ffn.fold.as_ref().and_then(|f| f.w8.as_ref())) {
                        // NATIVE T = 256: ONE launch for the 256 rows, every weight tile decoded once (`ops_f8::layer_sliding_t256_f8fb`, bit-identical per token to the two T = 128 launches)
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        prof::timed!("t_layer_s", sleepy_with(Slot::LayerS, launch(
                            ops_f8::layer_sliding_t256_f8pb,
                            (
                                &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                &l.pre_feedforward_norm, &w8.up, &f.up_gs, &w8.gate, &f.gate_gs,
                                &w8.down, &f.down_gs, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                            ),
                        ), &mut tick).await)?;
                    } else if let (true, false, true, Some(q)) = (native256bf(), fp8::ffn_fold(index), i8w::native_on(), ffn.i8.as_ref()) {
                        // all8b S2: unfused pass of an int8 layer -> the int8 native layer (int8 in every hybrid pass shape)
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        let lw_i8 = ops_i8::LayerW8 {
                                w0: &ao.w[0], w1: &ao.w[1], w2: &ao.w[2], s0: &ao.s[0], s1: &ao.s[1], s2: &ao.s[2], post_attn_norm: &l.post_attention_norm, pre_w: q.pre_w.as_ref().unwrap_or(&ffn.pre_norm),
                                up_q: &q.up_q, up_ws: &q.up_ws, gate_q: &q.gate_q, gate_ws: &q.gate_ws,
                                down_w: &ffn.down_w, down_s: &ffn.down_s, dgs: &l.mlp.down_global_scale, post_ff: &l.post_feedforward_norm, had: &q.had,
                            };
                        prof::timed!("t_layer_i8_s", sleepy_with(Slot::LayerS, launch(ops_i8::layer_sliding_t256_i8, (&mut *device, &attn, &lw_i8, &l.layer_scalar, &rin, &mut rout)), &mut tick).await)?;
                    } else if native256bf() && !fp8::ffn_fold(index) {
                        // NATIVE T = 256, bf16 FFN: ONE launch for the 256 rows, every up / gate / down weight tile decoded and loaded into the TRF once (`ops_ffnc::layer_sliding_t256p`,
                        // the same per-token arithmetic as the two `layer_sliding_t128` launches)
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&self.out_s, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        macro_rules! layer_s256 {
                            ($entry:path) => {
                                prof::timed!("t_layer_s", sleepy_with(Slot::LayerS, launch(
                                    $entry,
                                    (
                                        &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                        &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                        &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                    ),
                                ), &mut tick).await)?
                            };
                        }
                        layer_s256!(ops_fl::layer_sliding_t256r2);
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
                    if let (Some(xnq), Some(cos4), Some(sin4)) = (self.xnq8.as_mut(), t.cos_f4.as_ref(), t.sin_f4.as_ref()) {
                        // R13 qkv256: ONE launch for the four 64-row blocks of a full layer
                        let x: HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]> = piece(res_in, 0)?;
                        let mut q: HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]> = piece(&self.q_f, 0)?;
                        let mut k: HbmTensor<bf16, Chip, m![Ub, Tc, Df]> = piece(&self.k_f, 0)?;
                        let mut v: HbmTensor<bf16, Chip, m![Ub, Tc, Df]> = piece(&self.v_f, 0)?;
                        prof::timed!("t_qkv_f", launch(
                            ops_qkvc::full_qkv_quad,
                            (
                                &mut *device, &x, &l.input_norm, &mut *xnq, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                                &l.q_norm, &l.k_norm, cos4, sin4, &mut q, &mut k, &mut v,
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
                            FullPrefill::P4 { .. } => unreachable!("4-page prefill always uses the hg4 layout (P4h)"),
                            FullPrefill::P4h { index: idx, mask, pen } => {
                                prof::timed!("t_attn_f", launch(
                                    ops_rpa_full::full_ragged_prefill_s4_p4h,
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
                    if k03 && index + 1 == weights.layers.len() {
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
                    } else if native256bf() {
                        // NATIVE T = 256, bf16 FFN (`ops_ffnc::layer_full_t256p`): one launch, every weight tile decoded once for both 128-token halves
                        let attn: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, E256]> = piece(&self.out_f, 0)?;
                        let rin: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(res_in, 0)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]> = piece(&*res_out, 0)?;
                        macro_rules! layer_f256 {
                            ($entry:path) => {
                                prof::timed!("t_layer_f", sleepy_with(Slot::LayerF, launch(
                                    $entry,
                                    (
                                        &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &rin,
                                        &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                                        &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut rout,
                                    ),
                                ), &mut tick).await)?
                            };
                        }
                        // all8b v3: S3 (int8 full layer, ops_i8::layer_full_t256_i8) NOT dispatched: grader build budget (its image is not compiled when unreferenced)
                        layer_f256!(ops_fl::layer_full_t256r2);
                    } else {
                    for h in 0..2 {
                        let attn: HbmTensor<bf16, Chip, m![Tk128, Kc, E256]> = piece(&self.out_f, h)?;
                        let rin: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(res_in, h)?;
                        let mut rout: HbmTensor<bf16, Chip, m![Tk128, Kc, Ke]> = piece(&*res_out, h)?;
                        // b13s (slim build): full-attention layers never fold (the full-layer fold kernel was removed)
                        {
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
        // R14 d13e P0: the finishing rows go into the free lanes of the bucket (on the device), so riders + finishing rows are ONE res2 launch (no row D2H, no host group)
        if on_device && bd == 16 && !tail_only && res2_enabled() && d13e::level() >= 1 {
            let fin: Vec<(usize, usize)> = want.iter().copied().filter(|&(row, _)| row < ROWS - bd).collect();
            if let (true, Some((sidx, didx))) = (decs.len() + fin.len() <= d13e::LANES, t.d13e_copy.as_ref()) {
                let mut bucket: HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]> = piece(&self.res[cur], 15)?;
                prof::timed!("d13e_copy", launch(crate::ops_d13e::d13e_rowcopy8, (&mut *device, &self.res[cur], sidx, didx, &mut bucket)).await)?;
                let views: Vec<HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>> = vec![piece(&self.res[cur], 30)?, piece(&self.res[cur], 31)?];
                let slots: Vec<usize> = decs.iter().map(|dec| dec.slot).chain(fin.iter().map(|&(_, slot)| slot)).collect();
                for &slot in &slots {
                    hidden[slot].clear();
                }
                self.dev_rows = Some(DevRows { slots, views });
                prof::add("d13e_p0", std::time::Duration::ZERO);
                return Ok(());
            }
        }
        if on_device {
            // S2_HEAD_PACK (R14 s211host, from S2_211_tok): with a prompt finishing in a 16-rider pass the FIRST group of 8 riders stays on the device and the second group
            // (riders 8.., at most 7 when the pass has 15 riders) travels to the host together with the finishing rows: 15 riders + 1 finishing row = 8 device + 8 host rows = ONE
            // 16-row LM-head launch instead of two (7 device + 9 host). Same rows, same arithmetic, different grouping only.
            let first_group = bd == 16 && !tail_only && head_pack_enabled();
            let view: HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]> = piece(&self.res[cur], if first_group { 30 } else { 31 })?;
            // slots[b] = request slot of row b of the view (the rows 248..255 hold riders 8..15 when bd == 16, riders 0..7 when bd == 8; with `first_group` the view is rows 240..247 = riders 0..7)
            let first = if bd == 16 && !first_group { 8 } else { 0 };
            self.dev_rows = Some(DevRows { slots: decs.iter().skip(first).take(8).map(|dec| dec.slot).collect(), views: vec![view] });
            for dec in decs.iter().skip(first).take(8) {
                hidden[dec.slot].clear();
            }
            if bd == 16 {
                if first_group {
                    prof::add("head_pack_first", std::time::Duration::ZERO);
                }
                let rows: HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]> = piece(&self.res[cur], if first_group { 31 } else { 30 })?;
                let h: HostTensor<bf16, m![Rw8, Kc, Ke]> = prof::timed!("t_d2h", rows.to_host(&mut device.pdma).await)?;
                let host = h.into_inner().into_buf();
                for (b, dec) in decs.iter().skip(if first_group { 8 } else { 0 }).take(8).enumerate() {
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
