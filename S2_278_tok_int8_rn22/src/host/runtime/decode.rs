//! The batched decode path: up to 8 decode rows (one per request) through all 48 layers in ONE pass per layer with the T = 8 kernels
//! (qkvc `*_qkv_t8`, ffnc `layer_*_t8`) and the batched attention of the dattn tree over FLAT KV caches.
//!
//! Flat caches (one pair per layer): sliding `[Rf, Ns, Ds]` rows (request slot `r` owns rows `r * 1024 ..`, position `p` at row `r * 1024 + p % 1024`,
//! slot `Rn` is scratch) and full-layer page pools `[Pr, Df]` (physical pages of `Blocks::flat`, page `Pn` is scratch). A request prefills in the per-slot
//! caches of the chunk path (the chunk attention kernels read those) and is MIGRATED into the flat caches when its prompt is done (`Decode::migrate`:
//! `ops_mig` gather / scatter copies, ~3 ms for a 256-token prompt, ~12 ms for 2048); from then on it decodes in the flat caches only.
//!
//! Residual stream: `[8, 4096]` zero padded tensors, ping-pong between the layer kernel input and output, like the chunk path. Padding rows (fewer than 8
//! decode rows) repeat the last row for the QKV / layer kernels; their attention rows read the scratch slot / page (`dattn_host` padding rules).

use super::*;
use crate::axes::{Bd, Kc, Ke, Kx5, Pn, Pr, Rf, Rh, Rn, Tb, Tx};
use crate::device::ffnc::{E256, Tk16};
use crate::host::chunkw::ChunkLayer;
use crate::host::qkvc as qk;
use crate::host::views;
use crate::{dattn_host, ops_dattn, ops_ffnc, ops_qkvc, rpa_host};

type Res8 = HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>;
/// A 16-row residual as two 8-row halves.
type Res16 = HbmTensor<bf16, Chip, m![Rh, Tb, Kc, Ke]>;

/// Decode rows per pass (the T = 8 bucket).
pub(super) const ROWS: usize = 8;
/// Decode rows of the largest pass (two groups of 8).
pub(super) const ROWS16: usize = 16;
/// Positions the 5-page full-attention kernel covers; a request decodes in the flat caches only if prompt + new tokens fit.
pub const LIMIT: usize = 2560;
const WIDTH: usize = 4096;

// One 512 KiB / 1 MiB host transport buffer; these axes do not change device kernels.
axes![Dpk8 = 262144, Dpk16 = 524288];

fn packed_enabled(weights: &Weights) -> bool {
    // An odd debug-layer count leaves the packed input in the ping-pong output
    // slot. Use independent uploads there so the next pass cannot alias input/output.
    views::enabled()
        && weights.layers.len().min(super::debug_layers()) % 2 == 0
        && std::env::var("S2_DECODE_PACKED").map(|v| v != "0").unwrap_or(true)
}

/// Opt-in experiment: exact T8 arithmetic, one weight stream for two blocks.
fn qkv_pair_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| views::enabled() && std::env::var("S2_DECODE_QKV_PAIR").map(|v| v == "1").unwrap_or(true))
}

fn device_rows_enabled() -> bool {
    hybrid::n2_enabled() && std::env::var("S2_DECODE_N2").map(|v| v != "0").unwrap_or(true)
}

struct Upload<E: M> {
    bytes: Vec<u8>,
    shape: std::marker::PhantomData<E>,
}

impl<E: M> Upload<E> {
    fn new() -> Self { Self { bytes: Vec::new(), shape: std::marker::PhantomData } }

    fn put<D: Scalar, S: M>(&mut self, bytes: &[u8]) -> usize {
        assert_eq!(bytes.len(), HbmTensor::<D, Chip, S>::size());
        let offset = self.bytes.len().div_ceil(8192) * 8192;
        self.bytes.resize(offset, 0);
        self.bytes.extend_from_slice(bytes);
        offset
    }

    async fn upload(self, device: &mut Device, persistent: &mut Option<HbmTensor<bf16, Chip, E>>) -> Result<(), Error> {
        let mut bytes = self.bytes;
        assert!(bytes.len() <= HbmTensor::<bf16, Chip, E>::size(), "decode upload buffer too small");
        bytes.resize(HbmTensor::<bf16, Chip, E>::size(), 0);
        let host = HostTensor::<bf16, E>::from_buf(bytes);
        if let Some(big) = persistent.as_mut() {
            host.to_hbm(&mut device.pdma).output(big).await?;
        } else {
            *persistent = Some(host.to_hbm(&mut device.pdma).await?);
        }
        Ok(())
    }
}

/// Whether the batched decode path is available: <= `Rn` request slots, the chunk weights loaded, `S2_DECODE_BATCH` not 0.
pub(super) fn enabled(requests: usize, weights: &Weights) -> bool {
    requests <= Rn::SIZE
        && !weights.chunk.is_empty()
        && std::env::var("S2_DECODE_BATCH").map(|value| value != "0").unwrap_or(true)
}

/// Flat KV caches, indexed by layer (`Some` for the layers of the matching type).
pub(super) struct FlatCache {
    pub(super) sliding: Vec<Option<(HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>, HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>)>>,
    pub(super) full: Vec<Option<(HbmTensor<bf16, Chip, m![Pr, Df]>, HbmTensor<bf16, Chip, m![Pr, Df]>)>>,
}

impl FlatCache {
    pub(super) async fn new(device: &mut Device, weights: &Weights) -> Result<Self, Error> {
        let mut cache = Self { sliding: Vec::new(), full: Vec::new() };
        // one host zero buffer per shape, uploaded once per cache tensor
        let zero_s = HostTensor::<bf16, m![Rf, Ns, Ds]>::zero();
        let zero_f = HostTensor::<bf16, m![Pr, Df]>::zero();
        for layer in &weights.layers {
            match layer {
                Layer::Sliding(_) => {
                    cache.sliding.push(Some((zero_s.to_hbm(&mut device.pdma).await?, zero_s.to_hbm(&mut device.pdma).await?)));
                    cache.full.push(None);
                }
                Layer::Full(_) => {
                    cache.sliding.push(None);
                    cache.full.push(Some((zero_f.to_hbm(&mut device.pdma).await?, zero_f.to_hbm(&mut device.pdma).await?)));
                }
            }
        }
        Ok(cache)
    }
}

pub(super) struct DecodeEngine {
    pub(super) dev_rows: Option<hybrid::DevRows>,
    packed8: Option<HbmTensor<bf16, Chip, m![Dpk8]>>,
    packed16: Option<HbmTensor<bf16, Chip, m![Dpk16]>>,
    res2: Res8,
    xn: Res8,
    // Separate zero-initialized scratch prevents one block overwriting the other.
    xn_pair: Option<Res8>,
    q_s: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    k_s: HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    v_s: HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    attn_s: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    q_f: HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    k_f: HbmTensor<bf16, Chip, m![Tb, Df]>,
    v_f: HbmTensor<bf16, Chip, m![Tb, Df]>,
    attn_f: HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    // second group of a 16-row pass
    q_s2: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    k_s2: HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    v_s2: HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    attn_s2: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    q_f2: HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    k_f2: HbmTensor<bf16, Chip, m![Tb, Df]>,
    v_f2: HbmTensor<bf16, Chip, m![Tb, Df]>,
    attn_f2: HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    cat_s: HbmTensor<bf16, Chip, m![Rh, Tb, Ns, Gs, Ds]>,
    cat_f: HbmTensor<bf16, Chip, m![Rh, Tb, Gf, Df]>,
    res16_out: Res16,
    cos_s: Vec<u8>,
    sin_s: Vec<u8>,
    cos_f: Vec<u8>,
    sin_f: Vec<u8>,
}

impl DecodeEngine {
    pub(super) async fn new(device: &mut Device) -> Result<Self, Error> {
        let (cos_s, sin_s) = qk::rope_rows_bytes(Ds::SIZE, 10_000.0, Ds::SIZE, 0, LIMIT);
        let (cos_f, sin_f) = qk::rope_rows_bytes(Df::SIZE, 1_000_000.0, 128, 0, LIMIT);
        Ok(Self {
            dev_rows: None,
            packed8: None,
            packed16: None,
            res2: HostTensor::<bf16, m![Tb, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?,
            xn: HostTensor::<bf16, m![Tb, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?,
            xn_pair: if qkv_pair_enabled() { Some(HostTensor::<bf16, m![Tb, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?) } else { None },
            q_s: HostTensor::<bf16, m![Tb, Ns, Gs, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            k_s: HostTensor::<bf16, m![Tb, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            v_s: HostTensor::<bf16, m![Tb, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            attn_s: HostTensor::<bf16, m![Tb, Ns, Gs, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            q_f: HostTensor::<bf16, m![Tb, Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            k_f: HostTensor::<bf16, m![Tb, Df]>::zero().to_hbm(&mut device.pdma).await?,
            v_f: HostTensor::<bf16, m![Tb, Df]>::zero().to_hbm(&mut device.pdma).await?,
            attn_f: HostTensor::<bf16, m![Tb, Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            q_s2: HostTensor::<bf16, m![Tb, Ns, Gs, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            k_s2: HostTensor::<bf16, m![Tb, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            v_s2: HostTensor::<bf16, m![Tb, Ns, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            attn_s2: HostTensor::<bf16, m![Tb, Ns, Gs, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            q_f2: HostTensor::<bf16, m![Tb, Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            k_f2: HostTensor::<bf16, m![Tb, Df]>::zero().to_hbm(&mut device.pdma).await?,
            v_f2: HostTensor::<bf16, m![Tb, Df]>::zero().to_hbm(&mut device.pdma).await?,
            attn_f2: HostTensor::<bf16, m![Tb, Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            cat_s: HostTensor::<bf16, m![Rh, Tb, Ns, Gs, Ds]>::zero().to_hbm(&mut device.pdma).await?,
            cat_f: HostTensor::<bf16, m![Rh, Tb, Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            res16_out: HostTensor::<bf16, m![Rh, Tb, Kc, Ke]>::zero().to_hbm(&mut device.pdma).await?,
            cos_s,
            sin_s,
            cos_f,
            sin_f,
        })
    }

    /// One pass over `rows` (1..=8 decode rows, text tokens, distinct request slots, positions < `LIMIT`, `Blocks::flat` set). Returns the final hidden row
    /// (bf16 LE bytes) of every row, in order. With `keep_device` and `S2_DECODE_N2`,
    /// returns `None` and lends the final residual through `dev_rows` instead.
    pub(super) async fn run(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        rows: &[&DecodeRow<'_>],
        keep_device: bool,
    ) -> Result<Option<Vec<Vec<u8>>>, Error> {
        self.dev_rows = None;
        let k = rows.len();
        assert!((1..=ROWS).contains(&k));
        let slots: Vec<usize> = rows.iter().map(|row| row.blocks.sliding).collect();
        let positions: Vec<usize> = rows.iter().map(|row| row.pos).collect();
        let tables: Vec<Vec<usize>> = rows.iter().map(|row| row.blocks.flat.clone()).collect();
        let row_list: Vec<Row<'_>> = rows.iter().map(|row| row.row).collect();
        let padded_positions: Vec<usize> = (0..ROWS).map(|b| positions[b.min(k - 1)]).collect();

        let x = prof::timed!("d_embed", chunk::embed_rows(weights, &row_list, ROWS));
        let rope = |table: &[u8], dim: usize| -> Vec<u8> {
            let width = dim * 2;
            let mut out = Vec::with_capacity(ROWS * width);
            for &p in &padded_positions {
                out.extend_from_slice(&table[p * width..(p + 1) * width]);
            }
            out
        };
        let (mut res, cos_s, sin_s, cos_f, sin_f, s_gidx, s_mask, s_app, f_gidx, f_mask, f_app) = prof::timed!("d_h2d", async {
            if packed_enabled(weights) {
                let mut pk = Upload::<m![Dpk8]>::new();
                let res = pk.put::<bf16, m![Tb, Kc, Ke]>(&x);
                let cos_s = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.cos_s, Ds::SIZE));
                let sin_s = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.sin_s, Ds::SIZE));
                let cos_f = pk.put::<bf16, m![Tb, Df]>(&rope(&self.cos_f, Df::SIZE));
                let sin_f = pk.put::<bf16, m![Tb, Df]>(&rope(&self.sin_f, Df::SIZE));
                let s_gidx = pk.put::<i32, m![Bd, Tx]>(hybrid::i32_bytes(&rpa_host::decode_gather_index(&slots, &positions, ROWS)));
                let s_mask = pk.put::<bf16, m![Bd, Tx % 32, Tx / 32]>(hybrid::bf_bytes(&rpa_host::decode_mask(&positions, ROWS)));
                let s_app = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::decode_append_index(&slots, &positions, ROWS)));
                let f_gidx = pk.put::<i32, m![Bd, Kx5]>(hybrid::i32_bytes(&dattn_host::full_gather_index(&tables, ROWS, Kx5::SIZE)));
                let f_mask = pk.put::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>(hybrid::bf_bytes(&dattn_host::full_mask(&positions, ROWS, Kx5::SIZE, 64)));
                let f_app = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::full_decode_append_index(&tables, &positions, ROWS)));
                pk.upload(device, &mut self.packed8).await?;
                let big = self.packed8.as_ref().expect("uploaded decode tables");
                let res: HbmTensor<bf16, Chip, m![Tb, Kc, Ke]> = views::piece_at(big, res)?;
                let cos_s: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, cos_s)?;
                let sin_s: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, sin_s)?;
                let cos_f: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, cos_f)?;
                let sin_f: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, sin_f)?;
                let s_gidx: HbmTensor<i32, Chip, m![Bd, Tx]> = views::piece_at(big, s_gidx)?;
                let s_mask: HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]> = views::piece_at(big, s_mask)?;
                let s_app: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, s_app)?;
                let f_gidx: HbmTensor<i32, Chip, m![Bd, Kx5]> = views::piece_at(big, f_gidx)?;
                let f_mask: HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]> = views::piece_at(big, f_mask)?;
                let f_app: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, f_app)?;
                Ok::<_, Error>((res, cos_s, sin_s, cos_f, sin_f, s_gidx, s_mask, s_app, f_gidx, f_mask, f_app))
            } else {
                let res = HostTensor::<bf16, m![Tb, Kc, Ke]>::from_buf(x).to_hbm(&mut device.pdma).await?;
                let cos_s = HostTensor::<bf16, m![Tb, Ds]>::from_buf(rope(&self.cos_s, Ds::SIZE)).to_hbm(&mut device.pdma).await?;
                let sin_s = HostTensor::<bf16, m![Tb, Ds]>::from_buf(rope(&self.sin_s, Ds::SIZE)).to_hbm(&mut device.pdma).await?;
                let cos_f = HostTensor::<bf16, m![Tb, Df]>::from_buf(rope(&self.cos_f, Df::SIZE)).to_hbm(&mut device.pdma).await?;
                let sin_f = HostTensor::<bf16, m![Tb, Df]>::from_buf(rope(&self.sin_f, Df::SIZE)).to_hbm(&mut device.pdma).await?;
                let s_gidx = HostTensor::<i32, m![Bd, Tx]>::from_vec(rpa_host::decode_gather_index(&slots, &positions, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_mask = HostTensor::<bf16, m![Bd, Tx % 32, Tx / 32]>::from_vec(rpa_host::decode_mask(&positions, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_app = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::decode_append_index(&slots, &positions, ROWS)).to_hbm(&mut device.pdma).await?;
                let f_gidx = HostTensor::<i32, m![Bd, Kx5]>::from_vec(dattn_host::full_gather_index(&tables, ROWS, Kx5::SIZE)).to_hbm(&mut device.pdma).await?;
                let f_mask = HostTensor::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>::from_vec(dattn_host::full_mask(&positions, ROWS, Kx5::SIZE, 64)).to_hbm(&mut device.pdma).await?;
                let f_app = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::full_decode_append_index(&tables, &positions, ROWS)).to_hbm(&mut device.pdma).await?;
                Ok::<_, Error>((res, cos_s, sin_s, cos_f, sin_f, s_gidx, s_mask, s_app, f_gidx, f_mask, f_app))
            }
        }.await)?;

        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            let flat = ws.flat.as_mut().expect("flat caches");
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.sliding[index].as_mut().expect("sliding flat cache");
                    prof::timed!("d_qkv_s", launch(
                        ops_qkvc::sliding_qkv_t8,
                        (
                            device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                            &l.q_norm, &l.k_norm, &cos_s, &sin_s, &mut self.q_s, &mut self.k_s, &mut self.v_s,
                        ),
                    ).await)?;
                    prof::timed!("d_attn_s", launch(
                        ops_dattn::sliding_decode_attention_fused,
                        (device, &self.q_s, &self.k_s, &self.v_s, &s_app, &mut *fk, &mut *fv, &s_gidx, &s_mask, &mut self.attn_s),
                    ).await)?;
                    let attn = std::mem::replace(&mut self.attn_s, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tb, Kc, Ke]> = unsafe { attn.reshape() };
                    prof::timed!("d_layer_s", launch(
                        ops_ffnc::layer_sliding_t8,
                        (
                            device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut self.res2,
                        ),
                    ).await)?;
                    self.attn_s = unsafe { attn.reshape() };
                    std::mem::swap(&mut res, &mut self.res2);
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.full[index].as_mut().expect("full flat pool");
                    prof::timed!("d_qkv_f", launch(
                        ops_qkvc::full_qkv_t8,
                        (
                            device, &res, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                            &l.q_norm, &l.k_norm, &cos_f, &sin_f, &mut self.q_f, &mut self.k_f, &mut self.v_f,
                        ),
                    ).await)?;
                    prof::timed!("d_attn_f", launch(
                        ops_dattn::full_decode_attention_5_fused,
                        (device, &self.q_f, &self.k_f, &self.v_f, &f_app, &mut *fk, &mut *fv, &f_gidx, &f_mask, &mut self.attn_f),
                    ).await)?;
                    let attn = std::mem::replace(&mut self.attn_f, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tb, Kc, E256]> = unsafe { attn.reshape() };
                    prof::timed!("d_layer_f", launch(
                        ops_ffnc::layer_full_t8,
                        (
                            device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut self.res2,
                        ),
                    ).await)?;
                    self.attn_f = unsafe { attn.reshape() };
                    std::mem::swap(&mut res, &mut self.res2);
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
        }

        if keep_device && device_rows_enabled() {
            self.dev_rows = Some(hybrid::DevRows {
                slots: rows.iter().map(|row| row.blocks.sliding).collect(),
                views: vec![views::piece(&res, 0)?],
            });
            prof::add("d_n2", std::time::Duration::ZERO);
            return Ok(None);
        }
        let host: HostTensor<bf16, m![Tb, Kc, Ke]> = prof::timed!("d_d2h", res.to_host(&mut device.pdma).await)?;
        let bytes = host.into_inner().into_buf();
        Ok(Some((0..k).map(|b| bytes[b * WIDTH * 2..b * WIDTH * 2 + H::SIZE * 2].to_vec()).collect()))
    }

    /// A 16-row pass (9..=16 decode rows): the rows form two groups of 8 (group `b` has `k - 8` real rows). Per layer two 8-row QKV launches read the halves of ONE
    /// 16-row residual, the two-group batched attention appends and attends both groups in one launch, and ONE `layer_*_t16` launch (AO + FFN of all 16 rows)
    /// streams the FFN weights once instead of once per group.
    pub(super) async fn run16(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        weights: &Weights,
        rows: &[&DecodeRow<'_>],
        keep_device: bool,
    ) -> Result<Option<Vec<Vec<u8>>>, Error> {
        self.dev_rows = None;
        let k = rows.len();
        assert!(k > ROWS && k <= ROWS16);
        let (ga, gb) = rows.split_at(ROWS);
        let group = |g: &[&DecodeRow<'_>]| -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>) {
            (
                g.iter().map(|row| row.blocks.sliding).collect(),
                g.iter().map(|row| row.pos).collect(),
                g.iter().map(|row| row.blocks.flat.clone()).collect(),
            )
        };
        let ((slots_a, pos_a, tab_a), (slots_b, pos_b, tab_b)) = (group(ga), group(gb));
        let row_list: Vec<Row<'_>> = rows.iter().map(|row| row.row).collect();
        let padded: Vec<usize> = (0..ROWS16).map(|b| rows[b.min(k - 1)].pos).collect();

        let x = prof::timed!("d_embed", chunk::embed_rows(weights, &row_list, ROWS16));
        let rope = |table: &[u8], dim: usize, lo: usize| -> Vec<u8> {
            let width = dim * 2;
            let mut out = Vec::with_capacity(ROWS * width);
            for &p in &padded[lo..lo + ROWS] {
                out.extend_from_slice(&table[p * width..(p + 1) * width]);
            }
            out
        };
        let (mut res, cos_s, sin_s, cos_f, sin_f, inputs) = if packed_enabled(weights) {
            prof::timed!("d_h2d", async {
                let mut pk = Upload::<m![Dpk16]>::new();
                let res = pk.put::<bf16, m![Rh, Tb, Kc, Ke]>(&x);
                let cos_s_0 = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.cos_s, Ds::SIZE, 0));
                let cos_s_1 = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.cos_s, Ds::SIZE, 8));
                let sin_s_0 = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.sin_s, Ds::SIZE, 0));
                let sin_s_1 = pk.put::<bf16, m![Tb, Ds]>(&rope(&self.sin_s, Ds::SIZE, 8));
                let cos_f_0 = pk.put::<bf16, m![Tb, Df]>(&rope(&self.cos_f, Df::SIZE, 0));
                let cos_f_1 = pk.put::<bf16, m![Tb, Df]>(&rope(&self.cos_f, Df::SIZE, 8));
                let sin_f_0 = pk.put::<bf16, m![Tb, Df]>(&rope(&self.sin_f, Df::SIZE, 0));
                let sin_f_1 = pk.put::<bf16, m![Tb, Df]>(&rope(&self.sin_f, Df::SIZE, 8));
                let s_gidx_a = pk.put::<i32, m![Bd, Tx]>(hybrid::i32_bytes(&rpa_host::decode_gather_index(&slots_a, &pos_a, ROWS)));
                let s_gidx_b = pk.put::<i32, m![Bd, Tx]>(hybrid::i32_bytes(&rpa_host::decode_gather_index(&slots_b, &pos_b, ROWS)));
                let s_mask_a = pk.put::<bf16, m![Bd, Tx % 32, Tx / 32]>(hybrid::bf_bytes(&rpa_host::decode_mask(&pos_a, ROWS)));
                let s_mask_b = pk.put::<bf16, m![Bd, Tx % 32, Tx / 32]>(hybrid::bf_bytes(&rpa_host::decode_mask(&pos_b, ROWS)));
                let s_app_a = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::decode_append_index(&slots_a, &pos_a, ROWS)));
                let s_app_b = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::decode_append_index(&slots_b, &pos_b, ROWS)));
                let f_gidx_a = pk.put::<i32, m![Bd, Kx5]>(hybrid::i32_bytes(&dattn_host::full_gather_index(&tab_a, ROWS, Kx5::SIZE)));
                let f_gidx_b = pk.put::<i32, m![Bd, Kx5]>(hybrid::i32_bytes(&dattn_host::full_gather_index(&tab_b, ROWS, Kx5::SIZE)));
                let f_mask_a = pk.put::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>(hybrid::bf_bytes(&dattn_host::full_mask(&pos_a, ROWS, Kx5::SIZE, 64)));
                let f_mask_b = pk.put::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>(hybrid::bf_bytes(&dattn_host::full_mask(&pos_b, ROWS, Kx5::SIZE, 64)));
                let f_app_a = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::full_decode_append_index(&tab_a, &pos_a, ROWS)));
                let f_app_b = pk.put::<i32, m![Bd]>(hybrid::i32_bytes(&rpa_host::full_decode_append_index(&tab_b, &pos_b, ROWS)));
                pk.upload(device, &mut self.packed16).await?;
                let big = self.packed16.as_ref().expect("uploaded decode tables");
                let res: HbmTensor<bf16, Chip, m![Rh, Tb, Kc, Ke]> = views::piece_at(big, res)?;
                let cos_s_0: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, cos_s_0)?;
                let cos_s_1: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, cos_s_1)?;
                let sin_s_0: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, sin_s_0)?;
                let sin_s_1: HbmTensor<bf16, Chip, m![Tb, Ds]> = views::piece_at(big, sin_s_1)?;
                let cos_f_0: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, cos_f_0)?;
                let cos_f_1: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, cos_f_1)?;
                let sin_f_0: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, sin_f_0)?;
                let sin_f_1: HbmTensor<bf16, Chip, m![Tb, Df]> = views::piece_at(big, sin_f_1)?;
                let s_gidx_a: HbmTensor<i32, Chip, m![Bd, Tx]> = views::piece_at(big, s_gidx_a)?;
                let s_gidx_b: HbmTensor<i32, Chip, m![Bd, Tx]> = views::piece_at(big, s_gidx_b)?;
                let s_mask_a: HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]> = views::piece_at(big, s_mask_a)?;
                let s_mask_b: HbmTensor<bf16, Chip, m![Bd, Tx % 32, Tx / 32]> = views::piece_at(big, s_mask_b)?;
                let s_app_a: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, s_app_a)?;
                let s_app_b: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, s_app_b)?;
                let f_gidx_a: HbmTensor<i32, Chip, m![Bd, Kx5]> = views::piece_at(big, f_gidx_a)?;
                let f_gidx_b: HbmTensor<i32, Chip, m![Bd, Kx5]> = views::piece_at(big, f_gidx_b)?;
                let f_mask_a: HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]> = views::piece_at(big, f_mask_a)?;
                let f_mask_b: HbmTensor<bf16, Chip, m![Bd, Kx5 % 64, Kx5 / 64]> = views::piece_at(big, f_mask_b)?;
                let f_app_a: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, f_app_a)?;
                let f_app_b: HbmTensor<i32, Chip, m![Bd]> = views::piece_at(big, f_app_b)?;
                Ok::<_, Error>((res, [cos_s_0, cos_s_1], [sin_s_0, sin_s_1], [cos_f_0, cos_f_1], [sin_f_0, sin_f_1], (s_gidx_a, s_gidx_b, s_mask_a, s_mask_b, s_app_a, s_app_b, f_gidx_a, f_gidx_b, f_mask_a, f_mask_b, f_app_a, f_app_b)))
            }.await)?
        } else {
            let res: Res16 = prof::timed!("d_h2d", HostTensor::<bf16, m![Rh, Tb, Kc, Ke]>::from_buf(x).to_hbm(&mut device.pdma).await)?;
            let (cos_s, sin_s, cos_f, sin_f) = prof::timed!("d_h2d", async {
                let mut out = Vec::new();
                for lo in [0, ROWS] {
                    out.push((
                        HostTensor::<bf16, m![Tb, Ds]>::from_buf(rope(&self.cos_s, Ds::SIZE, lo)).to_hbm(&mut device.pdma).await?,
                        HostTensor::<bf16, m![Tb, Ds]>::from_buf(rope(&self.sin_s, Ds::SIZE, lo)).to_hbm(&mut device.pdma).await?,
                        HostTensor::<bf16, m![Tb, Df]>::from_buf(rope(&self.cos_f, Df::SIZE, lo)).to_hbm(&mut device.pdma).await?,
                        HostTensor::<bf16, m![Tb, Df]>::from_buf(rope(&self.sin_f, Df::SIZE, lo)).to_hbm(&mut device.pdma).await?,
                    ));
                }
                let mut it = out.into_iter();
                let (a0, a1, a2, a3) = it.next().unwrap();
                let (b0, b1, b2, b3) = it.next().unwrap();
                Ok::<_, Error>(([a0, b0], [a1, b1], [a2, b2], [a3, b3]))
            }
            .await)?;
            let inputs = prof::timed!("d_h2d", async {
                let s_gidx_a = HostTensor::<i32, m![Bd, Tx]>::from_vec(rpa_host::decode_gather_index(&slots_a, &pos_a, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_gidx_b = HostTensor::<i32, m![Bd, Tx]>::from_vec(rpa_host::decode_gather_index(&slots_b, &pos_b, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_mask_a = HostTensor::<bf16, m![Bd, Tx % 32, Tx / 32]>::from_vec(rpa_host::decode_mask(&pos_a, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_mask_b = HostTensor::<bf16, m![Bd, Tx % 32, Tx / 32]>::from_vec(rpa_host::decode_mask(&pos_b, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_app_a = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::decode_append_index(&slots_a, &pos_a, ROWS)).to_hbm(&mut device.pdma).await?;
                let s_app_b = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::decode_append_index(&slots_b, &pos_b, ROWS)).to_hbm(&mut device.pdma).await?;
                let f_gidx_a = HostTensor::<i32, m![Bd, Kx5]>::from_vec(dattn_host::full_gather_index(&tab_a, ROWS, Kx5::SIZE)).to_hbm(&mut device.pdma).await?;
                let f_gidx_b = HostTensor::<i32, m![Bd, Kx5]>::from_vec(dattn_host::full_gather_index(&tab_b, ROWS, Kx5::SIZE)).to_hbm(&mut device.pdma).await?;
                let f_mask_a = HostTensor::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>::from_vec(dattn_host::full_mask(&pos_a, ROWS, Kx5::SIZE, 64)).to_hbm(&mut device.pdma).await?;
                let f_mask_b = HostTensor::<bf16, m![Bd, Kx5 % 64, Kx5 / 64]>::from_vec(dattn_host::full_mask(&pos_b, ROWS, Kx5::SIZE, 64)).to_hbm(&mut device.pdma).await?;
                let f_app_a = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::full_decode_append_index(&tab_a, &pos_a, ROWS)).to_hbm(&mut device.pdma).await?;
                let f_app_b = HostTensor::<i32, m![Bd]>::from_vec(rpa_host::full_decode_append_index(&tab_b, &pos_b, ROWS)).to_hbm(&mut device.pdma).await?;
                Ok::<_, Error>((s_gidx_a, s_gidx_b, s_mask_a, s_mask_b, s_app_a, s_app_b, f_gidx_a, f_gidx_b, f_mask_a, f_mask_b, f_app_a, f_app_b))
            }
            .await)?;

            (res, cos_s, sin_s, cos_f, sin_f, inputs)
        };
        let (s_gidx_a, s_gidx_b, s_mask_a, s_mask_b, s_app_a, s_app_b, f_gidx_a, f_gidx_b, f_mask_a, f_mask_b, f_app_a, f_app_b) = inputs;

        for (index, (layer, chunk)) in weights.layers.iter().zip(&weights.chunk).enumerate().take(super::debug_layers()) {
            let flat = ws.flat.as_mut().expect("flat caches");
            match (layer, chunk) {
                (Layer::Sliding(l), ChunkLayer::Sliding { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.sliding[index].as_mut().expect("sliding flat cache");
                    // views (host/views.rs): the plain T = 8 QKV kernel runs on each half of the residual (one image, no `_h0` / `_h1`), and the attention writes straight into
                    // the halves of the concatenated AO input (no cat launch)
                    let res_a: Res8 = views::piece(&res, 0)?;
                    let res_b: Res8 = views::piece(&res, 1)?;
                    if let Some(xn1) = self.xn_pair.as_mut() {
                        prof::timed!("d_qkv_pair_s", launch(
                            ops_qkvc::sliding_qkv_pair_t8,
                            (
                                &mut *device, &res_a, &res_b, &l.input_norm, &mut self.xn, xn1,
                                &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                                &l.q_norm, &l.k_norm, &cos_s[0], &sin_s[0], &cos_s[1], &sin_s[1],
                                &mut self.q_s, &mut self.k_s, &mut self.v_s, &mut self.q_s2, &mut self.k_s2, &mut self.v_s2,
                            ),
                        ).await)?;
                    } else {
                    prof::timed!("d_qkv_s", launch(
                        ops_qkvc::sliding_qkv_t8,
                        (
                            &mut *device, &res_a, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                            &l.q_norm, &l.k_norm, &cos_s[0], &sin_s[0], &mut self.q_s, &mut self.k_s, &mut self.v_s,
                        ),
                    ).await)?;
                    prof::timed!("d_qkv_s", launch(
                        ops_qkvc::sliding_qkv_t8,
                        (
                            &mut *device, &res_b, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk, &qkv.wv, &qkv.sv,
                            &l.q_norm, &l.k_norm, &cos_s[1], &sin_s[1], &mut self.q_s2, &mut self.k_s2, &mut self.v_s2,
                        ),
                    ).await)?;
                    }
                    let mut attn_a: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]> = views::piece(&self.cat_s, 0)?;
                    let mut attn_b: HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]> = views::piece(&self.cat_s, 1)?;
                    prof::timed!("d_attn_s", launch(
                        ops_dattn::sliding_decode_attention_fused_x2,
                        (
                            &mut *device, &self.q_s, &self.q_s2, &self.k_s, &self.v_s, &self.k_s2, &self.v_s2, &s_app_a, &s_app_b, &mut *fk, &mut *fv,
                            &s_gidx_a, &s_gidx_b, &s_mask_a, &s_mask_b, &mut attn_a, &mut attn_b,
                        ),
                    ).await)?;
                    let attn = std::mem::replace(&mut self.cat_s, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = unsafe { attn.reshape() };
                    let res_in: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = unsafe { std::mem::replace(&mut res, HbmTensor::new()).reshape() };
                    let mut res_out: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = unsafe { std::mem::replace(&mut self.res16_out, HbmTensor::new()).reshape() };
                    prof::timed!("d_layer_s", launch(
                        ops_ffnc::layer_sliding_t16,
                        (
                            &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res_in,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut res_out,
                        ),
                    ).await)?;
                    self.cat_s = unsafe { attn.reshape() };
                    res = unsafe { res_out.reshape() };
                    self.res16_out = unsafe { res_in.reshape() };
                }
                (Layer::Full(l), ChunkLayer::Full { qkv, ao, ffn }) => {
                    let (fk, fv) = flat.full[index].as_mut().expect("full flat pool");
                    let res_a: Res8 = views::piece(&res, 0)?;
                    let res_b: Res8 = views::piece(&res, 1)?;
                    prof::timed!("d_qkv_f", launch(
                        ops_qkvc::full_qkv_t8,
                        (
                            &mut *device, &res_a, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                            &l.q_norm, &l.k_norm, &cos_f[0], &sin_f[0], &mut self.q_f, &mut self.k_f, &mut self.v_f,
                        ),
                    ).await)?;
                    prof::timed!("d_qkv_f", launch(
                        ops_qkvc::full_qkv_t8,
                        (
                            &mut *device, &res_b, &l.input_norm, &mut self.xn, &qkv.wq, &qkv.sq, &qkv.wk, &qkv.sk,
                            &l.q_norm, &l.k_norm, &cos_f[1], &sin_f[1], &mut self.q_f2, &mut self.k_f2, &mut self.v_f2,
                        ),
                    ).await)?;
                    let mut attn_a: HbmTensor<bf16, Chip, m![Tb, Gf, Df]> = views::piece(&self.cat_f, 0)?;
                    let mut attn_b: HbmTensor<bf16, Chip, m![Tb, Gf, Df]> = views::piece(&self.cat_f, 1)?;
                    prof::timed!("d_attn_f", launch(
                        ops_dattn::full_decode_attention_5_fused_x2,
                        (
                            &mut *device, &self.q_f, &self.q_f2, &self.k_f, &self.v_f, &self.k_f2, &self.v_f2, &f_app_a, &f_app_b, &mut *fk, &mut *fv,
                            &f_gidx_a, &f_gidx_b, &f_mask_a, &f_mask_b, &mut attn_a, &mut attn_b,
                        ),
                    ).await)?;
                    let attn = std::mem::replace(&mut self.cat_f, HbmTensor::new());
                    let attn: HbmTensor<bf16, Chip, m![Tk16, Kc, E256]> = unsafe { attn.reshape() };
                    let res_in: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = unsafe { std::mem::replace(&mut res, HbmTensor::new()).reshape() };
                    let mut res_out: HbmTensor<bf16, Chip, m![Tk16, Kc, Ke]> = unsafe { std::mem::replace(&mut self.res16_out, HbmTensor::new()).reshape() };
                    prof::timed!("d_layer_f", launch(
                        ops_ffnc::layer_full_t16,
                        (
                            &mut *device, &attn, &ao.w[0], &ao.w[1], &ao.w[2], &ao.s[0], &ao.s[1], &ao.s[2], &l.post_attention_norm, &res_in,
                            &ffn.pre_norm, &ffn.up_w, &ffn.up_s, &l.mlp.up_global_scale, &ffn.gate_w, &ffn.gate_s, &l.mlp.gate_global_scale,
                            &ffn.down_w, &ffn.down_s, &l.mlp.down_global_scale, &l.post_feedforward_norm, &l.layer_scalar, &mut res_out,
                        ),
                    ).await)?;
                    self.cat_f = unsafe { attn.reshape() };
                    res = unsafe { res_out.reshape() };
                    self.res16_out = unsafe { res_in.reshape() };
                }
                _ => unreachable!("chunk weights follow the layer pattern"),
            }
        }

        if keep_device && device_rows_enabled() {
            self.dev_rows = Some(hybrid::DevRows {
                slots: rows.iter().map(|row| row.blocks.sliding).collect(),
                views: vec![views::piece(&res, 0)?, views::piece(&res, 1)?],
            });
            prof::add("d_n2", std::time::Duration::ZERO);
            return Ok(None);
        }
        let host: HostTensor<bf16, m![Rh, Tb, Kc, Ke]> = prof::timed!("d_d2h", res.to_host(&mut device.pdma).await)?;
        let bytes = host.into_inner().into_buf();
        Ok(Some((0..k).map(|b| bytes[b * WIDTH * 2..b * WIDTH * 2 + H::SIZE * 2].to_vec()).collect()))
    }
}
