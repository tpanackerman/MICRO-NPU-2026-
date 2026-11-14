//! The model as the scheduler sees it: chunked prefill of one request, one decode step for many
//! requests, and the choice of each next token. Every device-facing body lives in
//! `runtime.rs` (`Decode::run_chunk`, `Decode::run_decode_batch`, `encode_image`, `encode_audio`)
//! or behind `Model::next_tokens`, so kernel work replaces only those bodies.

use super::{
    audio::AudioFrames,
    generate::{Next, SoftToken, read_greedy_all, read_logits},
    image::ImagePatches,
    load, runtime,
};
use crate::axes::{H, Rw8, Tf, W};
use crate::Chip;
use furiosa_opt_std::prelude::*;

use crate::host::lmhead::ArgmaxHead;
use runtime::{Blocks, DecodeRow, HybDec, HybRun, Hidden, Row, Workspace};
pub use runtime::{Logits, SoftEmbeddings};

type BoxError = Box<dyn std::error::Error>;

/// `GEMMA4_MMENC=old` runs the per-patch / per-frame encoder kernels instead of the batched ones.
fn batched_encoders() -> bool {
    static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OLD.get_or_init(|| std::env::var("GEMMA4_MMENC").map(|v| v == "old").unwrap_or(false))
}

/// One request's next row. `logits` is false for prompt tokens whose output is discarded.
pub struct Input<'a> {
    pub token: usize,
    pub soft: Option<SoftToken>,
    pub logits: bool,
    /// The request's precomputed soft-token embeddings (required when `soft` is set).
    pub embeddings: Option<&'a SoftEmbeddings>,
    /// Identifies the request (generation) this row belongs to; unique, never reused across requests.
    pub request: u64,
}

impl<'a> Input<'a> {
    fn row(&self) -> Result<Row<'a>, BoxError> {
        let soft = match (self.soft, self.embeddings) {
            (Some(token), Some(embeddings)) if embeddings.contains(token) => Some((embeddings, token)),
            (Some(_), _) => return Err("soft token without precomputed embeddings".into()),
            (None, _) => None,
        };
        Ok(Row { token: self.token, soft })
    }
}

/// One prefill run of a hybrid pass: `inputs` are consecutive prompt rows at positions `pos0 ..` of the request that owns `blocks` (flat KV pages set).
pub struct HybPrefill<'a> {
    pub blocks: &'a Blocks,
    pub pos0: usize,
    pub inputs: Vec<Input<'a>>,
    /// The run ends the prompt: leave its last row's final hidden row in the request slot for `next_tokens`.
    pub want_next: bool,
}

/// One decode row of a hybrid pass (`Input::logits` selects whether its next token is needed).
pub struct HybDecode<'a> {
    pub blocks: &'a Blocks,
    pub input: Input<'a>,
    pub pos: usize,
}

/// Which next token the model must produce for one request slot.
#[derive(Clone, Copy, Debug)]
pub struct SampleRequest {
    /// The request's slot (`Blocks::sliding`): its logits are `Model::logits(slot)`.
    pub slot: usize,
    /// `true`: the greedy argmax, returned as `Next::Greedy` (the token under every ban variant; the request keeps its own).
    /// `false`: the full logits as f32 for host sampling, returned as `Next::Logits`.
    pub greedy: bool,
}

/// The device picks greedy tokens (final norm + LM head + argmax in one launch, 32 KB back) unless `GEMMA4_ARGMAX=0`
/// (host argmax over the logits, the pre-lmhead path).
fn device_argmax() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("GEMMA4_ARGMAX").map(|value| value != "0").unwrap_or(true))
}

pub struct Model {
    weights: load::Weights,
    decode: runtime::Decode,
    /// One final-hidden-row buffer per request slot (`Blocks::sliding`): what the last row of a chunk / decode step leaves for `next_tokens`.
    hidden: Vec<Hidden>,
    /// One logits buffer per request slot; only filled for sampled requests (and by `forward`, which indexes them by input row).
    logits: Vec<Logits>,
    argmax: ArgmaxHead,
    /// Prediction of the next hybrid pass (`set_hybrid_hint`): its tables are built on a thread while the current pass runs.
    hybrid_hint: Option<runtime::PassDesc>,
    /// The prefill rows of the predicted pass (`set_hybrid_hint_rows`): the 256-row engine pre-uploads the pass while the current one runs.
    hybrid_hint_rows: Option<Vec<runtime::HintRun>>,
    /// R14 d13e: the feed plan of the transition into the predicted pass (`set_hybrid_hint_feed`).
    hybrid_hint_feed: Option<runtime::d13e::FeedPlan>,
    /// R14 d13e P1 device tensors (`S2_D13E=2`; `d13e_init`).
    d13e: Option<runtime::d13e::FeedDev>,
}

impl Model {
    pub async fn new(
        device: &mut Device,
        weights: load::Weights,
        workspace: &mut Workspace,
    ) -> Result<Self, BoxError> {
        let mut logits = Vec::with_capacity(workspace.requests());
        let mut hidden = Vec::with_capacity(workspace.requests());
        for _ in 0..workspace.requests() {
            logits.push(HostTensor::<bf16, m![W]>::zero().to_hbm(&mut device.pdma).await?);
            hidden.push(Vec::new());
        }
        Ok(Self {
            weights,
            decode: workspace.begin_decode(device).await?,
            hidden,
            logits,
            argmax: ArgmaxHead::new(device).await?,
            hybrid_hint: None,
            hybrid_hint_rows: None,
            hybrid_hint_feed: None,
            d13e: None,
        })
    }

    /// R14 d13e: builds the P1 device tensors (2 GiB row table) when `S2_D13E=2`. Call after the model and the workspace exist; a failure only disables P1.
    pub async fn d13e_init(&mut self, device: &mut Device) {
        if runtime::d13e::level() < 2 || self.d13e.is_some() {
            return;
        }
        match runtime::d13e::FeedDev::new(device, &self.weights.embed_host).await {
            Ok(fd) => self.d13e = Some(fd),
            Err(error) => eprintln!("S2 d13e disabled: {error}"),
        }
    }

    /// R14 d13e P2: the boundary of a 256-row pass WITHOUT the token readback: LM head (res2 views) -> device reduce -> feed of the pre-uploaded next pass; the tokens are read
    /// during the next pass (`hybrid_step_ext` callback or `d13e_late_take`). Returns the lane slots (lane j = the request slot whose token is lane j), or `None` (nothing done:
    /// use `next_tokens`) when the last pass did not leave both res2 views, the next pass is not device-fed, or `wanted` holds a slot outside the lanes.
    pub async fn next_tokens_blind(&mut self, device: &mut Device, wanted: &[SampleRequest]) -> Result<Option<Vec<usize>>, BoxError> {
        if runtime::d13e::level() < 3 {
            return Ok(None);
        }
        let Some(slots) = self.decode.d13e_dev_slots() else { return Ok(None) };
        if self.d13e.is_none() || !self.decode.d13e_feed_ready(&slots) || !wanted.iter().all(|r| r.greedy && slots.contains(&r.slot)) {
            return Ok(None);
        }
        let dev = self.decode.take_hybrid_dev_rows().expect("checked above");
        let fd = self.d13e.as_ref().expect("checked above");
        self.argmax.launch_res2(device, &dev.views[0], &dev.views[1], &self.weights.final_norm, &self.weights.embedding_table).await?;
        self.decode.d13e_boundary(device, fd, self.argmax.keys16(), true).await?;
        Ok(Some(slots))
    }

    /// R14 d13e P2: see `runtime::Decode::d13e_blind_ready`.
    pub fn d13e_blind_ready(&self, desc: &runtime::PassDesc, requests: &[u64]) -> bool {
        self.decode.d13e_blind_ready(desc, requests)
    }

    /// R14 d13e P2: the lane tokens of the last blind boundary if the callback did not consume them (waits for the read).
    pub async fn d13e_late_take(&mut self, device: &mut Device) -> Result<Option<Vec<usize>>, BoxError> {
        Ok(self.decode.d13e_late_take(device).await?)
    }

    /// Whether P1 can run (its device tensors exist).
    pub fn d13e_feed_on(&self) -> bool {
        self.d13e.is_some()
    }

    /// The feed plan of the transition into the predicted pass (with `set_hybrid_hint`).
    pub fn set_hybrid_hint_feed(&mut self, feed: Option<runtime::d13e::FeedPlan>) {
        self.hybrid_hint_feed = feed;
    }

    /// Compute every soft-token embedding of a request once (admission time): the batched encoders
    /// (`runtime::encode_image` / `encode_audio`, tiles of 64 rows) or, with `GEMMA4_MMENC=old`, the per-patch kernels.
    pub async fn embed_media(
        &mut self,
        device: &mut Device,
        image: Option<&ImagePatches>,
        audio: Option<&AudioFrames>,
    ) -> Result<SoftEmbeddings, BoxError> {
        use crate::host::prof::timed;
        // R15 LQ: the encoders launch synchronously; start them on an empty launch queue
        runtime::lq::drain()?;
        if batched_encoders() {
            let image = match image {
                Some(image) => Some(timed!("encode_image", runtime::encode_image(device, &self.weights, image).await)?),
                None => None,
            };
            let audio = match audio {
                Some(audio) => Some(timed!("encode_audio", runtime::encode_audio(device, &self.weights, audio).await)?),
                None => None,
            };
            let mut embeddings = SoftEmbeddings::new(image, audio);
            if crate::host::chunkw::chunk_path_enabled() {
                timed!("fetch_soft_host", embeddings.fetch_host(device).await)?;
            }
            Ok(embeddings)
        } else {
            let image = match image {
                Some(_) => panic!("c64L2: GEMMA4_MMENC=old (per-patch image encoder) retired from the build"),
                None => None,
            };
            let audio = match audio {
                Some(_) => panic!("c64L2: GEMMA4_MMENC=old (per-frame audio encoder) retired from the build"),
                None => None,
            };
            let mut embeddings = SoftEmbeddings::from_rows(image, audio);
            if crate::host::chunkw::chunk_path_enabled() {
                timed!("fetch_soft_host", embeddings.fetch_host(device).await)?;
            }
            Ok(embeddings)
        }
    }

    fn check(&self, workspace: &Workspace, blocks: &Blocks, last_position: usize) -> Result<(), BoxError> {
        if last_position >= workspace.capacity()
            || blocks.full.len() * Tf::SIZE <= last_position
            || blocks.sliding >= workspace.requests()
        {
            return Err("invalid position or blocks".into());
        }
        Ok(())
    }

    /// PREFILL CHUNK: append `rows.len()` consecutive positions `pos0..` of ONE request.
    ///
    /// - The chunk must lie inside one full-attention page (`Tf` = 512 positions); the scheduler
    ///   aligns chunks to a multiple of its chunk size (a divisor of 512), so this also keeps a
    ///   chunk from wrapping the 1024 sliding ring.
    /// - `blocks.full` must cover the last position.
    /// - If `want_next`, the LAST row's final hidden row is left in `self.hidden[blocks.sliding]`;
    ///   `next_tokens` turns it into the next token. Other rows produce no output.
    pub async fn prefill_chunk(
        &mut self,
        device: &mut Device,
        rows: &[Input<'_>],
        pos0: usize,
        blocks: &Blocks,
        workspace: &mut Workspace,
        want_next: bool,
    ) -> Result<(), BoxError> {
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        self.check(workspace, blocks, pos0 + n - 1)?;
        if pos0 / Tf::SIZE != (pos0 + n - 1) / Tf::SIZE {
            return Err("a prefill chunk must not cross a full-attention page".into());
        }
        let rows: Vec<Row> = rows.iter().map(Input::row).collect::<Result<_, _>>()?;
        let hidden = want_next.then(|| &mut self.hidden[blocks.sliding]);
        self.decode
            .run_chunk(device, workspace, &self.weights, blocks, pos0, &rows, hidden)
            .await?;
        Ok(())
    }

    /// DECODE BATCH: one pending token per request, each at its own position with its own blocks.
    /// Rows whose `Input::logits` is set leave their final hidden row in `self.hidden[blocks.sliding]`.
    /// The request slots (`blocks.sliding`) of one batch must be distinct.
    pub async fn decode_batch(
        &mut self,
        device: &mut Device,
        inputs: &[Input<'_>],
        positions: &[usize],
        blocks: &[&Blocks],
        workspace: &mut Workspace,
    ) -> Result<(), BoxError> {
        if inputs.len() != positions.len() || inputs.len() != blocks.len() {
            return Err("incompatible batch dimensions".into());
        }
        let mut seen = vec![false; workspace.requests()];
        let mut rows = Vec::with_capacity(inputs.len());
        for ((input, &pos), &blocks) in inputs.iter().zip(positions).zip(blocks) {
            self.check(workspace, blocks, pos)?;
            if std::mem::replace(&mut seen[blocks.sliding], true) {
                return Err("two rows of a decode batch share a request slot".into());
            }
            rows.push(DecodeRow { row: input.row()?, pos, blocks, logits: input.logits });
        }
        self.decode
            .run_decode_batch(device, workspace, &self.weights, &rows, &mut self.hidden)
            .await?;
        Ok(())
    }

    /// HYBRID PASS: up to two prefill runs plus up to 16 decode rows in ONE 128-row pass over the weights (`runtime::hybrid_fits` tells whether a combination fits).
    /// Requests must own flat KV pages (`Blocks::flat`); their K / V are written straight into the flat caches. Rows with `want_next` / `Input::logits` leave their final
    /// hidden row in `self.hidden[blocks.sliding]` for `next_tokens`.
    pub fn set_hybrid_hint(&mut self, mut hint: Option<runtime::PassDesc>) {
        // R14 d13b: the decode rows of a pass are ordered by window blocks (descending) under S2_DECTIER, so the prediction must use the same order
        if crate::rpa_host::dectier() {
            if let Some(h) = hint.as_mut() {
                h.decs.sort_by_key(|d| std::cmp::Reverse(crate::rpa_host::blk_need(d.pos)));
            }
        }
        self.hybrid_hint = hint;
    }

    /// The prefill rows of the pass `set_hybrid_hint` predicted (same run order); used only to pre-upload that pass.
    pub fn set_hybrid_hint_rows(&mut self, rows: Option<Vec<runtime::HintRun>>) {
        self.hybrid_hint_rows = rows;
    }

    pub async fn hybrid_step(
        &mut self,
        device: &mut Device,
        prefills: &[HybPrefill<'_>],
        decodes: &[HybDecode<'_>],
        workspace: &mut Workspace,
    ) -> Result<(), BoxError> {
        self.hybrid_step_ext(device, prefills, decodes, workspace, false, None).await
    }

    /// `hybrid_step` with the R14 d13e P2 options: `blind` = the pass is the pre-uploaded, device-fed one (its decode-row tokens are not on the host; the engine refuses
    /// anything else); `late_cb` receives the lane tokens of the previous boundary when the late read lands during this pass.
    pub async fn hybrid_step_ext(
        &mut self,
        device: &mut Device,
        prefills: &[HybPrefill<'_>],
        decodes: &[HybDecode<'_>],
        workspace: &mut Workspace,
        blind: bool,
        late_cb: Option<&mut dyn FnMut(&[usize])>,
    ) -> Result<(), BoxError> {
        let mut seen = vec![false; workspace.requests()];
        let mut runs = Vec::with_capacity(prefills.len());
        for p in prefills {
            let n = p.inputs.len();
            if n == 0 || p.blocks.flat.is_empty() || p.blocks.sliding >= workspace.requests() {
                return Err("invalid hybrid prefill run".into());
            }
            if std::mem::replace(&mut seen[p.blocks.sliding], true) {
                return Err("two rows of a hybrid pass share a request slot".into());
            }
            let rows: Vec<Row> = p.inputs.iter().map(Input::row).collect::<Result<_, _>>()?;
            runs.push(HybRun { request: p.inputs.first().map_or(0, |input| input.request), slot: p.blocks.sliding, table: p.blocks.flat.clone(), pos0: p.pos0, rows, want_hidden: p.want_next });
        }
        let mut decs = Vec::with_capacity(decodes.len());
        for d in decodes {
            if d.blocks.flat.is_empty() || d.blocks.sliding >= workspace.requests() || d.input.soft.is_some() {
                return Err("invalid hybrid decode row".into());
            }
            if std::mem::replace(&mut seen[d.blocks.sliding], true) {
                return Err("two rows of a hybrid pass share a request slot".into());
            }
            decs.push(HybDec { slot: d.blocks.sliding, table: d.blocks.flat.clone(), pos: d.pos, row: d.input.row()?, want_hidden: d.input.logits });
        }
        if crate::rpa_host::dectier() {
            // R14 d13b: rows ordered by window blocks, descending (stable): every per-row table, kernel and the per-slot hidden write-back follow this order
            decs.sort_by_key(|d| std::cmp::Reverse(crate::rpa_host::blk_need(d.pos)));
        }
        let counts: Vec<usize> = runs.iter().map(|r| r.rows.len()).collect();
        if !runtime::hybrid_fits_any(&counts, decs.len()) {
            return Err("prefill runs and decode rows do not fit one hybrid pass".into());
        }
        let hint = self.hybrid_hint.take();
        let hint_rows = self.hybrid_hint_rows.take();
        let hint_feed = self.hybrid_hint_feed.take();
        self.decode.run_hybrid(device, workspace, &self.weights, &runs, &decs, &mut self.hidden, hint, hint_rows, hint_feed, blind, late_cb).await?;
        Ok(())
    }

    /// The final hidden row (bf16 LE bytes) the last `prefill_chunk` / `decode_batch` row left for request slot `slot` (tests).
    pub fn hidden_row(&self, slot: usize) -> &[u8] {
        &self.hidden[slot]
    }

    /// The logits buffer of request slot `slot` (filled by `next_tokens` for sampled requests and by `forward`).
    pub fn logits(&self, slot: usize) -> &Logits {
        &self.logits[slot]
    }

    /// The next token of each request in `requests`, in order, from the final hidden row its last
    /// `prefill_chunk` / `decode_batch` row left in the request slot.
    ///
    /// Greedy requests (`SampleRequest::greedy`) get `Next::Greedy`: the argmax of the soft-capped logits with ties to the lowest index under EACH ban variant (`Ban::ALL`), computed ON
    /// THE DEVICE by `final_norm_and_argmax_rows8`: up to 8 rows share one pass over the 2 GB LM-head table (2.15 ms), 32 KB of keys come back. The kernel computes every variant for every
    /// row, so nothing here depends on what a request is; `Generation::advance` takes the variant of its own request. `GEMMA4_ARGMAX=0` (or an impossible all-banned row) falls back to
    /// the logits + host argmax. Sampled requests (temperature > 0) read the full logits.
    pub async fn next_tokens(&mut self, device: &mut Device, requests: &[SampleRequest]) -> Result<Vec<Next>, BoxError> {
        let mut out: Vec<Option<Next>> = (0..requests.len()).map(|_| None).collect();
        // N2: the decode rows of the last hybrid pass still sit in the device residual (taken here: valid for this call only). Greedy requests that are one of them are served in place by
        // the argmax kernel; anything else (the finishing prompt rows, sampled requests) goes through the host rows.
        let dev_rows = self.decode.take_hybrid_dev_rows();
        if device_argmax() {
            self.next_tokens_device(device, requests, dev_rows.as_ref(), &mut out).await?;
        } else if dev_rows.is_some() {
            crate::host::prof::add("n2_nouse", std::time::Duration::ZERO);
        }
        // a rider that was not served on the device (sampled / banned-out / device argmax off) needs its host copy for what follows
        if let Some(dev) = &dev_rows {
            if (0..requests.len()).any(|i| out[i].is_none() && dev.slots.contains(&requests[i].slot)) {
                dev.flush(device, &mut self.hidden).await?;
            }
        }
        for (i, request) in requests.iter().enumerate() {
            if out[i].is_some() {
                continue;
            }
            let slot = request.slot;
            runtime::final_logits(device, &self.weights, &self.hidden[slot], &mut self.logits[slot]).await?;
            let logits = &self.logits[slot];
            out[i] = Some(if request.greedy {
                Next::Greedy(crate::host::prof::timed!("read_greedy", read_greedy_all(device, logits).await)?)
            } else {
                Next::Logits(crate::host::prof::timed!("read_logits", read_logits(device, logits).await)?)
            });
        }
        Ok(out.into_iter().map(|next| next.expect("every request has a next token")).collect())
    }

    /// The greedy requests of `requests`, with the multi-group argmax: the rows that still sit on the device (`dev`: the decode rows of the last hybrid pass, one 8-row view) and the rows whose hidden
    /// row is on the host (`self.hidden`: prompts that finish, decode-only iterations) are served by as FEW launches over the 2 GB table as possible: one 8-row launch for <= 8 rows (the same
    /// kernels as before), one 16-row launch (`final_norm_and_argmax_res1_h1` / `_h2`) for 9..16 rows, and so on in pairs. The launches never depend on the requests: the kernels compute every ban
    /// variant for every row and `Generation::advance` keeps the variant of its own request. Requests it cannot serve stay `None`.
    async fn next_tokens_device(
        &mut self,
        device: &mut Device,
        requests: &[SampleRequest],
        dev: Option<&runtime::DevRows>,
        out: &mut [Option<Next>],
    ) -> Result<(), BoxError> {
        let (dev_idx, host_idx): (Vec<usize>, Vec<usize>) =
            (0..requests.len()).filter(|&i| requests[i].greedy).partition(|&i| dev.is_some_and(|d| d.slots.contains(&requests[i].slot)));
        if dev_idx.is_empty() && host_idx.is_empty() {
            return Ok(());
        }
        if dev.is_some() {
            crate::host::prof::add(if dev_idx.is_empty() { "n2_nouse" } else { "n2_use" }, std::time::Duration::ZERO);
        }
        // the 8-row groups of host rows (unused lanes repeat the first row of the group)
        let mut uploaded: Vec<HbmTensor<bf16, Chip, m![Rw8, H]>> = Vec::new();
        for batch in host_idx.chunks(8) {
            let mut bytes = Vec::with_capacity(8 * H::SIZE * 2);
            for r in 0..8 {
                bytes.extend_from_slice(&self.hidden[requests[batch[r.min(batch.len() - 1)]].slot]);
            }
            uploaded.push(HostTensor::<bf16, m![Rw8, H]>::from_buf(bytes).to_hbm(&mut device.pdma).await?);
        }
        let mut next_host = 0usize;
        if let (false, Some(dev)) = (dev_idx.is_empty(), dev) {
            let tokens = if let (2, Some(fd), true) = (dev.views.len(), self.d13e.as_ref(), self.decode.d13e_feed_ready(&dev.slots)) {
                // R14 d13e P1: LM head (keys stay on the device) -> device reduce + feed of the next pass's decode rows -> 2 x 8 tokens back (already of each lane's ban variant)
                self.argmax.launch_res2(device, &dev.views[0], &dev.views[1], &self.weights.final_norm, &self.weights.embedding_table).await?;
                let lane = self.decode.d13e_boundary(device, fd, self.argmax.keys16(), false).await?;
                crate::host::prof::add("d13e_p1", std::time::Duration::ZERO);
                lane.into_iter().map(|t| Some([t; crate::host::generate::BAN_VARIANTS])).collect::<Vec<_>>()
            } else if dev.views.len() > 2 {
                // Wide standalone decode lends three/four groups. Preserve per-group
                // order before mapping every live request slot to its token.
                let mut tokens = Vec::with_capacity(dev.views.len() * 8);
                for groups in dev.views.chunks(2) {
                    if groups.len() == 2 {
                        tokens.extend(self.argmax.argmax_groups(device, &[&groups[0], &groups[1]], &[], &self.weights.final_norm, &self.weights.embedding_table).await?);
                    } else {
                        // b32q: an odd last group goes through the same two-group LM-head kernel as the 16-rider pass (the group twice, first 8 tokens kept)
                        let t = self.argmax.argmax_groups(device, &[&groups[0], &groups[0]], &[], &self.weights.final_norm, &self.weights.embedding_table).await?;
                        tokens.extend(t.into_iter().take(8));
                    }
                }
                tokens
            } else if dev.views.len() == 2 {
                // the 16 riders of a 256-row pass: two device groups in ONE launch (host rows, if any, are served by the loop below)
                self.argmax.argmax_groups(device, &[&dev.views[0], &dev.views[1]], &[], &self.weights.final_norm, &self.weights.embedding_table).await?
            } else if uploaded.is_empty() {
                self.argmax.argmax_tokens8_res(device, &dev.views[0], &self.weights.final_norm, &self.weights.embedding_table).await?
            } else {
                self.argmax.argmax_groups(device, &[&dev.views[0]], &[&uploaded[0]], &self.weights.final_norm, &self.weights.embedding_table).await?
            };
            for &i in &dev_idx {
                let row = dev.slots.iter().position(|&s| s == requests[i].slot).expect("slot checked above");
                if let Some(tokens) = tokens[row] {
                    out[i] = Some(Next::Greedy(tokens));
                }
            }
            if dev.views.len() == 1 && !uploaded.is_empty() {
                for (k, &i) in host_idx[..host_idx.len().min(8)].iter().enumerate() {
                    if let Some(tokens) = tokens[8 + k] {
                        out[i] = Some(Next::Greedy(tokens));
                    }
                }
                next_host = 1;
            }
        }
        while next_host < uploaded.len() {
            let t = (uploaded.len() - next_host).min(2);
            let hosts: Vec<&HbmTensor<bf16, Chip, m![Rw8, H]>> = uploaded[next_host..next_host + t].iter().collect();
            let idx = &host_idx[8 * next_host..host_idx.len().min(8 * (next_host + t))];
            let tokens = if t == 1 {
                self.argmax.argmax_tokens8(device, hosts[0], &self.weights.final_norm, &self.weights.embedding_table).await?
            } else {
                self.argmax.argmax_groups(device, &[], &hosts, &self.weights.final_norm, &self.weights.embedding_table).await?
            };
            for (k, &i) in idx.iter().enumerate() {
                if let Some(tokens) = tokens[k] {
                    out[i] = Some(Next::Greedy(tokens));
                }
            }
            next_host += t;
        }
        Ok(())
    }

    /// Append each input to its history and return one logits buffer per input, in input order.
    /// Buffers of inputs with `logits == false` are stale. Rows run serially; this is the
    /// reference path (`bin/gemma4.rs` and tests) and does not use the request-slot logits.
    pub async fn forward(
        &mut self,
        device: &mut Device,
        inputs: &[Input<'_>],
        positions: &[usize],
        blocks: &[&Blocks],
        workspace: &mut Workspace,
    ) -> Result<&[Logits], BoxError> {
        if inputs.len() > self.logits.len() || inputs.len() != positions.len() || inputs.len() != blocks.len() {
            return Err("incompatible batch dimensions".into());
        }
        for (row, ((input, &position), &blocks)) in inputs.iter().zip(positions).zip(blocks).enumerate() {
            self.check(workspace, blocks, position)?;
            let row_input = input.row()?;
            let hidden = input.logits.then(|| &mut self.hidden[row]);
            self.decode
                .run_chunk(device, workspace, &self.weights, blocks, position, &[row_input], hidden)
                .await?;
            if input.logits {
                runtime::final_logits(device, &self.weights, &self.hidden[row], &mut self.logits[row]).await?;
            }
        }
        Ok(&self.logits[..inputs.len()])
    }
}
