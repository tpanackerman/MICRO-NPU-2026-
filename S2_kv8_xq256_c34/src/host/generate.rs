use crate::Chip;
use crate::axes::W;
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::model::{Input, Model, SampleRequest};
use crate::host::runtime::{Blocks, SoftEmbeddings, Workspace};
use crate::host::sampling::{SamplingConfig, argmax_bf16, sample};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use llm_tokenizer::DecodeStream;

pub const EOS_TOKEN: usize = 1;
pub const END_OF_TURN_TOKEN: usize = 106;
pub const TOOL_RESPONSE_TOKEN: usize = 50;

pub const EOS_TOKENS: [usize; 3] = [EOS_TOKEN, END_OF_TURN_TOKEN, TOOL_RESPONSE_TOKEN];

pub const CHANNEL_OPEN_TOKEN: usize = 100;
pub const CHANNEL_CLOSE_TOKEN: usize = 101;

const CHANNEL_OPEN_TEXT: &str = "<|channel>";
const CHANNEL_CLOSE_TEXT: &str = "<channel|>";

pub const DEFAULT_MAX_NEW_TOKENS: usize = 300;
pub const BOI_TOKEN: usize = 255999;
pub const IMAGE_TOKEN: usize = 258880;
pub const EOI_TOKEN: usize = 258882;
pub const BOA_TOKEN: usize = 256000;
pub const AUDIO_TOKEN: usize = 258881;
pub const EOA_TOKEN: usize = 258883;

/// Ids no request may ever produce (sorted).
const BAN_SUPPRESSED: [usize; 2] = [EOI_TOKEN, EOA_TOKEN];
/// Ids a request with `ignore_eos` must not produce either: the end-of-sequence tokens and the
/// channel markers (text in the reasoning channel is not part of the answer, so it would shorten
/// the visible output). Sorted.
const BAN_IGNORE_EOS: [usize; 7] = [
    EOS_TOKEN,
    TOOL_RESPONSE_TOKEN,
    CHANNEL_OPEN_TOKEN,
    CHANNEL_CLOSE_TOKEN,
    END_OF_TURN_TOKEN,
    EOI_TOKEN,
    EOA_TOKEN,
];

/// Which token ids a greedy argmax must exclude. The device argmax computes the greedy token of EVERY row under EACH variant (`Ban::ALL`, always the same instructions for every row,
/// whatever the request is), and `Generation::advance` keeps the one of its own request, the same place and the same predicate (`ignore_eos`) as the baseline's host-side ban set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ban {
    /// `EOI` and `EOA` only.
    Suppressed,
    /// `Suppressed` plus the end-of-sequence tokens and the channel markers (`ignore_eos`).
    IgnoreEos,
}

/// Number of ban variants the device argmax computes for every row.
pub const BAN_VARIANTS: usize = 2;

impl Ban {
    /// Every variant, in the order of the device tables / key blocks and of `Next::Greedy`.
    pub const ALL: [Ban; BAN_VARIANTS] = [Ban::Suppressed, Ban::IgnoreEos];

    pub fn ids(self) -> &'static [usize] {
        match self {
            Ban::Suppressed => &BAN_SUPPRESSED,
            Ban::IgnoreEos => &BAN_IGNORE_EOS,
        }
    }

    /// Position of this variant in `Ban::ALL`.
    pub fn index(self) -> usize {
        match self {
            Ban::Suppressed => 0,
            Ban::IgnoreEos => 1,
        }
    }
}

/// What the device side delivers for the row that finishes a request's prompt or decodes a token.
pub enum Next {
    /// Nothing was asked for (a prompt row whose output is discarded).
    None,
    /// The greedy token of the row under EACH ban variant (`Ban::ALL` order), all computed the same way for every row (device argmax, or the host argmax fallback): the row does not know which
    /// variant its request wants; `Generation::advance` takes `tokens[ban.index()]`.
    Greedy([usize; BAN_VARIANTS]),
    /// Full logits for a sampled (temperature > 0) request.
    Logits(Vec<f32>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Channel {
    Answer,
    Name,
    Reasoning,
}

#[derive(Clone, Copy, Debug)]
pub enum Delta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

#[derive(Clone, Copy)]
pub enum SoftToken {
    Image(usize),
    Audio(usize),
}

pub fn splice_multimodal_tokens(
    ids: Vec<usize>,
    image: Option<&ImagePatches>,
    audio: Option<&AudioFrames>,
) -> Result<(Vec<usize>, Vec<Option<SoftToken>>), String> {
    let image_markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN).count();
    let audio_markers = ids.iter().filter(|&&id| id == AUDIO_TOKEN).count();
    if image.is_some() && image_markers != 1 {
        return Err(format!("expected one image placeholder, found {image_markers}"));
    }
    if audio.is_some() && audio_markers != 1 {
        return Err(format!("expected one audio placeholder, found {audio_markers}"));
    }

    let image_len = image.map_or(0, |value| value.pixels.len());
    let audio_len = audio.map_or(0, AudioFrames::len);
    let mut out_ids = Vec::with_capacity(ids.len() + image_len + audio_len + 4);
    let mut soft_at = Vec::with_capacity(out_ids.capacity());
    for id in ids {
        match id {
            IMAGE_TOKEN if image.is_some() => {
                out_ids.push(BOI_TOKEN);
                soft_at.push(None);
                for index in 0..image_len {
                    out_ids.push(IMAGE_TOKEN);
                    soft_at.push(Some(SoftToken::Image(index)));
                }
                out_ids.push(EOI_TOKEN);
                soft_at.push(None);
            }
            AUDIO_TOKEN if audio.is_some() => {
                out_ids.push(BOA_TOKEN);
                soft_at.push(None);
                for index in 0..audio_len {
                    out_ids.push(AUDIO_TOKEN);
                    soft_at.push(Some(SoftToken::Audio(index)));
                }
                out_ids.push(EOA_TOKEN);
                soft_at.push(None);
            }
            _ => {
                out_ids.push(id);
                soft_at.push(None);
            }
        }
    }
    Ok((out_ids, soft_at))
}

fn pending_stop_overlap(text: &str, stop_strings: &[&str]) -> usize {
    let mut hold = 0usize;
    for &stop in stop_strings {
        let max_len = (stop.len() - 1).min(text.len());
        for len in (1..=max_len).rev() {
            if stop.is_char_boundary(len) && text.ends_with(&stop[..len]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    hold
}

fn step_over_marker(
    stream: &mut DecodeStream,
    id: usize,
    marker: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(delta) = stream.step(id as u32)? else {
        return Ok(None);
    };
    let held = delta.strip_suffix(marker).unwrap_or(&delta);
    Ok((!held.is_empty()).then(|| held.to_owned()))
}

/// All logits as f32, for requests that sample (temperature > 0).
pub async fn read_logits(
    device: &mut Device,
    logits: &HbmTensor<bf16, Chip, m![W]>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let logits: HostTensor<bf16, m![W]> = logits.to_host(&mut device.pdma).await?;
    Ok(logits.into_vec().into_iter().map(bf16::to_f32).collect())
}

/// The greedy token under each ban variant (`Ban::ALL` order): one D2H of the bf16 logits and an argmax straight over them per variant (no f32 copy, no per-element set lookup). Exactly the
/// semantics of `sampling::argmax` over the f32 values.
pub async fn read_greedy_all(
    device: &mut Device,
    logits: &HbmTensor<bf16, Chip, m![W]>,
) -> Result<[usize; BAN_VARIANTS], Box<dyn std::error::Error>> {
    let logits: HostTensor<bf16, m![W]> = logits.to_host(&mut device.pdma).await?;
    let logits = logits.into_vec();
    Ok(Ban::ALL.map(|ban| argmax_bf16(&logits, ban.ids())))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct GenerationRequest {
    pub prompt_ids: Vec<usize>,
    pub soft_at: Vec<Option<SoftToken>>,
    pub image: Option<ImagePatches>,
    pub audio: Option<AudioFrames>,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_strings: Vec<String>,
    pub separate_reasoning: bool,
    /// Never stop at an end-of-sequence token: generate exactly `max_new_tokens` tokens.
    /// Set only by performance tests, to give every request the same output length.
    pub ignore_eos: bool,
}

pub struct GenerationOutput {
    pub text: String,
    pub reasoning: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One conversation's progress, independent of the order in which requests run.
pub struct Generation {
    request: GenerationRequest,
    phase: Phase,
    stream: DecodeStream,
    channel: Channel,
    output: GenerationOutput,
    emitted: usize,
    /// Soft-token embeddings computed once at admission; dropped when the prompt is done.
    embeddings: Option<SoftEmbeddings>,
    /// Unique per generation (`Input::request`): keys any per-request device state.
    id: u64,
    /// R14 d13e P2: the output token of the last computed row is not known on the host yet (`advance_blind`; `fill_token` completes the step).
    pending: bool,
    /// R14 d13e P2: that pending token is the last one (`max_new_tokens` reached): `fill_token` finishes the request.
    pending_finish: bool,
}

/// R14 d13e P2: the pending input token of a `Phase::Decode` whose token is still on the device.
pub const PENDING_TOKEN: usize = usize::MAX;

/// The next `len()` prompt tokens of one request, to be computed as one chunk.
pub struct PromptChunk<'a> {
    /// Position of the first token; the chunk fills positions `pos0..pos0 + len()`.
    pub pos0: usize,
    pub ids: &'a [usize],
    pub soft: &'a [Option<SoftToken>],
    /// The chunk ends with the last prompt token: that row's logits give the first output token.
    pub last: bool,
    wants_logits: bool,
    embeddings: Option<&'a SoftEmbeddings>,
    request: u64,
}

impl<'a> PromptChunk<'a> {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// The chunk as model rows. Only the final row of a `last` chunk asks for logits.
    pub fn inputs(&self) -> Vec<Input<'a>> {
        let n = self.ids.len();
        (0..n)
            .map(|i| Input {
                token: self.ids[i],
                soft: self.soft[i],
                logits: self.last && self.wants_logits && i + 1 == n,
                embeddings: self.embeddings,
                request: self.request,
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Prompt(usize),
    Decode(usize),
    Complete,
}

impl Generation {
    pub fn new(
        tokenizer: &Tokenizer,
        request: GenerationRequest,
        capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if request.prompt_ids.is_empty() {
            return Err("prompt produced no tokens".into());
        }
        if request.prompt_ids.len().saturating_add(request.max_new_tokens) > capacity {
            return Err(format!("prompt plus generation exceeds {capacity} positions").into());
        }
        if request.prompt_ids.len() != request.soft_at.len() {
            return Err(format!(
                "prompt_ids ({}) and soft_at ({}) must be parallel",
                request.prompt_ids.len(),
                request.soft_at.len()
            )
            .into());
        }
        Ok(Self {
            stream: tokenizer.decode_stream(&request.prompt_ids.iter().map(|&id| id as u32).collect::<Vec<_>>()),
            output: GenerationOutput {
                text: String::new(),
                reasoning: String::new(),
                prompt_tokens: request.prompt_ids.len(),
                completion_tokens: 0,
                finish_reason: FinishReason::Length,
            },
            request,
            phase: Phase::Prompt(0),
            channel: Channel::Answer,
            emitted: 0,
            embeddings: None,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            pending: false,
            pending_finish: false,
        })
    }

    /// Unique id of this generation (never shared across requests).
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn image(&self) -> Option<&ImagePatches> {
        self.request.image.as_ref()
    }

    pub fn audio(&self) -> Option<&AudioFrames> {
        self.request.audio.as_ref()
    }

    /// Hand over the precomputed soft-token embeddings. The raw pixels and samples are no longer
    /// needed and are released.
    pub fn attach_embeddings(&mut self, embeddings: SoftEmbeddings) {
        self.embeddings = Some(embeddings);
        self.request.image = None;
        self.request.audio = None;
    }

    /// Whether the request is still feeding its prompt.
    pub fn is_prefill(&self) -> bool {
        matches!(self.phase, Phase::Prompt(_))
    }

    /// Whether the request is generating (its pending input is its own previous output).
    pub fn is_decode(&self) -> bool {
        matches!(self.phase, Phase::Decode(_))
    }

    pub fn prompt_len(&self) -> usize {
        self.request.prompt_ids.len()
    }

    /// The request's `max_new_tokens`.
    pub fn max_new_tokens(&self) -> usize {
        self.request.max_new_tokens
    }

    /// Prompt tokens not yet computed (0 once decoding).
    pub fn prompt_remaining(&self) -> usize {
        match self.phase {
            Phase::Prompt(pos) => self.request.prompt_ids.len() - pos,
            _ => 0,
        }
    }

    /// Position the pending input is written at.
    pub fn position(&self) -> usize {
        match self.phase {
            Phase::Prompt(pos) => pos,
            Phase::Decode(_) => self.request.prompt_ids.len() + self.output.completion_tokens - 1,
            Phase::Complete => unreachable!("completed requests have no position"),
        }
    }

    /// Which ids this request's greedy argmax excludes.
    pub fn ban(&self) -> Ban {
        if self.request.ignore_eos { Ban::IgnoreEos } else { Ban::Suppressed }
    }

    /// Whether the next token is the argmax of the logits (so a device argmax may produce it).
    pub fn is_greedy(&self) -> bool {
        self.request.sampling.is_greedy()
    }

    /// What the model must return for the pending input's row, if anything.
    pub fn sample_request(&self, slot: usize) -> SampleRequest {
        SampleRequest { slot, greedy: self.is_greedy() }
    }

    /// The next `max` prompt tokens (fewer at the end of the prompt), or `None` once decoding.
    /// Does not advance: call `consume_prompt` or `finish_chunk` after the chunk was computed.
    pub fn prompt_chunk(&self, max: usize) -> Option<PromptChunk<'_>> {
        let Phase::Prompt(pos) = self.phase else { return None };
        self.prompt_chunk_from(pos, max)
    }

    /// Like `prompt_chunk`, but starting at position `pos` (at or after the next position), for a
    /// scheduler that plans several chunks of one request before any of them has run.
    pub fn prompt_chunk_from(&self, pos: usize, max: usize) -> Option<PromptChunk<'_>> {
        let Phase::Prompt(next) = self.phase else { return None };
        if pos < next || pos >= self.request.prompt_ids.len() {
            return None;
        }
        let n = max.min(self.request.prompt_ids.len() - pos);
        if n == 0 {
            return None;
        }
        Some(PromptChunk {
            pos0: pos,
            ids: &self.request.prompt_ids[pos..pos + n],
            soft: &self.request.soft_at[pos..pos + n],
            last: pos + n == self.request.prompt_ids.len(),
            wants_logits: self.request.max_new_tokens > 0,
            embeddings: self.embeddings.as_ref(),
            request: self.id,
        })
    }

    /// R14 d13e P2z: whether the step that consumes this request's next greedy token can be taken before the token is known on the host (same rule for every request:
    /// greedy, a token is due, none pending). If the unread token ends the request (end-of-sequence or stop string), the request rides the already-launched next pass once
    /// as a zombie whose output is discarded (`fill_token` finishes it with exactly `advance`'s output).
    pub fn blind_ok(&self) -> bool {
        self.is_greedy() && self.request.max_new_tokens > 0 && !self.pending && !matches!(self.phase, Phase::Complete)
    }

    /// The output token of the last computed row is still pending (`advance_blind`).
    pub fn is_pending(&self) -> bool {
        self.pending
    }

    /// R14 d13e P2: `advance` for a `blind_ok` request whose token is not on the host yet: the position and length bookkeeping of the step now, the token later (`fill_token`).
    /// Returns true when this step reaches `max_new_tokens` (the request ends once the token is filled in).
    pub fn advance_blind(&mut self) -> bool {
        debug_assert!(self.blind_ok());
        match std::mem::replace(&mut self.phase, Phase::Complete) {
            Phase::Prompt(pos) if pos + 1 < self.request.prompt_ids.len() => {
                self.phase = Phase::Prompt(pos + 1);
                return false;
            }
            Phase::Complete => panic!("advance_blind on a completed generation"),
            _ => {}
        }
        self.embeddings = None;
        self.pending = true;
        self.output.completion_tokens += 1;
        if self.output.completion_tokens == self.request.max_new_tokens {
            self.pending_finish = true;
            return true;
        }
        self.phase = Phase::Decode(PENDING_TOKEN);
        false
    }

    /// R14 d13e P2: `finish_chunk` with the first output token pending.
    pub fn finish_chunk_blind(&mut self, n: usize) -> Result<bool, Box<dyn std::error::Error>> {
        let Phase::Prompt(pos) = self.phase else { return Err("finish_chunk_blind outside the prompt".into()) };
        if n == 0 || pos + n != self.request.prompt_ids.len() {
            return Err("finish_chunk_blind must end at the last prompt token".into());
        }
        self.phase = Phase::Prompt(pos + n - 1);
        Ok(self.advance_blind())
    }

    /// R14 d13e P2: completes the pending step with its greedy token (`tokens` = every ban variant, as `Next::Greedy`): the same emission and finishing as `advance`.
    pub fn fill_token(&mut self, tokens: [usize; BAN_VARIANTS], mut on_delta: impl FnMut(Delta) -> bool) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        if !self.pending {
            return Err("fill_token without a pending token".into());
        }
        self.pending = false;
        let next = tokens[self.ban().index()];
        // `advance` counts the token only after the end-of-sequence check and the emission: undo the blind count on those two early ends
        if EOS_TOKENS.contains(&next) {
            self.pending_finish = false;
            self.output.completion_tokens -= 1;
            return Ok(Some(self.finish(FinishReason::Stop, &mut on_delta)));
        }
        if let Some(reason) = self.emit(next, &mut on_delta)? {
            self.pending_finish = false;
            self.output.completion_tokens -= 1;
            return Ok(Some(self.finish(reason, &mut on_delta)));
        }
        if std::mem::replace(&mut self.pending_finish, false) {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)));
        }
        if let Phase::Decode(token) = &mut self.phase {
            if *token == PENDING_TOKEN {
                *token = next;
            }
        }
        Ok(None)
    }

    /// Advance past `n` computed prompt tokens that do NOT include the last one.
    pub fn consume_prompt(&mut self, n: usize) {
        let Phase::Prompt(pos) = self.phase else { panic!("consume_prompt outside the prompt") };
        assert!(pos + n < self.request.prompt_ids.len(), "the last prompt token goes through finish_chunk");
        self.phase = Phase::Prompt(pos + n);
    }

    /// Advance past the final `n` prompt tokens (the chunk ends with the last token) and turn that
    /// row's output into the first generated token, exactly as `advance` does for a single token.
    pub fn finish_chunk(
        &mut self,
        n: usize,
        next: Next,
        on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        let Phase::Prompt(pos) = self.phase else { return Err("finish_chunk outside the prompt".into()) };
        if n == 0 || pos + n != self.request.prompt_ids.len() {
            return Err("finish_chunk must end at the last prompt token".into());
        }
        self.phase = Phase::Prompt(pos + n - 1);
        self.advance(next, on_delta)
    }

    /// R14 d13e P2z fix (bigp): `input` for a row planned into a pass whose request was finished by the late token fill (end-of-sequence / stop string read during
    /// the previous blind pass) before this pass reads its inputs: a dummy token at its last position, the zombie ride of the blind path (output discarded, slot
    /// freed after the pass). Without it the legacy decode batch / the non-blind hybrid pass hit `input`'s Complete arm (fleurs_en C=22: worker panic).
    pub fn input_or_zombie(&self) -> (Input<'_>, usize) {
        if matches!(self.phase, Phase::Complete) {
            let pos = (self.request.prompt_ids.len() + self.output.completion_tokens).saturating_sub(1);
            return (Input { token: 0, soft: None, logits: true, embeddings: None, request: self.id }, pos);
        }
        self.input()
    }

    /// The next token to compute and the position it is written at.
    pub fn input(&self) -> (Input<'_>, usize) {
        let (token, soft, logits, position) = match self.phase {
            Phase::Prompt(pos) => (
                self.request.prompt_ids[pos],
                self.request.soft_at[pos],
                pos + 1 == self.request.prompt_ids.len(),
                pos,
            ),
            Phase::Decode(token) => (
                token,
                None,
                true,
                self.request.prompt_ids.len() + self.output.completion_tokens - 1,
            ),
            Phase::Complete => unreachable!("completed requests have no input"),
        };
        (
            Input {
                token,
                soft,
                logits,
                embeddings: self.embeddings.as_ref(),
                request: self.id,
            },
            position,
        )
    }

    /// Whether the pending input produces logits that `advance` needs.
    pub fn wants_logits(&self) -> bool {
        self.request.max_new_tokens > 0 && self.input().0.logits
    }

    /// Consume the output of the pending input (`Next::None` if `wants_logits` was false).
    pub fn advance(
        &mut self,
        next: Next,
        mut on_delta: impl FnMut(Delta) -> bool,
    ) -> Result<Option<GenerationOutput>, Box<dyn std::error::Error>> {
        let phase = std::mem::replace(&mut self.phase, Phase::Complete);
        if matches!(phase, Phase::Complete) {
            return Err("generation is already complete".into());
        }
        if self.request.max_new_tokens == 0 {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)));
        }
        if let Phase::Prompt(pos) = phase
            && pos + 1 < self.request.prompt_ids.len()
        {
            self.phase = Phase::Prompt(pos + 1);
            return Ok(None);
        }
        // The prompt is fully computed: its soft-token embeddings are no longer needed.
        self.embeddings = None;
        let next = match next {
            Next::Greedy(tokens) => tokens[self.ban().index()],
            Next::Logits(logits) => sample(&logits, &self.request.sampling, self.ban().ids(), &mut rand::rng()),
            Next::None => return Err("the model returned no output for a sampling row".into()),
        };
        if EOS_TOKENS.contains(&next) {
            return Ok(Some(self.finish(FinishReason::Stop, &mut on_delta)));
        }
        if let Some(reason) = self.emit(next, &mut on_delta)? {
            return Ok(Some(self.finish(reason, &mut on_delta)));
        }
        self.output.completion_tokens += 1;
        if self.output.completion_tokens == self.request.max_new_tokens {
            return Ok(Some(self.finish(FinishReason::Length, &mut on_delta)));
        }
        // The next turn computes the token just emitted, so every turn has one forward.
        self.phase = Phase::Decode(next);
        Ok(None)
    }

    fn emit(
        &mut self,
        token: usize,
        on_delta: &mut impl FnMut(Delta) -> bool,
    ) -> Result<Option<FinishReason>, Box<dyn std::error::Error>> {
        let (delta, switch_to) = match token {
            CHANNEL_OPEN_TOKEN if self.request.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_OPEN_TEXT)?,
                Some(Channel::Name),
            ),
            CHANNEL_CLOSE_TOKEN if self.request.separate_reasoning => (
                step_over_marker(&mut self.stream, token, CHANNEL_CLOSE_TEXT)?,
                Some(Channel::Answer),
            ),
            _ => (self.stream.step(token as u32)?, None),
        };
        if let Some(delta) = delta {
            let keep = match self.channel {
                Channel::Name => {
                    if let Some((_, rest)) = delta.split_once('\n') {
                        self.channel = Channel::Reasoning;
                        self.output.reasoning.push_str(rest);
                        rest.is_empty() || on_delta(Delta::Reasoning(rest))
                    } else {
                        true
                    }
                }
                Channel::Reasoning => {
                    self.output.reasoning.push_str(&delta);
                    on_delta(Delta::Reasoning(&delta))
                }
                Channel::Answer => {
                    self.output.text.push_str(&delta);
                    let stops: Vec<_> = self
                        .request
                        .stop_strings
                        .iter()
                        .map(String::as_str)
                        .filter(|text| !text.is_empty())
                        .collect();
                    if let Some(cut) = stops.iter().filter_map(|s| self.output.text.find(s)).min() {
                        if cut > self.emitted {
                            let _ = on_delta(Delta::Content(&self.output.text[self.emitted..cut]));
                        }
                        self.output.text.truncate(cut);
                        self.emitted = cut;
                        return Ok(Some(FinishReason::Stop));
                    }
                    let safe = self.output.text.len() - pending_stop_overlap(&self.output.text, &stops);
                    if safe > self.emitted {
                        let keep = on_delta(Delta::Content(&self.output.text[self.emitted..safe]));
                        self.emitted = safe;
                        keep
                    } else {
                        true
                    }
                }
            };
            if !keep {
                return Ok(Some(FinishReason::Length));
            }
        }
        if let Some(channel) = switch_to {
            self.channel = channel;
        }
        Ok(None)
    }

    fn finish(&mut self, reason: FinishReason, on_delta: &mut impl FnMut(Delta) -> bool) -> GenerationOutput {
        if self.emitted < self.output.text.len() {
            let _ = on_delta(Delta::Content(&self.output.text[self.emitted..]));
        }
        self.phase = Phase::Complete;
        GenerationOutput {
            text: std::mem::take(&mut self.output.text),
            reasoning: std::mem::take(&mut self.output.reasoning),
            prompt_tokens: self.output.prompt_tokens,
            completion_tokens: self.output.completion_tokens,
            finish_reason: reason,
        }
    }
}

/// Length of the next chunk of a prompt that has `remaining` tokens left, with the next token at
/// `pos`: at most `limit` tokens, never crossing a multiple of `chunk` (so a chunk never wraps the
/// sliding ring or crosses a full-attention page when `chunk` divides 512).
pub fn aligned_chunk(pos: usize, remaining: usize, chunk: usize, limit: usize) -> usize {
    remaining.min(limit).min(chunk - pos % chunk)
}

/// Prompt chunk length used by `generate` (the worker has its own, configurable).
pub const GENERATE_CHUNK: usize = 128;

/// Chunk size of `generate` (CLI and tests): `S2_CHUNK` (a divisor of 512; 128 uses the T = 128 chunk path), default `GENERATE_CHUNK`.
pub fn generate_chunk() -> usize {
    std::env::var("S2_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|&c: &usize| c > 0 && 512 % c == 0).unwrap_or(GENERATE_CHUNK)
}

/// Run one request alone, using the first `blocks_for(capacity)` pool blocks.
pub async fn generate(
    device: &mut Device,
    model: &mut Model,
    tokenizer: &Tokenizer,
    workspace: &mut Workspace,
    request: GenerationRequest,
    mut on_delta: impl FnMut(Delta) -> bool,
) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
    let empty = request.max_new_tokens == 0;
    let mut generation = Generation::new(tokenizer, request, workspace.capacity())?;
    if empty {
        return Ok(generation.advance(Next::None, &mut on_delta)?.expect("empty request finishes"));
    }
    let embeddings = model.embed_media(device, generation.image(), generation.audio()).await?;
    generation.attach_embeddings(embeddings);
    // Fast path (flat caches): the single request takes pool pages 0.., its prompt runs through hybrid passes (no decode rows) and it decodes through the batched decode
    // pass. `GEMMA4_DECODE=chunk` keeps the legacy per-slot path (A/B).
    let positions = generation.prompt_len() + generation.max_new_tokens();
    let flat = if workspace.has_flat()
        && positions <= crate::host::runtime::decode_limit()
        && generation.prompt_len() <= crate::host::runtime::HYBRID_PREFILL_LIMIT
        && std::env::var("GEMMA4_DECODE").map(|v| v != "chunk").unwrap_or(true)
    {
        (0..Workspace::blocks_for(positions)).collect()
    } else {
        Vec::new()
    };
    let blocks = Blocks {
        full: (0..Workspace::blocks_for(workspace.capacity())).collect(),
        sliding: 0,
        flat,
    };
    loop {
        let output = if generation.is_prefill() {
            let n = aligned_chunk(
                generation.position(),
                generation.prompt_remaining(),
                generate_chunk(),
                generate_chunk(),
            );
            let (n, last, next) = {
                let chunk = generation.prompt_chunk(n).expect("prefill has a chunk");
                let inputs = chunk.inputs();
                if blocks.flat.is_empty() {
                    model.prefill_chunk(device, &inputs, chunk.pos0, &blocks, workspace, chunk.last).await?;
                } else {
                    let run = crate::host::model::HybPrefill { blocks: &blocks, pos0: chunk.pos0, inputs, want_next: chunk.last };
                    model.hybrid_step(device, &[run], &[], workspace).await?;
                }
                let next = if chunk.last {
                    model.next_tokens(device, &[generation.sample_request(blocks.sliding)]).await?.pop()
                } else {
                    None
                };
                (chunk.len(), chunk.last, next)
            };
            if last {
                generation.finish_chunk(n, next.unwrap_or(Next::None), &mut on_delta)?
            } else {
                generation.consume_prompt(n);
                None
            }
        } else {
            let (input, position) = generation.input();
            model.decode_batch(device, &[input], &[position], &[&blocks], workspace).await?;
            let next = model
                .next_tokens(device, &[generation.sample_request(blocks.sliding)])
                .await?
                .pop()
                .unwrap_or(Next::None);
            generation.advance(next, &mut on_delta)?
        };
        if let Some(output) = output {
            return Ok(output);
        }
    }
}
