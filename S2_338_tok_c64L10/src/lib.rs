//! Gemma-4-12B-it on the Furiosa NPU: text, vision, and audio, targeting the Tensor
//! Contraction Processor via the `furiosa-opt-std` mapping-expression DSL. See the
//! crate's README for the device/host split and directory layout.

#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

pub mod api;
pub mod axes;
pub mod host;
pub mod mm_norm;
pub mod mm_pack;
pub mod ops;
pub mod chunk_host;
pub mod dattn_host;
pub mod ops_dattn;
pub mod ops_chunk;
pub mod ops_mig;
pub mod ops_rpa;
pub mod ops_rpa_full;
pub mod rpa_host;
pub mod ops_t128;
pub mod ops_ffnc;
pub mod ops_f8;
pub mod ops_qkvc;
pub mod ops_lmhead;
pub mod lmhead_host;
pub mod lmh_trace;
pub mod ops_mm;
pub mod ops_ref;
pub mod ops_audio;
pub mod ops_vision;
pub mod ops_fl;
pub mod ops_d13e;
pub mod ops_i8;
pub mod qkv8;

pub(crate) mod device;

pub type Chip = m![1];
/// Element type of the flat sliding K / V caches `[Rf, Ns, Ds]` (kv8: e4m3, decoded to bf16 in the Fetch Adapter of every reader).
pub type Kv = f8e4m3;
/// Bytes of one `Kv` element (host byte offsets into the flat sliding caches).
pub const KV_BYTES: usize = 1;

pub const LAYERS: usize = 48;
pub const EMBED_SCALE: f32 = 61.967_734;
pub const LOGIT_SOFTCAP: f32 = 30.0;
pub const EPS: f32 = 1e-6;
/// Added to the softmax denominator of the chunk / batched-decode / ragged attention kernels (the sum of the bf16 weights `exp(s - max)`). The max is taken over ALL key
/// slots, masked ones included, so a masked key whose score exceeds the visible maximum by `d` scales every visible weight by `e^-d`: with `EPS` = 1e-6 the denominator is
/// dominated by EPS from `d` ~ 14 and the attention output collapses (drift investigation: up to 10% error of the prompt's first row). This only guards the all-masked row
/// (0 / EPS = 0, no NaN), so it must be far below any real weight (f32 exp underflows at `d` ~ 87).
pub const SOFTMAX_EPS: f32 = 1e-30;
/// Additive score penalty of the attention key slots that can never be visible (never written / stale rows of the slot's ring, pages or pool pages, zero-filled
/// staging slots, rows of an unallocated page, keys of other requests that padding rows scattered): the host builds the (per key slot) penalty tensors with `0` for a key
/// that may be visible and `MASK_PEN` for the others, the kernels add it to the scores BEFORE the row max, so the max (and with it every bf16 rounding of `exp(s - max)`)
/// depends only on the request's own data, never on what a previous request left in the slot. `exp(s + MASK_PEN - max)` underflows to exactly 0 (|s - max| is far below 1024).
pub const MASK_PEN: f32 = -1024.0;
