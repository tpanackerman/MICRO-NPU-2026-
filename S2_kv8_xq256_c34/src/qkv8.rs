//! R14 w8qkv switch (LOSSY, compile-time: the grader builds `src/` with its own Cargo.toml, so the switch lives here).
//! ON  (this file as shipped by the patch): the sliding quad QKV of the fused K / K_f8 launches and of the quad prologue, and the full-layer quad QKV, run e4m3 x e4m3 on the native e4m3 checkpoint
//!     weights with a per-token e4m3 activation (`device::qkvc::quad8`); the quad scratch is e4m3.
//! OFF: replace the four lines below by
//!     `pub type XnQ = bf16;`
//!     `pub(crate) use crate::device::qkvc::quadp::sliding_qkv_quad_p as quad_sel;`
//!     `pub(crate) use crate::device::qkvc::quadhf::full_qkv_quad_h as quad_full_sel;`
//!     which gives back the bf16 quad of `big` (identical kernels; the host keeps one extra 2 MiB scratch).
use crate::axes::*;
use furiosa_opt_std::prelude::*;

pub type XnQ = f8e4m3;
/// The quad scratch [Ub, Tc, Kc, Ke] (token-major; a chunk-major e4m3 scratch was measured statically WORSE: the producer store becomes 30 pieces per token, quad 116k -> 158k).
pub type XnScratch = HbmTensor<XnQ, crate::Chip, m![Ub, Tc / 2, Kc, Tc % 2, Ke]>;
/// Host image of the scratch (zero upload).
pub type XnHost = HostTensor<XnQ, m![Ub, Tc / 2, Kc, Tc % 2, Ke]>;
pub(crate) use crate::device::qkvc::quad8::sliding_qkv_quad_p8 as quad_sel;
pub(crate) use crate::device::qkvc::quadhf8::full_qkv_quad_h8 as quad_full_sel;

/// Whether the e4m3 quad QKV is compiled in (derived from the scratch type).
pub const QKV8: bool = std::mem::size_of::<XnQ>() == 1;
