//! R14 w8a8 (`S2_I8_LAYERS`, LOSSY, default OFF): int8 W8A8 up / gate projections of the native T = 256 sliding layer inside the fused K launch
//! (stage2/w8a8/DESIGN.md). Weights: int8 per (output row x 128-element K chunk) with an f32 scale, re-quantised from the provided NVFP4 tensors at load
//! time on the host (`host/i8w.rs`); activations: int8 per (token x K chunk), quantised in the pre-FF norm of the K-split layout (`ffn::prenorm_q8`). Everything else
//! (AO, GeGLU, down projection, norms, attention, QKV) is the production bf16 code.
pub(crate) mod ffn;
pub(crate) mod qkvq;
