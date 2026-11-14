//! Batched (multi-row) bf16 GEMMs for the image / audio encoders.  See `gemm.rs`.
pub(crate) mod gemm;
pub(crate) mod norm;
