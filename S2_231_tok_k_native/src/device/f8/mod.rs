//! Lossy FP8 weight fold of the FFN (`S2_FP8`, default ON in fold1; the kernels are in `ffn.rs`, the token-sharded quantiser helpers in `tok.rs`, entries in `ops_f8.rs`).
//!
//! Quantiser contract (bit-exact vs the host round-to-nearest-even e4m3 quantiser): per token t, `s_t = sqrt(max_k x[t,k]^2) * (1/448) + 1e-30`, `xq = e4m3(x / s_t)` (saturating cast); the GEMM
//! output is multiplied by `s_t` before the post-FFN RMSNorm, so everything downstream sees the same dtype and layout as the bf16 path.
use furiosa_opt_std::prelude::*;

// part axes of the token-sharded layout: a token row of the pre-FF norm output (3840 = 2 x 1920) and of the GeGLU output (2 parts of 8192 padded columns)
axes![Xa1920 = 1920, Xa8192 = 8192];

pub(crate) mod ffn;
pub(crate) mod quant;
pub(crate) mod tok;
