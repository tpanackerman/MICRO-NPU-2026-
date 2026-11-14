//! Chunk (multi-token) QKV projection kernels for chunked prefill and batched decode (tree qkvc).
//! T = 64 (`Tc`) bodies live in this directory; the T = 8 bucket (`Tb`) is generated from them by tools/gen_t8.py into `t8/`.
pub(crate) mod full;
pub(crate) mod gemm;
pub(crate) mod pair;
pub(crate) mod pair2;
pub(crate) mod post;
pub(crate) mod post_full;
pub(crate) mod quad;
pub(crate) mod sliding;
pub(crate) mod t8;
pub(crate) mod xnorm;
