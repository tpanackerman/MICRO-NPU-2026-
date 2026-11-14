//! Chunk (multi-token) attention-output and FFN kernels (tree `ffnc`).  Axes live here so `src/axes.rs` stays untouched.
use furiosa_opt_std::prelude::*;

// Token axes, one per compiled bucket.
axes![Tk16 = 16, Tk32 = 32, Tk128 = 128, Tk256 = 256];
// The T = 64 / T = 8 token axes and the residual-stream chunk axes ARE the qkvc ones (same types), so one `[T, 32, 128]` HBM tensor feeds
// the QKV kernels (reading 30 chunks through a tile view) and the AO / FFN kernels without a conversion.
pub use crate::axes::{Kc as C32, Ke as E32, Tb as Tk8, Tc as Tk64};
// Sliding/full AO (fp8): K split 32 (chunk 128 / 256), 16 row groups, 3 row tiles of 80 rows; H = [G32, Hi3, Rt80].
axes![G32 = 16, E256 = 256, Rt80 = 80, Hi3 = 3];
// FFN up/gate (NVFP4): K = 3840 padded to 4096 (32 chunks of 128, 8 scale blocks each), 4 row tiles of 240 rows per group.
axes![Fb8 = 8, Rt240 = 240, Fi4 = 4, Dummy32 = 32];
// FFN GeGLU output / FFN down input chunk: 480 real columns per chunk (= 2 up tiles), stored as [Fh2, Rt256] with 16 pad columns per half.
axes![Fc2 = 2, Fh2 = 2, Rt256 = 256];
// FFN down (NVFP4): 5 row tiles of 48 rows per group, 16 scale blocks per half chunk.
axes![Rt48 = 48, Fi5 = 5, Fb16 = 16];
// FFN down with the K split over 64 slices (chunk 256) and 4 row groups per cluster (8 in total): 5 row tiles of 96 rows per group, `h` replicated 8x instead of 16x.
axes![Gd8 = 8, C64 = 64, Rt96 = 96];

// padded residual row (4096 columns), part-layout epilogue axes (P parts of a token row: Pd8 / Pd4 / Pd2 = broadcast axes of the all-reduce)
axes![Hp = 4096, Pd8 = 8, Pd4 = 4, Pd2 = 2];

pub(crate) mod ao;
pub(crate) mod normp;
pub(crate) mod ffn;
pub(crate) mod norm;
