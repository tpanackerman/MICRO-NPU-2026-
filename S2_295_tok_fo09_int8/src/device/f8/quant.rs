//! Constants shared by the dynamic-FP8 kernels.
/// 1 / (largest e4m3 magnitude 448): the per-token scale is `amax / 448` (+ tiny, so an all-zero row gives a finite scale).
pub(crate) const QINV: f32 = 1.0 / 448.0;
pub(crate) const TINY: f32 = 1e-30;
