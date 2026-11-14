//! `#[device]` entry points of the batched image / audio encoder GEMMs (kernels in `src/device/mm/gemm.rs`).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::{Cluster, Slice};
use crate::device::mm::{gemm, norm};

#[device(chip = 1)]
pub fn mm_dense_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ev]>,
    w0: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    w1: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    b0: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    b1: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    y0: &mut HbmTensor<bf16, Chip, m![Mg, Tm64, Rt120]>,
    y1: &mut HbmTensor<bf16, Chip, m![Mg, Tm64, Rt120]>,
) {
    gemm::dense_t64(device, x, w0, w1, b0, b1, y0, y1);
}

#[device(chip = 1)]
pub fn mm_proj_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ep]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    gemm::proj_t64(device, x, w, y);
}

#[device(chip = 1)]
pub fn mm_audio_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ea]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    gemm::audio_t64(device, x, w, y);
}

/// Copies row `offset / (2 * H)` of a `[Tm64, H]` tile into its own `[H]` tensor (dynamic row offset, bytes).
#[device(chip = 1)]
pub fn mm_copy_row(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![Tm64, H]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = table
        .gather::<m![Tm64], m![H]>()
        .by_byte_offsets(offset)
        .to_dm(&mut device.tdma);
    row.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Stage A of the device encoder pipeline (see `device/mm/norm.rs`): LN1 of 64 pixel rows into the zero padded dense input.
#[device(chip = 1)]
pub fn mm_ln1_tile(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Rv]>,
    w: &HbmTensor<bf16, Chip, m![Rv]>,
    b: &HbmTensor<bf16, Chip, m![Rv]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
) {
    norm::ln1_tile(device, x, w, b, xn);
}

/// Stage C: dense output rows -> LN2 -> + position tile -> LN3 -> rmsnorm -> projection input (see `device/mm/norm.rs`).
#[device(chip = 1)]
pub fn mm_ln23_tile(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
    pos: &HbmTensor<bf16, Chip, m![Tm64, Mv]>,
    w2: &HbmTensor<bf16, Chip, m![Mv]>,
    b2: &HbmTensor<bf16, Chip, m![Mv]>,
    w3: &HbmTensor<bf16, Chip, m![Mv]>,
    b3: &HbmTensor<bf16, Chip, m![Mv]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
) {
    norm::ln23_tile(device, unsafe { y.view().reshape::<Chip, m![Tm64, Mv]>() }, pos, w2, b2, w3, b3, xn);
}

/// Audio: rmsnorm of 64 frames -> audio projection input.
#[device(chip = 1)]
pub fn mm_rms_audio_tile(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Aa]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
) {
    norm::rms_audio_tile(device, x, xn);
}

/// Dense (K = 6912 -> 3840, + bias) on a token-major x[Tm64, Mc, Ev], natural output [Tm64, 3840].
#[device(chip = 1)]
pub fn mm_dense_nat(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
    w0: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    w1: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    b0: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    b1: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
) {
    gemm::dense_t64_nat(device, unsafe { x.view().reshape::<Chip, m![Tm64, Mc, Ev]>() }, w0, w1, b0, b1, y);
}

/// Projection (3840 -> 3840) on a token-major x[Tm64, Mc, Ep].
#[device(chip = 1)]
pub fn mm_proj_tm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    gemm::proj_t64_tm(device, x, w, y);
}

/// Audio projection (640 -> 3840) on a token-major x[Tm64, Mc, Ea].
#[device(chip = 1)]
pub fn mm_audio_tm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    gemm::audio_t64_tm(device, x, w, y);
}

/// The whole image tile in ONE launch: LN1 -> dense (+ bias) -> LN2 + position -> LN3 -> rmsnorm -> projection (see `device/mm/norm.rs`, `gemm.rs`).
/// `xn1` / `y` / `xn2` are persistent HBM scratch (xn1 and xn2 pad columns must be zero: they are never written).
#[device(chip = 1)]
pub fn mm_image_tile(
    device: &mut Device,
    px: &HbmTensor<bf16, Chip, m![Tm64, Rv]>,
    pos: &HbmTensor<bf16, Chip, m![Tm64, Mv]>,
    ln1_w: &HbmTensor<bf16, Chip, m![Rv]>,
    ln1_b: &HbmTensor<bf16, Chip, m![Rv]>,
    w0: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    w1: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    b0: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    b1: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    ln2_w: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_b: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_w: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_b: &HbmTensor<bf16, Chip, m![Mv]>,
    proj_w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tm64, Rv # 7168]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
    xn2: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
    out: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    norm::ln1_tile(device, px, ln1_w, ln1_b, xn1);
    gemm::dense_t64_nat(device, unsafe { xn1.view().reshape::<Chip, m![Tm64, Mc, Ev]>() }, w0, w1, b0, b1, y);
    norm::ln23_tile(device, unsafe { y.view().reshape::<Chip, m![Tm64, Mv]>() }, pos, ln2_w, ln2_b, ln3_w, ln3_b, xn2);
    gemm::proj_t64_tm(device, xn2, proj_w, out);
}

/// The whole audio tile in ONE launch: rmsnorm (no weight) -> audio projection.
#[device(chip = 1)]
pub fn mm_audio_tile(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Aa]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
    out: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    norm::rms_audio_tile(device, x, xn);
    gemm::audio_t64_tm(device, xn, w, out);
}
