//! Chunked-prefill attention entry points (one launch = one chunk of `Tc` consecutive positions of
//! one request, one layer). See RESULTS.md.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::chunk;

#[device(chip = 1)]
pub fn sliding_chunk_attention(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_ring: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_ring: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    k_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    mask: &HbmTensor<bf16, Chip, m![Ts % 32, Tc, Seg, Ts / 32]>,
    pen: &HbmTensor<bf16, Chip, m![Ts % 32, Seg, Ts / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
) {
    chunk::sliding_chunk::sliding_chunk_attend(device, q, k_ring, v_ring, k_stage, v_stage, mask, pen, out);
}

/// Appends the chunk's staged k/v rows into the sliding ring. `index[c]` = ring_row(p0 + c) * Ns * Ds * 2 (bytes).
#[device(chip = 1)]
pub fn sliding_chunk_append(
    device: &mut Device,
    k_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Ts = 64]>,
    k_ring: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_ring: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    chunk::append::append_rows(device, k_stage, index, k_ring);
    chunk::append::append_rows(device, v_stage, index, v_ring);
}

/// Full-layer chunk attention over at most two 512-key pages (page 0 = k0/v0 first). Output is normalised.
#[device(chip = 1)]
pub fn full_chunk_attention_2g(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    chunk::full_chunk::full_chunk_attend_2g(device, q, k0, k1, v0, v1, mask, out);
}

/// Variant of `full_chunk_attention_2` that loads K/V pages by replicated DMA (no switch gather).
#[device(chip = 1)]
pub fn full_chunk_attention_2(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>,
    pen: &HbmTensor<bf16, Chip, m![Tf % 32, Kf]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    chunk::full_chunk::full_chunk_attend_2(device, q, k0, k1, v0, v1, mask, pen, out);
}

/// Four-page full-layer chunk attention (page order 0..3), normalised output.
#[device(chip = 1)]
pub fn full_chunk_attention_4(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k2: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k3: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v2: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v3: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf64]>,
    pen: &HbmTensor<bf16, Chip, m![Tf % 32, Kf64]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    chunk::full_chunk::full_chunk_attend_4(device, q, k0, k1, k2, k3, v0, v1, v2, v3, mask, pen, out);
}

/// Appends the chunk's full-layer k/v rows [Tc, Df] into the current page: `index[c]` = (row_in_page(p0 + c)) * Df * 2 (bytes).
#[device(chip = 1)]
pub fn full_chunk_append(
    device: &mut Device,
    k_chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    k_page: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_page: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    chunk::append::append_full_rows(device, k_chunk, index, k_page);
    chunk::append::append_full_rows(device, v_chunk, index, v_page);
}
