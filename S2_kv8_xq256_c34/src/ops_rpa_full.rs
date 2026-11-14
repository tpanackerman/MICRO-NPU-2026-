//! Ragged attention, FULL layers. The pool typing forbids one launch: the chunk-attention kernels gather 8-row blocks from the pool viewed as `[Pq, R8, Df]`
//! (3x cheaper than a row gather), the decode kernels gather / scatter single rows of the same pool viewed as `[Pr, Df]`, and one device function cannot hold both views of one
//! tensor (`reshape` consumes the tensor). So a hybrid iteration costs TWO launches per full layer:
//!   1. `full_ragged_prefill_s{2,4}_p{2,4}`: scatter of the staged chunk rows (whole 8-row blocks, chunks are 64-row aligned) + attention of S 64-row units, units in pairs on the 2 clusters;
//!   2. the existing fused decode launch (`ops_dattn::full_decode_attention_5_fused` for 8 rows, `_x2` for 16 rows): row append + 5-page decode attention.
//! The two launches touch disjoint requests (a request is either prefilling or decoding), so their order does not matter.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::rpa;

/// S = 2 units (one wave), 2-page bucket (all positions < 1024).
#[device(chip = 1)]
pub fn full_ragged_prefill_s2_p2(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_chunk: &HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    append_p: &HbmTensor<i32, Chip, m![Pb2]>,
    index_p: &HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
) {
    rpa::full_pair::ragged_full_append_2(device, k_chunk, append_p, k_pool);
    rpa::full_pair::ragged_full_append_2(device, v_chunk, append_p, v_pool);
    rpa::full_pair::ragged_full_pair_2(device, q_p, k_pool, v_pool, index_p, mask_p, pen_p, out_p);
}

/// S = 2 units (one wave), 4-page bucket (positions < 2048).
#[device(chip = 1)]
pub fn full_ragged_prefill_s2_p4(
    device: &mut Device,
    q_p: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_chunk: &HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Pb2, R8, Df]>,
    append_p: &HbmTensor<i32, Chip, m![Pb2]>,
    index_p: &HbmTensor<i32, Chip, m![Pw, Hg, Seg4, Bk]>,
    mask_p: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf64]>,
    pen_p: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf64]>,
    out_p: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
) {
    rpa::full_pair::ragged_full_append_2(device, k_chunk, append_p, k_pool);
    rpa::full_pair::ragged_full_append_2(device, v_chunk, append_p, v_pool);
    rpa::full_pair::ragged_full_pair_4(device, q_p, k_pool, v_pool, index_p, mask_p, pen_p, out_p);
}

/// S = 4 units (two waves), 2-page bucket.
#[device(chip = 1)]
pub fn full_ragged_prefill_s4_p2(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_chunk: &HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    append_p: &HbmTensor<i32, Chip, m![Pb4]>,
    index_p0: &HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>,
    index_p1: &HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
) {
    rpa::full_pair::ragged_full_append_4(device, k_chunk, append_p, k_pool);
    rpa::full_pair::ragged_full_append_4(device, v_chunk, append_p, v_pool);
    rpa::full_pair::ragged_full_pair_2(device, q_p0, k_pool, v_pool, index_p0, mask_p0, pen_p0, out_p0);
    rpa::full_pair::ragged_full_pair_2(device, q_p1, k_pool, v_pool, index_p1, mask_p1, pen_p1, out_p1);
}

/// R14 attnre item E: `full_ragged_prefill_s4_p4` in the hg4 slice layout (`rpa::full_pair_h4`): K / V blocks gathered 4x instead of 8x. Tables:
/// `rpa_host::h4_index / h4_mask / h4_pen` (lane = 8-row block of a page, slot = (page, row)). Token-identical (row sum / PV reduction grouping changes).
#[device(chip = 1)]
pub fn full_ragged_prefill_s4_p4h(
    device: &mut Device,
    q_p0: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    q_p1: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_chunk: &HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    v_chunk: &HbmTensor<bf16, Chip, m![Pb4, R8, Df]>,
    append_p: &HbmTensor<i32, Chip, m![Pb4]>,
    index_p0: &HbmTensor<i32, Chip, m![Pw, Hg4, Seg4, Bk]>,
    index_p1: &HbmTensor<i32, Chip, m![Pw, Hg4, Seg4, Bk]>,
    mask_p0: &HbmTensor<bf16, Chip, m![Pw, Bk, Tc, Kq32]>,
    pen_p0: &HbmTensor<bf16, Chip, m![Pw, Bk, Kq32]>,
    mask_p1: &HbmTensor<bf16, Chip, m![Pw, Bk, Tc, Kq32]>,
    pen_p1: &HbmTensor<bf16, Chip, m![Pw, Bk, Kq32]>,
    out_p0: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    out_p1: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
) {
    rpa::full_pair::ragged_full_append_4(device, k_chunk, append_p, k_pool);
    rpa::full_pair::ragged_full_append_4(device, v_chunk, append_p, v_pool);
    rpa::full_pair_h4::ragged_full_pair_4h(device, q_p0, k_pool, v_pool, index_p0, mask_p0, pen_p0, out_p0);
    rpa::full_pair_h4::ragged_full_pair_4h(device, q_p1, k_pool, v_pool, index_p1, mask_p1, pen_p1, out_p1);
}
