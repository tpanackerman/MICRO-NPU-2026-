//! `#[device]` entry points of the chunk QKV kernels (kernels in `src/device/qkvc/`).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::qkvc::{full, gemm, pair, pair2, quad, quad_full, sliding, t8, xnorm};

#[device(chip = 1)]
pub fn qk_input_norm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
) {
    xnorm::input_norm(device, x, rms_weight, xn);
}

/// GEMM-stage test: raw Q projection of the normalised activations, output [Tc, Gg, Ds] (token-major, head-major inside).
#[device(chip = 1)]
pub fn qk_q_gemm(
    device: &mut Device,
    xn: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gg, Ds]>,
) {
    let x_dm = gemm::load_x(device, xn);
    let sq = gemm::load_s_q(device, sq);
    let t0 = gemm::load_w_q(device, wq, 0);
    let t1 = gemm::load_w_q(device, wq, 1);
    let t2 = gemm::load_w_q(device, wq, 2);
    let t3 = gemm::load_w_q(device, wq, 3);
    let mut y: DmTensor<bf16, Chip, gemm::Cl, gemm::SlOut, m![Tc, Rq4, Rt]> = DmTensor::new();
    gemm::proc_tile_q(device, &x_dm, t0, &sq, &mut y, 0);
    gemm::proc_tile_q(device, &x_dm, t1, &sq, &mut y, 1);
    gemm::proc_tile_q(device, &x_dm, t2, &sq, &mut y, 2);
    gemm::proc_tile_q(device, &x_dm, t3, &sq, &mut y, 3);
    let tok: DmTensor<bf16, Chip, gemm::Cl, m![Tc, 1 # 4], m![Gg % 8, Rq4, Rt]> = y.to_dm(&mut device.tdma);
    let tokv: DmTensorView<'_, bf16, Chip, gemm::Cl, m![Tc, 1 # 4], m![Gg % 8, Ds]> = unsafe { tok.view().reshape() };
    tokv.to_hbm_view(&mut device.tdma, out.view_mut());
}


/// Sliding-layer chunk QKV (T = Tc = 64 rows): x[Tc, H] (residual-stream rows; rows may be 64 consecutive positions of one request or 64 rows of different
/// requests) -> q [Tc, Ns, Gs, Ds] (norm + RoPE), k_stage / v_stage [Ts = 64, Ns, Ds] (k norm + RoPE, v norm; row t = input row t), the chunk attention's
/// consumer layout.  `cos` / `sin` are the baseline table rows of each ROW's position: row t = `ws.cos_s[pos_t]` / `ws.sin_s[pos_t]` (bf16 [Tc, Ds]; any
/// positions, assembled by the host once per iteration, shared by all 40 sliding layers).  `xn` is a zero-initialised scratch (see host::qkvc).
#[device(chip = 1)]
pub fn sliding_qkv_chunk(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    sliding::sliding_qkv(device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// Sliding-layer QKV of BOTH 64-row halves of a 128-row residual in ONE launch (the weights are streamed / decoded / loaded into the TRF once for 128 rows): x = the residual viewed as
/// `[Rh, Tc, Kc, Ke]`, `xn0` / `xn1` = zero-initialised scratches of the two halves, per-half cos / sin tables and per-half 64-row outputs (the stage pieces). Same arithmetic as two
/// `sliding_qkv_chunk` launches (bit-identical results expected).
#[device(chip = 1)]
pub fn sliding_qkv_pair(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn0: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    pair::sliding_qkv_pair(
        device, x, input_rms_weight, xn0, xn1, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos0, sin0, cos1, sin1, q_out0, k_out0, v_out0, q_out1, k_out1, v_out1,
    );
}

/// KVD: `sliding_qkv_pair` that scatters the K / V rows of both blocks straight into the flat sliding ring caches (`index0` / `index1`: byte offset of the cache row of each token of the block).
#[device(chip = 1)]
pub fn sliding_qkv_pair_direct(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn0: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    pair::sliding_qkv_pair_direct(
        device, x, input_rms_weight, xn0, xn1, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos0, sin0, cos1, sin1, q_out0, q_out1, index0, index1, k_cache, v_cache,
    );
}

/// Joint pair QKV: `sliding_qkv_pair` with ONE input-norm pass and ONE 128-token contraction per weight tile; both blocks share one zero-initialised scratch `xn` = [Rh, Tc, Kc, Ke].
#[device(chip = 1)]
pub fn sliding_qkv_pair_j(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out0: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_out1: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
) {
    pair2::sliding_qkv_pair_j(
        device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos0, sin0, cos1, sin1, q_out0, k_out0, v_out0, q_out1, k_out1, v_out1,
    );
}

/// Joint KVD pair QKV: `sliding_qkv_pair_direct` with the joint norm / contraction of `sliding_qkv_pair_j`.
#[device(chip = 1)]
pub fn sliding_qkv_pair_direct_j(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    pair2::sliding_qkv_pair_direct_j(
        device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos0, sin0, cos1, sin1, q_out0, q_out1, index0, index1, k_cache, v_cache,
    );
}

/// R13 qkv256: QUAD KVD sliding QKV, the four 64-row blocks of the 256-row hybrid pass in ONE launch (every weight tile decoded and loaded into the TRF once for 256 tokens).
/// `cos` / `sin` = the rope rows of the 256 rows `[Ub, Tc, Ds]`, `q_out` = the 256 q rows, `index0..3` = cache row byte offsets of the four blocks (as `sliding_qkv_pair_direct_j`).
#[device(chip = 1)]
pub fn sliding_qkv_quad_direct(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Ns, Gs, Ds]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    index2: &HbmTensor<i32, Chip, m![Tc]>,
    index3: &HbmTensor<i32, Chip, m![Tc]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    quad::sliding_qkv_quad_direct(
        device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, index0, index1, index2, index3, k_cache, v_cache,
    );
}

/// Full-attention-layer chunk QKV (T = 64 rows): x[Tc, H] -> q [Tc, Gf, Df] (norm + partial RoPE), k_chunk [Tc, Df] (norm + RoPE), v_chunk [Tc, Df] (the K
/// projection normalised without weight, no RoPE).  `cos` / `sin` = per-row rows of the full tables `ws.cos_f[pos_t]` / `ws.sin_f[pos_t]` ([Tc, Df]).
#[device(chip = 1)]
pub fn full_qkv_chunk(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
) {
    full::full_qkv(device, x, input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// R13 qkv256: QUAD full-layer QKV, the four 64-row blocks of the 256-row hybrid pass in ONE launch (`quad_full::full_qkv_quad`; outputs are the 256 stage rows).
#[device(chip = 1)]
pub fn full_qkv_quad(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Ub, Tc, Df]>,
) {
    // R14 qkvre QHF (b14/big): the half-outer schedule replaces the b11 quad body in this entry (bit-identical per token; no extra device entry)
    crate::device::qkvc::quadhf::full_qkv_quad_h(device, x.view(), input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// T = 8 bucket of `sliding_qkv_chunk` (decode batches of up to 8 rows; unused rows may repeat any valid row).  Outputs [Tb, ...].
#[device(chip = 1)]
pub fn sliding_qkv_t8(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
) {
    t8::sliding::sliding_qkv(device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

/// T = 8 bucket of `full_qkv_chunk`.
#[device(chip = 1)]
pub fn full_qkv_t8(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    cos: &HbmTensor<bf16, Chip, m![Tb, Df]>,
    sin: &HbmTensor<bf16, Chip, m![Tb, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    k_out: &mut HbmTensor<bf16, Chip, m![Tb, Df]>,
    v_out: &mut HbmTensor<bf16, Chip, m![Tb, Df]>,
) {
    t8::full::full_qkv(device, x, input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
}

// ---- T = 128 chunks: two 64-row QKV passes over the halves of one 128-row residual `[Rh, Tc, Kc, Ke]` (outputs are the usual 64-row tensors) ----
macro_rules! entry_qkv_half {
    ($name:ident, $body:path) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
            input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
            xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
            wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
            sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
            wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
            sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
            wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
            sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
            q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
            k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
            cos: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
            sin: &HbmTensor<bf16, Chip, m![Tc, Ds]>,
            q_out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
            k_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
            v_out: &mut HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
        ) {
            $body(device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
entry_qkv_half!(sliding_qkv_chunk_h0, sliding::sliding_qkv_h0);
entry_qkv_half!(sliding_qkv_chunk_h1, sliding::sliding_qkv_h1);

macro_rules! entry_qkv_half_full {
    ($name:ident, $body:path) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
            input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
            xn: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
            wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
            sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
            wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
            sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
            q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
            k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
            cos: &HbmTensor<bf16, Chip, m![Tc, Df]>,
            sin: &HbmTensor<bf16, Chip, m![Tc, Df]>,
            q_out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
            k_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
            v_out: &mut HbmTensor<bf16, Chip, m![Tc, Df]>,
        ) {
            $body(device, x, input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
entry_qkv_half_full!(full_qkv_chunk_h0, full::full_qkv_h0);
entry_qkv_half_full!(full_qkv_chunk_h1, full::full_qkv_h1);

// ---- T = 16 decode: two 8-row QKV passes over the halves of one 16-row residual `[Rh, Tb, Kc, Ke]` ----
macro_rules! entry_qkv8_half {
    ($name:ident, $body:path) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, Tb, Kc, Ke]>,
            input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
            xn: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
            wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
            sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
            wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
            sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
            wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
            sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
            q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
            k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
            cos: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
            sin: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
            q_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
            k_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
            v_out: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
        ) {
            $body(device, x, input_rms_weight, xn, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
entry_qkv8_half!(sliding_qkv_t8_h0, t8::sliding::sliding_qkv_h0);
entry_qkv8_half!(sliding_qkv_t8_h1, t8::sliding::sliding_qkv_h1);

macro_rules! entry_qkv8_half_full {
    ($name:ident, $body:path) => {
        #[device(chip = 1)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, Tb, Kc, Ke]>,
            input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
            xn: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
            wq: &HbmTensor<f8e4m3, Chip, m![Rq8, Gg, Kc, Rt, Ke]>,
            sq: &HbmTensor<bf16, Chip, m![Rq8, Gg, Rt]>,
            wk: &HbmTensor<f8e4m3, Chip, m![Gg, Kc, Rk, Ke]>,
            sk: &HbmTensor<bf16, Chip, m![Gg, Rk]>,
            q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
            k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
            cos: &HbmTensor<bf16, Chip, m![Tb, Df]>,
            sin: &HbmTensor<bf16, Chip, m![Tb, Df]>,
            q_out: &mut HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
            k_out: &mut HbmTensor<bf16, Chip, m![Tb, Df]>,
            v_out: &mut HbmTensor<bf16, Chip, m![Tb, Df]>,
        ) {
            $body(device, x, input_rms_weight, xn, wq, sq, wk, sk, q_rms_weight, k_rms_weight, cos, sin, q_out, k_out, v_out);
        }
    };
}
entry_qkv8_half_full!(full_qkv_t8_h0, t8::full::full_qkv_h0);
entry_qkv8_half_full!(full_qkv_t8_h1, t8::full::full_qkv_h1);

/// Decode-only QKV pair; cache writes remain in the subsequent attention launch.
#[device(chip = 1)]
pub fn sliding_qkv_pair_t8(
    device: &mut Device,
    res0: &HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    res1: &HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    xn0: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    xn1: &mut HbmTensor<bf16, Chip, m![Tb, Kc, Ke]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    sq: &HbmTensor<bf16, Chip, m![Rq4, Gg, Rt]>,
    wk: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sk: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    wv: &HbmTensor<f8e4m3, Chip, m![Rq2, Gg, Kc, Rt, Ke]>,
    sv: &HbmTensor<bf16, Chip, m![Rq2, Gg, Rt]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    cos0: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    sin0: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    cos1: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    sin1: &HbmTensor<bf16, Chip, m![Tb, Ds]>,
    q_out0: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    k_out0: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    v_out0: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    q_out1: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    k_out1: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
    v_out1: &mut HbmTensor<bf16, Chip, m![Tb, Ns, Ds]>,
) {
    t8::sliding::sliding_qkv_pair(device, res0, res1, input_rms_weight, xn0, xn1, wq, sq, wk, sk, wv, sv, q_rms_weight, k_rms_weight, cos0, sin0, cos1, sin1, q_out0, k_out0, v_out0, q_out1, k_out1, v_out1);
}
