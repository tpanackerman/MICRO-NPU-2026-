//! `#[device]` entry points of the lossy FP8 weight fold (`S2_FP8`, default ON in fold1; kernels in `src/device/f8/`).  Argument lists mirror `ops_ffnc::layer_{sliding,full}_t128` except that
//! `pre_w` is the unpadded `[H]` pre-FF norm weight, `up_s` / `gate_s` / `down_s` are the block scales divided by 8 (down in the fin6 8-row-group layout) and `ugs` / `ggs` / `dgs` are 8 x the
//! reciprocal global scales.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Rh};
use crate::device::f8::ffn;
use crate::device::ffnc::{C32, C64, E32, E256, Fb8, Fb16, Fi4, Fi5, G32, Gd8, Rt80, Rt96, Rt240, Rt256, Tk128};

macro_rules! entry_layer_f8f {
    ($name:ident, $helper:path, $E:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            attn: &HbmTensor<bf16, Chip, m![Tk128, C32, $E]>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_attn_norm: &HbmTensor<bf16, Chip, m![H]>,
            res_in: &HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, m![H]>,
            up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ugs: &HbmTensor<f32, Chip, m![1]>,
            gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ggs: &HbmTensor<f32, Chip, m![1]>,
            down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
            down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
            dgs: &HbmTensor<f32, Chip, m![1]>,
            post_ff: &HbmTensor<bf16, Chip, m![H]>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            res_out: &mut HbmTensor<bf16, Chip, m![Tk128, C32, E32]>,
        ) {
            $helper(
                device, attn, w0, w1, w2, s0, s1, s2, post_attn_norm, res_in, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff,
                layer_scalar, res_out,
            );
        }
    };
}
entry_layer_f8f!(layer_sliding_t128_f8fb, ffn::layer_sliding_f8fb, E32);
// b13s (slim build): layer_full_t128_f8fb removed (full layers never fold)

// R14 fixedcost: `layer_sliding_t256_f8fb` (per-pass f4 decode) replaced by `layer_sliding_t256_f8pb` (pre-decoded e4m3 weights, bit-identical).

/// R14 fixedcost PREBAKE: `layer_sliding_t256_f8fb` on pre-decoded e4m3 FFN weights (`prebake_up8` / `prebake_down8` outputs). Bit-identical TRF contents => same results.
#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn layer_sliding_t256_f8pb(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, E32]>,
    s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
    post_attn_norm: &HbmTensor<bf16, Chip, m![H]>,
    res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
    pre_w: &HbmTensor<bf16, Chip, m![H]>,
    up_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ugs: &HbmTensor<f32, Chip, m![1]>,
    gate_w8: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    ggs: &HbmTensor<f32, Chip, m![1]>,
    down_w8: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    dgs: &HbmTensor<f32, Chip, m![1]>,
    post_ff: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
) {
    ffn::layer_sliding_f8pb_pair(
        device, attn, w0, w1, w2, s0, s1, s2, post_attn_norm, res_in, pre_w, up_w8, ugs, gate_w8, ggs, down_w8, dgs, post_ff, layer_scalar, res_out,
    );
}

/// R14 fixedcost PREBAKE (load time): decode an up / gate weight matrix (all 4 `Fi4` tiles) to e4m3 exactly as `ffu_dec8` does at run time.
#[device(chip = 1)]
pub fn prebake_up8(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
    s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
    out: &mut HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, E32]>,
) {
    ffn::prebake_up8_tile(device, w, s, out, 0);
    ffn::prebake_up8_tile(device, w, s, out, 1);
    ffn::prebake_up8_tile(device, w, s, out, 2);
    ffn::prebake_up8_tile(device, w, s, out, 3);
}

/// R14 fixedcost PREBAKE (load time): decode a down weight matrix (all 5 `Fi5` tiles) to e4m3 exactly as `ffd_dec8` does at run time.
#[device(chip = 1)]
pub fn prebake_down8(
    device: &mut Device,
    w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    out: &mut HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
) {
    ffn::prebake_down8_tile(device, w, s, out, 0);
    ffn::prebake_down8_tile(device, w, s, out, 1);
    ffn::prebake_down8_tile(device, w, s, out, 2);
    ffn::prebake_down8_tile(device, w, s, out, 3);
    ffn::prebake_down8_tile(device, w, s, out, 4);
}
