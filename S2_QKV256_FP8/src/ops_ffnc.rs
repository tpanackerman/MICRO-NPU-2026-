//! `#[device]` entry points of the chunk AO / FFN kernels (kernels in `src/device/ffnc/`).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Rh};
use crate::device::ffnc::ao;
pub use crate::device::ffnc::{C32, C64, E32, E256, Fb8, Fb16, Fc2, Fh2, Fi4, Fi5, G32, Gd8, Hi3, Rt48, Rt80, Rt96, Rt240, Rt256, Tk8, Tk16, Tk32, Tk64, Tk128, Tk256};
use crate::device::ffnc::ffn;

macro_rules! entry_ao {
    ($name:ident, $helper:path, $T:ident, $E:ident, $XL:ty) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, $XL>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
        ) {
            $helper(device, x, w0, w1, w2, s0, s1, s2, post_norm, res_in, res_out);
        }
    };
}

macro_rules! entry_ffn_a {
    ($name:ident, $helper:path, $T:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
            uw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            uw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            uw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            uw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            us0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            us1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            us2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            us3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            ugs: &HbmTensor<f32, Chip, m![1]>,
            gw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            gw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            gw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            gw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
            gs0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            gs1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            gs2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            gs3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
            ggs: &HbmTensor<f32, Chip, m![1]>,
            h_out: &mut HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]>,
        ) {
            $helper(device, res_in, pre_w, uw0, uw1, uw2, uw3, us0, us1, us2, us3, ugs, gw0, gw1, gw2, gw3, gs0, gs1, gs2, gs3, ggs, h_out);
        }
    };
}
entry_ffn_a!(ffn_a_t64, ffn::f_t64::ffn_a, Tk64);
entry_ffn_a!(ffn_a_t8, ffn::f_t8::ffn_a, Tk8);

macro_rules! entry_ffn_b {
    ($name:ident, $helper:path, $T:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            h: &HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]>,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            post_w: &HbmTensor<bf16, Chip, m![H]>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            dw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
            dw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
            dw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
            dw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
            dw4: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
            ds0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
            ds1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
            ds2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
            ds3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
            ds4: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
            dgs: &HbmTensor<f32, Chip, m![1]>,
            res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
        ) {
            $helper(device, h, res_in, post_w, layer_scalar, dw0, dw1, dw2, dw3, dw4, ds0, ds1, ds2, ds3, ds4, dgs, res_out);
        }
    };
}
entry_ffn_b!(ffn_b_t64, ffn::f_t64::ffn_b, Tk64);
entry_ffn_b!(ffn_b_t8, ffn::f_t8::ffn_b, Tk8);

macro_rules! entry_ffn_chunk {
    ($name:ident, $helper:path, $T:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
            up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ugs: &HbmTensor<f32, Chip, m![1]>,
            gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
            gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
            ggs: &HbmTensor<f32, Chip, m![1]>,
            down_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi5, Rt48, Fh2, Rt256]>,
            down_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi5, Rt48, Fh2, Fb16]>,
            dgs: &HbmTensor<f32, Chip, m![1]>,
            post_w: &HbmTensor<bf16, Chip, m![H]>,
            layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
        ) {
            $helper(device, res_in, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_w, layer_scalar, res_out);
        }
    };
}
entry_ffn_chunk!(ffn_chunk_t64, ffn::f_t64::ffn_chunk, Tk64);
entry_ffn_chunk!(ffn_chunk_t8, ffn::f_t8::ffn_chunk, Tk8);
entry_ffn_a!(ffn_a_t128, ffn::f_t128::ffn_a, Tk128);
entry_ffn_b!(ffn_b_t128, ffn::f_t128::ffn_b, Tk128);
entry_ffn_chunk!(ffn_chunk_t128, ffn::f_t128::ffn_chunk, Tk128);
entry_ffn_a!(ffn_a_t16, ffn::f_t16::ffn_a, Tk16);
entry_ffn_a!(ffn_a_t32, ffn::f_t32::ffn_a, Tk32);
entry_ffn_b!(ffn_b_t16, ffn::f_t16::ffn_b, Tk16);
entry_ffn_b!(ffn_b_t32, ffn::f_t32::ffn_b, Tk32);
entry_ffn_chunk!(ffn_chunk_t16, ffn::f_t16::ffn_chunk, Tk16);
entry_ffn_chunk!(ffn_chunk_t32, ffn::f_t32::ffn_chunk, Tk32);
entry_ao!(sliding_ao_t8, ao::sliding_ao_t8, Tk8, E32, m![Tk8, C32, E32]);
entry_ao!(sliding_ao_t16, ao::sliding_ao_t16, Tk16, E32, m![Tk16, C32, E32]);
entry_ao!(sliding_ao_t32, ao::sliding_ao_t32, Tk32, E32, m![Tk32, C32, E32]);
entry_ao!(sliding_ao_t64, ao::sliding_ao_t64, Tk64, E32, m![Tk64, C32, E32]);
entry_ao!(sliding_ao_t128, ao::sliding_ao_t128, Tk128, E32, m![Tk128, C32, E32]);
entry_ao!(sliding_ao_t256, ao::sliding_ao_t256, Tk256, E32, m![Tk256, C32, E32]);
entry_ao!(full_ao_t8, ao::full_ao_t8, Tk8, E256, m![Tk8, C32, E256]);
entry_ao!(full_ao_t16, ao::full_ao_t16, Tk16, E256, m![Tk16, C32, E256]);
entry_ao!(full_ao_t32, ao::full_ao_t32, Tk32, E256, m![Tk32, C32, E256]);
entry_ao!(full_ao_t64, ao::full_ao_t64, Tk64, E256, m![Tk64, C32, E256]);
entry_ao!(full_ao_t128, ao::full_ao_t128, Tk128, E256, m![Tk128, C32, E256]);
entry_ao!(full_ao_t256, ao::full_ao_t256, Tk256, E256, m![Tk256, C32, E256]);

/// AO + FFN of a chunk in one launch (`E` = E32 sliding / E256 full).
macro_rules! entry_layer {
    ($name:ident, $helper:path, $T:ident, $E:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            attn: &HbmTensor<bf16, Chip, m![$T, C32, $E]>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_attn_norm: &HbmTensor<bf16, Chip, m![H]>,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
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
            res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
        ) {
            $helper(
                device, attn, w0, w1, w2, s0, s1, s2, post_attn_norm, res_in, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff,
                layer_scalar, res_out,
            );
        }
    };
}
entry_layer!(layer_sliding_t8, ffn::f_t8::layer_sliding, Tk8, E32);
entry_layer!(layer_full_t8, ffn::f_t8::layer_full, Tk8, E256);
entry_layer!(layer_sliding_t16, ffn::f_t16::layer_sliding, Tk16, E32);
entry_layer!(layer_full_t16, ffn::f_t16::layer_full, Tk16, E256);
entry_layer!(layer_sliding_t32, ffn::f_t32::layer_sliding, Tk32, E32);
entry_layer!(layer_full_t32, ffn::f_t32::layer_full, Tk32, E256);
entry_layer!(layer_sliding_t64, ffn::f_t64::layer_sliding, Tk64, E32);
entry_layer!(layer_full_t64, ffn::f_t64::layer_full, Tk64, E256);
entry_layer!(layer_sliding_t128, ffn::f_t128::layer_sliding, Tk128, E32);
entry_layer!(layer_full_t128, ffn::f_t128::layer_full, Tk128, E256);

/// (T = 256) AO + FFN of TWO 128-token halves in ONE launch, bf16 FFN (`E` = E32 sliding / E256 full); the activation tensors are `[Rh, Tk128, ..]` (= 256 consecutive rows).
macro_rules! entry_layer_pair {
    ($name:ident, $helper:path, $E:ident) => {
        #[device(chip = 1)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            attn: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, $E]>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_attn_norm: &HbmTensor<bf16, Chip, m![H]>,
            res_in: &HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
            pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
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
            res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, C32, E32]>,
        ) {
            $helper(
                device, attn, w0, w1, w2, s0, s1, s2, post_attn_norm, res_in, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff, layer_scalar, res_out,
            );
        }
    };
}
entry_layer_pair!(layer_sliding_t256p, ffn::f_t128::layer_sliding_pair, E32);
entry_layer_pair!(layer_full_t256p, ffn::f_t128::layer_full_pair, E256);
