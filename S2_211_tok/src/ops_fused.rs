//! `#[device]` entry of the FUSED sliding layer (`device/fused`): QKV + attention + AO + FFN of one sliding layer of the 256-row hybrid pass in ONE launch,
//! so the sliding layers between two full layers launch the same image back to back (the device reloads the image whenever it changes).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::fused::{self, Gs3, Sw, Tb, Tt12, Tt14, Tt18, Wkv, Wug, tb};
use crate::ops_ffnc::{C32, C64, E32, Fb8, Fb16, Fi4, Fi5, G32, Gd8, Hi3, Rt80, Rt96, Rt240, Rt256, Tk128, Tk256};

/// One sliding layer of the 256-row pass with 16 decode riders (the KVD-256 path): `res_in` -> `res_out`, K / V of the pass scattered into the ring
/// caches. `index{u}` = cache rows of unit u, `win{w}` = key window of wave w, `idx_d` = the decode rows' key rows; `small` / `tab` / the grouped weights
/// as packed by `host::fused`.
#[device(chip = 1)]
#[allow(clippy::too_many_arguments)]
pub fn sliding_layer_fused_b16(
    device: &mut Device,
    res_in: &HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>,
    res_out: &mut HbmTensor<bf16, Chip, m![Rh, Tk128, Kc, Ke]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    small: &HbmTensor<bf16, Chip, m![Sw]>,
    wq: &HbmTensor<f8e4m3, Chip, m![Rq4, Gg, Kc, Rt, Ke]>,
    wkv: &HbmTensor<f8e4m3, Chip, m![Wkv, Rq2, Gg, Kc, Rt, Ke]>,
    ao_w: &HbmTensor<f8e4m3, Chip, m![Hi3, G32, C32, Rt80, E32]>,
    ug_w: &HbmTensor<f4e2m1, Chip, m![Wug, G32, C32, Fi4, Rt240, E32]>,
    ug_s: &HbmTensor<f8e4m3, Chip, m![Wug, G32, C32, Fi4, Rt240, Fb8]>,
    down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
    down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
    gs3: &HbmTensor<f32, Chip, m![Gs3, Dummy8]>,
    tab: &HbmTensor<bf16, Chip, m![Tb]>,
    index0: &HbmTensor<i32, Chip, m![Tc]>,
    index1: &HbmTensor<i32, Chip, m![Tc]>,
    index2: &HbmTensor<i32, Chip, m![Tc]>,
    index3: &HbmTensor<i32, Chip, m![Tc]>,
    win0: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    win1: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    idx_d: &HbmTensor<i32, Chip, m![Bd16, Tx]>,
) {
    // ---- QKV: q -> `q` ([Rh = wave, Pw = unit of the wave]), K / V -> the ring caches ----
    let mut q: HbmTensor<bf16, Chip, m![Rh, Pw, Tc, Ns, Gs, Ds]> = HbmTensor::new();
    fused::qkv::qkv(device, res_in, &mut q, small, wq, wkv, tab, index0, index1, index2, index3, k_cache, v_cache);

    // ---- attention: wave 0, wave 1 (+ the 16 decode rows, stored through the same output handle) ----
    // [Rh = wave, Pw = unit, Rg4 x Bd16 = the 64 rows of the unit]: the layout of the wave-1 store; a write view cannot be reshaped
    let mut o: HbmTensor<bf16, Chip, m![Rh, Pw, Rg4, Bd16, Ns, Gs, Ds]> = HbmTensor::new();
    let masks: HbmTensorView<'_, bf16, Chip, m![Tt18, Pw, Ts % 32, Tc, Seg, Ts / 32]> = unsafe { tab.view().reshape() };
    let pens: HbmTensorView<'_, bf16, Chip, m![Tt12, Pw, Ts % 32, Seg, Ts / 32]> = unsafe { tab.view().reshape() };
    fused::attn::prefill_wave0(
        device,
        q.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Pw, Tc, Ns, Gs, Ds]>(0),
        &*k_cache,
        &*v_cache,
        win0,
        unsafe { tab.view().reshape::<Chip, m![Tt18, Pw, Ts % 32, Tc, Seg, Ts / 32]>() }.tile::<m![Tt18], 1, m![Tt18 = 1 # 3, Pw, Ts % 32, Tc, Seg, Ts / 32]>(tb::MASK),
        unsafe { tab.view().reshape::<Chip, m![Tt12, Pw, Ts % 32, Seg, Ts / 32]>() }.tile::<m![Tt12], 1, m![Tt12 = 1 # 192, Pw, Ts % 32, Seg, Ts / 32]>(tb::PEN),
        o.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Pw, Rg4, Bd16, Ns, Gs, Ds]>(0),
    );
    {
        fused::attn::prefill_wave1_r16(
            device,
            q.view().tile::<m![Rh], 1, m![Rh = 1 # 2, Pw, Tc, Ns, Gs, Ds]>(1),
            &*k_cache,
            &*v_cache,
            win1,
            masks.tile::<m![Tt18], 1, m![Tt18 = 1 # 3, Pw, Ts % 32, Tc, Seg, Ts / 32]>(tb::MASK + 1),
            pens.tile::<m![Tt12], 1, m![Tt12 = 1 # 192, Pw, Ts % 32, Seg, Ts / 32]>(tb::PEN + 1),
            o.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Pw, Rg4, Bd16, Ns, Gs, Ds]>(1),
        );
    }
    {
        let qd: HbmTensorView<'_, bf16, Chip, m![Rh, Pw, Rg4, Bd16, Ns, Gs, Ds]> = unsafe { q.view().reshape() };
        let mask_d: HbmTensorView<'_, bf16, Chip, m![Tt14, Bd16, Tx % 32, Tx / 32]> = unsafe { tab.view().reshape() };
        fused::attn::decode_attend16(
            device,
            qd.tile::<m![Rh], 1, m![Rh = 1 # 2, Pw, Rg4, Bd16, Ns, Gs, Ds]>(1)
                .tile::<m![Pw], 1, m![Rh = 1 # 2, Pw = 1 # 2, Rg4, Bd16, Ns, Gs, Ds]>(1)
                .tile::<m![Rg4], 1, m![Rh = 1 # 2, Pw = 1 # 2, Rg4 = 1 # 4, Bd16, Ns, Gs, Ds]>(3),
            &*k_cache,
            &*v_cache,
            idx_d,
            mask_d.tile::<m![Tt14], 1, m![Tt14 = 1 # 48, Bd16, Tx % 32, Tx / 32]>(tb::MASK_D),
            o.view_mut()
                .tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Pw, Rg4, Bd16, Ns, Gs, Ds]>(1)
                .tile::<m![Pw], 1, m![Rh = 1 #{!} 2, Pw = 1 #{!} 2, Rg4, Bd16, Ns, Gs, Ds]>(1)
                .tile::<m![Rg4], 1, m![Rh = 1 #{!} 2, Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ns, Gs, Ds]>(3),
        );
    }

    // ---- AO + FFN (t256r) ----
    crate::device::ffnc::ffn::f_t128::layer_sliding_fv(device, &o, ao_w, small, res_in, ug_w, ug_s, gs3, down_w, down_s, res_out);
}

