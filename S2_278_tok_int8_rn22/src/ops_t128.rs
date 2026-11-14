//! T = 128 chunk plumbing: two 64-row attention outputs -> one 128-row attention input for the T = 128 AO + FFN kernels.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Ds, Gf, Gs, Ns, Rh, Tb, Tc};

/// out[0] = a, out[1] = b (sliding attention outputs `[Tc, Ns, Gs, Ds]`).
#[device(chip = 1)]
pub fn cat_attn_s(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    b: &HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    out: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Ns, Gs, Ds]>,
) {
    let da: DmTensor<bf16, Chip, m![1 # 2], m![Tc, 1 # 4], m![Ns, Gs, Ds]> = a.to_dm(&mut device.tdma);
    da.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tc, Ns, Gs, Ds]>(0));
    let db: DmTensor<bf16, Chip, m![1 # 2], m![Tc, 1 # 4], m![Ns, Gs, Ds]> = b.to_dm(&mut device.tdma);
    db.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tc, Ns, Gs, Ds]>(1));
}

/// Same for full-layer attention outputs `[Tc, Gf, Df]`.
#[device(chip = 1)]
pub fn cat_attn_f(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    b: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    out: &mut HbmTensor<bf16, Chip, m![Rh, Tc, Gf, Df]>,
) {
    let da: DmTensor<bf16, Chip, m![1 # 2], m![Tc, 1 # 4], m![Gf, Df]> = a.to_dm(&mut device.tdma);
    da.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tc, Gf, Df]>(0));
    let db: DmTensor<bf16, Chip, m![1 # 2], m![Tc, 1 # 4], m![Gf, Df]> = b.to_dm(&mut device.tdma);
    db.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tc, Gf, Df]>(1));
}

// ---- T = 16 decode: two 8-row attention outputs -> one 16-row AO input ----
#[device(chip = 1)]
pub fn cat8_attn_s(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    b: &HbmTensor<bf16, Chip, m![Tb, Ns, Gs, Ds]>,
    out: &mut HbmTensor<bf16, Chip, m![Rh, Tb, Ns, Gs, Ds]>,
) {
    let da: DmTensor<bf16, Chip, m![1 # 2], m![Tb, 1 # 32], m![Ns, Gs, Ds]> = a.to_dm(&mut device.tdma);
    da.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tb, Ns, Gs, Ds]>(0));
    let db: DmTensor<bf16, Chip, m![1 # 2], m![Tb, 1 # 32], m![Ns, Gs, Ds]> = b.to_dm(&mut device.tdma);
    db.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tb, Ns, Gs, Ds]>(1));
}

#[device(chip = 1)]
pub fn cat8_attn_f(
    device: &mut Device,
    a: &HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    b: &HbmTensor<bf16, Chip, m![Tb, Gf, Df]>,
    out: &mut HbmTensor<bf16, Chip, m![Rh, Tb, Gf, Df]>,
) {
    let da: DmTensor<bf16, Chip, m![1 # 2], m![Tb, 1 # 32], m![Gf, Df]> = a.to_dm(&mut device.tdma);
    da.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tb, Gf, Df]>(0));
    let db: DmTensor<bf16, Chip, m![1 # 2], m![Tb, 1 # 32], m![Gf, Df]> = b.to_dm(&mut device.tdma);
    db.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tb, Gf, Df]>(1));
}
