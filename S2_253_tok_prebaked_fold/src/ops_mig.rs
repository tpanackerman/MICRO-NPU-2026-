//! KV migration kernels: copy the K / V rows a prefill wrote into a request-slot cache (`[Ts, Ns, Ds]` ring per sliding layer, `[Tf, Df]` pages per full
//! layer) into the flat caches of the batched decode kernels (`[Rf, Ns, Ds]` sliding rows, `[Pr, Df]` full-layer page pool; see `dattn_host`).
//!
//! One launch moves `Tq` = 256 rows of K and of V: rows are gathered out of the source by a byte-offset index and scattered into the destination by another
//! (both built on the host per request and shared by all layers).

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Ds, Ns, Pr, Rf, Tf, Tq, Ts};

/// Sliding layers: source rows `src_index[i] / (Ns * Ds * 2)` of the ring -> flat cache rows `dst_index[i] / (Ns * Ds * 2)`.
#[device(chip = 1)]
pub fn mig_ring(
    device: &mut Device,
    src_k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    src_v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    src_index: &HbmTensor<i32, Chip, m![Tq]>,
    dst_index: &HbmTensor<i32, Chip, m![Tq]>,
    dst_k: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    dst_v: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    let k: DmTensor<bf16, Chip, m![1 # 2], m![Ns, 1 # 32], m![Tq, Ds]> = src_k.dma_gather_scaled(src_index);
    k.dma_scatter::<m![Tq], _, _>(dst_index, dst_k);
    let v: DmTensor<bf16, Chip, m![1 # 2], m![Ns, 1 # 32], m![Tq, Ds]> = src_v.dma_gather_scaled(src_index);
    v.dma_scatter::<m![Tq], _, _>(dst_index, dst_v);
}

/// Full layers: source page rows `src_index[i] / (Df * 2)` -> pool rows `dst_index[i] / (Df * 2)`.
#[device(chip = 1)]
pub fn mig_page(
    device: &mut Device,
    src_k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    src_v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    src_index: &HbmTensor<i32, Chip, m![Tq]>,
    dst_index: &HbmTensor<i32, Chip, m![Tq]>,
    dst_k: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
    dst_v: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
) {
    let k: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Tq, Df]> = src_k.dma_gather_scaled(src_index);
    k.dma_scatter::<m![Tq], _, _>(dst_index, dst_k);
    let v: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Tq, Df]> = src_v.dma_gather_scaled(src_index);
    v.dma_scatter::<m![Tq], _, _>(dst_index, dst_v);
}

// ---- probe: tile views on a leading token-half axis (T = 128 chunk plumbing) ----
use crate::axes::{Kc, Ke, Rh, Tc};

type PCl = m![Tc / 32];
type PSl = m![Tc % 32, 1 # 8];

/// Reads half `HALF` of a `[Rh = 2, Tc, Kc, Ke]` residual (30 real chunks) token-per-slice and writes it back to a `[Tc, Kc, Ke]` tensor (identity copy of the half).
#[device(chip = 1)]
pub fn probe_half1(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Rh, Tc, Kc, Ke]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Kc, Ke]>,
) {
    let t: DmTensor<bf16, Chip, PCl, PSl, m![Kc = 30, Ke]> = x
        .view()
        .tile::<m![Rh], 1, m![Rh = 1 # 2, Tc, Kc, Ke]>(1)
        .tile::<m![Kc], 30, m![Rh = 1 # 2, Tc, Kc = 30 # 32, Ke]>(0)
        .to_dm(&mut device.tdma);
    t.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![Kc], 30, m![Tc, Kc = 30 #{!} 32, Ke]>(0));
}
