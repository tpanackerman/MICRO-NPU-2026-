use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Ds, Ns, Tc, Tf, Ts};
use crate::device::layout::Cluster;

/// Copies the chunk's `Tc = 64` staged key (or value) rows into the 1024-row ring: staged row c goes to ring
/// row `index[c] / (Ns * Ds * 2)`, where `index[c]` is a byte offset the host builds per chunk.
pub(crate) fn append_rows(
    device: &mut Device,
    stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Ts = 64]>,
    ring: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![Ns, 1 # 32], m![Ts = 64, Ds]> = stage.to_dm(&mut device.tdma);
    rows.dma_scatter::<m![Ts = 64], _, _>(index, ring);
}

/// Copies the chunk's `Tc = 64` key (or value) rows [Tc, Df] into the current 512-row page of a full-attention layer:
/// row c goes to page row `index[c] / (Df * 2)`.
pub(crate) fn append_full_rows(
    device: &mut Device,
    chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    page: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![1 # 256], m![Tc, Df]> = chunk.to_dm(&mut device.tdma);
    rows.dma_scatter::<m![Tc], _, _>(index, page);
}
