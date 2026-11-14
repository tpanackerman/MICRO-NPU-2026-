use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

macro_rules! scatter_rows_impl {
    ($fname:ident, $Bd:ident) => {
        /// Scatters the new key (or value) rows `[Bd, Ns, Ds]` into the flat unified sliding cache `[Rf, Ns, Ds]`:
        /// row b goes to cache row `index[b] / (Ns * Ds * 2)` (host builds the byte offsets, see `dattn_host::sliding_append_index`).
        pub(crate) fn $fname(
            device: &mut Device,
            new_rows: &HbmTensor<bf16, Chip, m![$Bd, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd]>,
            cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
        ) {
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![Ns, 1 # 32], m![$Bd, Ds]> = new_rows.to_dm(&mut device.tdma);
            let rows = crate::device::kv8::kv8_t(device, &rows);
            cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
        }
    };
}

scatter_rows_impl!(scatter_sliding_rows, Bd);
scatter_rows_impl!(scatter_sliding_rows16, Bd16);

macro_rules! scatter_full_rows_impl {
    ($fname:ident, $Bd:ident) => {
        /// Scatters the new key (or value) rows `[Bd, Df]` into the flat page pool `[Pr, Df]`: row b goes to pool row
        /// `index[b] / (Df * 2)` (`dattn_host::full_append_index`).
        pub(crate) fn $fname(
            device: &mut Device,
            new_rows: &HbmTensor<bf16, Chip, m![$Bd, Df]>,
            index: &HbmTensor<i32, Chip, m![$Bd]>,
            pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
        ) {
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![$Bd, Df]> = new_rows.to_dm(&mut device.tdma);
            pool.view_mut().scatter::<m![Pr], m![Df]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
        }
    };
}

scatter_full_rows_impl!(scatter_full_rows, Bd);
scatter_full_rows_impl!(scatter_full_rows16, Bd16);
