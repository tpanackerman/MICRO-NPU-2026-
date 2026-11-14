//! kv8: the flat sliding K / V caches hold e4m3 (`crate::Kv`). Writers that do not fold the rounding into their own epilogue round the bf16 rows here
//! once (bf16 -> f32 -> e4m3, round to nearest even) right before the scatter; every reader decodes e4m3 -> bf16 (exact) in its Fetch Adapter.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

/// bf16 row `[Ds]` per slice -> e4m3 (one Main pass).
pub(crate) fn kv8_0<Cl: M, Sl: M>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![Ds]>) -> DmTensor<crate::Kv, Chip, Cl, Sl, m![Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<crate::Kv, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

/// bf16 rows `[T, Ds]` per slice -> e4m3 (one Main pass).
pub(crate) fn kv8_t<Cl: M, Sl: M, T: AxisName>(device: &mut Device, x: &DmTensor<bf16, Chip, Cl, Sl, m![T, Ds]>) -> DmTensor<crate::Kv, Chip, Cl, Sl, m![T, Ds]> {
    device
        .main
        .begin(x.view())
        .fetch::<m![T, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![T, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .cast::<crate::Kv, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
