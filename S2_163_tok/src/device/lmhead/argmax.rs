//! Device-side greedy argmax over the LM-head logits (see `RESULTS.md`).
//!
//! Works on the contraction's own layout (no cross-slice redistribution): a slice (cluster `C`, slice `S`) holds 512 logits as one flat axis
//! `F = a * ROWS + r` (`a` = band, `r` = row inside the band; `w` increases with `F` inside a slice). Every logit becomes one i32 key and the
//! keys of a slice are Max-reduced to one key per slice (and row):
//!
//! ```text
//! b   = f32 bits of the (bf16-exact) logit
//! m   = b >> 31 (arithmetic)                 0 or -1
//! V   = b ^ (m & 0x7FFF_0000)                order-preserving i32 image of the logit (= f32::total_cmp, non-NaN; -0.0 < +0.0)
//! key = (V | P[F]) << S[F]                   P[F] = 2 * (511 - F) + 1   (odd; larger for lower w inside a slice)
//!                                            S[F] = 0, or 31 at a banned id (key lsb is 1, so key = i32::MIN)
//! ```
//!
//! The low 16 bits of V are zero (bf16), so the payload (< 2^10) never carries into the value. Max over keys = largest logit, ties to the
//! lowest `w` inside the slice. Banned ids sort below every non-NaN logit (-inf included). The host picks among the 512 slice keys
//! (`lmhead_host::decode_keys`): largest value, ties to the lowest id. (An inter-slice Max reduce over 16 slices would cut the output to 32
//! keys, but it hangs the device in 0.8.1 - see PROGRESS.md - and the output store costs the same either way.)

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Fl1, Rw8};

/// A per-slice `[F]` i32 table as a VRF operand (the HBM table is slice-contiguous: `[C, S, F]`).
pub(crate) type TableVrf<C, S> = VrfTensor<i32, Chip, C, S, m![Fl1]>;

pub(crate) fn load_table<C: M, S: M>(device: &mut Device, table: &HbmTensor<i32, Chip, m![{ C }, { S }, Fl1]>) -> TableVrf<C, S> {
    let dm: DmTensor<i32, Chip, C, S, m![Fl1]> = table.to_dm(&mut device.tdma);
    device
        .sub
        .begin(dm.view())
        .fetch::<m![Fl1 / 8], m![Fl1 % 8]>()
        .collect::<m![Fl1 / 8], m![Fl1 % 8]>()
        .to_vrf()
}

/// One pass over one row of the slice's 512 logits: per-slice Max key, `[1 # 8]` i32 (32 B cell).
pub(crate) fn slice_keys<C: M, S: M>(
    device: &mut Device,
    logits: &DmTensor<bf16, Chip, C, S, m![Fl1]>,
    payload: &TableVrf<C, S>,
    shift: &TableVrf<C, S>,
) -> DmTensor<i32, Chip, C, S, m![1, 1 # 8]> {
    device
        .main
        .begin(logits.view())
        .fetch::<m![Fl1 / 16], m![Fl1 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Fl1 / 8], m![Fl1 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_stash()
        .vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7FFF_0000i32)
        .vector_logic(LogicBinaryOpI32::BitXor, Stash)
        .vector_logic(LogicBinaryOpI32::BitOr, payload)
        .vector_logic(LogicBinaryOpI32::LeftShift, shift)
        .vector_narrow_split::<m![Fl1 / 4], m![Fl1 % 4]>()
        .vector_intra_slice_reduce::<Fl1, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Eight rows (tile `[Fl1, Rw8]`, row = TRF lane): per-slice and row Max key, `[Rw8]` i32 (32 B per slice). The tables broadcast over the
/// row lanes (one address per packet).
pub(crate) fn slice_keys_rows8<C: M, S: M>(
    device: &mut Device,
    logits: &DmTensor<bf16, Chip, C, S, m![Fl1, Rw8]>,
    payload: &TableVrf<C, S>,
    shift: &TableVrf<C, S>,
) -> DmTensor<i32, Chip, C, S, m![1, Rw8]> {
    device
        .main
        .begin(logits.view())
        .fetch::<m![Fl1], m![Rw8]>()
        .fetch_cast::<f32>()
        .collect::<m![Fl1], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_stash()
        .vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7FFF_0000i32)
        .vector_logic(LogicBinaryOpI32::BitXor, Stash)
        .vector_logic(LogicBinaryOpI32::BitOr, payload)
        .vector_logic(LogicBinaryOpI32::LeftShift, shift)
        .vector_narrow_split::<m![Fl1, Rw8 / 4], m![Rw8 % 4]>()
        .vector_intra_slice_reduce::<Fl1, m![Rw8 / 4], m![Rw8 % 4]>(IntraSliceReduceOpI32::Max)
        .vector_widen_concat::<m![1], m![Rw8]>()
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit()
}
