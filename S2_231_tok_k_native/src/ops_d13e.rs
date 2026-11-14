//! R14 D13e: the next-token boundary on the device (design: stage2/d13e/DESIGN.md). Switch `S2_D13E` (host side, `host::d13e`).
//!
//! - `d13e_rowcopy8`: copies up to 8 rows of the final 256-row residual (the rows that finish a prompt) into free rows of the 16-row decode bucket of the SAME residual (gather by byte
//!   offset, scatter by byte offset; source and destination rows are disjoint), so ONE `final_norm_and_argmax_res2` launch serves riders and finishing rows (P0).
//! - `d13e_reduce`: the exact device argmax reduce of R10/R11 asyncloop (`asyncloop/src/ops_reduce.rs::rider_reduce`, verbatim: 12,000 CPU emulator cases incl. forced bf16 ties and banned
//!   winners, 1600 HW rows with every top value tied, job 166113): per row, the token of the requested ban variant with the host rule (largest value, ties to the lowest id), written as the
//!   byte offset `token << 13` of the 8192 B pitch row table, then `max(min(off, hi), lo)` (lo = hi = h forces a host-known offset).
//! - `d13e_feed16`: gathers 2 x 8 rows of the pre-scaled embedding row table (`bf16(e * EMBED_SCALE)`, 4096 columns, pad columns 0) by those offsets and scatters them into the 16 decode rows of
//!   the NEXT pass's packed buffer (`ops_mig::mig_ring` gather + scatter pattern).

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bd16, Bn2, Fl1, Kc, Ke, Rw8, W};
use crate::device::ffnc::Tk256;

/// The pre-scaled row-major embedding table, rows padded to 4096 columns (8192 B pitch: the largest byte offset 262143 * 8192 = 2,147,475,456 fits an i32).
pub type RowTable4k = HbmTensor<bf16, Chip, m![W, Kc, Ke]>;
/// The 16 decode-bucket rows of a 256-row residual (rows 240..255).
pub type Bucket = HbmTensor<bf16, Chip, m![Bd16, Kc, Ke]>;
/// The final residual of a 256-row pass.
pub type Res256 = HbmTensor<bf16, Chip, m![Tk256, Kc, Ke]>;
/// One 8-row group of the two-group LM-head keys (`ops_lmhead::Keys8x2` shape).
pub type Keys8x2 = HbmTensor<i32, Chip, m![Bn2, W / 8192 % 2, W / 32 % 256, Rw8]>;

/// P0: residual rows `sidx[i] / 8192` -> bucket rows `didx[i] / 8192` (byte offsets; unused entries repeat a valid (source, destination) pair: same bytes written twice).
#[device(chip = 1)]
pub fn d13e_rowcopy8(device: &mut Device, src: &Res256, sidx: &HbmTensor<i32, Chip, m![Rw8]>, didx: &HbmTensor<i32, Chip, m![Rw8]>, dst: &mut Bucket) {
    let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Rw8, Kc, Ke]> = src.dma_gather_scaled(sidx);
    rows.dma_scatter::<m![Rw8], _, _>(didx, dst);
}

type Cl = m![1 # 2];
type Sl = m![1 # 256];
type KeyT = DmTensor<i32, Chip, Cl, Sl, m![Bn2, Fl1, Rw8]>;
type SmallT = DmTensor<i32, Chip, Cl, Sl, m![Bn2, Rw8]>;
type RowT = DmTensor<i32, Chip, Cl, Sl, m![Rw8]>;
type TabV = VrfTensor<i32, Chip, Cl, Sl, m![Fl1]>;
type RowV = VrfTensor<i32, Chip, Cl, Sl, m![Rw8]>;
type SmallV = VrfTensor<i32, Chip, Cl, Sl, m![Bn2, Rw8]>;

fn load_tab(device: &mut Device, t: &HbmTensor<i32, Chip, m![Fl1]>) -> TabV {
    let dm: DmTensor<i32, Chip, Cl, Sl, m![Fl1]> = t.to_dm(&mut device.tdma);
    device.sub.begin(dm.view()).fetch::<m![Fl1 / 8], m![Fl1 % 8]>().collect::<m![Fl1 / 8], m![Fl1 % 8]>().to_vrf()
}

fn row_vrf(device: &mut Device, t: &HbmTensor<i32, Chip, m![Rw8]>) -> RowV {
    let dm: RowT = t.to_dm(&mut device.tdma);
    device.sub.begin(dm.view()).fetch::<m![1], m![Rw8]>().collect::<m![1], m![Rw8]>().to_vrf()
}

fn small_vrf(device: &mut Device, t: &SmallT) -> SmallV {
    device.sub.begin(t.view()).fetch::<m![Bn2], m![Rw8]>().collect::<m![Bn2], m![Rw8]>().to_vrf()
}
