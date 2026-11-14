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

/// P1: table rows `off_g[i] / 8192` -> bucket rows `dst_g[i] / 8192` for both 8-row groups.
#[device(chip = 1)]
pub fn d13e_feed16(
    device: &mut Device,
    table: &RowTable4k,
    off_a: &HbmTensor<i32, Chip, m![Rw8]>,
    dst_a: &HbmTensor<i32, Chip, m![Rw8]>,
    off_b: &HbmTensor<i32, Chip, m![Rw8]>,
    dst_b: &HbmTensor<i32, Chip, m![Rw8]>,
    out: &mut Bucket,
) {
    let a: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Rw8, Kc, Ke]> = table.dma_gather_scaled(off_a);
    a.dma_scatter::<m![Rw8], _, _>(dst_a, out);
    let b: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Rw8, Kc, Ke]> = table.dma_gather_scaled(off_b);
    b.dma_scatter::<m![Rw8], _, _>(dst_b, out);
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

/// asyncloop `rider_reduce`, unchanged (see the module docs and `asyncloop/src/ops_reduce.rs` for the derivation of the five passes).
#[device(chip = 1)]
pub fn d13e_reduce(
    device: &mut Device,
    keys: &Keys8x2,
    g3: &HbmTensor<i32, Chip, m![Fl1]>,
    g6: &HbmTensor<i32, Chip, m![Fl1]>,
    pen: &HbmTensor<i32, Chip, m![Bn2, Rw8]>,
    lo: &HbmTensor<i32, Chip, m![Rw8]>,
    hi: &HbmTensor<i32, Chip, m![Rw8]>,
    out: &mut HbmTensor<i32, Chip, m![Rw8]>,
) {
    let kview: HbmTensorView<'_, i32, Chip, m![Bn2, Fl1, Rw8]> = unsafe { keys.view().reshape() };
    let k: KeyT = kview.to_dm(&mut device.tdma);
    let g3v = load_tab(device, g3);
    let g6v = load_tab(device, g6);
    let lov = row_vrf(device, lo);
    let hiv = row_vrf(device, hi);
    // A
    let kq: KeyT = device
        .main
        .begin(k.view())
        .fetch::<m![Bn2, Fl1], m![Rw8]>()
        .collect::<m![Bn2, Fl1], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_stash()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x3C0i32)
        .vector_fxp(FxpBinaryOp::MulInt, 63i32)
        .vector_fxp(FxpBinaryOp::AddFxp, Stash)
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit();
    // B
    let c1: SmallT = device
        .main
        .begin(kq.view())
        .fetch::<m![Bn2, Fl1], m![Rw8]>()
        .collect::<m![Bn2, Fl1], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, -64i32)
        .vector_logic(LogicBinaryOpI32::BitOr, &g3v)
        .vector_narrow_split::<m![Bn2, Fl1, Rw8 / 4], m![Rw8 % 4]>()
        .vector_intra_slice_reduce::<Fl1, m![Bn2, Rw8 / 4], m![Rw8 % 4]>(IntraSliceReduceOpI32::Max)
        .vector_widen_concat::<m![Bn2], m![Rw8]>()
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit();
    // D1
    let wm: SmallT = device
        .main
        .begin(c1.view())
        .fetch::<m![Bn2], m![Rw8]>()
        .collect::<m![Bn2], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitXor, 0xFFF8i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0xFFF8i32)
        .vector_logic(LogicBinaryOpI32::LeftShift, 3i32)
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit();
    let wmv = small_vrf(device, &wm);
    // C
    let lw: SmallT = device
        .main
        .begin(kq.view())
        .fetch::<m![Bn2, Fl1], m![Rw8]>()
        .collect::<m![Bn2, Fl1], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 63i32)
        .vector_logic(LogicBinaryOpI32::BitOr, &g6v)
        .vector_logic(LogicBinaryOpI32::BitXor, &wmv)
        .vector_narrow_split::<m![Bn2, Fl1, Rw8 / 4], m![Rw8 % 4]>()
        .vector_intra_slice_reduce::<Fl1, m![Bn2, Rw8 / 4], m![Rw8 % 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_concat::<m![Bn2], m![Rw8]>()
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit();
    let pen_dm: SmallT = pen.to_dm(&mut device.tdma);
    let penv = small_vrf(device, &pen_dm);
    // D2
    let off: RowT = device
        .main
        .begin(lw.view())
        .fetch::<m![Bn2], m![Rw8]>()
        .collect::<m![Bn2], m![Rw8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, 62i32)
        .vector_logic(LogicBinaryOpI32::BitXor, 62i32)
        .vector_logic(LogicBinaryOpI32::BitOr, &wmv)
        .vector_logic(LogicBinaryOpI32::LogicRightShift, 1i32)
        .vector_fxp(FxpBinaryOp::LeftShift, 13i32)
        .vector_fxp(FxpBinaryOp::AddFxpSat, &penv)
        .vector_narrow_split::<m![Bn2, Rw8 / 4], m![Rw8 % 4]>()
        .vector_intra_slice_reduce::<Bn2, m![Rw8 / 4], m![Rw8 % 4]>(IntraSliceReduceOpI32::Min)
        .vector_widen_concat::<m![1], m![Rw8]>()
        .vector_clip(ClipBinaryOpI32::Min, &hiv)
        .vector_clip(ClipBinaryOpI32::Max, &lov)
        .vector_final()
        .commit_trim::<m![Rw8]>()
        .commit();
    off.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
