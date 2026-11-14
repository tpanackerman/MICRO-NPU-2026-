//! LM-head kernels with a device-side greedy argmax (host contract: `lmhead_host`, `host::lmhead`; design: `device::lmhead::argmax`).

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bn2, Fl1, Fl8, Gb4, H, Kc, Ke, Rw8, W};
use crate::device::layout::Slice;
use crate::device::lmhead::{argmax, norm};
use crate::device::shared;
use crate::device::shared::lm_head::{Cluster, VocabRows};

/// The LM-head table in HBM stream order (the checkpoint's `[W, H]` table, permuted at load time by `host::load::lm_table_bytes`).
pub type LmTable = shared::lm_head::LmTable;

/// A `[W]` i32 table, slice-contiguous (`[c, v, F]`, see `lmhead_host::slice_layout_index`).
pub type Table = HbmTensor<i32, Chip, m![W / 8192 % 2, W / 32 % 256, Fl1]>;

/// Output of the argmax kernels for one row: one key per slice, `[c, v, 1 # 8]`.
pub type Keys1 = HbmTensor<i32, Chip, m![W / 8192 % 2, W / 32 % 256, 1 # 8]>;

/// Standalone argmax over bf16 logits `[W]` (natural order). `out[(c * 256 + v) * 2]` is the Max key of slice `(c, v)` (see `lmhead_host::decode_keys`).
#[device(chip = 1)]
pub fn argmax_logits(
    device: &mut Device,
    logits: &HbmTensor<bf16, Chip, m![W]>,
    payload: &Table,
    shift: &Table,
    out: &mut Keys1,
) {
    let l: DmTensor<bf16, Chip, Cluster, VocabRows, m![W / 16384, W % 32]> = logits.to_dm(&mut device.tdma);
    let l: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1]> = unsafe { l.reshape() };
    let payload = argmax::load_table::<Cluster, VocabRows>(device, payload);
    let shift = argmax::load_table::<Cluster, VocabRows>(device, shift);
    let keys = argmax::slice_keys(device, &l, &payload, &shift);
    let keys: DmTensor<i32, Chip, Cluster, VocabRows, m![1 # 8]> = unsafe { keys.reshape() };
    keys.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Output of the one-row ban-variant kernel: both variants' keys, `[Bn2, c, v, 1 # 8]` (one contiguous block per variant, in `Ban::ALL` order).
pub type Keys1x2 = HbmTensor<i32, Chip, m![Bn2, W / 8192 % 2, W / 32 % 256, 1 # 8]>;

/// Variant `$v` (a literal) of a ban-variant keys output: one contiguous `[c, v, F]` block per variant (the trailing tokens are `F`: `1 # 8` or `Rw8`).
macro_rules! store_variant_keys {
    ($device:ident, $keys:ident, $out:ident, $v:literal, $($f:tt)+) => {
        $keys.view().to_hbm_view(
            &mut $device.tdma,
            $out.view_mut().tile::<m![Bn2], 1, m![Bn2 = 1 #{!} 2, W / 8192 % 2, W / 32 % 256, $($f)+]>($v),
        );
    };
}

/// Final RMSNorm + LM head + soft-cap + per-slice argmax keys (one row), for BOTH ban variants (`shift0` / `shift1`: the variant tables of `host::lmhead`). Same logits as
/// `final_norm_and_logits`, bit for bit. The kernel never sees which variant a request wants: the host decodes the block it needs.
#[device(chip = 1)]
pub fn final_norm_and_argmax(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys1x2,
) {
    let payload = argmax::load_table::<Cluster, VocabRows>(device, payload);
    let shift0 = argmax::load_table::<Cluster, VocabRows>(device, shift0);
    let shift1 = argmax::load_table::<Cluster, VocabRows>(device, shift1);
    let row: HbmTensorView<'_, bf16, Chip, m![1, H]> = unsafe { input.view().reshape() };
    let normed = norm::normalize_rows::<m![1], Cluster>(device, row, rms_weight);
    let x_trf = norm::rows_to_trf::<m![1], Cluster, VocabRows>(device, &normed);
    let logits = shared::lm_head::contract_bands(device, &x_trf, lm_head_weight);
    let logits: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1]> = unsafe { logits.reshape() };
    let capped = shared::lm_head::softcap_flat::<Fl1, Cluster, VocabRows>(device, &logits);
    let keys0 = argmax::slice_keys(device, &capped, &payload, &shift0);
    let keys0: DmTensor<i32, Chip, Cluster, VocabRows, m![1 # 8]> = unsafe { keys0.reshape() };
    store_variant_keys!(device, keys0, out, 0, 1 # 8);
    let keys1 = argmax::slice_keys(device, &capped, &payload, &shift1);
    let keys1: DmTensor<i32, Chip, Cluster, VocabRows, m![1 # 8]> = unsafe { keys1.reshape() };
    store_variant_keys!(device, keys1, out, 1, 1 # 8);
}

/// Output of the 8-row kernels: one key per slice and row, `[c, v, row]`.
pub type Keys8 = HbmTensor<i32, Chip, m![W / 8192 % 2, W / 32 % 256, Rw8]>;

/// Standalone argmax over 8 rows of bf16 logits in the batched kernel's tile layout, `[c, v, a, r, row]` (`lmhead_host::to_rows8_layout`).
#[device(chip = 1)]
pub fn argmax_logits_rows8(
    device: &mut Device,
    logits: &HbmTensor<bf16, Chip, m![W / 8192 % 2, W / 32 % 256, W / 16384, W % 32, Rw8]>,
    payload: &Table,
    shift: &Table,
    out: &mut Keys8,
) {
    let l: DmTensor<bf16, Chip, Cluster, VocabRows, shared::lm_head::VocabTile8> = logits.to_dm(&mut device.tdma);
    let l: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1, Rw8]> = unsafe { l.reshape() };
    let payload = argmax::load_table::<Cluster, VocabRows>(device, payload);
    let shift = argmax::load_table::<Cluster, VocabRows>(device, shift);
    let keys = argmax::slice_keys_rows8(device, &l, &payload, &shift);
    let keys: DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]> = unsafe { keys.reshape() };
    keys.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Output of the 8-row ban-variant kernels: both variants' keys, `[Bn2, c, v, row]` (one contiguous 16 KB block per variant; interleaving the variants inside one line lost lanes on
/// HW in the t256b group-keys experiment, job 150254).
pub type Keys8x2 = HbmTensor<i32, Chip, m![Bn2, W / 8192 % 2, W / 32 % 256, Rw8]>;

/// Final RMSNorm + LM head + soft-cap + per-slice argmax keys for 8 rows sharing one pass over the table, for BOTH ban variants (`shift0` / `shift1`). `input` rows `[Rw8, H]`.
/// Every row runs the same instructions on the same operands; the variant a row needs is chosen by the host when it decodes the keys.
#[device(chip = 1)]
pub fn final_norm_and_argmax_rows8(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Rw8, H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys8x2,
) {
    let payload = argmax::load_table::<Cluster, VocabRows>(device, payload);
    let shift0 = argmax::load_table::<Cluster, VocabRows>(device, shift0);
    let shift1 = argmax::load_table::<Cluster, VocabRows>(device, shift1);
    let normed = norm::normalize_rows::<m![Rw8], Cluster>(device, input.view(), rms_weight);
    let x_trf = norm::rows_to_trf::<m![Rw8], Cluster, VocabRows>(device, &normed);
    let logits = shared::lm_head::contract_bands_rows8(device, &x_trf, lm_head_weight);
    let logits: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl8]> = unsafe { logits.reshape() };
    let capped = shared::lm_head::softcap_flat::<Fl8, Cluster, VocabRows>(device, &logits);
    let capped: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1, Rw8]> = unsafe { capped.reshape() };
    let keys0 = argmax::slice_keys_rows8(device, &capped, &payload, &shift0);
    let keys0: DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]> = unsafe { keys0.reshape() };
    store_variant_keys!(device, keys0, out, 0, Rw8);
    let keys1 = argmax::slice_keys_rows8(device, &capped, &payload, &shift1);
    let keys1: DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]> = unsafe { keys1.reshape() };
    store_variant_keys!(device, keys1, out, 1, Rw8);
}

/// `final_norm_and_argmax_rows8` reading its 8 rows straight from the zero padded residual stream (`[Rw8, Kc, Ke]` = 8 rows of 4096 columns, the last 8 rows of the hybrid pass's
/// final residual as a view): saves the hidden-row D2H and the re-upload of the host path. Same arithmetic, so the same keys as the host path for the same rows.
#[device(chip = 1)]
pub fn final_norm_and_argmax_rows8_res(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys8x2,
) {
    let payload = argmax::load_table::<Cluster, VocabRows>(device, payload);
    let shift0 = argmax::load_table::<Cluster, VocabRows>(device, shift0);
    let shift1 = argmax::load_table::<Cluster, VocabRows>(device, shift1);
    let rows: HbmTensorView<'_, bf16, Chip, m![Rw8, H #{!} 4096]> = unsafe { input.view().reshape() };
    let normed = norm::normalize_rows_padded::<m![Rw8], Cluster>(device, rows, rms_weight);
    let x_trf = norm::rows_to_trf::<m![Rw8], Cluster, VocabRows>(device, &normed);
    let logits = shared::lm_head::contract_bands_rows8(device, &x_trf, lm_head_weight);
    let logits: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl8]> = unsafe { logits.reshape() };
    let capped = shared::lm_head::softcap_flat::<Fl8, Cluster, VocabRows>(device, &logits);
    let capped: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1, Rw8]> = unsafe { capped.reshape() };
    let keys0 = argmax::slice_keys_rows8(device, &capped, &payload, &shift0);
    let keys0: DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]> = unsafe { keys0.reshape() };
    store_variant_keys!(device, keys0, out, 0, Rw8);
    let keys1 = argmax::slice_keys_rows8(device, &capped, &payload, &shift1);
    let keys1: DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]> = unsafe { keys1.reshape() };
    store_variant_keys!(device, keys1, out, 1, Rw8);
}

/// Output of the 16-row ban-variant kernels: one 16 KB block of keys per (group of 8 rows, ban variant), `[Gb4, c, v, lane]`, block `2 * group + variant` (variant order = `Ban::ALL`).
pub type Keys16x2 = HbmTensor<i32, Chip, m![Gb4, W / 8192 % 2, W / 32 % 256, Rw8]>;

type Normed = DmTensor<bf16, Chip, Cluster, m![Rw8 # 32, H / 480], m![H % 480]>;

/// Normalised rows of a group that sits in the padded residual stream (`[Rw8, Kc, Ke]` view).
fn norm_res(device: &mut Device, input: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>, rms_weight: &HbmTensor<bf16, Chip, m![H]>) -> Normed {
    let rows: HbmTensorView<'_, bf16, Chip, m![Rw8, H #{!} 4096]> = unsafe { input.view().reshape() };
    norm::normalize_rows_padded::<m![Rw8], Cluster>(device, rows, rms_weight)
}

/// Normalised rows of a group uploaded from the host (`[Rw8, H]`).
fn norm_host(device: &mut Device, input: &HbmTensor<bf16, Chip, m![Rw8, H]>, rms_weight: &HbmTensor<bf16, Chip, m![H]>) -> Normed {
    norm::normalize_rows::<m![Rw8], Cluster>(device, input.view(), rms_weight)
}

type Gath = norm::Gathered<m![Rw8], Cluster, VocabRows>;
type TableV = argmax::TableVrf<Cluster, VocabRows>;
type KeysG = DmTensor<i32, Chip, Cluster, VocabRows, m![Rw8]>;
type Capped = DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl1, Rw8]>;

/// Soft-cap of the logits of one 8-row group.
fn group_capped(device: &mut Device, l: DmTensor<bf16, Chip, Cluster, VocabRows, shared::lm_head::VocabTile8>) -> Capped {
    let l: DmTensor<bf16, Chip, Cluster, VocabRows, m![Fl8]> = unsafe { l.reshape() };
    let capped = shared::lm_head::softcap_flat::<Fl8, Cluster, VocabRows>(device, &l);
    unsafe { capped.reshape() }
}

/// Per-slice argmax keys of one 8-row group's soft-capped logits under one ban variant's shift table.
fn group_keys(device: &mut Device, capped: &Capped, payload: &TableV, shift: &TableV) -> KeysG {
    let keys = argmax::slice_keys_rows8(device, capped, payload, shift);
    unsafe { keys.reshape() }
}

fn gather_one(device: &mut Device, x: &Normed) -> Gath {
    norm::gather_rows::<m![Rw8], Cluster, VocabRows>(device, x)
}

/// Block `$b` (a literal) of the multi-block keys output: one contiguous `[c, v, Rw8]` block per (group, ban variant) (`$Gb` = the block axis, `$n` its size).
macro_rules! store_group_keys {
    ($device:ident, $k:ident, $out:ident, $Gb:ident, $n:literal, $b:literal) => {
        $k.view().to_hbm_view(
            &mut $device.tdma,
            $out.view_mut().tile::<m![$Gb], 1, m![$Gb = 1 #{!} $n, W / 8192 % 2, W / 32 % 256, Rw8]>($b),
        );
    };
}

/// Both groups, both ban variants: 4 key chains over the 2 groups' soft-capped logits, blocks `2 * group + variant`. Every row runs the same instructions on the same operands.
macro_rules! two_group_body {
    ($device:ident, $a:ident, $b:ident, $lm:ident, $payload:ident, $shift0:ident, $shift1:ident, $out:ident) => {{
        let pv = argmax::load_table::<Cluster, VocabRows>($device, $payload);
        let s0 = argmax::load_table::<Cluster, VocabRows>($device, $shift0);
        let s1 = argmax::load_table::<Cluster, VocabRows>($device, $shift1);
        let ga = gather_one($device, &$a);
        let gb = gather_one($device, &$b);
        let (la, lb) = shared::lm_head::contract_bands_groups2($device, &ga, &gb, $lm);
        let ca = group_capped($device, la);
        let cb = group_capped($device, lb);
        let ka0 = group_keys($device, &ca, &pv, &s0);
        store_group_keys!($device, ka0, $out, Gb4, 4, 0);
        let ka1 = group_keys($device, &ca, &pv, &s1);
        store_group_keys!($device, ka1, $out, Gb4, 4, 1);
        let kb0 = group_keys($device, &cb, &pv, &s0);
        store_group_keys!($device, kb0, $out, Gb4, 4, 2);
        let kb1 = group_keys($device, &cb, &pv, &s1);
        store_group_keys!($device, kb1, $out, Gb4, 4, 3);
    }};
}

/// 8 + 8 rows: one device view and one host-uploaded group in ONE pass, both ban variants (`shift0` / `shift1`).
#[device(chip = 1)]
pub fn final_norm_and_argmax_res1_h1(
    device: &mut Device,
    in_a: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>,
    in_h: &HbmTensor<bf16, Chip, m![Rw8, H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys16x2,
) {
    let a = norm_res(device, in_a, rms_weight);
    let h = norm_host(device, in_h, rms_weight);
    two_group_body!(device, a, h, lm_head_weight, payload, shift0, shift1, out);
}

/// 8 + 8 rows: TWO device views of the padded residual stream (the last 16 rows of the 256-row hybrid pass = its 16 riders) in ONE pass, both ban variants (`shift0` / `shift1`): no hidden-row
/// D2H and no re-upload.
#[device(chip = 1)]
pub fn final_norm_and_argmax_res2(
    device: &mut Device,
    in_a: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>,
    in_b: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys16x2,
) {
    let a = norm_res(device, in_a, rms_weight);
    let b = norm_res(device, in_b, rms_weight);
    two_group_body!(device, a, b, lm_head_weight, payload, shift0, shift1, out);
}

/// 16 host rows (two uploaded groups) in ONE pass, both ban variants (`shift0` / `shift1`).
#[device(chip = 1)]
pub fn final_norm_and_argmax_h2(
    device: &mut Device,
    in_a: &HbmTensor<bf16, Chip, m![Rw8, H]>,
    in_b: &HbmTensor<bf16, Chip, m![Rw8, H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &shared::lm_head::LmTable,
    payload: &Table,
    shift0: &Table,
    shift1: &Table,
    out: &mut Keys16x2,
) {
    let a = norm_host(device, in_a, rms_weight);
    let b = norm_host(device, in_b, rms_weight);
    two_group_body!(device, a, b, lm_head_weight, payload, shift0, shift1, out);
}
