//! Device-side greedy argmax with the LM head: constant tables, launch and decode (see `RESULTS.md`).
//!
//! The kernels compute the per-slice argmax keys of EVERY ban variant (`Ban::ALL`) for every row: one launch, always the same kernel, the same inputs and the same instructions for every
//! row, whatever the requests are. This module never learns what a request wants: it returns the token of each row under each variant (`[Option<usize>; BAN_VARIANTS]`, `Ban::ALL`
//! order), and `host::generate::Generation` keeps the one of its own request.

use furiosa_opt_std::prelude::*;

use crate::axes::{Bn2, Gb4, H, Kc, Ke, Rw8, W};
use crate::host::generate::{BAN_VARIANTS, Ban};
use crate::host::prof;
use crate::lmhead_host as lh;
use crate::{Chip, ops_lmhead};

// The kernels take exactly two shift tables and write two key blocks (`Bn2`).
const _: () = assert!(BAN_VARIANTS == 2);

pub type Keys = crate::ops_lmhead::Keys1x2;
type Table = crate::ops_lmhead::Table;
type LmTable = crate::ops_lmhead::LmTable;

/// The tokens of one row under every ban variant (`Ban::ALL` order); `None` only if every id is banned under some variant (impossible for a real model: the caller then falls back to the
/// logits path).
pub type RowTokens = Option<[usize; BAN_VARIANTS]>;

/// Row `row` of the keys read back from the device (`[Bn2, c, v, row]`, `rows` rows per block), decoded under every variant.
fn row_tokens(keys: &[i32], rows: usize, row: usize) -> RowTokens {
    let tokens: [Option<usize>; BAN_VARIANTS] = std::array::from_fn(|variant| lh::decode_variant_keys(keys, variant, rows, row));
    tokens.iter().all(Option::is_some).then(|| tokens.map(|token| token.expect("checked")))
}

/// The constant HBM tables of the argmax kernels plus the output buffers.
pub struct ArgmaxHead {
    payload: Table,
    /// One shift table per variant, in `Ban::ALL` order; every launch passes both.
    shift0: Table,
    shift1: Table,
    keys8: crate::ops_lmhead::Keys8x2,
    /// Output of the 16-row (two group) kernels: block `2 * group + variant`.
    keys16: crate::ops_lmhead::Keys16x2,
    /// Output of the one-row kernel: created on first use (tools only; serving never launches it), so serving allocates exactly what the 8-row kernels need.
    keys1: Option<Keys>,
}

impl ArgmaxHead {
    pub async fn new(device: &mut Device) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            payload: lh::upload_table(device, lh::payload_table()).await?,
            shift0: lh::upload_table(device, lh::shift_table(Ban::ALL[0].ids())).await?,
            shift1: lh::upload_table(device, lh::shift_table(Ban::ALL[1].ids())).await?,
            keys8: HostTensor::<i32, m![Bn2, W / 8192 % 2, W / 32 % 256, Rw8]>::zero().to_hbm(&mut device.pdma).await?,
            keys16: HostTensor::<i32, m![Gb4, W / 8192 % 2, W / 32 % 256, Rw8]>::zero().to_hbm(&mut device.pdma).await?,
            keys1: None,
        })
    }

    /// Final RMSNorm + LM head + soft-cap + argmax of one hidden row under every ban variant.
    pub async fn argmax_token(
        &mut self,
        device: &mut Device,
        hidden: &HbmTensor<bf16, Chip, m![H]>,
        final_norm: &HbmTensor<bf16, Chip, m![H]>,
        lm_head: &LmTable,
    ) -> Result<RowTokens, Error> {
        if self.keys1.is_none() {
            self.keys1 = Some(HostTensor::<i32, m![Bn2, W / 8192 % 2, W / 32 % 256, 1 # 8]>::zero().to_hbm(&mut device.pdma).await?);
        }
        let keys = self.keys1.as_mut().expect("created above");
        prof::timed!("argmax_launch", crate::host::sleepwait::sleepy(crate::host::sleepwait::Slot::Head, launch(
            ops_lmhead::final_norm_and_argmax,
            (device, hidden, final_norm, lm_head, &self.payload, &self.shift0, &self.shift1, &mut *keys),
        ))
        .await)?;
        let keys = prof::timed!("argmax_d2h", keys.to_host::<m![Bn2, W / 8192 % 2, W / 32 % 256, 1 # 8]>(&mut device.pdma).await)?.into_vec();
        Ok(prof::timed!("argmax_decode", row_tokens(&keys, 1, 0)))
    }

    /// Eight hidden rows (rows `[Rw8, H]`; unused rows may hold anything, e.g. zeros) in one pass over the LM head table: entry `r` of the result is the token of row `r` under every ban
    /// variant.
    pub async fn argmax_tokens8(
        &mut self,
        device: &mut Device,
        hidden: &HbmTensor<bf16, Chip, m![Rw8, H]>,
        final_norm: &HbmTensor<bf16, Chip, m![H]>,
        lm_head: &LmTable,
    ) -> Result<Vec<RowTokens>, Error> {
        prof::timed!("argmax_launch", crate::host::sleepwait::sleepy(crate::host::sleepwait::Slot::Head, launch(
            ops_lmhead::final_norm_and_argmax_rows8,
            (device, hidden, final_norm, lm_head, &self.payload, &self.shift0, &self.shift1, &mut self.keys8),
        ))
        .await)?;
        self.decode8(device).await
    }

    /// `argmax_tokens8` with the 8 rows read straight from the residual stream of the hybrid pass (`[Rw8, Kc, Ke]`: 8 rows of 4096 columns, the pad columns are zero): no hidden-row
    /// D2H and no re-upload (`final_norm_and_argmax_rows8_res`). Entry `r` is the token of residual row `r` under every ban variant.
    pub async fn argmax_tokens8_res(
        &mut self,
        device: &mut Device,
        rows: &HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>,
        final_norm: &HbmTensor<bf16, Chip, m![H]>,
        lm_head: &LmTable,
    ) -> Result<Vec<RowTokens>, Error> {
        prof::timed!("argmax_launch", crate::host::sleepwait::sleepy(crate::host::sleepwait::Slot::Head, launch(
            ops_lmhead::final_norm_and_argmax_rows8_res,
            (device, rows, final_norm, lm_head, &self.payload, &self.shift0, &self.shift1, &mut self.keys8),
        ))
        .await)?;
        self.decode8(device).await
    }

    /// Reads back the keys of all variants and decodes every row under every variant.
    async fn decode8(&mut self, device: &mut Device) -> Result<Vec<RowTokens>, Error> {
        let keys = prof::timed!(
            "argmax_d2h",
            self.keys8.to_host::<m![Bn2, W / 8192 % 2, W / 32 % 256, Rw8]>(&mut device.pdma).await
        )?
        .into_vec();
        Ok((0..lh::ROWS).map(|row| row_tokens(&keys, lh::ROWS, row)).collect())
    }

    /// TWO groups of 8 hidden rows in ONE pass over the LM-head table, every ban variant of every row: `res` = device views of the padded residual stream (`[Rw8, Kc, Ke]`), `host` =
    /// uploaded groups (`[Rw8, H]`). Supported combinations (device views, host groups): (1, 1), (0, 2). Entry `8 * g + r` is the token of row `r` of group `g` (device groups first, then host
    /// groups) under every ban variant.
    pub async fn argmax_groups(
        &mut self,
        device: &mut Device,
        res: &[&HbmTensor<bf16, Chip, m![Rw8, Kc, Ke]>],
        host: &[&HbmTensor<bf16, Chip, m![Rw8, H]>],
        final_norm: &HbmTensor<bf16, Chip, m![H]>,
        lm_head: &LmTable,
    ) -> Result<Vec<RowTokens>, Error> {
        let rows = 8 * (res.len() + host.len());
        match (res.len(), host.len()) {
            (1, 1) => {
                prof::timed!("argmax_launch", crate::host::sleepwait::sleepy(crate::host::sleepwait::Slot::Head, launch(
                    ops_lmhead::final_norm_and_argmax_res1_h1,
                    (device, res[0], host[0], final_norm, lm_head, &self.payload, &self.shift0, &self.shift1, &mut self.keys16),
                ))
                .await)?;
            }
            (0, 2) => {
                prof::timed!("argmax_launch", crate::host::sleepwait::sleepy(crate::host::sleepwait::Slot::Head, launch(
                    ops_lmhead::final_norm_and_argmax_h2,
                    (device, host[0], host[1], final_norm, lm_head, &self.payload, &self.shift0, &self.shift1, &mut self.keys16),
                ))
                .await)?;
            }
            (r, h) => panic!("unsupported argmax group combination ({r} device views, {h} host groups)"),
        }
        let keys = prof::timed!("argmax_d2h", self.keys16.to_host::<m![Gb4, W / 8192 % 2, W / 32 % 256, Rw8]>(&mut device.pdma).await)?.into_vec();
        // keys = [group, variant, slice, row8]: every (group, variant) is one contiguous block of SLICES * 8 keys; a group is one two-variant `[Bn2, c, v, Rw8]` buffer
        let group = BAN_VARIANTS * lh::SLICES * lh::ROWS;
        Ok((0..rows).map(|row| row_tokens(&keys[(row / 8) * group..(row / 8 + 1) * group], lh::ROWS, row % 8)).collect())
    }
}
