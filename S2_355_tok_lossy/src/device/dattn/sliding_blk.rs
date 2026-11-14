//! R11 race4: BLOCK-GRANULAR UNSCALED K/V window gather for the sliding decode attention (generated from `sliding.rs` by tools/gen_blk.py, do not edit by hand).
//! The flat ring cache is viewed as `[Rb32, Rw32, Ns, Ds]` (32-row blocks; ring 1280 = 40 blocks, so a block never straddles the ring seam). One raw block number per block is staged on chip
//! (`to_dm`, compiler-tracked wait) and consumed by the unscaled block gather (`by_positions`): NO table-load command, no hidden DMA command pair. Destination slice (head, lane = row inside the block), element
//! (row of the cluster, block slot, d). Host: block slot j of a row holds ring block `B0 + j` (B0 = lo / 32, lo = max(0, p - 1023)); the mask follows POSITION (`blk_mask`).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub(crate) type HKb = m![Ns, Rw32];
pub(crate) type HX = m![Ns, Dm32];
pub(crate) type HOne = m![Ns, 1 # 32];

macro_rules! sliding_decode_blk_impl {
    ($fname:ident, $Bd:ident, $Bc:ident, $DQ:literal, $DK:literal) => {
        /// Batched sliding-layer decode attention over B = 8 rows (one launch). Row b attends the 1024 ring rows named by
        /// `index[b][..]` (byte offsets of rows of the flat cache `[Rf, Ns, Ds]`), with the additive penalty `mask[b]` (0 / `MASK_PEN`, see `dattn_host`); scale 1.
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Sb]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Rw32, Sb40]>,
            out: &mut HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
        ) {
            type Cl = m![$Bd % 2];

            // ---- q: each slice loads a (128 / 256 B) piece of the q rows of its head, all-gathered over the 32 key lanes of the head by the
            //      switch, then into the TRF (lane = query group) ----
            let q_sh: DmTensor<bf16, Chip, Cl, m![Ns, $Bd / 2, Gs, Ds / $DQ], m![Ds % $DQ]> = q.to_dm(&mut device.tdma);
            let q_dm: DmTensor<bf16, Chip, Cl, HX, m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ, Ds % 16]> = device
                .main
                .begin(q_sh.view())
                .fetch::<m![Ds / 16 % $DK], m![Ds % 16]>()
                .switch::<HX, m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
                .collect::<m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit();
            let q_dm: DmTensor<bf16, Chip, Cl, HKb, m![Ds / 16 % $DK, $Bc, Gs, Ds / $DQ, Ds % 16]> = unsafe { q_dm.reshape() };
            let q_trf: TrfTensor<bf16, Chip, Cl, HKb, m![Gs], m![$Bc, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .collect::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .to_trf();

            // ---- additive key penalty [$Bc, Sb] per slice (0 = key may be visible, MASK_PEN = never visible: unwritten / stale ring row) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Sb40]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb40]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb40]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, Sb40]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb40 / 8], m![Sb40 % 8]>()
                .to_vrf();
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb # 40]> = unsafe { mask_vrf.reshape() };

            // ---- block numbers of the window, staged on chip (ordinary DMA + compiler wait; no table-load command) ----
            let idx_dm: DmTensor<i32, Chip, Cl, m![1 # 256], m![$Bd / 2, Sb]> = index.to_dm(&mut device.tdma);
            let idx_spm: SpmTensor<i32, Chip, Cl, m![1 # 4], m![$Bd / 2, Sb]> = idx_dm.to_spm(&mut device.tdma);

            // ---- keys of the B rings: slice (head, lane), element (row, slot, d); streamed through the contraction ----
            let k_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd / 2, Sb, Ds]> = k_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd % 2, $Bd / 2, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let k_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, Sb], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, Sb]>()
                .contract_lane::<m![$Bc, Sb], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &mask_vrf)
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row max over the key slots that can be visible (penalised slots are far below) ---- (all-reduce Max over the 32 key lanes); packet = query group ----
            let m_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = unsafe { m_dm.reshape() };
            let m_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = device
                .sub
                .begin(m_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();

            // ---- p = bf16(exp(s - max)) (penalised slots underflow to exactly 0) ----
            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(p.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p as the (tiny) TRF operand (single lane, element = (row, slot, group)), V streamed ----
            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, Sb, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .collect::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd / 2, Sb, Ds]> = v_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd % 2, $Bd / 2, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let v_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, Sb, Ds / 32], m![Ds % 32]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Sb, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Sb, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![$Bc, Ds / 32, Gs]>()
                .contract_lane::<m![$Bc, Ds / 32, Gs, Ds / 8 % 4], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<HOne, m![$Bc, Ds / 32, Gs, Ds / 8 % 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            // ---- divide by the row sum, bf16 out ----
            let l_vrf: VrfTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();
            let res: DmTensor<bf16, Chip, Cl, HOne, m![$Bc, Gs, Ds]> = device
                .main
                .begin(o.view())
                .fetch::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .collect::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$Bc, Gs, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&l_vrf)
                .vector_widen_concat::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();
            let res: DmTensor<bf16, Chip, Cl, HOne, m![$Bd / 2, Gs, Ds]> = unsafe { res.reshape() };
            res.view().to_hbm_view(&mut device.tdma, out.view_mut());
        }

    };
}

macro_rules! sliding_decode_blk_view_impl {
    ($fname:ident, $core:ident, $Bd:ident, $Bc:ident, $DQ:literal, $DK:literal, $Rg:ident, $RgN:literal, $LastGroup:literal) => {
        /// Same as the plain variant, but the output is a mutable VIEW (the hybrid pass writes the decode rows into the last rows of the prefill output tensor through the SAME handle, so the
        /// compiler orders them after the prefill store). Batched sliding-layer decode attention over B = 8 rows (one launch). Row b attends the 1024 ring rows named by
        /// `index[b][..]` (byte offsets of rows of the flat cache `[Rf, Ns, Ds]`), with the additive penalty `mask[b]` (0 / `MASK_PEN`, see `dattn_host`); scale 1.
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Sb]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Rw32, Sb40]>,
            out: &mut HbmTensor<bf16, Chip, m![Pw, $Rg, $Bd, Ns, Gs, Ds]>,
        ) {
            let res = $core(device, q, k_cache, v_cache, index, mask);
            let res: DmTensor<bf16, Chip, m![$Bd % 2], HOne, m![$Bd / 2, Gs, Ds]> = unsafe { res.reshape() };
            // last group of 8 rows of unit 1 of the prefill output (rows 120..128), written through the prefill output's own handle
            let dst: HbmTensorViewMut<'_, bf16, Chip, m![Pw = 1 #{!} 2, $Rg = 1 #{!} $RgN, $Bd, Ns, Gs, Ds]> = out
                .view_mut()
                .tile::<m![Pw], 1, m![Pw = 1 #{!} 2, $Rg, $Bd, Ns, Gs, Ds]>(1)
                .tile::<m![$Rg], 1, m![Pw = 1 #{!} 2, $Rg = 1 #{!} $RgN, $Bd, Ns, Gs, Ds]>($LastGroup);
            res.view().to_hbm_view(&mut device.tdma, dst);
        }

        /// The body of `$fname` up to the bf16 result rows (per cluster `$Bd % 2`: rows `$Bc` x heads), before the store.
        fn $core(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Sb]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Rw32, Sb40]>,
        ) -> DmTensor<bf16, Chip, m![$Bd % 2], HOne, m![$Bc, Gs, Ds]> {
            type Cl = m![$Bd % 2];

            // ---- q: each slice loads a (128 / 256 B) piece of the q rows of its head, all-gathered over the 32 key lanes of the head by the
            //      switch, then into the TRF (lane = query group) ----
            let q_sh: DmTensor<bf16, Chip, Cl, m![Ns, $Bd / 2, Gs, Ds / $DQ], m![Ds % $DQ]> = q.to_dm(&mut device.tdma);
            let q_dm: DmTensor<bf16, Chip, Cl, HX, m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ, Ds % 16]> = device
                .main
                .begin(q_sh.view())
                .fetch::<m![Ds / 16 % $DK], m![Ds % 16]>()
                .switch::<HX, m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
                .collect::<m![Ds / 16 % $DK, $Bd / 2, Gs, Ds / $DQ], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit();
            let q_dm: DmTensor<bf16, Chip, Cl, HKb, m![Ds / 16 % $DK, $Bc, Gs, Ds / $DQ, Ds % 16]> = unsafe { q_dm.reshape() };
            let q_trf: TrfTensor<bf16, Chip, Cl, HKb, m![Gs], m![$Bc, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .collect::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .to_trf();

            // ---- additive key penalty [$Bc, Sb] per slice (0 = key may be visible, MASK_PEN = never visible: unwritten / stale ring row) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Sb40]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb40]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb40]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, Sb40]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb40 / 8], m![Sb40 % 8]>()
                .to_vrf();
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb # 40]> = unsafe { mask_vrf.reshape() };

            // ---- block numbers of the window, staged on chip (ordinary DMA + compiler wait; no table-load command) ----
            let idx_dm: DmTensor<i32, Chip, Cl, m![1 # 256], m![$Bd / 2, Sb]> = index.to_dm(&mut device.tdma);
            let idx_spm: SpmTensor<i32, Chip, Cl, m![1 # 4], m![$Bd / 2, Sb]> = idx_dm.to_spm(&mut device.tdma);

            // ---- keys of the B rings: slice (head, lane), element (row, slot, d); streamed through the contraction ----
            let k_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd / 2, Sb, Ds]> = k_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd % 2, $Bd / 2, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let k_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, Sb], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, Sb]>()
                .contract_lane::<m![$Bc, Sb], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &mask_vrf)
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row max over the key slots that can be visible (penalised slots are far below) ---- (all-reduce Max over the 32 key lanes); packet = query group ----
            let m_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = unsafe { m_dm.reshape() };
            let m_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = device
                .sub
                .begin(m_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();

            // ---- p = bf16(exp(s - max)) (penalised slots underflow to exactly 0) ----
            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(p.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p as the (tiny) TRF operand (single lane, element = (row, slot, group)), V streamed ----
            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, Sb, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .collect::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd / 2, Sb, Ds]> = v_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd % 2, $Bd / 2, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let v_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, Sb, Ds / 32], m![Ds % 32]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Sb, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Sb, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![$Bc, Ds / 32, Gs]>()
                .contract_lane::<m![$Bc, Ds / 32, Gs, Ds / 8 % 4], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<HOne, m![$Bc, Ds / 32, Gs, Ds / 8 % 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            // ---- divide by the row sum, bf16 out ----
            let l_vrf: VrfTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();
            let res: DmTensor<bf16, Chip, Cl, HOne, m![$Bc, Gs, Ds]> = device
                .main
                .begin(o.view())
                .fetch::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .collect::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$Bc, Gs, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&l_vrf)
                .vector_widen_concat::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();
            res
        }

    };
}

sliding_decode_blk_impl!(sliding_decode_attend8_blk, Bd, Bc, 64, 4);
sliding_decode_blk_impl!(sliding_decode_attend16_blk, Bd16, Bc8, 128, 8);
sliding_decode_blk_view_impl!(sliding_decode_attend8_blk_view, sliding_decode_attend8_blk_view_core, Bd, Bc, 64, 4, Rg8, 8, 7);
sliding_decode_blk_view_impl!(sliding_decode_attend16_blk_view, sliding_decode_attend16_blk_view_core, Bd16, Bc8, 128, 8, Rg4, 4, 3);

// act3 res8: the 16-row view core with the rows split over the clusters as `$Bd / 8` (rows 0..8 on cluster 0, 8..16 on cluster 1) instead of `$Bd % 2`.
// Every row runs the same instructions on the same data (only the cluster that holds it changes), so the result rows are bit-identical; the e4m3 AO8 store
// then writes 8 contiguous rows (1 KiB) per (head, group, 128-column half) instead of single interleaved 128 B rows (static 6,237 -> ~510 cycles).
macro_rules! sliding_decode_blk_view_core_c8 {
    ($core:ident, $Bd:ident, $Bc:ident, $DQ:literal, $DK:literal) => {
        /// `sliding_decode_attend16_blk_view_core` with cluster = row / 8 (rows `$Bc` x heads per cluster), before the store.
        fn $core(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Sb]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Rw32, Sb40]>,
        ) -> DmTensor<bf16, Chip, m![$Bd / 8], HOne, m![$Bc, Gs, Ds]> {
            type Cl = m![$Bd / 8];

            // ---- q: each slice loads a (128 / 256 B) piece of the q rows of its head, all-gathered over the 32 key lanes of the head by the
            //      switch, then into the TRF (lane = query group) ----
            let q_sh: DmTensor<bf16, Chip, Cl, m![Ns, $Bd % 8, Gs, Ds / $DQ], m![Ds % $DQ]> = q.to_dm(&mut device.tdma);
            let q_dm: DmTensor<bf16, Chip, Cl, HX, m![Ds / 16 % $DK, $Bd % 8, Gs, Ds / $DQ, Ds % 16]> = device
                .main
                .begin(q_sh.view())
                .fetch::<m![Ds / 16 % $DK], m![Ds % 16]>()
                .switch::<HX, m![Ds / 16 % $DK, $Bd % 8, Gs, Ds / $DQ]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
                .collect::<m![Ds / 16 % $DK, $Bd % 8, Gs, Ds / $DQ], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit();
            let q_dm: DmTensor<bf16, Chip, Cl, HKb, m![Ds / 16 % $DK, $Bc, Gs, Ds / $DQ, Ds % 16]> = unsafe { q_dm.reshape() };
            let q_trf: TrfTensor<bf16, Chip, Cl, HKb, m![Gs], m![$Bc, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .collect::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .to_trf();

            // ---- additive key penalty [$Bc, Sb] per slice (0 = key may be visible, MASK_PEN = never visible: unwritten / stale ring row) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bd % 8, Sb40]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb40]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb40]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, Sb40]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb40 / 8], m![Sb40 % 8]>()
                .to_vrf();
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sb # 40]> = unsafe { mask_vrf.reshape() };

            // ---- block numbers of the window, staged on chip (ordinary DMA + compiler wait; no table-load command) ----
            let idx_dm: DmTensor<i32, Chip, Cl, m![1 # 256], m![$Bd % 8, Sb]> = index.to_dm(&mut device.tdma);
            let idx_spm: SpmTensor<i32, Chip, Cl, m![1 # 4], m![$Bd % 8, Sb]> = idx_dm.to_spm(&mut device.tdma);

            // ---- keys of the B rings: slice (head, lane), element (row, slot, d); streamed through the contraction ----
            let k_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd % 8, Sb, Ds]> = k_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd / 8, $Bd % 8, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let k_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Ds / 32, Sb, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, Sb], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, Sb]>()
                .contract_lane::<m![$Bc, Sb], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &mask_vrf)
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row max over the key slots that can be visible (penalised slots are far below) ---- (all-reduce Max over the 32 key lanes); packet = query group ----
            let m_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = unsafe { m_dm.reshape() };
            let m_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Gs # 8]> = device
                .sub
                .begin(m_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();

            // ---- p = bf16(exp(s - max)) (penalised slots underflow to exactly 0) ----
            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sb, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(p.view())
                .fetch::<m![$Bc, Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p as the (tiny) TRF operand (single lane, element = (row, slot, group)), V streamed ----
            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, Sb, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .collect::<m![$Bc, Sb / 2], m![Sb % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bd % 8, Sb, Ds]> = v_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$Bd / 8, $Bd % 8, Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let v_dm: DmTensor<crate::Kv, Chip, Cl, HKb, m![$Bc, Sb, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, Sb, Ds / 32], m![Ds % 32]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![$Bc, Sb, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Sb, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![$Bc, Ds / 32, Gs]>()
                .contract_lane::<m![$Bc, Ds / 32, Gs, Ds / 8 % 4], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<HOne, m![$Bc, Ds / 32, Gs, Ds / 8 % 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            // ---- divide by the row sum, bf16 out ----
            let l_vrf: VrfTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![$Bc], m![Gs # 8]>()
                .collect::<m![$Bc], m![Gs # 8]>()
                .to_vrf();
            let res: DmTensor<bf16, Chip, Cl, HOne, m![$Bc, Gs, Ds]> = device
                .main
                .begin(o.view())
                .fetch::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .collect::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$Bc, Gs, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&l_vrf)
                .vector_widen_concat::<m![$Bc, Gs, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();
            res
        }
    };
}
sliding_decode_blk_view_core_c8!(sliding_decode_attend16_blk_view_core_c8, Bd16, Bc8, 128, 8);

/// R15 gapimpl F1 (AO8 pre-cast): `sliding_decode_attend16_blk_view` whose 16 rider rows go to the e4m3 AO8 input of the fused K launch
/// (`rpa::prefill_kvd256::O8P1`, last row group of unit 1, through the prefill output's own handle): e4m3(f32(bf16 res) x 16), the arithmetic of `ffnc::ao::cvt8_t128`.
pub(crate) fn sliding_decode_attend16_blk_view_o8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd16, Sb]>,
    mask: &HbmTensor<bf16, Chip, m![Bd16, Rw32, Sb40]>,
    out: &mut HbmTensor<f8e4m3, Chip, crate::device::rpa::prefill_kvd256::O8P1>,
) {
    // act3 res8: rows split over the clusters as Bd16 / 8 (see `sliding_decode_blk_view_core_c8`)
    type Cl = m![Bd16 / 8];
    let res = sliding_decode_attend16_blk_view_core_c8(device, q, k_cache, v_cache, index, mask);
    // read in the order of the slice-contiguous HBM layout (`Ds / 128` outside the rows)
    let res8: DmTensor<f8e4m3, Chip, Cl, HOne, m![Gs, Ds / 128, Bc8, Ds % 128]> = device
        .main
        .begin(res.view())
        .fetch::<m![Gs, Ds / 128, Bc8, Ds % 128 / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 128, Bc8, Ds % 128 / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 128, Bc8, Ds % 128 / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Gs, Ds / 128, Bc8, Ds % 128 / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let res8: DmTensor<f8e4m3, Chip, Cl, HOne, m![Gs, Ds / 128, Bd16 % 8, Ds % 128]> = unsafe { res8.reshape() };
    let dst: HbmTensorViewMut<'_, f8e4m3, Chip, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ds % 128]> = out
        .view_mut()
        .tile::<m![Pw], 1, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4, Bd16, Ds % 128]>(1)
        .tile::<m![Rg4], 1, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ds % 128]>(3);
    res8.view().to_hbm_view(&mut device.tdma, dst);
}

// Hybrid32 first16 riders: same arithmetic, store absolute224..239 through the shared wave1 handle.
sliding_decode_blk_view_impl!(sliding_decode_attend16_blk_view_g2, sliding_decode_attend16_blk_view_g2_core, Bd16, Bc8, 128, 8, Rg4, 4, 2);

pub(crate) fn sliding_decode_attend16_blk_view_g2_o8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Bd16, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rb32, Rw32, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Bd16, Sb]>,
    mask: &HbmTensor<bf16, Chip, m![Bd16, Rw32, Sb40]>,
    out: &mut HbmTensor<f8e4m3, Chip, crate::device::rpa::prefill_kvd256::O8P1>,
) {
    // act3 res8: rows split over the clusters as Bd16 / 8 (see `sliding_decode_blk_view_core_c8`)
    type Cl = m![Bd16 / 8];
    let res = sliding_decode_attend16_blk_view_core_c8(device, q, k_cache, v_cache, index, mask);
    // read in the order of the slice-contiguous HBM layout (`Ds / 128` outside the rows)
    let res8: DmTensor<f8e4m3, Chip, Cl, HOne, m![Gs, Ds / 128, Bc8, Ds % 128]> = device
        .main
        .begin(res.view())
        .fetch::<m![Gs, Ds / 128, Bc8, Ds % 128 / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 128, Bc8, Ds % 128 / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 128, Bc8, Ds % 128 / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Gs, Ds / 128, Bc8, Ds % 128 / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let res8: DmTensor<f8e4m3, Chip, Cl, HOne, m![Gs, Ds / 128, Bd16 % 8, Ds % 128]> = unsafe { res8.reshape() };
    let dst: HbmTensorViewMut<'_, f8e4m3, Chip, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ds % 128]> = out
        .view_mut()
        .tile::<m![Pw], 1, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4, Bd16, Ds % 128]>(1)
        .tile::<m![Rg4], 1, m![Ns, Gs, Ds / 128, Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ds % 128]>(2);
    res8.view().to_hbm_view(&mut device.tdma, dst);
}