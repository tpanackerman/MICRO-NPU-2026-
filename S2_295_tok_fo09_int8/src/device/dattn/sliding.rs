use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

/// 256 slices per cluster = 8 KV heads x 32 key lanes (ring row t lives in lane t % 32, slot t / 32).
pub(crate) type HKb = m![Ns, Tx % 32];
/// Row statistics after an all-reduce over the 32 key lanes of a head (fresh broadcast axis).
pub(crate) type HX = m![Ns, Dm32];
/// After an inter-slice Add reduce the result lives in key lane 0 of each head.
pub(crate) type HOne = m![Ns, 1 # 32];

macro_rules! sliding_decode_impl {
    ($fname:ident, $Bd:ident, $Bc:ident, $DQ:literal, $DK:literal) => {
        /// Batched sliding-layer decode attention over B = 8 rows (one launch). Row b attends the 1024 ring rows named by
        /// `index[b][..]` (byte offsets of rows of the flat cache `[Rf, Ns, Ds]`), with the additive penalty `mask[b]` (0 / `MASK_PEN`, see `dattn_host`); scale 1.
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            v_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Tx]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Tx % 32, Tx / 32]>,
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

            // ---- additive key penalty [$Bc, Sk] per slice (0 = key may be visible, MASK_PEN = never visible: unwritten / stale ring row) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sk]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, Sk]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk / 8], m![Sk % 8]>()
                .to_vrf();

            // ---- keys of the B rings: slice (head, lane), element (row, slot, d); streamed through the contraction ----
            let k_raw: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let k_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, Sk, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, Ds / 32, Sk, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, Sk], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, Sk]>()
                .contract_lane::<m![$Bc, Sk], m![Gs # 8]>(LaneMode::Interleaved)
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
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sk, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
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
            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
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
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sk, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p as the (tiny) TRF operand (single lane, element = (row, slot, group)), V streamed ----
            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, Sk, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, Sk / 2], m![Sk % 2, Gs # 8]>()
                .collect::<m![$Bc, Sk / 2], m![Sk % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let v_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, Sk, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, Sk, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Sk, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
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

macro_rules! sliding_decode_view_impl {
    ($fname:ident, $Bd:ident, $Bc:ident, $DQ:literal, $DK:literal, $Rg:ident, $RgN:literal, $LastGroup:literal) => {
        /// Same as the plain variant, but the output is a mutable VIEW (the hybrid pass writes the decode rows into the last rows of the prefill output tensor through the SAME handle, so the
        /// compiler orders them after the prefill store). Batched sliding-layer decode attention over B = 8 rows (one launch). Row b attends the 1024 ring rows named by
        /// `index[b][..]` (byte offsets of rows of the flat cache `[Rf, Ns, Ds]`), with the additive penalty `mask[b]` (0 / `MASK_PEN`, see `dattn_host`); scale 1.
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            v_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Bd, Tx]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, Tx % 32, Tx / 32]>,
            out: &mut HbmTensor<bf16, Chip, m![Pw, $Rg, $Bd, Ns, Gs, Ds]>,
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

            // ---- additive key penalty [$Bc, Sk] per slice (0 = key may be visible, MASK_PEN = never visible: unwritten / stale ring row) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, Sk]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, Sk]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk / 8], m![Sk % 8]>()
                .to_vrf();

            // ---- keys of the B rings: slice (head, lane), element (row, slot, d); streamed through the contraction ----
            let k_raw: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let k_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, Sk, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, Ds / 32, Sk, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, Sk], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, Sk]>()
                .contract_lane::<m![$Bc, Sk], m![Gs # 8]>(LaneMode::Interleaved)
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
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sk, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
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
            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
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
                .fetch::<m![$Bc, Sk], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, Sk], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<Sk, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p as the (tiny) TRF operand (single lane, element = (row, slot, group)), V streamed ----
            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, Sk, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, Sk / 2], m![Sk % 2, Gs # 8]>()
                .collect::<m![$Bc, Sk / 2], m![Sk % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<bf16, Chip, Cl, HKb, m![$Bd / 2, Tx / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let v_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, Sk, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, Sk, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, Sk, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Sk, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
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
            // last group of 8 rows of unit 1 of the prefill output (rows 120..128), written through the prefill output's own handle
            let dst: HbmTensorViewMut<'_, bf16, Chip, m![Pw = 1 #{!} 2, $Rg = 1 #{!} $RgN, $Bd, Ns, Gs, Ds]> = out
                .view_mut()
                .tile::<m![Pw], 1, m![Pw = 1 #{!} 2, $Rg, $Bd, Ns, Gs, Ds]>(1)
                .tile::<m![$Rg], 1, m![Pw = 1 #{!} 2, $Rg = 1 #{!} $RgN, $Bd, Ns, Gs, Ds]>($LastGroup);
            res.view().to_hbm_view(&mut device.tdma, dst);
        }

    };
}


sliding_decode_impl!(sliding_decode_attend8, Bd, Bc, 64, 4);
sliding_decode_impl!(sliding_decode_attend16, Bd16, Bc8, 128, 8);
sliding_decode_view_impl!(sliding_decode_attend8_view, Bd, Bc, 64, 4, Rg8, 8, 7);
sliding_decode_view_impl!(sliding_decode_attend16_view, Bd16, Bc8, 128, 8, Rg4, 4, 3);
