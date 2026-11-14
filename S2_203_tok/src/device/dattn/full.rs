use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

macro_rules! full_decode_impl {
    ($fname:ident, $Bd:ident, $Kx:ident, $Sf:ident, $L:literal, $Dx:ident, $PW:literal, $PK:literal) => {
        /// Batched full-layer decode attention over `Bd` rows (one launch). Row b attends the `Kx` pool rows named by
        /// `index[b][..]` (byte offsets of rows of the flat page pools `[Pr, Df]`), with the additive penalty `mask[b]` (0 / `MASK_PEN`, see `dattn_host`); scale 1.
        /// Cluster = row parity, slice = (row / 2, key lane = key % L), `Sf = Kx / 64` key slots per lane.
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$Bd, Gf, Df]>,
            k_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
            v_pool: &HbmTensor<bf16, Chip, m![Pr, Df]>,
            index: &HbmTensor<i32, Chip, m![$Bd, $Kx]>,
            mask: &HbmTensor<bf16, Chip, m![$Bd, $Kx % $L, $Kx / $L]>,
            out: &mut HbmTensor<bf16, Chip, m![$Bd, Gf, Df]>,
        ) {
            type Cl = m![$Bd % 2];
            type FS = m![$Bd / 2, $Kx % $L];
            type FX = m![$Bd / 2, $Dx];
            type FOne = m![$Bd / 2, 1 # $L];

            // ---- q: each slice loads a 256 B piece of its row, all-gathered over the 64 key lanes of the row by the switch ----
            let q_sh: DmTensor<bf16, Chip, Cl, m![$Bd / 2, Gf, Df / $PW], m![Df % $PW]> = q.to_dm(&mut device.tdma);
            let q_all: DmTensor<bf16, Chip, Cl, FX, m![Df / 16 % $PK, Gf, Df / $PW, Df % 16]> = device
                .main
                .begin(q_sh.view())
                .fetch::<m![Df / 16 % $PK], m![Df % 16]>()
                .switch::<FX, m![Df / 16 % $PK, Gf, Df / $PW]>(SwitchConfig::Broadcast1 { slice1: $L, slice0: 1 })
                .collect::<m![Df / 16 % $PK, Gf, Df / $PW], m![Df % 16]>()
                .commit_trim::<m![Df % 16]>()
                .commit();
            let q_all: DmTensor<bf16, Chip, Cl, FS, m![Df / 16 % $PK, Gf, Df / $PW, Df % 16]> = unsafe { q_all.reshape() };
            // TRF: lane = head % 8, element = (head / 8, d)
            let q_trf: TrfTensor<bf16, Chip, Cl, FS, m![Gf % 8], m![Gf / 8, Df]> = device
                .sub
                .begin(q_all.view())
                .fetch::<m![Gf % 8, Gf / 8, Df / $PW, Df / 16 % $PK], m![Df % 16]>()
                .collect::<m![Gf % 8, Gf / 8, Df / $PW, Df / 16 % $PK], m![Df % 16]>()
                .to_trf();

            // ---- additive key penalty [Sf] per slice (0 = key may be visible, MASK_PEN = never visible: position beyond the request / unallocated page) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, FS, m![$Kx / $L]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, FS, m![$Sf]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, FS, m![$Sf]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Sf]>()
                .fetch_cast::<f32>()
                .collect::<m![$Sf / 8], m![$Sf % 8]>()
                .to_vrf();

            // ---- keys: slice (row, lane), element (slot, d); streamed through the contraction ----
            let k_raw: DmTensor<bf16, Chip, Cl, FS, m![$Kx / $L, Df]> = k_pool.dma_gather_scaled(index);
            let k_dm: DmTensor<bf16, Chip, Cl, FS, m![$Sf, Df]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, FS, m![Gf / 8, $Sf, Gf % 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![Df / 32, $Sf, Df / 16 % 2], m![Df % 16]>()
                .collect::<m![Df / 32, $Sf, Df / 16 % 2], m![Df % 16]>()
                .contract_outer::<m![Df / 32, $Sf, Gf / 8], m![Df % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Sf, Gf / 8]>()
                .contract_lane::<m![$Sf, Gf / 8], m![Gf % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &mask_vrf)
                .vector_final()
                .cast::<bf16, m![Gf % 8 # 16]>()
                .commit_trim::<m![Gf % 8]>()
                .commit();

            // ---- row max over the key slots that can be visible (all-reduce Max over the 64 key lanes of the row; penalised slots are far below) ----
            let m_dm: DmTensor<f32, Chip, Cl, FX, m![Gf / 8, Gf % 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf / 8, $Sf, Gf / 4 % 2], m![Gf % 4]>()
                .vector_intra_slice_reduce::<$Sf, m![Gf / 8, Gf / 4 % 2], m![Gf % 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_concat::<m![Gf / 8], m![Gf % 8]>()
                .vector_inter_slice_reduce::<FX, m![Gf / 8]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .commit_trim::<m![Gf % 8]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, FS, m![Gf / 8, Gf % 8]> = unsafe { m_dm.reshape() };
            let m_vrf: VrfTensor<f32, Chip, Cl, FS, m![Gf / 8, Gf % 8]> = device
                .sub
                .begin(m_dm.view())
                .fetch::<m![Gf / 8], m![Gf % 8]>()
                .collect::<m![Gf / 8], m![Gf % 8]>()
                .to_vrf();

            // ---- p = bf16(exp(s - max)) (penalised slots underflow to exactly 0) ----
            let p: DmTensor<bf16, Chip, Cl, FS, m![Gf / 8, $Sf, Gf % 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf / 8, $Sf, Gf / 4 % 2], m![Gf % 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_widen_concat::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .vector_final()
                .cast::<bf16, m![Gf % 8 # 16]>()
                .commit_trim::<m![Gf % 8]>()
                .commit();

            // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, FX, m![Gf / 8, Gf % 8]> = device
                .main
                .begin(p.view())
                .fetch::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf / 8, $Sf], m![Gf % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf / 8, $Sf, Gf / 4 % 2], m![Gf % 4]>()
                .vector_intra_slice_reduce::<$Sf, m![Gf / 8, Gf / 4 % 2], m![Gf % 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_concat::<m![Gf / 8], m![Gf % 8]>()
                .vector_clip(ClipBinaryOpF32::Add, const { crate::SOFTMAX_EPS / ($L as f32) })
                .vector_inter_slice_reduce::<FX, m![Gf / 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gf % 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, FOne, m![Gf / 8, Gf % 8]> = unsafe { l_dm.reshape() };

            // ---- PV: p is the TRF operand (lane = head / 8, element = (slot, head % 8)), V streamed in 8 column chunks ----
            let p_trf: TrfTensor<bf16, Chip, Cl, FS, m![Gf / 8], m![$Sf, Gf % 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![Gf / 8, $Sf / 2], m![$Sf % 2, Gf % 8]>()
                .collect::<m![Gf / 8, $Sf / 2], m![$Sf % 2, Gf % 8]>()
                .to_trf();
            let v_raw: DmTensor<bf16, Chip, Cl, FS, m![$Kx / $L, Df]> = v_pool.dma_gather_scaled(index);
            let v_dm: DmTensor<bf16, Chip, Cl, FS, m![$Sf, Dc, Dw]> = unsafe { v_raw.reshape() };
            let l_vrf: VrfTensor<f32, Chip, Cl, FOne, m![Gf / 8, Gf % 8]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![Gf / 8], m![Gf % 8]>()
                .collect::<m![Gf / 8], m![Gf % 8]>()
                .to_vrf();
            let mut res: DmTensor<bf16, Chip, Cl, FOne, m![Gf / 8, Gf % 8, Dc, Dw]> = DmTensor::new();
            for c in 0..Dc::SIZE {
                let v_tile = v_dm.view().tile::<m![Dc], 1, m![$Sf, Dc = 1 # 8, Dw]>(c);
                let o: DmTensor<f32, Chip, Cl, FOne, m![Dw / 32, Gf % 8, Gf / 8, Dw % 32]> = device
                    .main
                    .begin(v_tile)
                    .fetch::<m![$Sf], m![Dc = 1, Dw]>()
                    .collect::<m![$Sf, Dw / 32, Dw / 16 % 2], m![Dw % 16]>()
                    .contract_outer::<m![$Sf, Dw / 32, Gf % 8], m![Dw % 32], _, _, _>(&p_trf)
                    .contract_packet::<m![Dw % 32]>()
                    .contract_time::<m![Dw / 32, Gf % 8]>()
                    .contract_lane::<m![Dw / 32, Gf % 8, Gf / 8, Dw / 8 % 4], m![Dw % 8]>(LaneMode::Sequential)
                    .vector_init()
                    .vector_inter_slice_reduce::<FOne, m![Dw / 32, Gf % 8, Gf / 8, Dw / 8 % 4]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![Dw % 8]>()
                    .commit();
                device
                    .main
                    .begin(o.view())
                    .fetch::<m![Gf / 8, Gf % 8, Dw / 8], m![Dw % 8]>()
                    .collect::<m![Gf / 8, Gf % 8, Dw / 8], m![Dw % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gf / 8, Gf % 8, Dw / 4], m![Dw % 4]>()
                    .vector_fp_div(&l_vrf)
                    .vector_widen_concat::<m![Gf / 8, Gf % 8, Dw / 8], m![Dw % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Dw % 8 # 16]>()
                    .commit_trim::<m![Dw % 8]>()
                    .commit_view(res.view_mut().tile::<m![Dc], 1, m![Gf / 8, Gf % 8, Dc = 1 #{!} 8, Dw]>(c));
            }
            let res: DmTensor<bf16, Chip, Cl, FOne, m![Gf, Df]> = unsafe { res.reshape() };
            res.view().to_hbm_view(&mut device.tdma, out.view_mut());
        }
    };
}

full_decode_impl!(full_decode_attend8_2, Bd, Kx2, Sf2, 64, Dm64, 128, 8);
full_decode_impl!(full_decode_attend8_3, Bd, Kx3, Sf3, 64, Dm64, 128, 8);
full_decode_impl!(full_decode_attend8_4, Bd, Kx4, Sf4, 64, Dm64, 128, 8);
full_decode_impl!(full_decode_attend8_5, Bd, Kx5, Sf5, 64, Dm64, 128, 8);
