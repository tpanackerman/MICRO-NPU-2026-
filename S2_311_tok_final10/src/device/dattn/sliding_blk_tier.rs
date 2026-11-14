//! R14 d13b (S2_DECTIER): TIERED block-granular decode attention for the 16 rider rows of the fused 256-row pass. The host sorts the riders by the number of
//! 32-row blocks their window needs (descending) and the 16 rows are processed as three tiers: rows 0..8 with the full `Sb = 34` block slots, rows 8..12
//! with `Sb26`, rows 12..16 with `Sb18` slots. A rider needs blocks `lo / 32 ..= p / 32` (lo = max(0, p - 1023)); every slot past the last needed block is
//! DEAD (fully masked, and always TRAILING), so a tier with fewer slots gathers fewer key / value rows and drops only trailing slots whose softmax weight is
//! exactly 0. The arithmetic of every row is the same chain as `sliding_blk::sliding_decode_attend16_blk_view` (same code, generated per tier): scores
//! (contraction over Ds), max / sum over the slot (time) axis then over the 32 key lanes, PV accumulated over the slots, divide. Identity vs the untiered
//! kernel is checked bit for bit by the lab harness (d13b_k) before anything else.
//! The three tiers write their rows of the decode group through tile views of the SAME output handle (rows typed `Bd16 = 8` / `Bd16 = 4`) (the prefill r16 output), so the compiler orders the stores.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

use super::sliding_blk::{HKb, HOne, HX};

macro_rules! sliding_decode_tier_impl {
    (
        $fname:ident, [$($Bd:tt)*], $Bc:ident, $DQ:literal, $DK:literal, $Sb:ident, $SbP:ident, $Len:literal, $Start:literal
    ) => {
        /// One tier of the tiered decode attention (see the module doc): `$($Bd)*` rows, `$Sb` block slots per row (`$SbP` = mask slots padded to 8),
        /// output = its rows of the decode group of `out` (unit 1, last 16-row group of the r16 prefill output).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![$($Bd)*, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<bf16, Chip, m![Rb32, Rw32, Ns, Ds]>,
            v_cache: &HbmTensor<bf16, Chip, m![Rb32, Rw32, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$($Bd)*, $Sb]>,
            mask: &HbmTensor<bf16, Chip, m![$($Bd)*, Rw32, $SbP]>,
            out: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
        ) {
            type Cl = m![$($Bd)* % 2];

            let q_sh: DmTensor<bf16, Chip, Cl, m![Ns, $($Bd)* / 2, Gs, Ds / $DQ], m![Ds % $DQ]> = q.to_dm(&mut device.tdma);
            let q_dm: DmTensor<bf16, Chip, Cl, HX, m![Ds / 16 % $DK, $($Bd)* / 2, Gs, Ds / $DQ, Ds % 16]> = device
                .main
                .begin(q_sh.view())
                .fetch::<m![Ds / 16 % $DK], m![Ds % 16]>()
                .switch::<HX, m![Ds / 16 % $DK, $($Bd)* / 2, Gs, Ds / $DQ]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
                .collect::<m![Ds / 16 % $DK, $($Bd)* / 2, Gs, Ds / $DQ], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit();
            let q_dm: DmTensor<bf16, Chip, Cl, HKb, m![Ds / 16 % $DK, $Bc, Gs, Ds / $DQ, Ds % 16]> = unsafe { q_dm.reshape() };
            let q_trf: TrfTensor<bf16, Chip, Cl, HKb, m![Gs], m![$Bc, Ds]> = device
                .sub
                .begin(q_dm.view())
                .fetch::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .collect::<m![Gs, $Bc, Ds / $DQ, Ds / 16 % $DK], m![Ds % 16]>()
                .to_trf();

            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$($Bd)* / 2, $SbP]> = mask.to_dm(&mut device.tdma);
            let mask_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, $SbP]> = unsafe { mask_dm.reshape() };
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, $SbP]> = device
                .sub
                .begin(mask_dm.view())
                .fetch::<m![1], m![$Bc, $SbP]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, $SbP / 8], m![$SbP % 8]>()
                .to_vrf();
            let mask_vrf: VrfTensor<f32, Chip, Cl, HKb, m![$Bc, $Sb # $SbP]> = unsafe { mask_vrf.reshape() };

            let idx_dm: DmTensor<i32, Chip, Cl, m![1 # 256], m![$($Bd)* / 2, $Sb]> = index.to_dm(&mut device.tdma);
            let idx_spm: SpmTensor<i32, Chip, Cl, m![1 # 4], m![$($Bd)* / 2, $Sb]> = idx_dm.to_spm(&mut device.tdma);

            let k_raw: DmTensor<bf16, Chip, Cl, HKb, m![$($Bd)* / 2, $Sb, Ds]> = k_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$($Bd)* % 2, $($Bd)* / 2, $Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let k_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, $Sb, Ds]> = unsafe { k_raw.reshape() };
            let s: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, $Sb, Gs # 8]> = device
                .main
                .begin(k_dm.view())
                .fetch::<m![$Bc, Ds / 32, $Sb, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, Ds / 32, $Sb, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, Ds / 32, $Sb], m![Ds % 32], _, _, _>(&q_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$Bc, $Sb]>()
                .contract_lane::<m![$Bc, $Sb], m![Gs # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &mask_vrf)
                .vector_final()
                .cast::<bf16, m![Gs # 16]>()
                .commit_trim::<m![Gs # 8]>()
                .commit();

            let m_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, $Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, $Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<$Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Max)
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

            let p: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, $Sb, Gs # 8]> = device
                .main
                .begin(s.view())
                .fetch::<m![$Bc, $Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, $Sb], m![Gs # 8]>()
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

            let l_dm: DmTensor<f32, Chip, Cl, HX, m![$Bc, Gs # 8]> = device
                .main
                .begin(p.view())
                .fetch::<m![$Bc, $Sb], m![Gs # 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Bc, $Sb], m![Gs # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![Gs # 4]>()
                .vector_intra_slice_reduce::<$Sb, m![$Bc], m![Gs # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![Gs # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HX, m![$Bc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Gs # 8]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Gs # 8]> = unsafe { l_dm.reshape() };

            let p_trf: TrfTensor<bf16, Chip, Cl, HKb, m![1], m![$Bc, $Sb, Gs # 8]> = device
                .sub
                .begin(p.view())
                .fetch::<m![$Bc, $Sb / 2], m![$Sb % 2, Gs # 8]>()
                .collect::<m![$Bc, $Sb / 2], m![$Sb % 2, Gs # 8]>()
                .to_trf();
            let v_raw: DmTensor<bf16, Chip, Cl, HKb, m![$($Bd)* / 2, $Sb, Ds]> = v_cache.gather::<m![Rb32], m![Rw32, Ns, Ds]>().by_positions::<m![$($Bd)* % 2, $($Bd)* / 2, $Sb]>(&idx_spm).to_dm(&mut device.tdma);
            let v_dm: DmTensor<bf16, Chip, Cl, HKb, m![$Bc, $Sb, Ds]> = unsafe { v_raw.reshape() };
            let o: DmTensor<f32, Chip, Cl, HOne, m![$Bc, Ds / 32, Gs, Ds % 32]> = device
                .main
                .begin(v_dm.view())
                .fetch::<m![$Bc, $Sb, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .collect::<m![$Bc, $Sb, Ds / 32, Ds / 16 % 2], m![Ds % 16]>()
                .contract_outer::<m![$Bc, $Sb, Ds / 32, Gs], m![Ds % 32], _, _, _>(&p_trf)
                .contract_packet::<m![Ds % 32]>()
                .contract_time::<m![$Bc, Ds / 32, Gs]>()
                .contract_lane::<m![$Bc, Ds / 32, Gs, Ds / 8 % 4], m![Ds % 8]>(LaneMode::Sequential)
                .vector_init()
                .vector_inter_slice_reduce::<HOne, m![$Bc, Ds / 32, Gs, Ds / 8 % 4]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<m![Ds % 8]>()
                .commit();

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
            let res: DmTensor<bf16, Chip, Cl, HOne, m![$($Bd)* / 2, Gs, Ds]> = unsafe { res.reshape() };
            // this tier's rows of the decode group (unit 1 = Pw 1, last 16-row group = Rg4 3, rows $Start .. $Start + $Len), through the SAME output handle as the prefill store
            let dst: HbmTensorViewMut<'_, bf16, Chip, m![Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16 = $Len #{!} 16, Ns, Gs, Ds]> = out
                .view_mut()
                .tile::<m![Pw], 1, m![Pw = 1 #{!} 2, Rg4, Bd16, Ns, Gs, Ds]>(1)
                .tile::<m![Rg4], 1, m![Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16, Ns, Gs, Ds]>(3)
                .tile::<m![Bd16], $Len, m![Pw = 1 #{!} 2, Rg4 = 1 #{!} 4, Bd16 = $Len #{!} 16, Ns, Gs, Ds]>($Start);
            res.view().to_hbm_view(&mut device.tdma, dst);
        }
    };
}

// tier A: rows 0..8 of the decode group, the full 34 block slots
sliding_decode_tier_impl!(sliding_decode_tier_a, [Bd16 = 8], Bc, 64, 4, Sb, Sb40, 8, 0);
// tier B: rows 8..12, 26 block slots (windows of <= 26 blocks: p <= 831)
sliding_decode_tier_impl!(sliding_decode_tier_b, [Bd16 = 4], Bc2, 32, 2, Sb26, Sp32, 4, 8);
// tier C: rows 12..16, 18 block slots (p <= 575)
sliding_decode_tier_impl!(sliding_decode_tier_c, [Bd16 = 4], Bc2, 32, 2, Sb18, Sp24, 4, 12);
