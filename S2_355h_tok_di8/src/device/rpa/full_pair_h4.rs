// R14 attnre item E: the full-layer prefill pair (`full_pair::ragged_full_pair_4`, 4-page bucket) in the hg4 slice layout.
// b11 puts the 256 slices of a cluster on (8 query-head groups x 32 key lanes) and gathers every 8-row K / V block once per head group: the full layers have ONE KV head,
// so the pool is read 8 times. Here the slices are (4 head groups of 4 heads x 64 key lanes): lane = 8-row block `bk` of each page, slot = (page seg, row r) (32 keys
// per slice, key position seg * 512 + 8 bk + r), every block is read 4 times (half the gather bytes; lean probe: 20,842 -> 10,926 static per tensor). 4 heads x 64 rows
// of q per slice = 256 KiB (2 head groups would not fit the slice). QK / PV time steps are those of b11 (32 keys per slice keeps the 32-wide PV packet).
// Numerics: max exact; the row sum and PV are reduced over 64 lanes x 32 slots instead of 32 x 64 (f32 grouping differs): TOKEN identity, not bit identity.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bk, Df, Dummy64, Gf, Hg4, Kq32, Pq, Pw, R8, Seg4, Tc};

type Cl = m![Pw];
/// 256 slices = 4 query-head groups (4 heads each) x 64 key lanes (8-row blocks).
pub(crate) type FKb4 = m![Gf / 4, Bk];
/// Row statistics after an all-reduce over the 64 key lanes of a head group.
pub(crate) type FX4 = m![Gf / 4, Dummy64];
/// After an inter-slice Add reduce the result lives in lane 0 of each head group.
pub(crate) type FOne4 = m![Gf / 4, 1 # 64];

macro_rules! full_pair_h4_impl {
    ($fname:ident, $S:ident, $K:ident) => {
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
            k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
            v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
            index: &HbmTensor<i32, Chip, m![Pw, Hg4, $S, Bk]>,
            mask: &HbmTensor<bf16, Chip, m![Pw, Bk, Tc, $K]>,
            pen: &HbmTensor<bf16, Chip, m![Pw, Bk, $K]>,
            out: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
        ) {
            // ---- q all-gather over the 64 key lanes of each head group (slice (h4, t) holds row t of the 4 heads of group h4) ----
            let q_shard: DmTensor<bf16, Chip, Cl, m![Gf / 4, Tc], m![Gf % 4, Df]> = q.to_dm(&mut device.tdma);
            let q_all: DmTensor<bf16, Chip, Cl, FX4, m![Gf % 4, Df / 16, Tc, Df % 16]> = device
                .main
                .begin(q_shard.view())
                .fetch::<m![Gf % 4, Df / 16], m![Df % 16]>()
                .switch::<FX4, m![Gf % 4, Df / 16, Tc]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
                .collect::<m![Gf % 4, Df / 16, Tc], m![Df % 16]>()
                .commit_trim::<m![Df % 16]>()
                .commit();
            let q_all: DmTensor<bf16, Chip, Cl, FKb4, m![Gf % 4, Df / 16, Tc, Df % 16]> = unsafe { q_all.reshape() };

            // ---- keys / values: one gather each (lane = block, element = (page, row, d)), K into the TRF (lane = row) ----
            let k_g: DmTensor<bf16, Chip, Cl, m![Hg4, Bk], m![$S, R8, Df]> = k_pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let k_both: DmTensor<bf16, Chip, Cl, FKb4, m![$S, R8, Df]> = unsafe { k_g.reshape() };
            let v_g: DmTensor<bf16, Chip, Cl, m![Hg4, Bk], m![$S, R8, Df]> = v_pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
            let v_both: DmTensor<bf16, Chip, Cl, FKb4, m![$S, R8, Df]> = unsafe { v_g.reshape() };
            let k_trf: TrfTensor<bf16, Chip, Cl, FKb4, m![R8], m![$S, Df]> = device
                .sub
                .begin(k_both.view())
                .fetch::<m![R8, $S, Df / 16], m![Df % 16]>()
                .collect::<m![R8, $S, Df / 16], m![Df % 16]>()
                .to_trf();

            // ---- per-key additive penalty [$K] = ($S, R8): time $S, packet R8 ----
            let pen_dm: DmTensor<bf16, Chip, Cl, FKb4, m![$K]> = pen.to_dm(&mut device.tdma);
            let pen_dm: DmTensor<bf16, Chip, Cl, FKb4, m![$S, R8]> = unsafe { pen_dm.reshape() };
            let pen_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![$S, R8]> = device
                .sub
                .begin(pen_dm.view())
                .fetch::<m![1], m![$S, R8]>()
                .fetch_cast::<f32>()
                .collect::<m![$S], m![R8]>()
                .to_vrf();

            // ---- scores S[Gf % 4, Tc, $K] (bf16), penalty added in the VE ----
            let s: DmTensor<bf16, Chip, Cl, FKb4, m![Gf % 4, Tc, $S, R8]> = device
                .main
                .begin(q_all.view())
                .fetch::<m![Gf % 4, Tc, Df / 16], m![Df % 16]>()
                .collect::<m![Gf % 4, Tc, Df / 16], m![Df % 16]>()
                .contract_outer::<m![Gf % 4, Tc, Df / 32, $S], m![Df % 32], _, _, _>(&k_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Gf % 4, Tc, $S]>()
                .contract_lane::<m![Gf % 4, Tc, $S], m![R8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &pen_vrf)
                .vector_final()
                .cast::<bf16, m![R8 # 16]>()
                .commit_trim::<m![R8]>()
                .commit();
            let s: DmTensor<bf16, Chip, Cl, FKb4, m![Gf % 4, Tc, $K]> = unsafe { s.reshape() };

            // ---- row max over all key slots (all-reduce Max over the 64 key lanes of a head group) ----
            let m_dm: DmTensor<f32, Chip, Cl, FX4, m![Gf % 4, Tc]> = device
                .main
                .begin(s.view())
                .fetch::<m![Gf % 4, Tc, $K / 16], m![$K % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf % 4, Tc, $K / 8], m![$K % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf % 4, Tc, $K / 4], m![$K % 4]>()
                .vector_intra_slice_reduce::<$K, m![Gf % 4, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<FX4, m![Gf % 4, Tc]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .transpose::<m![Gf % 4, Tc / 2], m![Tc % 2 # 8]>()
                .commit_trim::<m![Tc % 2]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, FKb4, m![Gf % 4, Tc]> = unsafe { m_dm.reshape() };

            // ---- masks [Tc, $K] per slice ----
            let mask_dm: DmTensor<bf16, Chip, Cl, FKb4, m![Tc, $K]> = mask.to_dm(&mut device.tdma);

            // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows ----
            let mut p: DmTensor<bf16, Chip, Cl, FKb4, m![Gf % 4, Tc, $K]> = DmTensor::new();
            // The 4 tiles are expanded by hand (all tiles visible to the scheduler); `#[unroll]` is avoided on purpose (see full_pair.rs / prefill.rs).
            const _: () = assert!(Tc::SIZE == 64);
            {
                let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, $K]>(0);
                let mask_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Tc = 16, $K]> = device
                    .sub
                    .begin(mask_tile)
                    .fetch::<m![1], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Tc = 16, $K / 8], m![$K % 8]>()
                    .to_vrf();
                let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64]>(0);
                let m_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Gf % 4, Tc = 16]> = device
                    .sub
                    .begin(m_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16]>()
                    .collect::<m![Gf % 4, Tc = 16 / 8], m![Tc = 16 % 8]>()
                    .to_vrf();
                let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64, $K]>(0);
                device
                    .main
                    .begin(s_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gf % 4, Tc = 16, $K / 4], m![$K % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
                    .vector_final()
                    .cast::<bf16, m![$K % 8 # 16]>()
                    .commit_trim::<m![$K % 8]>()
                    .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 #{!} 64, $K]>(0));
            }
            {
                let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, $K]>(16);
                let mask_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Tc = 16, $K]> = device
                    .sub
                    .begin(mask_tile)
                    .fetch::<m![1], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Tc = 16, $K / 8], m![$K % 8]>()
                    .to_vrf();
                let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64]>(16);
                let m_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Gf % 4, Tc = 16]> = device
                    .sub
                    .begin(m_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16]>()
                    .collect::<m![Gf % 4, Tc = 16 / 8], m![Tc = 16 % 8]>()
                    .to_vrf();
                let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64, $K]>(16);
                device
                    .main
                    .begin(s_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gf % 4, Tc = 16, $K / 4], m![$K % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
                    .vector_final()
                    .cast::<bf16, m![$K % 8 # 16]>()
                    .commit_trim::<m![$K % 8]>()
                    .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 #{!} 64, $K]>(16));
            }
            {
                let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, $K]>(32);
                let mask_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Tc = 16, $K]> = device
                    .sub
                    .begin(mask_tile)
                    .fetch::<m![1], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Tc = 16, $K / 8], m![$K % 8]>()
                    .to_vrf();
                let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64]>(32);
                let m_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Gf % 4, Tc = 16]> = device
                    .sub
                    .begin(m_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16]>()
                    .collect::<m![Gf % 4, Tc = 16 / 8], m![Tc = 16 % 8]>()
                    .to_vrf();
                let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64, $K]>(32);
                device
                    .main
                    .begin(s_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gf % 4, Tc = 16, $K / 4], m![$K % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
                    .vector_final()
                    .cast::<bf16, m![$K % 8 # 16]>()
                    .commit_trim::<m![$K % 8]>()
                    .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 #{!} 64, $K]>(32));
            }
            {
                let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, $K]>(48);
                let mask_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Tc = 16, $K]> = device
                    .sub
                    .begin(mask_tile)
                    .fetch::<m![1], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Tc = 16, $K / 8], m![$K % 8]>()
                    .to_vrf();
                let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64]>(48);
                let m_vrf: VrfTensor<f32, Chip, Cl, FKb4, m![Gf % 4, Tc = 16]> = device
                    .sub
                    .begin(m_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16]>()
                    .collect::<m![Gf % 4, Tc = 16 / 8], m![Tc = 16 % 8]>()
                    .to_vrf();
                let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 # 64, $K]>(48);
                device
                    .main
                    .begin(s_tile)
                    .fetch::<m![Gf % 4], m![Tc = 16, $K]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gf % 4, Tc = 16, $K / 4], m![$K % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Gf % 4, Tc = 16, $K / 8], m![$K % 8]>()
                    .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
                    .vector_final()
                    .cast::<bf16, m![$K % 8 # 16]>()
                    .commit_trim::<m![$K % 8]>()
                    .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 4, Tc = 16 #{!} 64, $K]>(48));
            }

            // ---- row sum of the (bf16) weights, all-reduced over the 64 lanes; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, FX4, m![Gf % 4, Tc]> = device
                .main
                .begin(p.view())
                .fetch::<m![Gf % 4, Tc, $K / 16], m![$K % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Gf % 4, Tc, $K / 8], m![$K % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf % 4, Tc, $K / 4], m![$K % 4]>()
                .vector_intra_slice_reduce::<$K, m![Gf % 4, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 64.0)
                .vector_inter_slice_reduce::<FX4, m![Gf % 4, Tc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![Gf % 4, Tc / 2], m![Tc % 2 # 8]>()
                .commit_trim::<m![Tc % 2]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, FOne4, m![Gf % 4, Tc]> = unsafe { l_dm.reshape() };
            let l_vrf: VrfTensor<f32, Chip, Cl, FOne4, m![Gf % 4, Tc]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![Gf % 4, Tc / 8], m![Tc % 8]>()
                .collect::<m![Gf % 4, Tc / 8], m![Tc % 8]>()
                .to_vrf();

            // ---- values: transpose to [Df, $K], TRF (lane = Df % 8) ----
            let v_t: DmTensor<bf16, Chip, Cl, FKb4, m![Df, $S, R8]> = device
                .main
                .begin(v_both.view())
                .fetch::<m![Df / 8, $S, R8], m![Df % 8 # 16]>()
                .collect::<m![Df / 8, $S, R8], m![Df % 8 # 16]>()
                .transpose::<m![Df / 8, $S, R8 / 4, Df % 8], m![R8 % 4 # 16]>()
                .commit_trim::<m![R8 % 4]>()
                .commit();
            let v_t: DmTensor<bf16, Chip, Cl, FKb4, m![Df, $K]> = unsafe { v_t.reshape() };
            let v_trf: TrfTensor<bf16, Chip, Cl, FKb4, m![Df % 8], m![Df / 8, $K]> = device
                .sub
                .begin(v_t.view())
                .fetch::<m![Df % 8, Df / 8, $K / 16], m![$K % 16]>()
                .collect::<m![Df % 8, Df / 8, $K / 16], m![$K % 16]>()
                .to_trf();

            // ---- PV: contraction over the 32 key slots (one 32-wide packet), all-reduce Add over the 64 key lanes, divide by the row sum ----
            let o: DmTensor<bf16, Chip, Cl, FOne4, m![Gf % 4, Tc, Df]> = device
                .main
                .begin(p.view())
                .fetch::<m![Gf % 4, Tc, $K / 16], m![$K % 16]>()
                .collect::<m![Gf % 4, Tc, $K / 16], m![$K % 16]>()
                .contract_outer::<m![Gf % 4, Tc, Df / 8], m![$K], _, _, _>(&v_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Gf % 4, Tc, Df / 8]>()
                .contract_lane::<m![Gf % 4, Tc, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<FOne4, m![Gf % 4, Tc, Df / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gf % 4, Tc, Df / 4], m![Df % 4]>()
                .vector_fp_div(&l_vrf)
                .vector_widen_concat::<m![Gf % 4, Tc, Df / 8], m![Df % 8]>()
                .vector_final()
                .cast::<bf16, m![Df % 8 # 16]>()
                .commit_trim::<m![Df % 8]>()
                .commit();

            o.view().to_hbm_view(&mut device.tdma, out.view_mut());
        }
    };
}

full_pair_h4_impl!(ragged_full_pair_4h, Seg4, Kq32);
// (a 2-page instance `(ragged_full_pair_2h, Seg, Kq16)` compiles but is +9.3 % on HW, job 201918: 16-key PV packets double the PV steps)
