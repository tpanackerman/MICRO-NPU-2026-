// Ragged-attention prefill part: a PAIR of 64-row prefill units (one per cluster) on the unified flat sliding cache [Rf, Ns, Ds].
// Derived from device/dattn/chunk_flat.rs (same math, same rounding points) with three changes:
//   * cluster = unit of the pair (Pw), so the two clusters work on two independent 64-row segments at the same time
//     (the single-segment kernel keeps the second cluster idle);
//   * the chunk keys of a unit are GATHERED from the flat stage tensor [Pt, Ns, Ds] by an index [Pw, Cx] (a run of up to Cx / 64 = 4
//     segments of one request: segment j of a run sees the staged keys of the segments before it, no append between segments needed);
//   * the ring part is read BEFORE any append of the launch (the appends are issued after the attention), so the ring-overwrite hazard
//     of chunks at p0 >= 1024 is impossible by construction.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

/// Cluster = unit of the pair.
type Cl = m![Pw];
/// 256 slices: head x key lane (ring row r lives in slice (h, r % 32), slot r / 32).
type HeadKb = m![Ns, Ts % 32];
/// Row statistics after an inter-slice all-reduce (fresh broadcast axis).
type HeadX = m![Ns, Dummy32];
/// After an inter-slice Add reduce the result lives in key lane 0 of each head.
type HeadOne = m![Ns, 1 # 32];

/// Stores of the prefill result `o` (`[Cl = Pw][HeadOne][Gs, Tc, Ds]`): plain rows `[Pw, Tc, Ns, Gs, Ds]`, or the same rows with each unit's 64 rows split in 8 groups of 8 (`[Pw, Rg8, Bd, Ns, Gs, Ds]`:
/// the layout in which the hybrid pass's fused kernel writes the decode rows = the last group of unit 1 through the SAME handle).
macro_rules! store_plain {
    ($device:ident, $o:ident, $out:ident) => {
        $o.view().to_hbm_view(&mut $device.tdma, $out.view_mut());
    };
}
macro_rules! store_rows8 {
    ($device:ident, $o:ident, $out:ident) => {{
        let ov: DmTensorView<'_, bf16, Chip, Cl, HeadOne, m![Gs, Rg8, Bd, Ds]> = unsafe { $o.view().reshape() };
        ov.to_hbm_view(&mut $device.tdma, $out.view_mut());
    }};
}

/// Same for the 16-row decode bucket: each unit's 64 rows split in 4 groups of 16 (`[Pw, Rg4, Bd16, Ns, Gs, Ds]`).
macro_rules! store_rows16 {
    ($device:ident, $o:ident, $out:ident) => {{
        let ov: DmTensorView<'_, bf16, Chip, Cl, HeadOne, m![Gs, Rg4, Bd16, Ds]> = unsafe { $o.view().reshape() };
        ov.to_hbm_view(&mut $device.tdma, $out.view_mut());
    }};
}

macro_rules! ragged_prefill_pair_impl {
    ($fname:ident, $Pt:ident, $OutTy:ty, $store:ident) => {
        /// Attention of two 64-row prefill units (unit c of the pair runs on cluster c). Slice = (head, key lane); each slice holds 64 key slots:
        /// slots 0..32 = ring rows r with r % 32 == lane (slot r / 32), slots 32..64 = chunk keys c with c % 32 == lane (slot 32 + c / 32; only
        /// the first Cx / 32 = 8 slots are loaded, the rest is zero).
        /// `ring_index[u][t]` = byte offset of ring row t of the unit's slot, `stage_index[u][c]` = byte offset of the stage row of chunk key c,
        /// `mask[u][lane][t][seg][slot]` = 0/1 visibility of the key in slot (seg, slot) of lane `lane` for query row t of the unit.
        /// `pen[u][lane][seg][slot]` = additive score penalty (0 / `MASK_PEN`) of the key slots that no row of the unit can ever see (ring slots before position 0, staged keys c >= n,
        /// zero-filled slots): added before the row max so the max never depends on stale ring rows or on other requests' rows (`rpa_host::unit_pen`).
        pub(crate) fn $fname(
            device: &mut Device,
            q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
            k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
            v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
            ring_index: &HbmTensor<i32, Chip, m![Pw, Tx]>,
            k_stage: &HbmTensor<bf16, Chip, m![$Pt, Ns, Ds]>,
            v_stage: &HbmTensor<bf16, Chip, m![$Pt, Ns, Ds]>,
            stage_index: &HbmTensor<i32, Chip, m![Pw, Cx]>,
            mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
            pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
            out: &mut $OutTy,
        ) {
            // ---- q all-gather over the 32 key-block slices of each head ----
            let q_shard: DmTensor<bf16, Chip, Cl, m![Ns, Tc % 32], m![Tc / 32, Gs, Ds]> = q.to_dm(&mut device.tdma);
            let q_all: DmTensor<bf16, Chip, Cl, HeadX, m![Tc / 32, Gs, Ds / 16, Tc % 32, Ds % 16]> = device
                .main
                .begin(q_shard.view())
                .fetch::<m![Tc / 32, Gs, Ds / 16], m![Ds % 16]>()
                .switch::<HeadX, m![Tc / 32, Gs, Ds / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
                .collect::<m![Tc / 32, Gs, Ds / 16, Tc % 32], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit();
            let q_all: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc / 32, Gs, Ds / 16, Tc % 32, Ds % 16]> = unsafe { q_all.reshape() };

            // ---- key slots: ring block + chunk block (zero filled where the unit has no keys) ----
            let mut k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
            k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            let k_ring: DmTensor<crate::Kv, Chip, Cl, m![Ns, Tx % 32], m![Tx / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(ring_index).to_dm(&mut device.tdma);
            let k_ring: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 32, Ds]> = unsafe { k_ring.reshape() };
            device
                .main
                .begin(k_ring.view())
                .fetch::<m![Kk48 = 32, Ds / 16], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![Kk48 = 32, Ds / 16], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit_view(k_both.view_mut().tile::<m![Kk48], 32, m![Kk48 = 32 #{!} 48, Ds]>(0));
            let k_chunk: DmTensor<bf16, Chip, Cl, m![Ns, Cx % 32], m![Cx / 32, Ds]> = k_stage.gather::<m![$Pt], m![Ns, Ds]>().by_byte_offsets(stage_index).to_dm(&mut device.tdma);
            let k_chunk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48 = 8, Ds]> = unsafe { k_chunk.reshape() };
            device
                .main
                .begin(k_chunk.view())
                .fetch::<m![Kk48 = 8, Ds / 16], m![Ds % 16]>()
                .collect::<m![Kk48 = 8, Ds / 16], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit_view(
                    k_both.view_mut().tile::<m![Kk48], 8, m![Kk48 = 8 #{!} 48, Ds]>(32),
                );
            let k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = unsafe { k_both.reshape() };
            let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
                .sub
                .begin(k_both.view())
                .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
                .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
                .to_trf();

            // ---- per-key additive penalty [Kk48] ----
            let pen_dm: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
            let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = unsafe { pen_dm.reshape() };
            let pen_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Kk48]> = device
                .sub
                .begin(pen_kk.view())
                .fetch::<m![1], m![Kk48]>()
                .fetch_cast::<f32>()
                .collect::<m![Kk48 / 8], m![Kk48 % 8]>()
                .to_vrf();

            // ---- scores S[Gs, Tc, Kk48] (bf16), penalty added in the VE ----
            let s: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = device
                .main
                .begin(q_all.view())
                .fetch::<m![Gs, Tc / 32, Tc % 32, Ds / 16], m![Ds % 16]>()
                .collect::<m![Gs, Tc / 32, Tc % 32, Ds / 16], m![Ds % 16]>()
                .contract_outer::<m![Gs, Tc, Ds / 32, Kk48 / 8], m![Ds % 32], _, _, _>(&k_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Gs, Tc, Kk48 / 8]>()
                .contract_lane::<m![Gs, Tc, Kk48 / 8], m![Kk48 % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, &pen_vrf)
                .vector_final()
                .cast::<bf16, m![Kk48 % 8 # 16]>()
                .commit_trim::<m![Kk48 % 8]>()
                .commit();

            // ---- row max over the key slots (penalised ones are far below; all-reduce Max over the head's 32 key-block slices) ----
            let m_dm: DmTensor<f32, Chip, Cl, HeadX, m![Gs, Tc]> = device
                .main
                .begin(s.view())
                .fetch::<m![Gs, Tc, Kk48 / 16], m![Kk48 % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Gs, Tc, Kk48 / 8], m![Kk48 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gs, Tc, Kk48 / 4], m![Kk48 % 4]>()
                .vector_intra_slice_reduce::<Kk48, m![Gs, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_inter_slice_reduce::<HeadX, m![Gs, Tc]>(InterSliceReduceOpF32::Max)
                .vector_final()
                .transpose::<m![Gs, Tc / 2], m![Tc % 2 # 8]>()
                .commit_trim::<m![Tc % 2]>()
                .commit();
            let m_dm: DmTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc]> = unsafe { m_dm.reshape() };

            // ---- masks: [Tc, Seg, 32 slots] per slice (host builds the zero padding of the chunk part) ----
            let mask_dm: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);
            let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = unsafe { mask_dm.reshape() };

            // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows (VRF holds one mask tile) ----
            let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
            // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
            // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
            const _: () = assert!(Tc::SIZE == 64);
            macro_rules! p_tile {
                ($i:expr) => {{
                let i: usize = $i;
                let mask_tile = mask_both.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kk48]>(16 * i);
                let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 16, Kk48]> = device
                    .sub
                    .begin(mask_tile)
                    .fetch::<m![1], m![Tc = 16, Kk48]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Tc = 16, Kk48 / 8], m![Kk48 % 8]>()
                    .to_vrf();
                let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gs, Tc = 16 # 64]>(16 * i);
                let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 16]> = device
                    .sub
                    .begin(m_tile)
                    .fetch::<m![Gs], m![Tc = 16]>()
                    .collect::<m![Gs, Tc = 16 / 8], m![Tc = 16 % 8]>()
                    .to_vrf();
                let s_tile = s.view().tile::<m![Tc], 16, m![Gs, Tc = 16 # 64, Kk48]>(16 * i);
                device
                    .main
                    .begin(s_tile)
                    .fetch::<m![Gs], m![Tc = 16, Kk48]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Gs, Tc = 16, Kk48 / 8], m![Kk48 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Gs, Tc = 16, Kk48 / 4], m![Kk48 % 4]>()
                    .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
                    .vector_fp_unary(FpUnaryOp::Exp)
                    .vector_widen_concat::<m![Gs, Tc = 16, Kk48 / 8], m![Kk48 % 8]>()
                    .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
                    .vector_final()
                    .cast::<bf16, m![Kk48 % 8 # 16]>()
                    .commit_trim::<m![Kk48 % 8]>()
                    .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gs, Tc = 16 #{!} 64, Kk48]>(16 * i));
                }};
            }
            p_tile!(0);
            p_tile!(1);
            p_tile!(2);
            p_tile!(3);

            // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
            let l_dm: DmTensor<f32, Chip, Cl, HeadX, m![Gs, Tc]> = device
                .main
                .begin(p.view())
                .fetch::<m![Gs, Tc, Kk48 / 16], m![Kk48 % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Gs, Tc, Kk48 / 8], m![Kk48 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gs, Tc, Kk48 / 4], m![Kk48 % 4]>()
                .vector_intra_slice_reduce::<Kk48, m![Gs, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
                .vector_inter_slice_reduce::<HeadX, m![Gs, Tc]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .transpose::<m![Gs, Tc / 2], m![Tc % 2 # 8]>()
                .commit_trim::<m![Tc % 2]>()
                .commit();
            let l_dm: DmTensor<f32, Chip, Cl, HeadOne, m![Gs, Tc]> = unsafe { l_dm.reshape() };
            let l_vrf: VrfTensor<f32, Chip, Cl, HeadOne, m![Gs, Tc]> = device
                .sub
                .begin(l_dm.view())
                .fetch::<m![Gs, Tc / 8], m![Tc % 8]>()
                .collect::<m![Gs, Tc / 8], m![Tc % 8]>()
                .to_vrf();

            // ---- PV over both segments (Seg is a reduced time axis) ----
            let mut v_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
            v_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            let v_ring: DmTensor<crate::Kv, Chip, Cl, m![Ns, Tx % 32], m![Tx / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(ring_index).to_dm(&mut device.tdma);
            let v_ring: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 32, Ds]> = unsafe { v_ring.reshape() };
            device
                .main
                .begin(v_ring.view())
                .fetch::<m![Kk48 = 32, Ds / 16], m![Ds % 16]>()
                .fetch_table_lookup::<bf16>()
                .collect::<m![Kk48 = 32, Ds / 16], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit_view(v_both.view_mut().tile::<m![Kk48], 32, m![Kk48 = 32 #{!} 48, Ds]>(0));
            let v_chunk: DmTensor<bf16, Chip, Cl, m![Ns, Cx % 32], m![Cx / 32, Ds]> = v_stage.gather::<m![$Pt], m![Ns, Ds]>().by_byte_offsets(stage_index).to_dm(&mut device.tdma);
            let v_chunk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48 = 8, Ds]> = unsafe { v_chunk.reshape() };
            device
                .main
                .begin(v_chunk.view())
                .fetch::<m![Kk48 = 8, Ds / 16], m![Ds % 16]>()
                .collect::<m![Kk48 = 8, Ds / 16], m![Ds % 16]>()
                .commit_trim::<m![Ds % 16]>()
                .commit_view(
                    v_both.view_mut().tile::<m![Kk48], 8, m![Kk48 = 8 #{!} 48, Ds]>(32),
                );
            let v_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = unsafe { v_both.reshape() };
            let v_t: DmTensor<bf16, Chip, Cl, HeadKb, m![Ds, Kk48]> = device
                .main
                .begin(v_both.view())
                .fetch::<m![Ds / 8, Kk48], m![Ds % 8 # 16]>()
                .collect::<m![Ds / 8, Kk48], m![Ds % 8 # 16]>()
                .transpose::<m![Ds / 8, Kk48 / 4, Ds % 8], m![Kk48 % 4 # 16]>()
                .commit_trim::<m![Kk48 % 4]>()
                .commit();
            let v_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Ds % 8], m![Ds / 8, Kk48]> = device
                .sub
                .begin(v_t.view())
                .fetch::<m![Ds % 8, Ds / 8, Kk48 / 16], m![Kk48 % 16]>()
                .collect::<m![Ds % 8, Ds / 8, Kk48 / 16], m![Kk48 % 16]>()
                .to_trf();
            let o: DmTensor<bf16, Chip, Cl, HeadOne, m![Gs, Tc, Ds]> = device
                .main
                .begin(p.view())
                .fetch::<m![Gs, Tc, Kk48 / 16], m![Kk48 % 16]>()
                .collect::<m![Gs, Tc, Kk48 / 16], m![Kk48 % 16]>()
                .contract_outer::<m![Gs, Tc, Kk48 / 16, Ds / 8], m![Kk48 % 16], _, _, _>(&v_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![Gs, Tc, Ds / 8]>()
                .contract_lane::<m![Gs, Tc, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<HeadOne, m![Gs, Tc, Ds / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Gs, Tc, Ds / 4], m![Ds % 4]>()
                .vector_fp_div(&l_vrf)
                .vector_widen_concat::<m![Gs, Tc, Ds / 8], m![Ds % 8]>()
                .vector_final()
                .cast::<bf16, m![Ds % 8 # 16]>()
                .commit_trim::<m![Ds % 8]>()
                .commit();

            $store!(device, o, out);
        }
    };
}

ragged_prefill_pair_impl!(ragged_prefill_pair_128, P128, HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>, store_plain);
ragged_prefill_pair_impl!(ragged_prefill_pair_256, P256, HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>, store_plain);
ragged_prefill_pair_impl!(ragged_prefill_pair_128_r8, P128, HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>, store_rows8);
ragged_prefill_pair_impl!(ragged_prefill_pair_256_r8, P256, HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>, store_rows8);
ragged_prefill_pair_impl!(ragged_prefill_pair_256_r16, P256, HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>, store_rows16);

macro_rules! ragged_append_impl {
    ($fname:ident, $Pt:ident) => {
        /// Appends the staged prefill rows of a launch into the flat sliding cache: staged row r goes to cache row `index[r] / (Ns * Ds * 2)`
        /// (padding rows go to the scratch slot).
        pub(crate) fn $fname(
            device: &mut Device,
            stage: &HbmTensor<bf16, Chip, m![$Pt, Ns, Ds]>,
            index: &HbmTensor<i32, Chip, m![$Pt]>,
            cache: &mut HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
        ) {
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![Ns, 1 # 32], m![$Pt, Ds]> = stage.to_dm(&mut device.tdma);
            let rows = crate::device::kv8::kv8_t(device, &rows);
            cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
        }
    };
}

ragged_append_impl!(ragged_append_128, P128);
ragged_append_impl!(ragged_append_256, P256);
