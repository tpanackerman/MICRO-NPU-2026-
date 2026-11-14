// Chunked-prefill sliding attention on the unified flat cache [Rf, Ns, Ds] (generated from device/chunk/sliding_chunk.rs).
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Dummy32, Ds, Gs, Kk, Ns, Pr, Rf, Seg, Tc, Ts, Tx};
use crate::device::layout::Cluster;

/// 256 slices: head x key lane (ring row r lives in slice (h, r % 32), slot r / 32).
type HeadKb = m![Ns, Ts % 32];
/// Row statistics after an inter-slice all-reduce (fresh broadcast axis).
type HeadX = m![Ns, Dummy32];
/// After an inter-slice Add reduce the result lives in key lane 0 of each head.
type HeadOne = m![Ns, 1 # 32];

/// Sliding-layer chunk attention. Each slice (head h, key lane l) holds 64 key slots:
/// slots 0..32 = ring rows r with r % 32 == l (slot r / 32), slots 32..64 = chunk keys c with
/// c % 32 == l (slot 32 + c / 32, only slots 32, 33 are non-zero for Tc = 64; the rest is zeroed by memset).
/// `mask[l][t][seg][s]` is the multiplicative 0/1 visibility of the key in slot (seg, s) of lane l for query row t.
pub(crate) fn sliding_chunk_attend_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
    ring_index: &HbmTensor<i32, Chip, m![Tx]>,
    k_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    v_stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    mask: &HbmTensor<bf16, Chip, m![Ts % 32, Tc, Seg, Ts / 32]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Ns, Gs, Ds]>,
) {
    // ---- q all-gather over the 32 key-block slices of each head ----
    let q_shard: DmTensor<bf16, Chip, Cluster, m![Ns, Tc % 32], m![Tc / 32, Gs, Ds]> = q.to_dm(&mut device.tdma);
    let q_all: DmTensor<bf16, Chip, Cluster, HeadX, m![Tc / 32, Gs, Ds / 16, Tc % 32, Ds % 16]> = device
        .main
        .begin(q_shard.view())
        .fetch::<m![Tc / 32, Gs, Ds / 16], m![Ds % 16]>()
        .switch::<HeadX, m![Tc / 32, Gs, Ds / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Tc / 32, Gs, Ds / 16, Tc % 32], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();
    let q_all: DmTensor<bf16, Chip, Cluster, HeadKb, m![Tc / 32, Gs, Ds / 16, Tc % 32, Ds % 16]> =
        unsafe { q_all.reshape() };

    // ---- key slots: ring block + chunk block (zero filled where the chunk has no keys) ----
    let mut k_both: DmTensor<bf16, Chip, Cluster, HeadKb, m![Seg, Ts / 32, Ds]> = DmTensor::new();
    k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let k_ring: DmTensor<bf16, Chip, Cluster, m![Ns, Tx % 32], m![Tx / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(ring_index).to_dm(&mut device.tdma);
    let k_ring: DmTensor<bf16, Chip, Cluster, HeadKb, m![Ts / 32, Ds]> = unsafe { k_ring.reshape() };
    device
        .main
        .begin(k_ring.view())
        .fetch::<m![Ts / 32, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ts / 32, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Ts / 32, Ds]>(0));
    k_stage.view().to_dm_view(
        &mut device.tdma,
        k_both
            .view_mut()
            .tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Ts / 32, Ds]>(1)
            .tile::<m![Ts / 32], 2, m![Seg = 1 #{!} 2, Ts / 32 = 2 #{!} 32, Ds]>(0),
    );
    let k_both: DmTensor<bf16, Chip, Cluster, HeadKb, m![Kk, Ds]> = unsafe { k_both.reshape() };
    let k_trf: TrfTensor<bf16, Chip, Cluster, HeadKb, m![Kk % 8], m![Kk / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk % 8, Kk / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk % 8, Kk / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- scores S[Gs, Tc, Kk] (bf16) ----
    let s: DmTensor<bf16, Chip, Cluster, HeadKb, m![Gs, Tc, Kk]> = device
        .main
        .begin(q_all.view())
        .fetch::<m![Gs, Tc / 32, Tc % 32, Ds / 16], m![Ds % 16]>()
        .collect::<m![Gs, Tc / 32, Tc % 32, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Gs, Tc, Ds / 32, Kk / 8], m![Ds % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gs, Tc, Kk / 8]>()
        .contract_lane::<m![Gs, Tc, Kk / 8], m![Kk % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Kk % 8 # 16]>()
        .commit_trim::<m![Kk % 8]>()
        .commit();

    // ---- row max over all key slots (all-reduce Max over the head's 32 key-block slices) ----
    let m_dm: DmTensor<f32, Chip, Cluster, HeadX, m![Gs, Tc]> = device
        .main
        .begin(s.view())
        .fetch::<m![Gs, Tc, Kk / 16], m![Kk % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Tc, Kk / 8], m![Kk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Tc, Kk / 4], m![Kk % 4]>()
        .vector_intra_slice_reduce::<Kk, m![Gs, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<HeadX, m![Gs, Tc]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .transpose::<m![Gs, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let m_dm: DmTensor<f32, Chip, Cluster, HeadKb, m![Gs, Tc]> = unsafe { m_dm.reshape() };

    // ---- masks: [Tc, Seg, 32 slots] per slice (host builds the zero padding of the chunk part) ----
    let mask_dm: DmTensor<bf16, Chip, Cluster, HeadKb, m![Tc, Seg, Ts / 32]> = mask.to_dm(&mut device.tdma);
    let mask_both: DmTensor<bf16, Chip, Cluster, HeadKb, m![Tc, Kk]> = unsafe { mask_dm.reshape() };

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows (VRF holds one mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cluster, HeadKb, m![Gs, Tc, Kk]> = DmTensor::new();
    for i in 0..Tc::SIZE / 16 {
        let mask_tile = mask_both.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kk]>(16 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cluster, HeadKb, m![Tc = 16, Kk]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 16, Kk]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 16, Kk / 8], m![Kk % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gs, Tc = 16 # 64]>(16 * i);
        let m_vrf: VrfTensor<f32, Chip, Cluster, HeadKb, m![Gs, Tc = 16]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 16]>()
            .collect::<m![Gs, Tc = 16 / 8], m![Tc = 16 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 16, m![Gs, Tc = 16 # 64, Kk]>(16 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 16, Kk]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 16, Kk / 8], m![Kk % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 16, Kk / 4], m![Kk % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 16, Kk / 8], m![Kk % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk % 8 # 16]>()
            .commit_trim::<m![Kk % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gs, Tc = 16 #{!} 64, Kk]>(16 * i));
    }

    // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
    let l_dm: DmTensor<f32, Chip, Cluster, HeadX, m![Gs, Tc]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gs, Tc, Kk / 16], m![Kk % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Tc, Kk / 8], m![Kk % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Tc, Kk / 4], m![Kk % 4]>()
        .vector_intra_slice_reduce::<Kk, m![Gs, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
        .vector_inter_slice_reduce::<HeadX, m![Gs, Tc]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![Gs, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let l_dm: DmTensor<f32, Chip, Cluster, HeadOne, m![Gs, Tc]> = unsafe { l_dm.reshape() };
    let l_vrf: VrfTensor<f32, Chip, Cluster, HeadOne, m![Gs, Tc]> = device
        .sub
        .begin(l_dm.view())
        .fetch::<m![Gs, Tc / 8], m![Tc % 8]>()
        .collect::<m![Gs, Tc / 8], m![Tc % 8]>()
        .to_vrf();

    // ---- PV over both segments (Seg is a reduced time axis) ----
    let mut v_both: DmTensor<bf16, Chip, Cluster, HeadKb, m![Seg, Ts / 32, Ds]> = DmTensor::new();
    v_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let v_ring: DmTensor<bf16, Chip, Cluster, m![Ns, Tx % 32], m![Tx / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(ring_index).to_dm(&mut device.tdma);
    let v_ring: DmTensor<bf16, Chip, Cluster, HeadKb, m![Ts / 32, Ds]> = unsafe { v_ring.reshape() };
    device
        .main
        .begin(v_ring.view())
        .fetch::<m![Ts / 32, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ts / 32, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Ts / 32, Ds]>(0));
    v_stage.view().to_dm_view(
        &mut device.tdma,
        v_both
            .view_mut()
            .tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Ts / 32, Ds]>(1)
            .tile::<m![Ts / 32], 2, m![Seg = 1 #{!} 2, Ts / 32 = 2 #{!} 32, Ds]>(0),
    );
    let v_both: DmTensor<bf16, Chip, Cluster, HeadKb, m![Kk, Ds]> = unsafe { v_both.reshape() };
    let v_t: DmTensor<bf16, Chip, Cluster, HeadKb, m![Ds, Kk]> = device
        .main
        .begin(v_both.view())
        .fetch::<m![Ds / 8, Kk], m![Ds % 8 # 16]>()
        .collect::<m![Ds / 8, Kk], m![Ds % 8 # 16]>()
        .transpose::<m![Ds / 8, Kk / 4, Ds % 8], m![Kk % 4 # 16]>()
        .commit_trim::<m![Kk % 4]>()
        .commit();
    let v_trf: TrfTensor<bf16, Chip, Cluster, HeadKb, m![Ds % 8], m![Ds / 8, Kk]> = device
        .sub
        .begin(v_t.view())
        .fetch::<m![Ds % 8, Ds / 8, Kk / 16], m![Kk % 16]>()
        .collect::<m![Ds % 8, Ds / 8, Kk / 16], m![Kk % 16]>()
        .to_trf();
    let o: DmTensor<bf16, Chip, Cluster, HeadOne, m![Gs, Tc, Ds]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gs, Tc, Kk / 16], m![Kk % 16]>()
        .collect::<m![Gs, Tc, Kk / 16], m![Kk % 16]>()
        .contract_outer::<m![Gs, Tc, Kk / 32, Ds / 8], m![Kk % 32], _, _, _>(&v_trf)
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

    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

#[allow(dead_code)]
type _Unused = Dummy32;

/// Appends the chunk's staged k (or v) rows into the flat unified cache: staged row c goes to cache row `index[c] / (Ns * Ds * 2)`
/// (`dattn_host::sliding_chunk_append_index`).
pub(crate) fn append_chunk_rows_flat(
    device: &mut Device,
    stage: &HbmTensor<bf16, Chip, m![Ts = 64, Ns, Ds]>,
    index: &HbmTensor<i32, Chip, m![Ts = 64]>,
    cache: &mut HbmTensor<bf16, Chip, m![Rf, Ns, Ds]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![Ns, 1 # 32], m![Ts = 64, Ds]> = stage.to_dm(&mut device.tdma);
    cache.view_mut().scatter::<m![Rf], m![Ns, Ds]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
}

/// Same for a full-layer chunk: rows [Tc, Df] into the flat page pool [Pr, Df].
pub(crate) fn append_full_chunk_rows_flat(
    device: &mut Device,
    chunk: &HbmTensor<bf16, Chip, m![Tc, Df]>,
    index: &HbmTensor<i32, Chip, m![Tc]>,
    pool: &mut HbmTensor<bf16, Chip, m![Pr, Df]>,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![1 # 256], m![Tc, Df]> = chunk.to_dm(&mut device.tdma);
    pool.view_mut().scatter::<m![Pr], m![Df]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
}
