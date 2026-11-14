// Chunked-prefill full-layer attention on the flat unified page pool (generated from device/chunk/full_chunk.rs).
// The pool is viewed as `[Pq, R8, Df]` (8-row blocks); K / V of the request's pages are gathered (replicated over the 8 head groups
// through a replicated index) with one block per index entry. Key assignment: key j of page `seg` lives in lane j / 16, slot
// (j / 8 % 2, j % 8); the mask is `dattn_host::full_chunk_mask_flat`.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bk, Df, Dummy32, Dummy8, Gf, Hg, Kf, Kf64, Pq, R8, Seg, Seg4, Tc, Tf};
use crate::device::layout::Cluster;

/// 256 slices = 8 query-head groups (2 heads each) x 32 key lanes.
pub(crate) type FKb = m![Gf / 2, Tf % 32];
/// Row statistics after an all-reduce over the 32 key lanes of a head group.
pub(crate) type FX = m![Gf / 2, Dummy32];
/// After an inter-slice Add reduce the result lives in lane 0 of each head group.
pub(crate) type FOne = m![Gf / 2, 1 # 32];
type FBothD = m![Seg, Tf / 256, Tf / 32 % 8, Df];
type FBothD4 = m![Seg4, Tf / 256, Tf / 32 % 8, Df];

fn gather_pages_flat2(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg, Bk]>,
) -> DmTensor<bf16, Chip, Cluster, FKb, FBothD> {
    let g: DmTensor<bf16, Chip, Cluster, m![Hg, Bk / 2], m![Seg, Bk % 2, R8, Df]> = pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
    unsafe { g.reshape() }
}

fn gather_pages_flat4(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg4, Bk]>,
) -> DmTensor<bf16, Chip, Cluster, FKb, FBothD4> {
    let g: DmTensor<bf16, Chip, Cluster, m![Hg, Bk / 2], m![Seg4, Bk % 2, R8, Df]> = pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
    unsafe { g.reshape() }
}

pub(crate) fn full_chunk_attend_2_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    // ---- q all-gather over the 32 key lanes of each head group ----
    let q_shard: DmTensor<bf16, Chip, Cluster, m![Gf / 2, Tc % 32], m![Tc / 32, Gf % 2, Df]> = q.to_dm(&mut device.tdma);
    let q_all: DmTensor<bf16, Chip, Cluster, FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> = device
        .main
        .begin(q_shard.view())
        .fetch::<m![Tc / 32, Gf % 2, Df / 16], m![Df % 16]>()
        .switch::<FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Tc / 32, Gf % 2, Df / 16, Tc % 32], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit();
    let q_all: DmTensor<bf16, Chip, Cluster, FKb, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> =
        unsafe { q_all.reshape() };

    // ---- keys: gather, then TRF (lane = key slot % 8) ----
    let k_both = gather_pages_flat2(device, k_pool, index);
    let v_both = gather_pages_flat2(device, v_pool, index);
    let k_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Tf / 32 % 8], m![Seg, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- scores S[Gf % 2, Tc, Kf] (bf16) ----
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Seg, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(q_all.view())
        .fetch::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .collect::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Df / 32, Seg, Tf / 256], m![Df % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Seg, Tf / 256]>()
        .contract_lane::<m![Gf % 2, Tc, Seg, Tf / 256], m![Tf / 32 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf / 32 % 8 # 16]>()
        .commit_trim::<m![Tf / 32 % 8]>()
        .commit();
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Kf]> = unsafe { s.reshape() };

    // ---- row max over all key slots (all-reduce Max over the 32 key lanes of a head group) ----
    let m_dm: DmTensor<f32, Chip, Cluster, FX, m![Gf % 2, Tc]> = device
        .main
        .begin(s.view())
        .fetch::<m![Gf % 2, Tc, Kf / 16], m![Kf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Tc, Kf / 8], m![Kf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Kf / 4], m![Kf % 4]>()
        .vector_intra_slice_reduce::<Kf, m![Gf % 2, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<FX, m![Gf % 2, Tc]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .transpose::<m![Gf % 2, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let m_dm: DmTensor<f32, Chip, Cluster, FKb, m![Gf % 2, Tc]> = unsafe { m_dm.reshape() };

    // ---- masks [Tc, Kf] per slice ----
    let mask_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Tc, Kf]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows ----
    let mut p: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Kf]> = DmTensor::new();
    for i in 0..Tc::SIZE / 16 {
        let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kf]>(16 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Tc = 16, Kf]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 16, Kf]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 16, Kf / 8], m![Kf % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64]>(16 * i);
        let m_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Gf % 2, Tc = 16]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gf % 2], m![Tc = 16]>()
            .collect::<m![Gf % 2, Tc = 16 / 8], m![Tc = 16 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64, Kf]>(16 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gf % 2], m![Tc = 16, Kf]>()
            .fetch_cast::<f32>()
            .collect::<m![Gf % 2, Tc = 16, Kf / 8], m![Kf % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gf % 2, Tc = 16, Kf / 4], m![Kf % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gf % 2, Tc = 16, Kf / 8], m![Kf % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kf % 8 # 16]>()
            .commit_trim::<m![Kf % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 #{!} 64, Kf]>(16 * i));
    }

    // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
    let l_dm: DmTensor<f32, Chip, Cluster, FX, m![Gf % 2, Tc]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gf % 2, Tc, Kf / 16], m![Kf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Tc, Kf / 8], m![Kf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Kf / 4], m![Kf % 4]>()
        .vector_intra_slice_reduce::<Kf, m![Gf % 2, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
        .vector_inter_slice_reduce::<FX, m![Gf % 2, Tc]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![Gf % 2, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let l_dm: DmTensor<f32, Chip, Cluster, FOne, m![Gf % 2, Tc]> = unsafe { l_dm.reshape() };
    let l_vrf: VrfTensor<f32, Chip, Cluster, FOne, m![Gf % 2, Tc]> = device
        .sub
        .begin(l_dm.view())
        .fetch::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .collect::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .to_vrf();

    // ---- values: gather, transpose to [Df, Kf], TRF (lane = Df % 8) ----
    let v_t: DmTensor<bf16, Chip, Cluster, FKb, m![Df, Seg, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(v_both.view())
        .fetch::<m![Df / 8, Seg, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Seg, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Seg, Tf / 256, Tf / 128 % 2, Df % 8], m![Tf / 32 % 4 # 16]>()
        .commit_trim::<m![Tf / 32 % 4]>()
        .commit();
    let v_t: DmTensor<bf16, Chip, Cluster, FKb, m![Df, Kf]> = unsafe { v_t.reshape() };
    let v_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Df % 8], m![Df / 8, Kf]> = device
        .sub
        .begin(v_t.view())
        .fetch::<m![Df % 8, Df / 8, Kf / 16], m![Kf % 16]>()
        .collect::<m![Df % 8, Df / 8, Kf / 16], m![Kf % 16]>()
        .to_trf();

    // ---- PV: contraction over the 32 key slots, then all-reduce Add over the key lanes and divide by the row sum ----
    let o: DmTensor<bf16, Chip, Cluster, FOne, m![Gf % 2, Tc, Df]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gf % 2, Tc, Kf / 16], m![Kf % 16]>()
        .collect::<m![Gf % 2, Tc, Kf / 16], m![Kf % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Df / 8], m![Kf], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Df / 8]>()
        .contract_lane::<m![Gf % 2, Tc, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<FOne, m![Gf % 2, Tc, Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Df / 4], m![Df % 4]>()
        .vector_fp_div(&l_vrf)
        .vector_widen_concat::<m![Gf % 2, Tc, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();

    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}


pub(crate) fn full_chunk_attend_4_flat(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Hg, Seg4, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf64]>,
    out: &mut HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
) {
    // ---- q all-gather over the 32 key lanes of each head group ----
    let q_shard: DmTensor<bf16, Chip, Cluster, m![Gf / 2, Tc % 32], m![Tc / 32, Gf % 2, Df]> = q.to_dm(&mut device.tdma);
    let q_all: DmTensor<bf16, Chip, Cluster, FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> = device
        .main
        .begin(q_shard.view())
        .fetch::<m![Tc / 32, Gf % 2, Df / 16], m![Df % 16]>()
        .switch::<FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Tc / 32, Gf % 2, Df / 16, Tc % 32], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit();
    let q_all: DmTensor<bf16, Chip, Cluster, FKb, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> =
        unsafe { q_all.reshape() };

    // ---- keys: gather, then TRF (lane = key slot % 8) ----
    let k_both = gather_pages_flat4(device, k_pool, index);
    let v_both = gather_pages_flat4(device, v_pool, index);
    let k_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Tf / 32 % 8], m![Seg4, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- scores S[Gf % 2, Tc, Kf64] (bf16) ----
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Seg4, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(q_all.view())
        .fetch::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .collect::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Df / 32, Seg4, Tf / 256], m![Df % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Seg4, Tf / 256]>()
        .contract_lane::<m![Gf % 2, Tc, Seg4, Tf / 256], m![Tf / 32 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf / 32 % 8 # 16]>()
        .commit_trim::<m![Tf / 32 % 8]>()
        .commit();
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Kf64]> = unsafe { s.reshape() };

    // ---- row max over all key slots (all-reduce Max over the 32 key lanes of a head group) ----
    let m_dm: DmTensor<f32, Chip, Cluster, FX, m![Gf % 2, Tc]> = device
        .main
        .begin(s.view())
        .fetch::<m![Gf % 2, Tc, Kf64 / 16], m![Kf64 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Tc, Kf64 / 8], m![Kf64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Kf64 / 4], m![Kf64 % 4]>()
        .vector_intra_slice_reduce::<Kf64, m![Gf % 2, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<FX, m![Gf % 2, Tc]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .transpose::<m![Gf % 2, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let m_dm: DmTensor<f32, Chip, Cluster, FKb, m![Gf % 2, Tc]> = unsafe { m_dm.reshape() };

    // ---- masks [Tc, Kf64] per slice ----
    let mask_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Tc, Kf64]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows ----
    let mut p: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Kf64]> = DmTensor::new();
    for i in 0..Tc::SIZE / 16 {
        let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kf64]>(16 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Tc = 16, Kf64]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 16, Kf64]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 16, Kf64 / 8], m![Kf64 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64]>(16 * i);
        let m_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Gf % 2, Tc = 16]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gf % 2], m![Tc = 16]>()
            .collect::<m![Gf % 2, Tc = 16 / 8], m![Tc = 16 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64, Kf64]>(16 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gf % 2], m![Tc = 16, Kf64]>()
            .fetch_cast::<f32>()
            .collect::<m![Gf % 2, Tc = 16, Kf64 / 8], m![Kf64 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gf % 2, Tc = 16, Kf64 / 4], m![Kf64 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gf % 2, Tc = 16, Kf64 / 8], m![Kf64 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kf64 % 8 # 16]>()
            .commit_trim::<m![Kf64 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 #{!} 64, Kf64]>(16 * i));
    }

    // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
    let l_dm: DmTensor<f32, Chip, Cluster, FX, m![Gf % 2, Tc]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gf % 2, Tc, Kf64 / 16], m![Kf64 % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf % 2, Tc, Kf64 / 8], m![Kf64 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Kf64 / 4], m![Kf64 % 4]>()
        .vector_intra_slice_reduce::<Kf64, m![Gf % 2, Tc], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, crate::SOFTMAX_EPS / 32.0)
        .vector_inter_slice_reduce::<FX, m![Gf % 2, Tc]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![Gf % 2, Tc / 2], m![Tc % 2 # 8]>()
        .commit_trim::<m![Tc % 2]>()
        .commit();
    let l_dm: DmTensor<f32, Chip, Cluster, FOne, m![Gf % 2, Tc]> = unsafe { l_dm.reshape() };
    let l_vrf: VrfTensor<f32, Chip, Cluster, FOne, m![Gf % 2, Tc]> = device
        .sub
        .begin(l_dm.view())
        .fetch::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .collect::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .to_vrf();

    // ---- values: gather, transpose to [Df, Kf64], TRF (lane = Df % 8) ----
    let v_t: DmTensor<bf16, Chip, Cluster, FKb, m![Df, Seg4, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(v_both.view())
        .fetch::<m![Df / 8, Seg4, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Seg4, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Seg4, Tf / 256, Tf / 128 % 2, Df % 8], m![Tf / 32 % 4 # 16]>()
        .commit_trim::<m![Tf / 32 % 4]>()
        .commit();
    let v_t: DmTensor<bf16, Chip, Cluster, FKb, m![Df, Kf64]> = unsafe { v_t.reshape() };
    let v_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Df % 8], m![Df / 8, Kf64]> = device
        .sub
        .begin(v_t.view())
        .fetch::<m![Df % 8, Df / 8, Kf64 / 16], m![Kf64 % 16]>()
        .collect::<m![Df % 8, Df / 8, Kf64 / 16], m![Kf64 % 16]>()
        .to_trf();

    // ---- PV: contraction over the 32 key slots, then all-reduce Add over the key lanes and divide by the row sum ----
    let o: DmTensor<bf16, Chip, Cluster, FOne, m![Gf % 2, Tc, Df]> = device
        .main
        .begin(p.view())
        .fetch::<m![Gf % 2, Tc, Kf64 / 16], m![Kf64 % 16]>()
        .collect::<m![Gf % 2, Tc, Kf64 / 16], m![Kf64 % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Kf64 / 32, Df / 8], m![Kf64 % 32], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Df / 8]>()
        .contract_lane::<m![Gf % 2, Tc, Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<FOne, m![Gf % 2, Tc, Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf % 2, Tc, Df / 4], m![Df % 4]>()
        .vector_fp_div(&l_vrf)
        .vector_widen_concat::<m![Gf % 2, Tc, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();

    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

