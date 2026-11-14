use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Dummy8, Dummy32, Gf, Kf, Kf64, Seg, Seg4, Tc, Tf};
use crate::device::layout::Cluster;

/// 256 slices = 8 query-head groups (2 heads each) x 32 key lanes.
pub(crate) type FKb = m![Gf / 2, Tf % 32];
/// Row statistics after an all-reduce over the 32 key lanes of a head group.
pub(crate) type FX = m![Gf / 2, Dummy32];
/// After an inter-slice Add reduce the result lives in lane 0 of each head group.
pub(crate) type FOne = m![Gf / 2, 1 # 32];
/// Slice axes after the K/V all-gather over the head groups.
pub(crate) type FRows = m![Dummy8, Tf % 32];
/// Key page pair as laid out in DM after the gather: key slot = (Seg, Tf / 256, Tf / 32 % 8), key position inside the
/// page = Tf / 256 * 256 + (Tf / 32 % 8) * 32 + lane.
type FBoth = m![Seg, Tf / 256, Df / 16, Tf / 32 % 8, Df % 16];

/// Loads one 512-row page unique-partitioned over the 256 slices and all-gathers it over the 8 head groups into
/// segment `i` of `both`.
fn gather_one(
    device: &mut Device,
    page: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    both: &mut DmTensor<bf16, Chip, Cluster, FRows, FBoth>,
    i: usize,
) {
    let rows: DmTensor<bf16, Chip, Cluster, m![Tf % 256], m![Tf / 256, Df]> = page.to_dm(&mut device.tdma);
    device
        .main
        .begin(rows.view())
        .fetch::<m![Tf / 256, Df / 16], m![Df % 16]>()
        .switch::<FRows, m![Tf / 256, Df / 16, Tf / 32 % 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 32 })
        .collect::<m![Tf / 256, Df / 16, Tf / 32 % 8], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit_view(
            both.view_mut()
                .tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Tf / 256, Df / 16, Tf / 32 % 8, Df % 16]>(i),
        );
}

fn gather_pages(
    device: &mut Device,
    page0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, FKb, FBoth> {
    let mut both: DmTensor<bf16, Chip, Cluster, FRows, FBoth> = DmTensor::new();
    gather_one(device, page0, &mut both, 0);
    gather_one(device, page1, &mut both, 1);
    unsafe { both.reshape() }
}

/// Full-layer chunk attention over (at most) two 512-key pages. `mask[lane][t][kf]` is the multiplicative 0/1
/// visibility of key slot kf = (page, Tf / 256, Tf / 32 % 8) of key lane `lane` for query row t.
pub(crate) fn full_chunk_attend_2g(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
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
    let k_both = gather_pages(device, k0, k1);
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
    let v_both = gather_pages(device, v0, v1);
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

// ===== variant: K/V loaded by replicated DMA instead of the switch gather =====

/// DMA variant: every slice loads the keys of its lane (all 8 head groups get the same data: replicated DMA).
type FBothD = m![Seg, Tf / 256, Tf / 32 % 8, Df];

fn gather_pages_d(
    device: &mut Device,
    page0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, FKb, FBothD> {
    let mut both: DmTensor<bf16, Chip, Cluster, FKb, FBothD> = DmTensor::new();
    page0.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Tf / 256, Tf / 32 % 8, Df]>(0),
    );
    page1.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg], 1, m![Seg = 1 #{!} 2, Tf / 256, Tf / 32 % 8, Df]>(1),
    );
    both
}

/// Full-layer chunk attention over (at most) two 512-key pages. `mask[lane][t][kf]` is the multiplicative 0/1
/// visibility of key slot kf = (page, Tf / 256, Tf / 32 % 8) of key lane `lane` for query row t.
pub(crate) fn full_chunk_attend_2(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf]>,
    pen: &HbmTensor<bf16, Chip, m![Tf % 32, Kf]>,
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
    let k_both = gather_pages_d(device, k0, k1);
    let v_both = gather_pages_d(device, v0, v1);
    let k_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Tf / 32 % 8], m![Seg, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kf] ----
    // (the VRF operand must carry exactly the stream axes (time Seg, Tf / 256; packet Tf / 32 % 8): the composite name Kf is not matched by the VE)
    let pen_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Kf]> = pen.to_dm(&mut device.tdma);
    let pen_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Seg, Tf / 256, Tf / 32 % 8]> = unsafe { pen_dm.reshape() };
    let pen_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Seg, Tf / 256, Tf / 32 % 8]> = device
        .sub
        .begin(pen_dm.view())
        .fetch::<m![1], m![Seg, Tf / 256, Tf / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Seg, Tf / 256], m![Tf / 32 % 8]>()
        .to_vrf();

    // ---- scores S[Gf % 2, Tc, Kf] (bf16), penalty added in the VE ----
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Seg, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(q_all.view())
        .fetch::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .collect::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Df / 32, Seg, Tf / 256], m![Df % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Seg, Tf / 256]>()
        .contract_lane::<m![Gf % 2, Tc, Seg, Tf / 256], m![Tf / 32 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &pen_vrf)
        .vector_final()
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

// ===== variant: four pages in one launch (DMA loads) =====
/// 4-page variant (Kf64 = 4 pages x 16 key slots).
type FBothD4 = m![Seg4, Tf / 256, Tf / 32 % 8, Df];

fn gather_pages_d4(
    device: &mut Device,
    page0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page2: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    page3: &HbmTensor<bf16, Chip, m![Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, FKb, FBothD4> {
    let mut both: DmTensor<bf16, Chip, Cluster, FKb, FBothD4> = DmTensor::new();
    page0.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg4], 1, m![Seg4 = 1 #{!} 4, Tf / 256, Tf / 32 % 8, Df]>(0),
    );
    page1.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg4], 1, m![Seg4 = 1 #{!} 4, Tf / 256, Tf / 32 % 8, Df]>(1),
    );
    page2.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg4], 1, m![Seg4 = 1 #{!} 4, Tf / 256, Tf / 32 % 8, Df]>(2),
    );
    page3.view().to_dm_view(
        &mut device.tdma,
        both.view_mut().tile::<m![Seg4], 1, m![Seg4 = 1 #{!} 4, Tf / 256, Tf / 32 % 8, Df]>(3),
    );
    both
}

/// Full-layer chunk attention over (at most) two 512-key pages. `mask[lane][t][kf]` is the multiplicative 0/1
/// visibility of key slot kf = (page, Tf / 256, Tf / 32 % 8) of key lane `lane` for query row t.
pub(crate) fn full_chunk_attend_4(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Tc, Gf, Df]>,
    k0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k2: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    k3: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v0: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v1: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v2: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v3: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<bf16, Chip, m![Tf % 32, Tc, Kf64]>,
    pen: &HbmTensor<bf16, Chip, m![Tf % 32, Kf64]>,
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
    let k_both = gather_pages_d4(device, k0, k1, k2, k3);
    let v_both = gather_pages_d4(device, v0, v1, v2, v3);
    let k_trf: TrfTensor<bf16, Chip, Cluster, FKb, m![Tf / 32 % 8], m![Seg4, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kf64] ----
    // (the VRF operand must carry exactly the stream axes (time Seg4, Tf / 256; packet Tf / 32 % 8): the composite name Kf64 is not matched by the VE)
    let pen_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Kf64]> = pen.to_dm(&mut device.tdma);
    let pen_dm: DmTensor<bf16, Chip, Cluster, FKb, m![Seg4, Tf / 256, Tf / 32 % 8]> = unsafe { pen_dm.reshape() };
    let pen_vrf: VrfTensor<f32, Chip, Cluster, FKb, m![Seg4, Tf / 256, Tf / 32 % 8]> = device
        .sub
        .begin(pen_dm.view())
        .fetch::<m![1], m![Seg4, Tf / 256, Tf / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Seg4, Tf / 256], m![Tf / 32 % 8]>()
        .to_vrf();

    // ---- scores S[Gf % 2, Tc, Kf64] (bf16), penalty added in the VE ----
    let s: DmTensor<bf16, Chip, Cluster, FKb, m![Gf % 2, Tc, Seg4, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(q_all.view())
        .fetch::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .collect::<m![Gf % 2, Tc / 32, Tc % 32, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf % 2, Tc, Df / 32, Seg4, Tf / 256], m![Df % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf % 2, Tc, Seg4, Tf / 256]>()
        .contract_lane::<m![Gf % 2, Tc, Seg4, Tf / 256], m![Tf / 32 % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &pen_vrf)
        .vector_final()
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
