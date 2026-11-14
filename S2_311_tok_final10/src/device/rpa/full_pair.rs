// Ragged-attention full-layer prefill part: a PAIR of 64-row prefill units (one per cluster) over the flat page pool viewed as [Pq, R8, Df].
// Generated from device/dattn/full_chunk_flat.rs by lab/gen_full_pair.py: identical math, the only changes are
//   * cluster = unit of the pair (Pw): `Cluster` -> `Cl = m![Pw]`, every HBM argument gets a leading Pw axis (the index is per unit, replicated over the 8 head groups);
//   * the unit's keys are read from the pool AFTER the append of the launch before (`full_chunk_append_flat` / the fused decode launch appends the staged rows first),
//     so a run of consecutive segments of one request needs no staging: unit j simply sees the pool rows of positions <= p0 + 64 j + t.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bk, Df, Dummy32, Dummy8, Gf, Hg, Kf, Kf64, Pb2, Pb4, Pq, Pw, R8, Seg, Seg4, Tc, Tf};

type Cl = m![Pw];
/// 256 slices = 8 query-head groups (2 heads each) x 32 key lanes.
pub(crate) type FKb = m![Gf / 2, Tf % 32];
/// Row statistics after an all-reduce over the 32 key lanes of a head group.
pub(crate) type FX = m![Gf / 2, Dummy32];
/// After an inter-slice Add reduce the result lives in lane 0 of each head group.
pub(crate) type FOne = m![Gf / 2, 1 # 32];
type FBothD = m![Seg, Tf / 256, Tf / 32 % 8, Df];
type FBothD4 = m![Seg4, Tf / 256, Tf / 32 % 8, Df];

fn gather_pages_pair2(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>,
) -> DmTensor<bf16, Chip, Cl, FKb, FBothD> {
    let g: DmTensor<bf16, Chip, Cl, m![Hg, Bk / 2], m![Seg, Bk % 2, R8, Df]> = pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
    unsafe { g.reshape() }
}

fn gather_pages_pair4(
    device: &mut Device,
    pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Pw, Hg, Seg4, Bk]>,
) -> DmTensor<bf16, Chip, Cl, FKb, FBothD4> {
    let g: DmTensor<bf16, Chip, Cl, m![Hg, Bk / 2], m![Seg4, Bk % 2, R8, Df]> = pool.gather::<m![Pq], m![R8, Df]>().by_byte_offsets(index).to_dm(&mut device.tdma);
    unsafe { g.reshape() }
}

pub(crate) fn ragged_full_pair_2(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Pw, Hg, Seg, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
) {
    // ---- q all-gather over the 32 key lanes of each head group ----
    let q_shard: DmTensor<bf16, Chip, Cl, m![Gf / 2, Tc % 32], m![Tc / 32, Gf % 2, Df]> = q.to_dm(&mut device.tdma);
    let q_all: DmTensor<bf16, Chip, Cl, FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> = device
        .main
        .begin(q_shard.view())
        .fetch::<m![Tc / 32, Gf % 2, Df / 16], m![Df % 16]>()
        .switch::<FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Tc / 32, Gf % 2, Df / 16, Tc % 32], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit();
    let q_all: DmTensor<bf16, Chip, Cl, FKb, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> =
        unsafe { q_all.reshape() };

    // ---- keys: gather, then TRF (lane = key slot % 8) ----
    let k_both = gather_pages_pair2(device, k_pool, index);
    let v_both = gather_pages_pair2(device, v_pool, index);
    let k_trf: TrfTensor<bf16, Chip, Cl, FKb, m![Tf / 32 % 8], m![Seg, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kf] (the VRF operand carries exactly the stream axes: time Seg, Tf / 256; packet Tf / 32 % 8) ----
    let pen_dm: DmTensor<bf16, Chip, Cl, FKb, m![Kf]> = pen.to_dm(&mut device.tdma);
    let pen_dm: DmTensor<bf16, Chip, Cl, FKb, m![Seg, Tf / 256, Tf / 32 % 8]> = unsafe { pen_dm.reshape() };
    let pen_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Seg, Tf / 256, Tf / 32 % 8]> = device
        .sub
        .begin(pen_dm.view())
        .fetch::<m![1], m![Seg, Tf / 256, Tf / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Seg, Tf / 256], m![Tf / 32 % 8]>()
        .to_vrf();

    // ---- scores S[Gf % 2, Tc, Kf] (bf16), penalty added in the VE ----
    let s: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Seg, Tf / 256, Tf / 32 % 8]> = device
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
    let s: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Kf]> = unsafe { s.reshape() };

    // ---- row max over all key slots (all-reduce Max over the 32 key lanes of a head group) ----
    let m_dm: DmTensor<f32, Chip, Cl, FX, m![Gf % 2, Tc]> = device
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
    let m_dm: DmTensor<f32, Chip, Cl, FKb, m![Gf % 2, Tc]> = unsafe { m_dm.reshape() };

    // ---- masks [Tc, Kf] per slice ----
    let mask_dm: DmTensor<bf16, Chip, Cl, FKb, m![Tc, Kf]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows ----
    let mut p: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Kf]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler); `#[unroll]` is avoided on purpose, see prefill.rs (it drops the decode attention of the N3 kernel).
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kf]>(16 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Tc = 16, Kf]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 16, Kf]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 16, Kf / 8], m![Kf % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64]>(16 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Gf % 2, Tc = 16]> = device
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
        }};
    }
    p_tile!(0);
    p_tile!(1);
    p_tile!(2);
    p_tile!(3);

    // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
    let l_dm: DmTensor<f32, Chip, Cl, FX, m![Gf % 2, Tc]> = device
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
    let l_dm: DmTensor<f32, Chip, Cl, FOne, m![Gf % 2, Tc]> = unsafe { l_dm.reshape() };
    let l_vrf: VrfTensor<f32, Chip, Cl, FOne, m![Gf % 2, Tc]> = device
        .sub
        .begin(l_dm.view())
        .fetch::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .collect::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .to_vrf();

    // ---- values: gather, transpose to [Df, Kf], TRF (lane = Df % 8) ----
    let v_t: DmTensor<bf16, Chip, Cl, FKb, m![Df, Seg, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(v_both.view())
        .fetch::<m![Df / 8, Seg, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Seg, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Seg, Tf / 256, Tf / 128 % 2, Df % 8], m![Tf / 32 % 4 # 16]>()
        .commit_trim::<m![Tf / 32 % 4]>()
        .commit();
    let v_t: DmTensor<bf16, Chip, Cl, FKb, m![Df, Kf]> = unsafe { v_t.reshape() };
    let v_trf: TrfTensor<bf16, Chip, Cl, FKb, m![Df % 8], m![Df / 8, Kf]> = device
        .sub
        .begin(v_t.view())
        .fetch::<m![Df % 8, Df / 8, Kf / 16], m![Kf % 16]>()
        .collect::<m![Df % 8, Df / 8, Kf / 16], m![Kf % 16]>()
        .to_trf();

    // ---- PV: contraction over the 32 key slots, then all-reduce Add over the key lanes and divide by the row sum ----
    let o: DmTensor<bf16, Chip, Cl, FOne, m![Gf % 2, Tc, Df]> = device
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


pub(crate) fn ragged_full_pair_4(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
    k_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    v_pool: &HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
    index: &HbmTensor<i32, Chip, m![Pw, Hg, Seg4, Bk]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Tc, Kf64]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Tf % 32, Kf64]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Gf, Df]>,
) {
    // ---- q all-gather over the 32 key lanes of each head group ----
    let q_shard: DmTensor<bf16, Chip, Cl, m![Gf / 2, Tc % 32], m![Tc / 32, Gf % 2, Df]> = q.to_dm(&mut device.tdma);
    let q_all: DmTensor<bf16, Chip, Cl, FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> = device
        .main
        .begin(q_shard.view())
        .fetch::<m![Tc / 32, Gf % 2, Df / 16], m![Df % 16]>()
        .switch::<FX, m![Tc / 32, Gf % 2, Df / 16, Tc % 32]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Tc / 32, Gf % 2, Df / 16, Tc % 32], m![Df % 16]>()
        .commit_trim::<m![Df % 16]>()
        .commit();
    let q_all: DmTensor<bf16, Chip, Cl, FKb, m![Tc / 32, Gf % 2, Df / 16, Tc % 32, Df % 16]> =
        unsafe { q_all.reshape() };

    // ---- keys: gather, then TRF (lane = key slot % 8) ----
    let k_both = gather_pages_pair4(device, k_pool, index);
    let v_both = gather_pages_pair4(device, v_pool, index);
    let k_trf: TrfTensor<bf16, Chip, Cl, FKb, m![Tf / 32 % 8], m![Seg4, Tf / 256, Df]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .collect::<m![Tf / 32 % 8, Seg4, Tf / 256, Df / 16], m![Df % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kf64] (the VRF operand carries exactly the stream axes: time Seg4, Tf / 256; packet Tf / 32 % 8) ----
    let pen_dm: DmTensor<bf16, Chip, Cl, FKb, m![Kf64]> = pen.to_dm(&mut device.tdma);
    let pen_dm: DmTensor<bf16, Chip, Cl, FKb, m![Seg4, Tf / 256, Tf / 32 % 8]> = unsafe { pen_dm.reshape() };
    let pen_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Seg4, Tf / 256, Tf / 32 % 8]> = device
        .sub
        .begin(pen_dm.view())
        .fetch::<m![1], m![Seg4, Tf / 256, Tf / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Seg4, Tf / 256], m![Tf / 32 % 8]>()
        .to_vrf();

    // ---- scores S[Gf % 2, Tc, Kf64] (bf16), penalty added in the VE ----
    let s: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Seg4, Tf / 256, Tf / 32 % 8]> = device
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
    let s: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Kf64]> = unsafe { s.reshape() };

    // ---- row max over all key slots (all-reduce Max over the 32 key lanes of a head group) ----
    let m_dm: DmTensor<f32, Chip, Cl, FX, m![Gf % 2, Tc]> = device
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
    let m_dm: DmTensor<f32, Chip, Cl, FKb, m![Gf % 2, Tc]> = unsafe { m_dm.reshape() };

    // ---- masks [Tc, Kf64] per slice ----
    let mask_dm: DmTensor<bf16, Chip, Cl, FKb, m![Tc, Kf64]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 16 query rows ----
    let mut p: DmTensor<bf16, Chip, Cl, FKb, m![Gf % 2, Tc, Kf64]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler); `#[unroll]` is avoided on purpose, see prefill.rs (it drops the decode attention of the N3 kernel).
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_dm.view().tile::<m![Tc], 16, m![Tc = 16 # 64, Kf64]>(16 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Tc = 16, Kf64]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 16, Kf64]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 16, Kf64 / 8], m![Kf64 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 16, m![Gf % 2, Tc = 16 # 64]>(16 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, FKb, m![Gf % 2, Tc = 16]> = device
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
        }};
    }
    p_tile!(0);
    p_tile!(1);
    p_tile!(2);
    p_tile!(3);

    // ---- row sum of the (bf16) weights, all-reduced; EPS added once in total ----
    let l_dm: DmTensor<f32, Chip, Cl, FX, m![Gf % 2, Tc]> = device
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
    let l_dm: DmTensor<f32, Chip, Cl, FOne, m![Gf % 2, Tc]> = unsafe { l_dm.reshape() };
    let l_vrf: VrfTensor<f32, Chip, Cl, FOne, m![Gf % 2, Tc]> = device
        .sub
        .begin(l_dm.view())
        .fetch::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .collect::<m![Gf % 2, Tc / 8], m![Tc % 8]>()
        .to_vrf();

    // ---- values: gather, transpose to [Df, Kf64], TRF (lane = Df % 8) ----
    let v_t: DmTensor<bf16, Chip, Cl, FKb, m![Df, Seg4, Tf / 256, Tf / 32 % 8]> = device
        .main
        .begin(v_both.view())
        .fetch::<m![Df / 8, Seg4, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Seg4, Tf / 256, Tf / 32 % 8], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Seg4, Tf / 256, Tf / 128 % 2, Df % 8], m![Tf / 32 % 4 # 16]>()
        .commit_trim::<m![Tf / 32 % 4]>()
        .commit();
    let v_t: DmTensor<bf16, Chip, Cl, FKb, m![Df, Kf64]> = unsafe { v_t.reshape() };
    let v_trf: TrfTensor<bf16, Chip, Cl, FKb, m![Df % 8], m![Df / 8, Kf64]> = device
        .sub
        .begin(v_t.view())
        .fetch::<m![Df % 8, Df / 8, Kf64 / 16], m![Kf64 % 16]>()
        .collect::<m![Df % 8, Df / 8, Kf64 / 16], m![Kf64 % 16]>()
        .to_trf();

    // ---- PV: contraction over the 32 key slots, then all-reduce Add over the key lanes and divide by the row sum ----
    let o: DmTensor<bf16, Chip, Cl, FOne, m![Gf % 2, Tc, Df]> = device
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


macro_rules! ragged_full_append_impl {
    ($fname:ident, $Pb:ident) => {
        /// Appends the staged chunk rows of a launch into the page pool viewed as 8-row blocks: staged block b goes to pool block `index[b] / (8 * Df * 2)`.
        /// Chunks are 64-row aligned (p0 % 64 == 0), so every unit writes whole blocks; padding blocks go to the scratch page.
        pub(crate) fn $fname(
            device: &mut Device,
            chunk: &HbmTensor<bf16, Chip, m![$Pb, R8, Df]>,
            index: &HbmTensor<i32, Chip, m![$Pb]>,
            pool: &mut HbmTensor<bf16, Chip, m![Pq, R8, Df]>,
        ) {
            let rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![$Pb, R8, Df]> = chunk.to_dm(&mut device.tdma);
            pool.view_mut().scatter::<m![Pq], m![R8, Df]>().by_byte_offsets(index.view()).from_dm(&mut device.tdma, rows);
        }
    };
}

ragged_full_append_impl!(ragged_full_append_2, Pb2);
ragged_full_append_impl!(ragged_full_append_4, Pb4);
