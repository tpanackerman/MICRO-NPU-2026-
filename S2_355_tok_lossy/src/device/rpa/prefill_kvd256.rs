// KVD-256 variant (generated from prefill_kvd.rs by wip/apply_kvd256.py: Kw 1152 -> Kw5 1280 = 40 key slots, so a run of up to 256 rows (4 units) reads all its chunk keys back from the ring).
// Original header: KVD variant of the ragged prefill pair (hybrid N3 launch only): the K / V of the launch's own rows are ALREADY in the ring (the QKV launch scattered them), so the unit's keys
// are ONE window gather of Kw = 1024 + 128 ring rows (`win_index[u][i]` = byte offset of ring row of position `p0 - 1024 + i`): window rows 0..1023 = the old ring part (slots 0..31 of
// each key lane), rows 1024..1151 = the chunk keys c = i - 1024 (slots 32..35; a run of the S = 2 bucket has <= 128 rows). The key SLOT layout (slot = i / 32, lane = i % 32, then
// the zero-filled rest of the 64 slots) is identical to the stage-gather version, so scores, softmax and PV see the same values in the same positions.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

type Cl = m![Pw];
type HeadKb = m![Ns, Ts % 32];
type HeadX = m![Ns, Dummy32];
type HeadOne = m![Ns, 1 # 32];

pub(crate) fn ragged_prefill_pair_kvd256(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
) {
    let o = ragged_prefill_pair_kvd256_o(device, q, k_cache, v_cache, win_index, mask, pen);
    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// R15 gapimpl F1 (AO8 pre-cast): `ragged_prefill_pair_kvd256` whose output is the AO8 input of the fused K launch: e4m3(f32(bf16 o) x 16) (the arithmetic of
/// `ffnc::ao::cvt8_t128`), stored slice-contiguous (`O8P0` = `[C32, Tk128, E32]` bytes) so each AO slice reads one contiguous run.
pub(crate) fn ragged_prefill_pair_kvd256_o8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<f8e4m3, Chip, O8P0>,
) {
    let o = ragged_prefill_pair_kvd256_o(device, q, k_cache, v_cache, win_index, mask, pen);
    let o8 = o8_cast(device, &o);
    o8.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// R15 gapimpl F1: the e4m3 attention outputs of the fused K launches, slice-contiguous for the AO (`[Ns, Gs, Ds / 128]` = C32, `[Pw, Tc]` = Tk128, `Ds % 128` = E32).
pub(crate) type O8P0 = m![Ns, Gs, Ds / 128, Pw, Tc, Ds % 128];
pub(crate) type O8P1 = m![Ns, Gs, Ds / 128, Pw, Rg4, Bd16, Ds % 128];

/// R15 gapimpl F1: `ffnc::ao::cvt8_t128`'s per-element arithmetic (bf16 -> f32, x 16 (exact, pow2), RNE cast to e4m3) on the committed bf16 attention output.
/// The pass reads o in the order of the slice-contiguous HBM layout (`Ds / 128` outside `Tc`), so the store moves contiguous 8 KB runs on both sides.
fn o8_cast(device: &mut Device, o: &DmTensor<bf16, Chip, Cl, HeadOne, m![Gs, Tc, Ds]>) -> DmTensor<f8e4m3, Chip, Cl, HeadOne, m![Gs, Ds / 128, Tc, Ds % 128]> {
    device
        .main
        .begin(o.view())
        .fetch::<m![Gs, Ds / 128, Tc, Ds % 128 / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 128, Tc, Ds % 128 / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 128, Tc, Ds % 128 / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16f32)
        .vector_widen_concat::<m![Gs, Ds / 128, Tc, Ds % 128 / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Ds % 8 # 32]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn ragged_prefill_pair_kvd256_o(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
) -> DmTensor<bf16, Chip, Cl, HeadOne, m![Gs, Tc, Ds]> {
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

    // ---- key slots: ONE window gather (ring + chunk keys), zero filled beyond the 36 slots ----
    let mut k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
    k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let k_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let k_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { k_win.reshape() };
    device
        .main
        .begin(k_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
    let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kk48] ----
    let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
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
    let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 32 query rows (k2p: 2 tiles instead of 4 → 12 fewer ops per launch image; VRF holds one 6 KiB mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
    // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_both.view().tile::<m![Tc], 32, m![Tc = 32 # 64, Kk48]>(32 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 32, Kk48]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64]>(32 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 32]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 32]>()
            .collect::<m![Gs, Tc = 32 / 8], m![Tc = 32 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64, Kk48]>(32 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 32, Kk48 / 4], m![Kk48 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk48 % 8 # 16]>()
            .commit_trim::<m![Kk48 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 32, m![Gs, Tc = 32 #{!} 64, Kk48]>(32 * i));
        }};
    }
    p_tile!(0);
    p_tile!(1);

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
    let v_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let v_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { v_win.reshape() };
    device
        .main
        .begin(v_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
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
    o
}

pub(crate) fn ragged_prefill_pair_kvd256_r8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Rg8, Bd, Ns, Gs, Ds]>,
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

    // ---- key slots: ONE window gather (ring + chunk keys), zero filled beyond the 36 slots ----
    let mut k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
    k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let k_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let k_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { k_win.reshape() };
    device
        .main
        .begin(k_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
    let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kk48] ----
    let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
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
    let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 32 query rows (k2p: 2 tiles instead of 4 → 12 fewer ops per launch image; VRF holds one 6 KiB mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
    // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_both.view().tile::<m![Tc], 32, m![Tc = 32 # 64, Kk48]>(32 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 32, Kk48]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64]>(32 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 32]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 32]>()
            .collect::<m![Gs, Tc = 32 / 8], m![Tc = 32 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64, Kk48]>(32 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 32, Kk48 / 4], m![Kk48 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk48 % 8 # 16]>()
            .commit_trim::<m![Kk48 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 32, m![Gs, Tc = 32 #{!} 64, Kk48]>(32 * i));
        }};
    }
    p_tile!(0);
    p_tile!(1);

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
    let v_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let v_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { v_win.reshape() };
    device
        .main
        .begin(v_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
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

    let ov: DmTensorView<'_, bf16, Chip, Cl, HeadOne, m![Gs, Rg8, Bd, Ds]> = unsafe { o.view().reshape() };
    ov.to_hbm_view(&mut device.tdma, out.view_mut());
}

pub(crate) fn ragged_prefill_pair_kvd256_r16(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
) {
    let o = ragged_prefill_pair_kvd256_r16_o(device, q, k_cache, v_cache, win_index, mask, pen);
    let ov: DmTensorView<'_, bf16, Chip, Cl, HeadOne, m![Gs, Rg4, Bd16, Ds]> = unsafe { o.view().reshape() };
    ov.to_hbm_view(&mut device.tdma, out.view_mut());
}

/// R15 gapimpl F1: `ragged_prefill_pair_kvd256_r16` with the e4m3 AO8-input output (see `ragged_prefill_pair_kvd256_o8`).
pub(crate) fn ragged_prefill_pair_kvd256_r16_o8(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<f8e4m3, Chip, O8P1>,
) {
    let o = ragged_prefill_pair_kvd256_r16_o(device, q, k_cache, v_cache, win_index, mask, pen);
    let o8 = o8_cast(device, &o);
    let ov: DmTensorView<'_, f8e4m3, Chip, Cl, HeadOne, m![Gs, Ds / 128, Rg4, Bd16, Ds % 128]> = unsafe { o8.view().reshape() };
    ov.to_hbm_view(&mut device.tdma, out.view_mut());
}

fn ragged_prefill_pair_kvd256_r16_o(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
) -> DmTensor<bf16, Chip, Cl, HeadOne, m![Gs, Tc, Ds]> {
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

    // ---- key slots: ONE window gather (ring + chunk keys), zero filled beyond the 36 slots ----
    let mut k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
    k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let k_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let k_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { k_win.reshape() };
    device
        .main
        .begin(k_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
    let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kk48] ----
    let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
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
    let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 32 query rows (k2p: 2 tiles instead of 4 → 12 fewer ops per launch image; VRF holds one 6 KiB mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
    // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_both.view().tile::<m![Tc], 32, m![Tc = 32 # 64, Kk48]>(32 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 32, Kk48]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64]>(32 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 32]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 32]>()
            .collect::<m![Gs, Tc = 32 / 8], m![Tc = 32 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64, Kk48]>(32 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 32, Kk48 / 4], m![Kk48 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk48 % 8 # 16]>()
            .commit_trim::<m![Kk48 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 32, m![Gs, Tc = 32 #{!} 64, Kk48]>(32 * i));
        }};
    }
    p_tile!(0);
    p_tile!(1);

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
    let v_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let v_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { v_win.reshape() };
    device
        .main
        .begin(v_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
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
    o
}

// ------------------------------------------------------------------------------------------------------------------
// R14 d13a S2_WINDOWONCE: every unit of a run gathers the SAME 1280-row window (the whole ring of its slot, `kvd256_tables`), so when both waves of the
// pass belong to one run (host check: wave index tables equal) wave 0 keeps its key / value slot tensors and wave 1 reuses them instead of gathering
// again. `_keep` = `ragged_prefill_pair_kvd256` returning `k_both` / `v_t`; `_r16_shared` = `ragged_prefill_pair_kvd256_r16` without the two window
// gathers, memsets, slot commits and the V transpose. All other instructions are the same code on the same values (bit-identical outputs).
// ------------------------------------------------------------------------------------------------------------------

pub(crate) fn ragged_prefill_pair_kvd256_keep(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw5]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
) -> (DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]>, DmTensor<bf16, Chip, Cl, HeadKb, m![Ds, Kk48]>) {
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

    // ---- key slots: ONE window gather (ring + chunk keys), zero filled beyond the 36 slots ----
    let mut k_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]> = DmTensor::new();
    k_both.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
    let k_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let k_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { k_win.reshape() };
    device
        .main
        .begin(k_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
    let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kk48] ----
    let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
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
    let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 32 query rows (k2p: 2 tiles instead of 4 → 12 fewer ops per launch image; VRF holds one 6 KiB mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
    // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_both.view().tile::<m![Tc], 32, m![Tc = 32 # 64, Kk48]>(32 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 32, Kk48]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64]>(32 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 32]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 32]>()
            .collect::<m![Gs, Tc = 32 / 8], m![Tc = 32 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64, Kk48]>(32 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 32, Kk48 / 4], m![Kk48 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk48 % 8 # 16]>()
            .commit_trim::<m![Kk48 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 32, m![Gs, Tc = 32 #{!} 64, Kk48]>(32 * i));
        }};
    }
    p_tile!(0);
    p_tile!(1);

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
    let v_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw5 % 32], m![Kw5 / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let v_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 40, Ds]> = unsafe { v_win.reshape() };
    device
        .main
        .begin(v_win.view())
        .fetch::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 40, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Kk48], 40, m![Kk48 = 40 #{!} 48, Ds]>(0));
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

    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
    (k_both, v_t)
}

pub(crate) fn ragged_prefill_pair_kvd256_r16_shared(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_both: &DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48, Ds]>,
    v_t: &DmTensor<bf16, Chip, Cl, HeadKb, m![Ds, Kk48]>,
    mask: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Tc, Kk48]>,
    pen: &HbmTensor<bf16, Chip, m![Pw, Ts % 32, Kk48]>,
    out: &mut HbmTensor<bf16, Chip, m![Pw, Rg4, Bd16, Ns, Gs, Ds]>,
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

    // ---- key slots: the wave-0 window of this cluster (S2_WINDOWONCE: the host launches this twin only when both waves gather the same window) ----
    let k_trf: TrfTensor<bf16, Chip, Cl, HeadKb, m![Kk48 % 8], m![Kk48 / 8, Ds]> = device
        .sub
        .begin(k_both.view())
        .fetch::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .collect::<m![Kk48 % 8, Kk48 / 8, Ds / 16], m![Ds % 16]>()
        .to_trf();

    // ---- per-key additive penalty [Kk48] ----
    let pen_kk: DmTensor<bf16, Chip, Cl, HeadKb, m![Kk48]> = pen.to_dm(&mut device.tdma);
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
    let mask_both: DmTensor<bf16, Chip, Cl, HeadKb, m![Tc, Kk48]> = mask.to_dm(&mut device.tdma);

    // ---- p = bf16(min(exp(s - max), mask)), in tiles of 32 query rows (k2p: 2 tiles instead of 4 → 12 fewer ops per launch image; VRF holds one 6 KiB mask tile) ----
    let mut p: DmTensor<bf16, Chip, Cl, HeadKb, m![Gs, Tc, Kk48]> = DmTensor::new();
    // The 4 tiles are expanded by hand (all tiles visible to the scheduler). Do NOT use `#[unroll]` here: it makes the compiler drop the whole decode attention of the N3 kernel
    // (`sliding_ragged_s2_b8_h`: the decode store goes through a tile of the same output handle), i.e. wrong decode rows.
    const _: () = assert!(Tc::SIZE == 64);
    macro_rules! p_tile {
        ($i:expr) => {{
        let i: usize = $i;
        let mask_tile = mask_both.view().tile::<m![Tc], 32, m![Tc = 32 # 64, Kk48]>(32 * i);
        let mask_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Tc = 32, Kk48]> = device
            .sub
            .begin(mask_tile)
            .fetch::<m![1], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .to_vrf();
        let m_tile = m_dm.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64]>(32 * i);
        let m_vrf: VrfTensor<f32, Chip, Cl, HeadKb, m![Gs, Tc = 32]> = device
            .sub
            .begin(m_tile)
            .fetch::<m![Gs], m![Tc = 32]>()
            .collect::<m![Gs, Tc = 32 / 8], m![Tc = 32 % 8]>()
            .to_vrf();
        let s_tile = s.view().tile::<m![Tc], 32, m![Gs, Tc = 32 # 64, Kk48]>(32 * i);
        device
            .main
            .begin(s_tile)
            .fetch::<m![Gs], m![Tc = 32, Kk48]>()
            .fetch_cast::<f32>()
            .collect::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Gs, Tc = 32, Kk48 / 4], m![Kk48 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &m_vrf)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Gs, Tc = 32, Kk48 / 8], m![Kk48 % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
            .vector_final()
            .cast::<bf16, m![Kk48 % 8 # 16]>()
            .commit_trim::<m![Kk48 % 8]>()
            .commit_view(p.view_mut().tile::<m![Tc], 32, m![Gs, Tc = 32 #{!} 64, Kk48]>(32 * i));
        }};
    }
    p_tile!(0);
    p_tile!(1);

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

    // ---- PV over both segments (V slots: the transposed wave-0 window) ----
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

    let ov: DmTensorView<'_, bf16, Chip, Cl, HeadOne, m![Gs, Rg4, Bd16, Ds]> = unsafe { o.view().reshape() };
    ov.to_hbm_view(&mut device.tdma, out.view_mut());
}
