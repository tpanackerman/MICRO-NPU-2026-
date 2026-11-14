// KVD variant of the ragged prefill pair (hybrid N3 launch only): the K / V of the launch's own rows are ALREADY in the ring (the QKV launch scattered them), so the unit's keys
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

pub(crate) fn ragged_prefill_pair_kvd(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![Pw, Tc, Ns, Gs, Ds]>,
    k_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    v_cache: &HbmTensor<crate::Kv, Chip, m![Rf, Ns, Ds]>,
    win_index: &HbmTensor<i32, Chip, m![Pw, Kw]>,
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
    let k_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw % 32], m![Kw / 32, Ds]> = k_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let k_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 36, Ds]> = unsafe { k_win.reshape() };
    device
        .main
        .begin(k_win.view())
        .fetch::<m![Kk48 = 36, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 36, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(k_both.view_mut().tile::<m![Kk48], 36, m![Kk48 = 36 #{!} 48, Ds]>(0));
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
    let v_win: DmTensor<crate::Kv, Chip, Cl, m![Ns, Kw % 32], m![Kw / 32, Ds]> = v_cache.gather::<m![Rf], m![Ns, Ds]>().by_byte_offsets(win_index).to_dm(&mut device.tdma);
    let v_win: DmTensor<crate::Kv, Chip, Cl, HeadKb, m![Kk48 = 36, Ds]> = unsafe { v_win.reshape() };
    device
        .main
        .begin(v_win.view())
        .fetch::<m![Kk48 = 36, Ds / 16], m![Ds % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Kk48 = 36, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(v_both.view_mut().tile::<m![Kk48], 36, m![Kk48 = 36 #{!} 48, Ds]>(0));
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
