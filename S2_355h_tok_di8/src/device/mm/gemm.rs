//! Weight-stationary bf16 GEMM  y[T, N] = x[T, K] . W[N, K]^T (+ bias[N]) for T rows at once (mapping of kprobe `wst`):
//!
//! * slice (per cluster, 256) = [Mg % 8, Mc], cluster = Mg / 8: 16 row groups x 32 K chunks;
//! * W is pre-packed on the host as W[g, c, r, e] (n = g * 240 + tile * Rt + r, k = c * E + e, zero padded), one HBM tensor
//!   per row tile, so every slice reads one contiguous run;
//! * x is chunk-major x[c, t, e] and DMA-replicated to the 16 row groups;
//! * the weight tile goes DM -> TRF (lane = r % 8, element [r / 8, e]); the contraction streams the T rows;
//! * the 32 K chunks are summed by the VE inter-slice reducer in f32, the epilogue adds the bias (optional) and rounds
//!   to bf16 once, written by the 8 live slices of each cluster.
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

type Cl = m![Mg / 8];
type Sl = m![Mg % 8, Mc];
type SlOut = m![Mg % 8, 1 # 32];

macro_rules! gemm_family {
    ($load:ident, $proc_plain:ident, $proc_bias:ident, $proc_bias_view:ident, $T:ident, $Rt:ident, $E:ident) => {
        pub(crate) fn $load(
            device: &mut Device,
            w: &HbmTensor<bf16, Chip, m![Mg, Mc, $Rt, $E]>,
        ) -> DmTensor<bf16, Chip, Cl, Sl, m![$Rt, $E]> {
            w.to_dm(&mut device.tdma)
        }

        /// Contraction of one weight tile without bias: output rows written natural `[T, Mg, Rt]`.
        pub(crate) fn $proc_plain(
            device: &mut Device,
            x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]>,
            w_dm: DmTensor<bf16, Chip, Cl, Sl, m![$Rt, $E]>,
        ) -> DmTensor<bf16, Chip, Cl, SlOut, m![$T, $Rt]> {
            let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![$Rt % 8], m![$Rt / 8, $E]> = device
                .sub
                .begin(w_dm.view())
                .fetch::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .collect::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .to_trf();

            device
                .main
                .begin(x_dm.view())
                .fetch::<m![$T, $E / 16], m![$E % 16]>()
                .collect::<m![$T, $E / 16], m![$E % 16]>()
                .contract_outer::<m![$T, $E / 32, $Rt / 8], m![$E % 32], _, _, _>(&w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$T, $Rt / 8]>()
                .contract_lane::<m![$T, $Rt / 8], m![$Rt % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<SlOut, m![$T, $Rt / 8]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![$Rt % 8 # 16]>()
                .commit_trim::<m![$Rt % 8]>()
                .commit()
        }

        /// Contraction of one weight tile with a per-row bias added in f32 before the bf16 rounding.
        pub(crate) fn $proc_bias(
            device: &mut Device,
            x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]>,
            w_dm: DmTensor<bf16, Chip, Cl, Sl, m![$Rt, $E]>,
            bias: &HbmTensor<bf16, Chip, m![Mg, $Rt]>,
        ) -> DmTensor<bf16, Chip, Cl, SlOut, m![$T, $Rt]> {
            let bias_dm: DmTensor<bf16, Chip, Cl, SlOut, m![$Rt]> = bias.to_dm(&mut device.tdma);
            let bias_vrf: VrfTensor<f32, Chip, Cl, SlOut, m![$Rt]> = device
                .sub
                .begin(bias_dm.view())
                .fetch::<m![$Rt / 8], m![$Rt % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Rt / 8], m![$Rt % 8]>()
                .to_vrf();

            let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![$Rt % 8], m![$Rt / 8, $E]> = device
                .sub
                .begin(w_dm.view())
                .fetch::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .collect::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .to_trf();

            device
                .main
                .begin(x_dm.view())
                .fetch::<m![$T, $E / 16], m![$E % 16]>()
                .collect::<m![$T, $E / 16], m![$E % 16]>()
                .contract_outer::<m![$T, $E / 32, $Rt / 8], m![$E % 32], _, _, _>(&w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$T, $Rt / 8]>()
                .contract_lane::<m![$T, $Rt / 8], m![$Rt % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<SlOut, m![$T, $Rt / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$T, $Rt / 8, $Rt / 4 % 2], m![$Rt % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, &bias_vrf)
                .vector_widen_concat::<m![$T, $Rt / 8], m![$Rt % 8]>()
                .vector_final()
                .cast::<bf16, m![$Rt % 8 # 16]>()
                .commit_trim::<m![$Rt % 8]>()
                .commit()
        }

        /// Same as `$proc_bias`, but the result is committed into tile `i` (a literal at every call site) of a combined token-major buffer `y[T, Dummy2, Rt]`
        /// (the two row tiles of one group side by side = natural column order `n = g * 2 * Rt + tile * Rt + r`).
        pub(crate) fn $proc_bias_view(
            device: &mut Device,
            x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]>,
            w_dm: DmTensor<bf16, Chip, Cl, Sl, m![$Rt, $E]>,
            bias: &HbmTensor<bf16, Chip, m![Mg, $Rt]>,
            y: &mut DmTensor<bf16, Chip, Cl, SlOut, m![$T, Dummy2, $Rt]>,
            i: usize,
        ) {
            let bias_dm: DmTensor<bf16, Chip, Cl, SlOut, m![$Rt]> = bias.to_dm(&mut device.tdma);
            let bias_vrf: VrfTensor<f32, Chip, Cl, SlOut, m![$Rt]> = device
                .sub
                .begin(bias_dm.view())
                .fetch::<m![$Rt / 8], m![$Rt % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![$Rt / 8], m![$Rt % 8]>()
                .to_vrf();

            let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![$Rt % 8], m![$Rt / 8, $E]> = device
                .sub
                .begin(w_dm.view())
                .fetch::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .collect::<m![$Rt % 8, $Rt / 8, $E / 16], m![$E % 16]>()
                .to_trf();

            device
                .main
                .begin(x_dm.view())
                .fetch::<m![$T, $E / 16], m![$E % 16]>()
                .collect::<m![$T, $E / 16], m![$E % 16]>()
                .contract_outer::<m![$T, $E / 32, $Rt / 8], m![$E % 32], _, _, _>(&w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$T, $Rt / 8]>()
                .contract_lane::<m![$T, $Rt / 8], m![$Rt % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<SlOut, m![$T, $Rt / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$T, $Rt / 8, $Rt / 4 % 2], m![$Rt % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, &bias_vrf)
                .vector_widen_concat::<m![$T, $Rt / 8], m![$Rt % 8]>()
                .vector_final()
                .cast::<bf16, m![$Rt % 8 # 16]>()
                .commit_trim::<m![$Rt % 8]>()
                .commit_view(y.view_mut().tile::<m![Dummy2], 1, m![$T, Dummy2 = 1 #{!} 2, $Rt]>(i));
        }
    };
}

gemm_family!(load_dense, proc_dense_plain, proc_dense_bias, proc_dense_bias_view, Tm64, Rt120, Ev);
gemm_family!(load_proj, proc_proj_plain, proc_proj_bias, proc_proj_bias_view, Tm64, Rt240, Ep);
gemm_family!(load_audio, proc_audio_plain, proc_audio_bias, proc_audio_bias_view, Tm64, Rt240, Ea);

/// Vision dense (K = 6912 -> 32 x 224, N = 3840 = 16 groups x 2 tiles x 120 rows), bias, group-major outputs y_i[g, t, r].
pub(crate) fn dense_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ev]>,
    w0: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    w1: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    b0: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    b1: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    y0: &mut HbmTensor<bf16, Chip, m![Mg, Tm64, Rt120]>,
    y1: &mut HbmTensor<bf16, Chip, m![Mg, Tm64, Rt120]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ev]> = x.to_dm(&mut device.tdma);
    let t0 = load_dense(device, w0);
    let t1 = load_dense(device, w1);
    let o0 = proc_dense_bias(device, &x_dm, t0, b0);
    o0.view().to_hbm_view(&mut device.tdma, y0.view_mut());
    let o1 = proc_dense_bias(device, &x_dm, t1, b1);
    o1.view().to_hbm_view(&mut device.tdma, y1.view_mut());
}

/// Vision projection (K = 3840 -> 32 x 128 with 2 zero chunks, N = 3840 = 16 groups x 240 rows), natural output y[t, n].
pub(crate) fn proj_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ep]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ep]> = x.to_dm(&mut device.tdma);
    let t = load_proj(device, w);
    let o = proc_proj_plain(device, &x_dm, t);
    o.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

/// Audio projection (K = 640 -> 32 x 32 with 12 zero chunks, N = 3840), natural output y[t, n].
pub(crate) fn audio_t64(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Mc, Tm64, Ea]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ea]> = x.to_dm(&mut device.tdma);
    let t = load_audio(device, w);
    let o = proc_audio_plain(device, &x_dm, t);
    o.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

// ---------------------------------------------------------------------------------------------------------------------
// Token-major x variants (input of the device norm stages, `norm.rs`): x[Tm64, Mc, E] is DMA-sliced chunk-wise and replicated to the 16 row groups
// (same read as `qkvc/gemm.rs::load_x`), so no host-side chunk-major packing is needed.

/// Vision dense, x[Tm64, Mc, Ev] token-major, natural output y[Tm64, Mg, Dummy2, Rt120] = [Tm64, 3840] (n = g * 240 + tile * 120 + r).
pub(crate) fn dense_t64_nat(
    device: &mut Device,
    x: HbmTensorView<'_, bf16, Chip, m![Tm64, Mc, Ev]>,
    w0: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    w1: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt120, Ev]>,
    b0: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    b1: &HbmTensor<bf16, Chip, m![Mg, Rt120]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Dummy2, Rt120]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ev]> = x.to_dm(&mut device.tdma);
    let t0 = load_dense(device, w0);
    let t1 = load_dense(device, w1);
    let mut o: DmTensor<bf16, Chip, Cl, SlOut, m![Tm64, Dummy2, Rt120]> = DmTensor::new();
    proc_dense_bias_view(device, &x_dm, t0, b0, &mut o, 0);
    proc_dense_bias_view(device, &x_dm, t1, b1, &mut o, 1);
    o.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

/// Vision projection with x[Tm64, Mc, Ep] token-major.
pub(crate) fn proj_t64_tm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Mc, Ep]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ep]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ep]> = x.to_dm(&mut device.tdma);
    let t = load_proj(device, w);
    let o = proc_proj_plain(device, &x_dm, t);
    o.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

/// Audio projection with x[Tm64, Mc, Ea] token-major.
pub(crate) fn audio_t64_tm(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Tm64, Mc, Ea]>,
    w: &HbmTensor<bf16, Chip, m![Mg, Mc, Rt240, Ea]>,
    y: &mut HbmTensor<bf16, Chip, m![Tm64, Mg, Rt240]>,
) {
    let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tm64, Ea]> = x.to_dm(&mut device.tdma);
    let t = load_audio(device, w);
    let o = proc_audio_plain(device, &x_dm, t);
    o.view().to_hbm_view(&mut device.tdma, y.view_mut());
}
