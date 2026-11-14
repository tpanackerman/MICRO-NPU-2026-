//! Token-sharded 2-part layout helpers of the T = 128 hybrid layer kernels (the layout of `normp::p1920` in `ffn_mod!(f_t128, ..)`: cluster = token / 64, slice = (token % 64, part),
//! 2 parts of 1920 columns of a 3840-wide row; the per-token statistics are all-reduced over the 2 part slices).  The fp8 kernels quantise activations here (per-token amax over the whole row,
//! 8x less data per slice than on the K-chunk-replicated layout) and apply the per-token dequantisation scale in the same layout.
use furiosa_opt_std::prelude::*;

use super::quant::{QINV, TINY};
use super::{Xa1920, Xa8192};
use crate::Chip;
use crate::axes::{H, Rh};
use crate::device::ffnc::normp;
use crate::device::ffnc::{C32, E32, G32, Pd2, Tk128};

pub(crate) type TCl = m![Tk128 / 64];
pub(crate) type Pre = m![1 # 2, Tk128 % 64];
/// y rows in the part layout.
pub(crate) type YDm<D> = DmTensor<D, Chip, TCl, m![{ Pre }, H / 1920], m![H % 1920]>;
/// The per-token scale on the part slices (every part slice of a token holds it): `[1 # 8]` f32, element 0.
pub(crate) type ScaleDm = DmTensor<f32, Chip, TCl, m![{ Pre }, H / 1920], m![1 # 8]>;

/// y (bf16, part layout) x per-token scale -> bf16(y * s): the dequantisation of an e4m3 x e4m3 GEMM output, right before an RMSNorm.
pub(crate) fn scale_tok(device: &mut Device, y: &YDm<bf16>, s: &ScaleDm) -> YDm<bf16> {
    let s_vrf: VrfTensor<f32, Chip, TCl, m![{ Pre }, H / 1920], m![1 # 8]> =
        device.sub.begin(s.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    device
        .main
        .begin(y.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &s_vrf)
        .vector_widen_concat::<m![H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// y (bf16, part layout), weight (HBM `[H]`) -> bf16(y / rms * w)  (the RMSNorm of `normp::p1920::norm_residual` without the residual add).
pub(crate) fn norm_only(device: &mut Device, y: &YDm<bf16>, weight: &HbmTensor<bf16, Chip, impl M>) -> YDm<bf16> {
    let rms = normp::p1920::row_rms::<TCl, Pre>(device, y);
    let rms_vrf: VrfTensor<f32, Chip, TCl, m![{ Pre }, H / 1920], m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let weight_dm: YDm<bf16> = weight.to_dm(&mut device.tdma);
    let weight_vrf: VrfTensor<f32, Chip, TCl, m![{ Pre }, H / 1920], m![H % 1920]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 240], m![H % 8]>()
        .to_vrf();
    device
        .main
        .begin(y.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Dynamic per-token quantiser on the part layout.  `$P` = the part axis expression of the input slices, `$Xa` = the single element axis of one slice.
/// Returns the e4m3 tensor (same layout) and the per-token scale `s = sqrt(amax^2) / 448 + tiny` (all-reduced Max over the 2 part slices), typed on the `H / 1920` part axis.
macro_rules! quant_part {
    ($name:ident, ($($P:tt)*), $Xa:ident) => {
        quant_part!($name, ($($P)*), $Xa, Pre);
    };
    ($name:ident, ($($P:tt)*), $Xa:ident, $Pre:ty) => {
        pub(crate) fn $name(
            device: &mut Device,
            x_dm: &DmTensor<bf16, Chip, TCl, m![{ $Pre }, $($P)*], m![$Xa]>,
        ) -> (DmTensor<f8e4m3, Chip, TCl, m![{ $Pre }, $($P)*], m![$Xa]>, DmTensor<f32, Chip, TCl, m![{ $Pre }, H / 1920], m![1 # 8]>) {
            let amax2: DmTensor<f32, Chip, TCl, m![{ $Pre }, $($P)*], m![1 # 8]> = device
                .main
                .begin(x_dm.view())
                .fetch::<m![$Xa / 16], m![$Xa % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$Xa / 8], m![$Xa % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$Xa / 4], m![$Xa % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<$Xa, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let s_b: DmTensor<f32, Chip, TCl, m![{ $Pre }, Pd2], m![1 # 8]> = device
                .main
                .begin(amax2.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<m![{ $Pre }, Pd2], m![1]>(InterSliceReduceOpF32::Max)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), QINV)
                .vector_fp_binary(FpBinaryOp::AddF, TINY)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let s_all: DmTensor<f32, Chip, TCl, m![{ $Pre }, $($P)*], m![1 # 8]> = unsafe { s_b.reshape() };
            let s_vrf: VrfTensor<f32, Chip, TCl, m![{ $Pre }, $($P)*], m![1 # 8]> =
                device.sub.begin(s_all.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
            let xq: DmTensor<f8e4m3, Chip, TCl, m![{ $Pre }, $($P)*], m![$Xa]> = device
                .main
                .begin(x_dm.view())
                .fetch::<m![$Xa / 16], m![$Xa % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![$Xa / 8], m![$Xa % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$Xa / 4], m![$Xa % 4]>()
                .vector_fp_binary(FpBinaryOp::DivF, &s_vrf)
                .vector_widen_concat::<m![$Xa / 8], m![$Xa % 8]>()
                .vector_final()
                .cast::<f8e4m3, m![$Xa % 8 # 32]>()
                .commit_trim::<m![$Xa % 8]>()
                .commit();
            (xq, unsafe { s_all.reshape() })
        }
    };
}
pub(crate) use quant_part;

quant_part!(quant_xn, (H / 1920), Xa1920);
quant_part!(quant_h, (G32 / 8), Xa8192);
/// act3 hsp: `quant_h` on the token axis spread over all 4 PEs of a cluster (`Pre2`, see below): same per-token arithmetic.
quant_part!(quant_h_sp, (G32 / 8), Xa8192, Pre2);

/// act3 hsp: the part layout with the token axis OUTSIDE the slice pad (slice = token * 4 + pad * 2 + part): `Pre` puts its `1 # 2` pad on the
/// outermost slice bit, i.e. on 2 of the 4 PEs (DMA queues) of a cluster; every DMA between HBM and a `Pre2` tensor is spread over all 4.
pub(crate) type Pre2 = m![Tk128 % 64, 1 # 2];

/// Store of the quantised x rows: the HBM tensor is `[C32, T, E32]` (every K chunk's tokens contiguous), written from ONE slice per cluster (DM -> DM gather of the cluster's 64 tokens, element order
/// `[C32 = 30, T % 64, E32]`, then 30 pieces of 8 KB per cluster).  The replicated up / gate x load then reads one 16 KB run per slice instead of 128 pieces of 128 B (static: store 18.2k -> ~6k, load 25.0k -> 9.9k).
pub(crate) fn store_rows_q_cm(device: &mut Device, x: YDm<f8e4m3>, out: &mut HbmTensor<f8e4m3, Chip, m![C32, Tk128, E32]>) {
    let xv: DmTensor<f8e4m3, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { x.reshape() };
    let g: DmTensor<f8e4m3, Chip, TCl, m![1 # 256], m![C32 = 30, Tk128 % 64, E32]> = xv.to_dm(&mut device.tdma);
    g.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![C32], 30, m![C32 = 30 #{!} 32, Tk128, E32]>(0));
}

/// The two zero pad chunks (30, 31) of a chunk-major `[C32, T, E32]` e4m3 tensor.
pub(crate) fn zero_pad_q_cm(device: &mut Device, out: &mut HbmTensor<f8e4m3, Chip, m![C32, Tk128, E32]>) {
    let mut zero_pad: DmTensor<f8e4m3, Chip, TCl, m![1 # 256], m![C32 = 2, Tk128 % 64, E32]> = DmTensor::new();
    zero_pad.view_mut().memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    zero_pad.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![C32], 2, m![C32 = 2 #{!} 32, Tk128, E32]>(30));
}

/// `store_rows_q_cm` into half `half` (0 / 1) of a `[C32, Rh, T, E32]` tensor (the T = 256 layer: both halves in ONE chunk-major tensor, so ONE replicated x load feeds the 256-token contraction).
/// act3 tq3: the gather lands on 16 slices per cluster (part = 15 chunks x 8-token group, 15 KiB each) instead of ONE slice: the DM -> DM gather
/// (4,658 -> 3,464 static) and the HBM store (60 runs of 8 KiB from one slice -> 480 runs of 1 KiB from 16 slices: 3,671 -> 960) both spread over the
/// cluster's DMA queues; same bytes at the same HBM addresses.
pub(crate) fn store_rows_q_cm_half(device: &mut Device, x: YDm<f8e4m3>, out: &mut HbmTensor<f8e4m3, Chip, m![C32, Rh, Tk128, E32]>, half: usize) {
    let xv: DmTensor<f8e4m3, Chip, TCl, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { x.reshape() };
    let g: DmTensor<f8e4m3, Chip, TCl, m![C32 = 30 / 15, Tk128 % 64 / 8, 1 # 16], m![C32 = 30 % 15, Tk128 % 8, E32]> = xv.to_dm(&mut device.tdma);
    g.view().to_hbm_view(
        &mut device.tdma,
        out.view_mut().tile::<m![Rh], 1, m![C32, Rh = 1 #{!} 2, Tk128, E32]>(half).tile::<m![C32], 30, m![C32 = 30 #{!} 32, Rh = 1 #{!} 2, Tk128, E32]>(0),
    );
}

/// The two zero pad chunks (30, 31) of both halves of a `[C32, Rh, T, E32]` tensor.
pub(crate) fn zero_pad_q_cm_half(device: &mut Device, out: &mut HbmTensor<f8e4m3, Chip, m![C32, Rh, Tk128, E32]>, half: usize) {
    let mut zero_pad: DmTensor<f8e4m3, Chip, TCl, m![1 # 256], m![C32 = 2, Tk128 % 64, E32]> = DmTensor::new();
    zero_pad.view_mut().memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    zero_pad.view().to_hbm_view(
        &mut device.tdma,
        out.view_mut().tile::<m![Rh], 1, m![C32, Rh = 1 #{!} 2, Tk128, E32]>(half).tile::<m![C32], 2, m![C32 = 2 #{!} 32, Rh = 1 #{!} 2, Tk128, E32]>(30),
    );
}
