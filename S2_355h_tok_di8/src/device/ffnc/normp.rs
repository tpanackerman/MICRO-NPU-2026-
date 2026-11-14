//! "Part" epilogue: every token row is spread over P slices (`H / PW` parts of PW columns, PW * P = 3840) so that all passes work on PW columns
//! per slice instead of the whole 3840-wide row; the per-token sum of squares is all-reduced over the P part slices (innermost slice axis).
//! Same numerics as `norm.rs` (baseline `normalize` + `residual::add`): out = bf16( bf16(y / sqrt(mean(y^2) + EPS) * w) + residual ).
//!
//! Slice mapping of every tensor here: `m![{ Pre }, H / PW]` (Pre = the token axes, padded to the slice count), element `m![H % PW]`.
use furiosa_opt_std::prelude::*;

use super::norm::EPS;
use super::{C32, E32};
use crate::Chip;
use crate::axes::H;

const H_F32: f32 = H::SIZE as f32;

macro_rules! norm_parts {
    ($m:ident, $PW:literal, $PW16:literal, $PW8:literal, $PW4:literal, $Pd:ident) => {
        pub(crate) mod $m {
            use super::*;
            use crate::device::ffnc::$Pd;

            /// rms[1 # 8] per slice (every part slice of a token holds the token's rms): sqrt(mean(y^2) + EPS).
            pub(crate) fn row_rms<Cluster: M, Pre: M>(
                device: &mut Device,
                y: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
            ) -> DmTensor<f32, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]> {
                let mean_square: DmTensor<f32, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]> = device
                    .main
                    .begin(y.view())
                    .fetch::<m![H / 16 % $PW16], m![H % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![H / 4 % $PW4], m![H % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                    .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(H_F32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let rms: DmTensor<f32, Chip, Cluster, m![{ Pre }, $Pd], m![1 # 8]> = device
                    .main
                    .begin(mean_square.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .vector_init()
                    .vector_inter_slice_reduce::<m![{ Pre }, $Pd], m![1]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(FpBinaryOp::AddF, EPS)
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                unsafe { rms.reshape() }
            }

            fn to_vrf_cols<Cluster: M, Pre: M>(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
            ) -> VrfTensor<f32, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                device
                    .sub
                    .begin(x.view())
                    .fetch::<m![H / 16 % $PW16], m![H % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H / 8 % $PW8], m![H % 8]>()
                    .to_vrf()
            }

            /// y (bf16, part layout), residual (same layout), norm weight (HBM `[H]`) -> bf16(bf16(y / rms * w) + residual).
            pub(crate) fn norm_residual<Cluster: M, Pre: M>(
                device: &mut Device,
                y: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                residual: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                weight: &HbmTensor<bf16, Chip, impl M>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                let weight_dm = load_weight_dm::<Cluster, Pre>(device, weight);
                norm_residual_dm(device, y, residual, &weight_dm)
            }

            /// cpk: one DMA of a replicated norm weight (`NormRep`) into the part layout, to be shared by several epilogue passes of a launch.
            pub(crate) fn load_weight_dm<Cluster: M, Pre: M>(
                device: &mut Device,
                weight: &HbmTensor<bf16, Chip, impl M>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                weight.to_dm(&mut device.tdma)
            }

            /// cpk: one DMA of a replicated scalar (`ScalarRep`) into the part layout.
            pub(crate) fn load_scalar_dm<Cluster: M, Pre: M>(
                device: &mut Device,
                scalar: &HbmTensor<bf16, Chip, impl M>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]> {
                scalar.to_dm(&mut device.tdma)
            }

            /// cpk: `norm_residual` with the norm weight ALREADY in DM (loaded once per launch, shared by both halves). y (bf16, part layout), residual (same layout), norm weight -> bf16(bf16(y / rms * w) + residual).
            pub(crate) fn norm_residual_dm<Cluster: M, Pre: M>(
                device: &mut Device,
                y: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                residual: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                weight_dm: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                let rms = row_rms(device, y);
                let rms_vrf: VrfTensor<f32, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]> = device
                    .sub
                    .begin(rms.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf();
                let weight_vrf = to_vrf_cols(device, weight_dm);
                let normalized: DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> = device
                    .main
                    .begin(y.view())
                    .fetch::<m![H / 16 % $PW16], m![H % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![H / 4 % $PW4], m![H % 4]>()
                    .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
                    .vector_widen_concat::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_final()
                    .cast::<bf16, m![H % 8 # 16]>()
                    .commit_trim::<m![H % 8]>()
                    .commit();
                let residual_vrf = to_vrf_cols(device, residual);
                device
                    .main
                    .begin(normalized.view())
                    .fetch::<m![H / 16 % $PW16], m![H % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
                    .vector_final()
                    .cast::<bf16, m![H % 8 # 16]>()
                    .commit_trim::<m![H % 8]>()
                    .commit()
            }

            /// x * scalar (the layer scalar), one bf16 rounding.
            pub(crate) fn scale_rows<Cluster: M, Pre: M>(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                scalar: &HbmTensor<bf16, Chip, impl M>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                let scalar_dm = load_scalar_dm::<Cluster, Pre>(device, scalar);
                scale_rows_dm(device, x, &scalar_dm)
            }

            /// cpk: `scale_rows` with the scalar ALREADY in DM (loaded once per launch).
            pub(crate) fn scale_rows_dm<Cluster: M, Pre: M>(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                scalar_dm: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                let scalar_vrf: VrfTensor<f32, Chip, Cluster, m![{ Pre }, H / $PW], m![1 # 8]> = device
                    .sub
                    .begin(scalar_dm.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf();
                device
                    .main
                    .begin(x.view())
                    .fetch::<m![H / 16 % $PW16], m![H % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![H / 4 % $PW4], m![H % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
                    .vector_widen_concat::<m![H / 8 % $PW8], m![H % 8]>()
                    .vector_final()
                    .cast::<bf16, m![H % 8 # 16]>()
                    .commit_trim::<m![H % 8]>()
                    .commit()
            }

            /// Residual rows of the zero padded stream `[T, C32, E32]` in the part layout (the 30 real chunks = H columns of every token).
            pub(crate) fn load_rows<T: AxisName, Cluster: M, Pre: M>(
                device: &mut Device,
                res: &HbmTensor<bf16, Chip, m![T, C32, E32]>,
            ) -> DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]> {
                // part = chunk / 15 (15 chunks = 1920 columns), element = [chunk % 15, E32]
                let x: DmTensor<bf16, Chip, Cluster, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = res
                    .view()
                    .tile::<m![C32], 30, m![T, C32 = 30 # 32, E32]>(0)
                    .to_dm(&mut device.tdma);
                unsafe { x.reshape() }
            }

            /// Writes part-layout rows into the first 3840 columns of the padded residual stream (the pad columns are never touched).
            pub(crate) fn store_rows<T: AxisName, Cluster: M, Pre: M>(
                device: &mut Device,
                x: &DmTensor<bf16, Chip, Cluster, m![{ Pre }, H / $PW], m![H % $PW]>,
                res_out: &mut HbmTensor<bf16, Chip, m![T, C32, E32]>,
            ) {
                let xv: DmTensorView<'_, bf16, Chip, Cluster, m![{ Pre }, C32 = 30 / 15], m![C32 = 30 % 15, E32]> = unsafe { x.view().reshape() };
                xv.to_hbm_view(
                    &mut device.tdma,
                    res_out.view_mut().tile::<m![C32], 30, m![T, C32 = 30 #{!} 32, E32]>(0),
                );
            }
        }
    };
}

norm_parts!(p1920, 1920, 120, 240, 480, Pd2);
