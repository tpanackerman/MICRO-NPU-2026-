//! FFN chunk kernels (one module per token bucket).
//!
//! `ffn_a` = pre-FF RMSNorm (in-slice; statistics by an inter-slice all-reduce over the 32 K-chunk slices) + up and gate projections
//! (NVFP4, weight stationary, K split 32, 4 row tiles of 240) + GeGLU on token pairs spread over the slices -> `h` in HBM, in the K-chunk
//! layout of the down projection.  `ffn_b` = down projection (5 row tiles of 48) + token-sharded epilogue (post-FF RMSNorm, residual,
//! layer scalar).  `ffn_chunk` = both in ONE device function: tile-merged weight tensors (argument tuple <= 24) and the `h` hop through a
//! compiler-placed HBM scratch tensor.
use furiosa_opt_std::prelude::*;

use super::norm::EPS;
use super::normp;
use super::{C32, C64, Dummy32, E32, E256, Fb8, Fb16, Fc2, Fh2, Fi4, Fi5, G32, Gd8, Hi3, Rt48, Rt80, Rt96, Rt240, Rt256, Tk8, Tk16, Tk32, Tk64, Tk128};
use crate::Chip;
use crate::axes::H;

type Cl = m![G32 / 8];
type Sl = m![G32 % 8, C32];
type SlOut = m![G32 % 8, 1 # 32];
// down projection with the K split over 64 slices (chunk 256) and 4 row groups per cluster (8 in total)
type Cl8 = m![Gd8 / 4];
type Sl8 = m![Gd8 % 4, C64];
type SlOut8 = m![Gd8 % 4, 1 # 64];

const INVSQRT2: f32 = 0.70710678118f32;
const INV_H: f32 = 1.0 / 3840.0;

macro_rules! ffn_mod {
    ($m:ident, $T:ident, $Tb:literal, $SpSl:ty, $TCl:ty, $Pre:ty, $aos:path, $aof:path) => {
        pub(crate) mod $m {
            use super::*;

            /// x [T, 128] (a K chunk of the residual, zero padded to 32 chunks) -> bf16(x / rms * w), rms from the whole row.
            fn prenorm(
                device: &mut Device,
                x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]>,
                w: &HbmTensor<bf16, Chip, m![C32, E32]>,
            ) -> DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]> {
                let ss: DmTensor<f32, Chip, Cl, Sl, m![$T, 1 # 8]> = device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, E32 / 16], m![E32 % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$T, E32 / 8], m![E32 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T, E32 / 4], m![E32 % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                    .vector_intra_slice_reduce::<E32, m![$T], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let rms: DmTensor<f32, Chip, Cl, m![G32 % 8, Dummy32], m![$T, 1 # 8]> = device
                    .main
                    .begin(ss.view())
                    .fetch::<m![$T], m![1 # 8]>()
                    .collect::<m![$T], m![1 # 8]>()
                    .vector_init()
                    .vector_inter_slice_reduce::<m![G32 % 8, Dummy32], m![$T]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INV_H)
                    .vector_fp_binary(FpBinaryOp::AddF, EPS)
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let rms: DmTensor<f32, Chip, Cl, Sl, m![$T, 1 # 8]> = unsafe { rms.reshape() };
                let rms_vrf: VrfTensor<f32, Chip, Cl, Sl, m![$T, 1 # 8]> = device
                    .sub
                    .begin(rms.view())
                    .fetch::<m![$T], m![1 # 8]>()
                    .collect::<m![$T], m![1 # 8]>()
                    .to_vrf();
                let w_dm: DmTensor<bf16, Chip, Cl, Sl, m![E32]> = w.to_dm(&mut device.tdma);
                let w_vrf: VrfTensor<f32, Chip, Cl, Sl, m![E32]> = device
                    .sub
                    .begin(w_dm.view())
                    .fetch::<m![E32 / 16], m![E32 % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![E32 / 8], m![E32 % 8]>()
                    .to_vrf();
                device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, E32 / 16], m![E32 % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$T, E32 / 8], m![E32 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T, E32 / 4], m![E32 % 4]>()
                    .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
                    .vector_widen_concat::<m![$T, E32 / 8], m![E32 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![E32 % 8 # 16]>()
                    .commit_trim::<m![E32 % 8]>()
                    .commit()
            }

            fn scalar_vrf(device: &mut Device, gs: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]> {
                let gs_dm: DmTensor<f32, Chip, Cl, SlOut, m![1 # 8]> = gs.to_dm(&mut device.tdma);
                device
                    .sub
                    .begin(gs_dm.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf()
            }

            /// the same scalar, replicated on the spread slices (`$SpSl`) of the GeGLU
            fn scalar_vrf_sp(device: &mut Device, gs: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, Cl, $SpSl, m![1 # 8]> {
                let gs_dm: DmTensor<f32, Chip, Cl, $SpSl, m![1 # 8]> = gs.to_dm(&mut device.tdma);
                device
                    .sub
                    .begin(gs_dm.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf()
            }

            /// One 240-row tile of up or gate: f4 -> f8 lookup, f8 x block scale -> bf16, TRF, contraction, K reduce, global scale;
            /// result: the tile `[T, Rt240]` on the live slices (bf16).
            #[allow(clippy::too_many_arguments)]
            fn ffu_tile(
                device: &mut Device,
                x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]>,
                w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]>,
                scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]>,
                gs_vrf: &VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, Cl, SlOut, m![$T, Rt240]> {
                let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt240, E32 / 16]> =
                    unsafe { scale_dm.view().reshape() };
                let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt240, E32 / 16]> = device
                    .sub
                    .begin(scale_view)
                    .fetch::<m![Rt240], m![E32 / 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt240], m![E32 / 16]>()
                    .to_vrf();

                let w_bf: DmTensor<bf16, Chip, Cl, Sl, m![Rt240, E32]> = device
                    .main
                    .begin(w4_dm.view())
                    .fetch::<m![Rt240], m![E32]>()
                    .fetch_table_lookup::<f8e4m3>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt240, E32 / 8], m![E32 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rt240, E32 / 4], m![E32 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                    .vector_widen_concat::<m![Rt240, E32 / 8], m![E32 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![E32 % 8 # 16]>()
                    .commit_trim::<m![E32 % 8]>()
                    .commit();

                let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]> = device
                    .sub
                    .begin(w_bf.view())
                    .fetch::<m![Rt240 % 8, Rt240 / 8, E32 / 16], m![E32 % 16]>()
                    .collect::<m![Rt240 % 8, Rt240 / 8, E32 / 16], m![E32 % 16]>()
                    .to_trf();

                device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, E32 / 16], m![E32 % 16]>()
                    .collect::<m![$T, E32 / 16], m![E32 % 16]>()
                    .contract_outer::<m![$T, E32 / 32, Rt240 / 8], m![E32 % 32], _, _, _>(&w_trf)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![$T, Rt240 / 8]>()
                    .contract_lane::<m![$T, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_inter_slice_reduce::<SlOut, m![$T, Rt240 / 8]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
                    .vector_widen_concat::<m![$T, Rt240 / 8], m![Rt240 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt240 % 8 # 16]>()
                    .commit_trim::<m![Rt240 % 8]>()
                    .commit()
            }

            /// One 240-row tile of up or gate: f4 -> f8 lookup, f8 x block scale -> bf16, TRF, contraction, K reduce, global scale;
            /// result: the tile spread over the slices by tokens (`$Tb` tokens per slice, bf16): the inter-slice reduce PROMOTES the token axis into the freed K slots (reduce-scatter).
            #[allow(clippy::too_many_arguments)]
            fn ffu_tile_sp(
                device: &mut Device,
                x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]>,
                w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]>,
                scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]>,
                gs_vrf: &VrfTensor<f32, Chip, Cl, $SpSl, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]> {
                let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt240, E32 / 16]> =
                    unsafe { scale_dm.view().reshape() };
                let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt240, E32 / 16]> = device
                    .sub
                    .begin(scale_view)
                    .fetch::<m![Rt240], m![E32 / 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt240], m![E32 / 16]>()
                    .to_vrf();

                let w_bf: DmTensor<bf16, Chip, Cl, Sl, m![Rt240, E32]> = device
                    .main
                    .begin(w4_dm.view())
                    .fetch::<m![Rt240], m![E32]>()
                    .fetch_table_lookup::<f8e4m3>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt240, E32 / 8], m![E32 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rt240, E32 / 4], m![E32 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                    .vector_widen_concat::<m![Rt240, E32 / 8], m![E32 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![E32 % 8 # 16]>()
                    .commit_trim::<m![E32 % 8]>()
                    .commit();

                let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![Rt240 % 8], m![Rt240 / 8, E32]> = device
                    .sub
                    .begin(w_bf.view())
                    .fetch::<m![Rt240 % 8, Rt240 / 8, E32 / 16], m![E32 % 16]>()
                    .collect::<m![Rt240 % 8, Rt240 / 8, E32 / 16], m![E32 % 16]>()
                    .to_trf();

                device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, E32 / 16], m![E32 % 16]>()
                    .collect::<m![$T, E32 / 16], m![E32 % 16]>()
                    .contract_outer::<m![$T, E32 / 32, Rt240 / 8], m![E32 % 32], _, _, _>(&w_trf)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![$T, Rt240 / 8]>()
                    .contract_lane::<m![$T, Rt240 / 8], m![Rt240 % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_inter_slice_reduce::<$SpSl, m![$T % $Tb, Rt240 / 8]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T % $Tb, Rt240 / 8, Rt240 / 4 % 2], m![Rt240 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
                    .vector_widen_concat::<m![$T % $Tb, Rt240 / 8], m![Rt240 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt240 % 8 # 16]>()
                    .commit_trim::<m![Rt240 % 8]>()
                    .commit()
            }

            /// GeGLU of one tile pair: up / gate tiles (live slices, [T, Rt240]) -> h = bf16(up * gelu2(gate) / 2) spread over the slices
            /// (`$Tb` tokens per slice), committed into tile `i` = (c2, h) of `h_buf` (`[Fc2, Fh2, T % Tb, Rt256]`, zero pad columns).
            fn geglu_tile(
                device: &mut Device,
                up_t: &DmTensor<bf16, Chip, Cl, SlOut, m![$T, Rt240]>,
                gate_t: &DmTensor<bf16, Chip, Cl, SlOut, m![$T, Rt240]>,
                h_buf: &mut DmTensor<bf16, Chip, Cl, $SpSl, m![Fc2, Fh2, $T % $Tb, Rt256]>,
                i: usize,
            ) {
                let up_sp: DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]> = up_t.to_dm(&mut device.tdma);
                let gate_sp: DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]> = gate_t.to_dm(&mut device.tdma);
                geglu_sp(device, up_sp, &gate_sp, h_buf, i);
            }

            /// GeGLU of one tile pair already spread over the slices (see `geglu_tile`).
            fn geglu_sp(
                device: &mut Device,
                up_sp: DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]>,
                gate_sp: &DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]>,
                h_buf: &mut DmTensor<bf16, Chip, Cl, $SpSl, m![Fc2, Fh2, $T % $Tb, Rt256]>,
                i: usize,
            ) {
                // gelu2 = gate * (1 + erf(gate / sqrt 2))  (f32, no 1/2: the baseline folds it into the product); on Sub
                let gelu: DmTensor<f32, Chip, Cl, $SpSl, m![$T % $Tb, Rt240]> = device
                    .sub
                    .begin(gate_sp.view())
                    .fetch::<m![$T % $Tb, Rt240 / 16], m![Rt240 % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$T % $Tb, Rt240 / 8], m![Rt240 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T % $Tb, Rt240 / 4], m![Rt240 % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
                    .vector_fp_unary(FpUnaryOp::Erf)
                    .vector_fp_binary(FpBinaryOp::AddF, 1f32)
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
                    .vector_widen_concat::<m![$T % $Tb, Rt240 / 8], m![Rt240 % 8]>()
                    .vector_final()
                    .commit_trim::<m![Rt240 % 8]>()
                    .commit();
                // relabel Rt240 -> Rt256 = 240 so that the output pass can commit into the first 240 columns of a live [.., Rt256] buffer
                let gelu: DmTensor<f32, Chip, Cl, $SpSl, m![$T % $Tb, Rt256 = 240]> = unsafe { gelu.reshape() };
                let up_sp: DmTensor<bf16, Chip, Cl, $SpSl, m![$T % $Tb, Rt256 = 240]> = unsafe { up_sp.reshape() };
                let gelu_vrf: VrfTensor<f32, Chip, Cl, $SpSl, m![$T % $Tb, Rt256 = 240]> = device
                    .sub
                    .begin(gelu.view())
                    .fetch::<m![$T % $Tb, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
                    .collect::<m![$T % $Tb, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
                    .to_vrf();

                device
                    .main
                    .begin(up_sp.view())
                    .fetch::<m![$T % $Tb, Rt256 = 240 / 16], m![Rt256 = 240 % 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![$T % $Tb, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T % $Tb, Rt256 = 240 / 4], m![Rt256 = 240 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
                    .vector_fp_div(2f32)
                    .vector_widen_concat::<m![$T % $Tb, Rt256 = 240 / 8], m![Rt256 = 240 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt256 = 240 % 8 # 16]>()
                    .commit_trim::<m![Rt256 = 240 % 8]>()
                    .commit_view(
                        h_buf
                            .view_mut()
                            .tile::<m![Fc2], 1, m![Fc2 = 1 #{!} 2, Fh2, $T % $Tb, Rt256]>(i / 2)
                            .tile::<m![Fh2], 1, m![Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, $T % $Tb, Rt256]>(i % 2)
                            .tile::<m![Rt256], 240, m![Fc2 = 1 #{!} 2, Fh2 = 1 #{!} 2, $T % $Tb, Rt256 = 240 #{!} 256]>(0),
                    );
            }

            /// zero-filled `[Fc2, Fh2, T % Tb, Rt256]` buffer for the GeGLU output (the 16 pad columns of every half chunk stay zero)
            fn new_h_buf(device: &mut Device) -> DmTensor<bf16, Chip, Cl, $SpSl, m![Fc2, Fh2, $T % $Tb, Rt256]> {
                let mut h_buf: DmTensor<bf16, Chip, Cl, $SpSl, m![Fc2, Fh2, $T % $Tb, Rt256]> = DmTensor::new();
                h_buf.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
                h_buf
            }

            /// One 48-row tile of the down projection (chunk = [Fh2, Rt256]: 2 halves of 240 real + 16 zero columns), result -> tile `i`
            /// of `buf` ([T, Fi5, Rt48]).
            #[allow(clippy::too_many_arguments)]
            fn ffd_tile(
                device: &mut Device,
                x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![Fh2, $T, Rt256]>,
                w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt48, Fh2, Rt256]>,
                scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt48, Fh2, Fb16]>,
                gs_vrf: &VrfTensor<f32, Chip, Cl, SlOut, m![1 # 8]>,
                buf: &mut DmTensor<bf16, Chip, Cl, SlOut, m![$T, Fi5, Rt48]>,
                i: usize,
            ) {
                let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl, Sl, m![Rt48, Fh2, Rt256 / 16]> =
                    unsafe { scale_dm.view().reshape() };
                let scale_vrf: VrfTensor<f32, Chip, Cl, Sl, m![Rt48, Fh2, Rt256 / 16]> = device
                    .sub
                    .begin(scale_view)
                    .fetch::<m![Rt48], m![Fh2, Rt256 / 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt48, Fh2, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
                    .to_vrf();

                let w_bf: DmTensor<bf16, Chip, Cl, Sl, m![Rt48, Fh2, Rt256]> = device
                    .main
                    .begin(w4_dm.view())
                    .fetch::<m![Rt48, Fh2], m![Rt256]>()
                    .fetch_table_lookup::<f8e4m3>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt48, Fh2, Rt256 / 8], m![Rt256 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rt48, Fh2, Rt256 / 4], m![Rt256 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                    .vector_widen_concat::<m![Rt48, Fh2, Rt256 / 8], m![Rt256 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt256 % 8 # 16]>()
                    .commit_trim::<m![Rt256 % 8]>()
                    .commit();

                let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![Rt48 % 8], m![Rt48 / 8, Fh2, Rt256]> = device
                    .sub
                    .begin(w_bf.view())
                    .fetch::<m![Rt48 % 8, Rt48 / 8, Fh2, Rt256 / 16], m![Rt256 % 16]>()
                    .collect::<m![Rt48 % 8, Rt48 / 8, Fh2, Rt256 / 16], m![Rt256 % 16]>()
                    .to_trf();

                device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, Fh2, Rt256 / 16], m![Rt256 % 16]>()
                    .collect::<m![$T, Fh2, Rt256 / 16], m![Rt256 % 16]>()
                    .contract_outer::<m![$T, Fh2, Rt256 / 32, Rt48 / 8], m![Rt256 % 32], _, _, _>(&w_trf)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![$T, Rt48 / 8]>()
                    .contract_lane::<m![$T, Rt48 / 8], m![Rt48 % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_inter_slice_reduce::<SlOut, m![$T, Rt48 / 8]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T, Rt48 / 8, Rt48 / 4 % 2], m![Rt48 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
                    .vector_widen_concat::<m![$T, Rt48 / 8], m![Rt48 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt48 % 8 # 16]>()
                    .commit_trim::<m![Rt48 % 8]>()
                    .commit_view(buf.view_mut().tile::<m![Fi5], 1, m![$T, Fi5 = 1 #{!} 5, Rt48]>(i));
            }

            /// y tiles of the down projection (live slices, [T, Fi5, Rt48]) + residual rows (part layout) -> post-FF norm, residual, layer scalar.
            fn down_tail(
                device: &mut Device,
                dbuf: &DmTensor<bf16, Chip, Cl, SlOut, m![$T, Fi5, Rt48]>,
                res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
                // 2-part epilogue: slice (token, part) holds the 1920 columns [part * 1920, ..) of y (n = g * 240 + i * 48 + rr, part = source cluster)
                let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, G32 / 8], m![G32 % 8, Fi5, Rt48]> = dbuf.to_dm(&mut device.tdma);
                let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
                let summed = normp::p1920::norm_residual::<$TCl, $Pre>(device, &yb, res, post_w);
                normp::p1920::scale_rows::<$TCl, $Pre>(device, &summed, layer_scalar)
            }

            fn down_epilogue(
                device: &mut Device,
                dbuf: &DmTensor<bf16, Chip, Cl, SlOut, m![$T, Fi5, Rt48]>,
                res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            ) {
                let res = normp::p1920::load_rows::<$T, $TCl, $Pre>(device, res_in);
                let out = down_tail(device, dbuf, &res, post_w, layer_scalar);
                normp::p1920::store_rows::<$T, $TCl, $Pre>(device, &out, res_out);
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn ffn_a(
                device: &mut Device,
                res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
                uw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                uw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                uw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                uw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                us0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                us1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                us2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                us3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                ugs: &HbmTensor<f32, Chip, m![1]>,
                gw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                gw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                gw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                gw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt240, E32]>,
                gs0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                gs1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                gs2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                gs3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt240, Fb8]>,
                ggs: &HbmTensor<f32, Chip, m![1]>,
                h_out: &mut HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]>,
            ) {
                let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]> = res_in.to_dm(&mut device.tdma);
                let x_n = prenorm(device, &x_dm, pre_w);

                let up_gs = scalar_vrf(device, ugs);
                let gate_gs = scalar_vrf(device, ggs);
                let mut h_buf = new_h_buf(device);
                macro_rules! pair {
                    ($uw:ident, $us:ident, $gw:ident, $gsc:ident, $i:expr) => {{
                        let w4_dm = $uw.to_dm(&mut device.tdma);
                        let scale_dm = $us.to_dm(&mut device.tdma);
                        let up_t = ffu_tile(device, &x_n, w4_dm, scale_dm, &up_gs);
                        let w4_dm = $gw.to_dm(&mut device.tdma);
                        let scale_dm = $gsc.to_dm(&mut device.tdma);
                        let gate_t = ffu_tile(device, &x_n, w4_dm, scale_dm, &gate_gs);
                        geglu_tile(device, &up_t, &gate_t, &mut h_buf, $i);
                    }};
                }
                pair!(uw0, us0, gw0, gs0, 0);
                pair!(uw1, us1, gw1, gs1, 1);
                pair!(uw2, us2, gw2, gs2, 2);
                pair!(uw3, us3, gw3, gs3, 3);
                h_buf.view().to_hbm_view(&mut device.tdma, h_out.view_mut());
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn ffn_b(
                device: &mut Device,
                h: &HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]>,
                res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                dw0: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
                dw1: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
                dw2: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
                dw3: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
                dw4: &HbmTensor<f4e2m1, Chip, m![G32, C32, Rt48, Fh2, Rt256]>,
                ds0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
                ds1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
                ds2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
                ds3: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
                ds4: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt48, Fh2, Fb16]>,
                dgs: &HbmTensor<f32, Chip, m![1]>,
                res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            ) {
                // chunk c = (G32, Fc2) of the GeGLU output, replicated 16x (one copy per row group)
                let hv: HbmTensorView<'_, bf16, Chip, m![C32, Fh2, $T, Rt256]> = unsafe { h.view().reshape() };
                let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Fh2, $T, Rt256]> = hv.to_dm(&mut device.tdma);
                let gs_vrf = scalar_vrf(device, dgs);
                let mut dbuf: DmTensor<bf16, Chip, Cl, SlOut, m![$T, Fi5, Rt48]> = DmTensor::new();
                macro_rules! tile {
                    ($w:ident, $s:ident, $i:expr) => {{
                        let w4_dm = $w.to_dm(&mut device.tdma);
                        let scale_dm = $s.to_dm(&mut device.tdma);
                        ffd_tile(device, &x_dm, w4_dm, scale_dm, &gs_vrf, &mut dbuf, $i);
                    }};
                }
                tile!(dw0, ds0, 0);
                tile!(dw1, ds1, 1);
                tile!(dw2, ds2, 2);
                tile!(dw3, ds3, 3);
                tile!(dw4, ds4, 4);
                down_epilogue(device, &dbuf, res_in, post_w, layer_scalar, res_out);
            }

            /// FFN core: x rows (HBM, replicated chunk layout) + residual rows (DM, part layout), tile-merged weight tensors (`[G32, C32, tile, rows, ..]`,
            /// tile `i` selected by a tile view); the GeGLU output `h` hops through a compiler-placed HBM scratch tensor. Returns the new residual rows (part layout).
            #[allow(clippy::too_many_arguments)]
            fn ffn_core(
                device: &mut Device,
                x_hbm: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
                pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
                up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ugs: &HbmTensor<f32, Chip, m![1]>,
                gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ggs: &HbmTensor<f32, Chip, m![1]>,
                down_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi5, Rt48, Fh2, Rt256]>,
                down_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi5, Rt48, Fh2, Fb16]>,
                dgs: &HbmTensor<f32, Chip, m![1]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
                let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]> = x_hbm.to_dm(&mut device.tdma);
                let x_n = prenorm(device, &x_dm, pre_w);

                let up_gs = scalar_vrf(device, ugs);
                let gate_gs = scalar_vrf(device, ggs);
                let mut h_buf = new_h_buf(device);
                macro_rules! mtile {
                    ($w:ident, $s:ident, $gs:ident, $i:expr) => {{
                        let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]> = $w
                            .view()
                            .tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i)
                            .to_dm(&mut device.tdma);
                        let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]> = $s
                            .view()
                            .tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, Fb8]>($i)
                            .to_dm(&mut device.tdma);
                        ffu_tile(device, &x_n, w4_dm, scale_dm, &$gs)
                    }};
                }
                macro_rules! mpair {
                    ($i:expr) => {{
                        let up_t = mtile!(up_w, up_s, up_gs, $i);
                        let gate_t = mtile!(gate_w, gate_s, gate_gs, $i);
                        geglu_tile(device, &up_t, &gate_t, &mut h_buf, $i);
                    }};
                }
                mpair!(0);
                mpair!(1);
                mpair!(2);
                mpair!(3);
                let h_buf = h_buf;
                // hop through HBM: chunk-major h, replicated 16x by the DMA of the down projection
                let h: HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]> = h_buf.to_hbm(&mut device.tdma);
                let hv: HbmTensorView<'_, bf16, Chip, m![C32, Fh2, $T, Rt256]> = unsafe { h.view().reshape() };
                let x_h: DmTensor<bf16, Chip, Cl, Sl, m![Fh2, $T, Rt256]> = hv.to_dm(&mut device.tdma);
                let d_gs = scalar_vrf(device, dgs);
                let mut dbuf: DmTensor<bf16, Chip, Cl, SlOut, m![$T, Fi5, Rt48]> = DmTensor::new();
                macro_rules! dtile {
                    ($i:expr) => {{
                        let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt48, Fh2, Rt256]> = down_w
                            .view()
                            .tile::<m![Fi5], 1, m![G32, C32, Fi5 = 1 # 5, Rt48, Fh2, Rt256]>($i)
                            .to_dm(&mut device.tdma);
                        let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt48, Fh2, Fb16]> = down_s
                            .view()
                            .tile::<m![Fi5], 1, m![G32, C32, Fi5 = 1 # 5, Rt48, Fh2, Fb16]>($i)
                            .to_dm(&mut device.tdma);
                        ffd_tile(device, &x_h, w4_dm, scale_dm, &d_gs, &mut dbuf, $i);
                    }};
                }
                dtile!(0);
                dtile!(1);
                dtile!(2);
                dtile!(3);
                dtile!(4);
                down_tail(device, &dbuf, res, post_w, layer_scalar)
            }

            /// the down-projection global scale on the live slices of the 8-group down layout
            fn scalar_vrf8(device: &mut Device, gs: &HbmTensor<f32, Chip, m![1]>) -> VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> {
                let gs_dm: DmTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]> = gs.to_dm(&mut device.tdma);
                device
                    .sub
                    .begin(gs_dm.view())
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf()
            }

            /// (E10) One 96-row tile of the down projection in the 8-row-group layout: K split over 64 slices (chunk = [Rt256]: 240 real + 16 zero columns),
            /// 4 row groups per cluster; result -> tile `i` of `buf` ([T, Fi5, Rt96]) on the 4 live slices of each cluster.
            #[allow(clippy::too_many_arguments)]
            fn ffd_tile8(
                device: &mut Device,
                x_dm: &DmTensor<bf16, Chip, Cl8, Sl8, m![$T, Rt256]>,
                w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]>,
                scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]>,
                gs_vrf: &VrfTensor<f32, Chip, Cl8, SlOut8, m![1 # 8]>,
                buf: &mut DmTensor<bf16, Chip, Cl8, SlOut8, m![$T, Fi5, Rt96]>,
                i: usize,
            ) {
                let scale_view: DmTensorView<'_, f8e4m3, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = unsafe { scale_dm.view().reshape() };
                let scale_vrf: VrfTensor<f32, Chip, Cl8, Sl8, m![Rt96, Rt256 / 16]> = device
                    .sub
                    .begin(scale_view)
                    .fetch::<m![Rt96], m![Rt256 / 16]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt96, Rt256 / 16 / 8], m![Rt256 / 16 % 8]>()
                    .to_vrf();

                let w_bf: DmTensor<bf16, Chip, Cl8, Sl8, m![Rt96, Rt256]> = device
                    .main
                    .begin(w4_dm.view())
                    .fetch::<m![Rt96], m![Rt256]>()
                    .fetch_table_lookup::<f8e4m3>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![Rt96, Rt256 / 4], m![Rt256 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                    .vector_widen_concat::<m![Rt96, Rt256 / 8], m![Rt256 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt256 % 8 # 16]>()
                    .commit_trim::<m![Rt256 % 8]>()
                    .commit();

                let w_trf: TrfTensor<bf16, Chip, Cl8, Sl8, m![Rt96 % 8], m![Rt96 / 8, Rt256]> = device
                    .sub
                    .begin(w_bf.view())
                    .fetch::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
                    .collect::<m![Rt96 % 8, Rt96 / 8, Rt256 / 16], m![Rt256 % 16]>()
                    .to_trf();

                device
                    .main
                    .begin(x_dm.view())
                    .fetch::<m![$T, Rt256 / 16], m![Rt256 % 16]>()
                    .collect::<m![$T, Rt256 / 16], m![Rt256 % 16]>()
                    .contract_outer::<m![$T, Rt256 / 32, Rt96 / 8], m![Rt256 % 32], _, _, _>(&w_trf)
                    .contract_packet::<m![1]>()
                    .contract_time::<m![$T, Rt96 / 8]>()
                    .contract_lane::<m![$T, Rt96 / 8], m![Rt96 % 8]>(LaneMode::Interleaved)
                    .vector_init()
                    .vector_inter_slice_reduce::<SlOut8, m![$T, Rt96 / 8]>(InterSliceReduceOpF32::Add)
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![$T, Rt96 / 8, Rt96 / 4 % 2], m![Rt96 % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gs_vrf)
                    .vector_widen_concat::<m![$T, Rt96 / 8], m![Rt96 % 8]>()
                    .vector_final()
                    .cast::<bf16, m![Rt96 % 8 # 16]>()
                    .commit_trim::<m![Rt96 % 8]>()
                    .commit_view(buf.view_mut().tile::<m![Fi5], 1, m![$T, Fi5 = 1 #{!} 5, Rt96]>(i));
            }

            /// y tiles of the 8-row-group down projection (4 live slices per cluster, [T, Fi5, Rt96]) + residual rows -> post-FF norm, residual, layer scalar.
            fn down_tail8(
                device: &mut Device,
                dbuf: &DmTensor<bf16, Chip, Cl8, SlOut8, m![$T, Fi5, Rt96]>,
                res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
                // 2-part epilogue: slice (token, part) holds the 1920 columns [part * 1920, ..) of y (n = g * 480 + i * 96 + rr, part = source cluster)
                let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, Gd8 / 4], m![Gd8 % 4, Fi5, Rt96]> = dbuf.to_dm(&mut device.tdma);
                let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
                let summed = normp::p1920::norm_residual::<$TCl, $Pre>(device, &yb, res, post_w);
                normp::p1920::scale_rows::<$TCl, $Pre>(device, &summed, layer_scalar)
            }

            /// (E10) E2 up / gate + the down projection in the 8-row-group layout (K split 64, chunk 256): the GeGLU hop `h` is replicated 8x instead of 16x. FFN core: x rows (HBM, replicated chunk layout) + residual rows (DM, part layout), tile-merged weight tensors,
            /// the down weights as `[Gd8, C64, Fi5, Rt96, Rt256]` (f4) / `[Gd8, C64, Fi5, Rt96, Fb16]` (scales). Returns the new residual rows (part layout).
            #[allow(clippy::too_many_arguments)]
            fn ffn_core_d8(
                device: &mut Device,
                x_hbm: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
                pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
                up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ugs: &HbmTensor<f32, Chip, m![1]>,
                gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ggs: &HbmTensor<f32, Chip, m![1]>,
                down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
                down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
                dgs: &HbmTensor<f32, Chip, m![1]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
            ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
                let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, E32]> = x_hbm.to_dm(&mut device.tdma);
                let x_n = prenorm(device, &x_dm, pre_w);

                let up_gs = scalar_vrf_sp(device, ugs);
                let gate_gs = scalar_vrf_sp(device, ggs);
                let mut h_buf = new_h_buf(device);
                macro_rules! mtile {
                    ($w:ident, $s:ident, $gs:ident, $i:expr) => {{
                        let w4_dm: DmTensor<f4e2m1, Chip, Cl, Sl, m![Rt240, E32]> = $w
                            .view()
                            .tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, E32]>($i)
                            .to_dm(&mut device.tdma);
                        let scale_dm: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt240, Fb8]> = $s
                            .view()
                            .tile::<m![Fi4], 1, m![G32, C32, Fi4 = 1 # 4, Rt240, Fb8]>($i)
                            .to_dm(&mut device.tdma);
                        ffu_tile_sp(device, &x_n, w4_dm, scale_dm, &$gs)
                    }};
                }
                macro_rules! mpair {
                    ($i:expr) => {{
                        let up_t = mtile!(up_w, up_s, up_gs, $i);
                        let gate_t = mtile!(gate_w, gate_s, gate_gs, $i);
                        geglu_sp(device, up_t, &gate_t, &mut h_buf, $i);
                    }};
                }
                mpair!(0);
                mpair!(1);
                mpair!(2);
                mpair!(3);
                let h_buf = h_buf;
                // hop through HBM: chunk-major h (64 chunks of 256 columns), replicated 8x by the DMA of the down projection
                let h: HbmTensor<bf16, Chip, m![G32, Fc2, Fh2, $T, Rt256]> = h_buf.to_hbm(&mut device.tdma);
                let hv: HbmTensorView<'_, bf16, Chip, m![C64, $T, Rt256]> = unsafe { h.view().reshape() };
                let x_h: DmTensor<bf16, Chip, Cl8, Sl8, m![$T, Rt256]> = hv.to_dm(&mut device.tdma);
                let d_gs = scalar_vrf8(device, dgs);
                let mut dbuf: DmTensor<bf16, Chip, Cl8, SlOut8, m![$T, Fi5, Rt96]> = DmTensor::new();
                macro_rules! dtile {
                    ($i:expr) => {{
                        let w4_dm: DmTensor<f4e2m1, Chip, Cl8, Sl8, m![Rt96, Rt256]> = down_w
                            .view()
                            .tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Rt256]>($i)
                            .to_dm(&mut device.tdma);
                        let scale_dm: DmTensor<f8e4m3, Chip, Cl8, Sl8, m![Rt96, Fb16]> = down_s
                            .view()
                            .tile::<m![Fi5], 1, m![Gd8, C64, Fi5 = 1 # 5, Rt96, Fb16]>($i)
                            .to_dm(&mut device.tdma);
                        ffd_tile8(device, &x_h, w4_dm, scale_dm, &d_gs, &mut dbuf, $i);
                    }};
                }
                dtile!(0);
                dtile!(1);
                dtile!(2);
                dtile!(3);
                dtile!(4);
                down_tail8(device, &dbuf, res, post_w, layer_scalar)
            }

            /// The whole FFN of a chunk in one launch (see `ffn_core`).
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn ffn_chunk(
                device: &mut Device,
                res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
                up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ugs: &HbmTensor<f32, Chip, m![1]>,
                gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                ggs: &HbmTensor<f32, Chip, m![1]>,
                down_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi5, Rt48, Fh2, Rt256]>,
                down_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi5, Rt48, Fh2, Fb16]>,
                dgs: &HbmTensor<f32, Chip, m![1]>,
                post_w: &HbmTensor<bf16, Chip, m![H]>,
                layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            ) {
                let res = normp::p1920::load_rows::<$T, $TCl, $Pre>(device, res_in);
                let out = ffn_core(device, res_in, &res, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_w, layer_scalar);
                normp::p1920::store_rows::<$T, $TCl, $Pre>(device, &out, res_out);
            }

            /// Attention output + FFN of a chunk in ONE launch (`$E` = `E32` sliding / `E256` full): the AO output (the mid residual) hops through a
            /// compiler-placed HBM scratch row tensor (zero pad columns written explicitly) for the replicated x of the FFN and stays in DM as the FFN residual.
            macro_rules! layer_fn {
                ($fname:ident, $E:ident, $core:path) => {
                    #[allow(clippy::too_many_arguments)]
                    pub(crate) fn $fname(
                        device: &mut Device,
                        attn: &HbmTensor<bf16, Chip, m![$T, C32, $E]>,
                        w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
                        w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
                        w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
                        s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
                        s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
                        s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
                        post_attn_norm: &HbmTensor<bf16, Chip, m![H]>,
                        res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                        pre_w: &HbmTensor<bf16, Chip, m![C32, E32]>,
                        up_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                        up_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                        ugs: &HbmTensor<f32, Chip, m![1]>,
                        gate_w: &HbmTensor<f4e2m1, Chip, m![G32, C32, Fi4, Rt240, E32]>,
                        gate_s: &HbmTensor<f8e4m3, Chip, m![G32, C32, Fi4, Rt240, Fb8]>,
                        ggs: &HbmTensor<f32, Chip, m![1]>,
                        down_w: &HbmTensor<f4e2m1, Chip, m![Gd8, C64, Fi5, Rt96, Rt256]>,
                        down_s: &HbmTensor<f8e4m3, Chip, m![Gd8, C64, Fi5, Rt96, Fb16]>,
                        dgs: &HbmTensor<f32, Chip, m![1]>,
                        post_ff: &HbmTensor<bf16, Chip, m![H]>,
                        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
                        res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
                    ) {
                        let t0: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w0.to_dm(&mut device.tdma);
                        let t1: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w1.to_dm(&mut device.tdma);
                        let t2: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> = w2.to_dm(&mut device.tdma);
                        let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
                        let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
                        let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
                        let res = normp::p1920::load_rows::<$T, $TCl, $Pre>(device, res_in);
                        let mid = $core(device, attn, t0, t1, t2, sc0, sc1, sc2, post_attn_norm, &res);

                        // mid residual rows -> HBM scratch [T, C32, E32] (30 real chunks + 2 explicit zero chunks)
                        let mut mid_hbm: HbmTensor<bf16, Chip, m![$T, C32, E32]> = HbmTensor::new();
                        normp::p1920::store_rows::<$T, $TCl, $Pre>(device, &mid, &mut mid_hbm);
                        let mut zero_pad: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, 1 # 2], m![C32 = 2, E32]> = DmTensor::new();
                        zero_pad.view_mut().memset(const { bf16::from_f32(0.0) }, &mut device.sub);
                        zero_pad.view().to_hbm_view(
                            &mut device.tdma,
                            mid_hbm.view_mut().tile::<m![C32], 2, m![$T, C32 = 2 #{!} 32, E32]>(30),
                        );
                        let out = ffn_core_d8(device, &mid_hbm, &mid, pre_w, up_w, up_s, ugs, gate_w, gate_s, ggs, down_w, down_s, dgs, post_ff, layer_scalar);
                        normp::p1920::store_rows::<$T, $TCl, $Pre>(device, &out, res_out);
                    }
                };
            }
            layer_fn!(layer_sliding, E32, $aos);
            layer_fn!(layer_full, E256, $aof);
        }
    };
}

ffn_mod!(f_t64, Tk64, 2, m![G32 % 8, Tk64 / 2], m![Tk64 / 32], m![1 # 4, Tk64 % 32], crate::device::ffnc::ao::sliding_core_t64, crate::device::ffnc::ao::full_core_t64);
ffn_mod!(f_t128, Tk128, 4, m![G32 % 8, Tk128 / 4], m![Tk128 / 64], m![1 # 2, Tk128 % 64], crate::device::ffnc::ao::sliding_core_t128, crate::device::ffnc::ao::full_core_t128);
ffn_mod!(f_t8, Tk8, 2, m![G32 % 8, Tk8 / 2, 1 # 8], m![1 # 2], m![1 # 16, Tk8], crate::device::ffnc::ao::sliding_core_t8, crate::device::ffnc::ao::full_core_t8);
ffn_mod!(f_t16, Tk16, 2, m![G32 % 8, Tk16 / 2, 1 # 4], m![1 # 2], m![1 # 8, Tk16], crate::device::ffnc::ao::sliding_core_t16, crate::device::ffnc::ao::full_core_t16);
ffn_mod!(f_t32, Tk32, 2, m![G32 % 8, Tk32 / 2, 1 # 2], m![1 # 2], m![1 # 4, Tk32], crate::device::ffnc::ao::sliding_core_t32, crate::device::ffnc::ao::full_core_t32);
