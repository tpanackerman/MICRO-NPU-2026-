//! Sliding / full attention-output chunk kernels: fp8 GEMM (weight stationary in the TRF, K split 32, row tiles of 80 as in kprobe `wst`) -> y on the 16 live slices ->
//! DM->DM re-shard to the 2-part layout (`normp`) -> RMSNorm + residual -> residual out.
use furiosa_opt_std::prelude::*;

use super::normp;
use super::{C32, E32, E256, G32, Hi3, Rt80, Tk8, Tk16, Tk32, Tk64, Tk128, Tk256};
use crate::Chip;
use crate::axes::{H, Rh};

type Cl = m![G32 / 8];
type Sl = m![G32 % 8, C32];
type SlOut = m![G32 % 8, 1 # 32];

/// One 80-row tile: decode (fetch_cast + Cast), TRF load, contraction, K reduce, row scale; result -> tile `i` of `ybuf`.
macro_rules! ao_proc {
    ($name:ident, $load:ident, $T:ident, $E:ident) => {
        pub(crate) fn $load(
            device: &mut Device,
            w: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
        ) -> DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]> {
            w.to_dm(&mut device.tdma)
        }

        pub(crate) fn $name(
            device: &mut Device,
            x_dm: &DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]>,
            w_f8: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]>,
            scale_dm: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]>,
            ybuf: &mut DmTensor<bf16, Chip, Cl, SlOut, m![$T, Hi3, Rt80]>,
            i: usize,
        ) {
            let scale_vrf: VrfTensor<f32, Chip, Cl, SlOut, m![Rt80]> = device
                .sub
                .begin(scale_dm.view())
                .fetch::<m![Rt80 / 8], m![Rt80 % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![Rt80 / 8], m![Rt80 % 8]>()
                .to_vrf();

            let w_bf: DmTensor<bf16, Chip, Cl, Sl, m![Rt80, $E]> = device
                .main
                .begin(w_f8.view())
                .fetch::<m![Rt80, $E / 32], m![$E % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![Rt80, $E / 32, $E % 32 / 8], m![$E % 8]>()
                .cast::<bf16, m![$E % 8 # 16]>()
                .commit_trim::<m![$E % 8]>()
                .commit();
            let w_trf: TrfTensor<bf16, Chip, Cl, Sl, m![Rt80 % 8], m![Rt80 / 8, $E]> = device
                .sub
                .begin(w_bf.view())
                .fetch::<m![Rt80 % 8, Rt80 / 8, $E / 16], m![$E % 16]>()
                .collect::<m![Rt80 % 8, Rt80 / 8, $E / 16], m![$E % 16]>()
                .to_trf();

            device
                .main
                .begin(x_dm.view())
                .fetch::<m![$T, $E / 16], m![$E % 16]>()
                .collect::<m![$T, $E / 16], m![$E % 16]>()
                .contract_outer::<m![$T, $E / 32, Rt80 / 8], m![$E % 32], _, _, _>(&w_trf)
                .contract_packet::<m![1]>()
                .contract_time::<m![$T, Rt80 / 8]>()
                .contract_lane::<m![$T, Rt80 / 8], m![Rt80 % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<SlOut, m![$T, Rt80 / 8]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![$T, Rt80 / 8, Rt80 / 4 % 2], m![Rt80 % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
                .vector_widen_concat::<m![$T, Rt80 / 8], m![Rt80 % 8]>()
                .vector_final()
                .cast::<bf16, m![Rt80 % 8 # 16]>()
                .commit_trim::<m![Rt80 % 8]>()
                .commit_view(ybuf.view_mut().tile::<m![Hi3], 1, m![$T, Hi3 = 1 #{!} 3, Rt80]>(i));
        }
    };
}

ao_proc!(proc_t8, load_t8, Tk8, E32);
ao_proc!(proc_t16, load_t16, Tk16, E32);
ao_proc!(proc_t32, load_t32, Tk32, E32);
ao_proc!(proc_t64, load_t64, Tk64, E32);
ao_proc!(proc_t128, load_t128, Tk128, E32);
ao_proc!(proc_t256, load_t256, Tk256, E32);
ao_proc!(proc_f_t8, load_f_t8, Tk8, E256);
ao_proc!(proc_f_t16, load_f_t16, Tk16, E256);
ao_proc!(proc_f_t32, load_f_t32, Tk32, E256);
ao_proc!(proc_f_t64, load_f_t64, Tk64, E256);
ao_proc!(proc_f_t128, load_f_t128, Tk128, E256);
ao_proc!(proc_f_t256, load_f_t256, Tk256, E256);



/// AO chunk core for one T: attention output `x` (natural layout `[T, C32, E]` = `[T, Qs]`), the three loaded o_proj tiles and row scales, the residual rows
/// (part layout) -> post-attention RMSNorm + residual, in the 2-part layout of `normp::p1920`: slice (token, part) holds the 1920 columns of 8 row groups
/// (part = source cluster); `$Pre` = token axes of the slice mapping (the slice count is `$Pre` x 2).
macro_rules! ao_chunk {
    ($name:ident, $core:ident, $proc:ident, $load:ident, $T:ident, $E:ident, $TCl:ty, $Pre:ty, $XL:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $core(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, $XL>,
            t0: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]>,
            t1: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]>,
            t2: DmTensor<f8e4m3, Chip, Cl, Sl, m![Rt80, $E]>,
            sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]>,
            sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]>,
            sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
        ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
            let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]> = x.to_dm(&mut device.tdma);
            let mut ybuf: DmTensor<bf16, Chip, Cl, SlOut, m![$T, Hi3, Rt80]> = DmTensor::new();
            $proc(device, &x_dm, t0, sc0, &mut ybuf, 0);
            $proc(device, &x_dm, t1, sc1, &mut ybuf, 1);
            $proc(device, &x_dm, t2, sc2, &mut ybuf, 2);

            // re-shard: slice (token, part) holds the 1920 columns [part * 1920, (part + 1) * 1920) of the token row
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, G32 / 8], m![G32 % 8, Hi3, Rt80]> = ybuf.to_dm(&mut device.tdma);
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
            normp::p1920::norm_residual::<$TCl, $Pre>(device, &yb, res, post_norm)
        }

        /// Whole AO chunk (separate tile tensors): packed weights, residual in / out.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, $XL>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res_in: &HbmTensor<bf16, Chip, m![$T, C32, E32]>,
            res_out: &mut HbmTensor<bf16, Chip, m![$T, C32, E32]>,
        ) {
            let t0 = $load(device, w0);
            let t1 = $load(device, w1);
            let t2 = $load(device, w2);
            let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
            let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
            let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
            let res = normp::p1920::load_rows::<$T, $TCl, $Pre>(device, res_in);
            let out = $core(device, x, t0, t1, t2, sc0, sc1, sc2, post_norm, &res);
            normp::p1920::store_rows::<$T, $TCl, $Pre>(device, &out, res_out);
        }
    };
}

ao_chunk!(sliding_ao_t8, sliding_core_t8, proc_t8, load_t8, Tk8, E32, m![1 # 2], m![1 # 16, Tk8], m![Tk8, C32, E32]);
ao_chunk!(sliding_ao_t16, sliding_core_t16, proc_t16, load_t16, Tk16, E32, m![1 # 2], m![1 # 8, Tk16], m![Tk16, C32, E32]);
ao_chunk!(sliding_ao_t32, sliding_core_t32, proc_t32, load_t32, Tk32, E32, m![1 # 2], m![1 # 4, Tk32], m![Tk32, C32, E32]);
ao_chunk!(sliding_ao_t64, sliding_core_t64, proc_t64, load_t64, Tk64, E32, m![Tk64 / 32], m![1 # 4, Tk64 % 32], m![Tk64, C32, E32]);
ao_chunk!(sliding_ao_t128, sliding_core_t128, proc_t128, load_t128, Tk128, E32, m![Tk128 / 64], m![1 # 2, Tk128 % 64], m![Tk128, C32, E32]);
ao_chunk!(sliding_ao_t256, sliding_core_t256, proc_t256, load_t256, Tk256, E32, m![Tk256 / 128], m![Tk256 % 128], m![Tk256, C32, E32]);
ao_chunk!(full_ao_t8, full_core_t8, proc_f_t8, load_f_t8, Tk8, E256, m![1 # 2], m![1 # 16, Tk8], m![Tk8, C32, E256]);
ao_chunk!(full_ao_t16, full_core_t16, proc_f_t16, load_f_t16, Tk16, E256, m![1 # 2], m![1 # 8, Tk16], m![Tk16, C32, E256]);
ao_chunk!(full_ao_t32, full_core_t32, proc_f_t32, load_f_t32, Tk32, E256, m![1 # 2], m![1 # 4, Tk32], m![Tk32, C32, E256]);
ao_chunk!(full_ao_t64, full_core_t64, proc_f_t64, load_f_t64, Tk64, E256, m![Tk64 / 32], m![1 # 4, Tk64 % 32], m![Tk64, C32, E256]);
ao_chunk!(full_ao_t128, full_core_t128, proc_f_t128, load_f_t128, Tk128, E256, m![Tk128 / 64], m![1 # 2, Tk128 % 64], m![Tk128, C32, E256]);
ao_chunk!(full_ao_t256, full_core_t256, proc_f_t256, load_f_t256, Tk256, E256, m![Tk256 / 128], m![Tk256 % 128], m![Tk256, C32, E256]);

/// (T = 256) AO core of ONE 128-token half of a `[Rh, T, ..]` activation tensor (`x` read through a view of half `half`); same arithmetic as `ao_chunk!`'s core.
macro_rules! ao_half {
    ($core:ident, $proc:ident, $load:ident, $T:ident, $E:ident, $TCl:ty, $Pre:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $core(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![Rh, $T, C32, $E]>,
            half: usize,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
        ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
            let t0 = $load(device, w0);
            let t1 = $load(device, w1);
            let t2 = $load(device, w2);
            let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
            let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
            let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
            let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]> = x.view().tile::<m![Rh], 1, m![Rh = 1 # 2, $T, C32, $E]>(half).to_dm(&mut device.tdma);
            let mut ybuf: DmTensor<bf16, Chip, Cl, SlOut, m![$T, Hi3, Rt80]> = DmTensor::new();
            $proc(device, &x_dm, t0, sc0, &mut ybuf, 0);
            $proc(device, &x_dm, t1, sc1, &mut ybuf, 1);
            $proc(device, &x_dm, t2, sc2, &mut ybuf, 2);
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, G32 / 8], m![G32 % 8, Hi3, Rt80]> = ybuf.to_dm(&mut device.tdma);
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
            normp::p1920::norm_residual::<$TCl, $Pre>(device, &yb, res, post_norm)
        }
    };
}
ao_half!(sliding_core_half_t128, proc_t128, load_t128, Tk128, E32, m![Tk128 / 64], m![1 # 2, Tk128 % 64]);
ao_half!(full_core_half_t128, proc_f_t128, load_f_t128, Tk128, E256, m![Tk128 / 64], m![1 # 2, Tk128 % 64]);

// R13 fusedlayer: the same half-AO taking the attention rows of ONE half as a view of the attention kernel's own output handle (ordering through the SAME handle inside a fused launch).
macro_rules! ao_half_v {
    ($core:ident, $proc:ident, $load:ident, $T:ident, $E:ident, $TCl:ty, $Pre:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $core(
            device: &mut Device,
            x: HbmTensorView<'_, bf16, Chip, m![$T, C32, $E]>,
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res: &DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]>,
        ) -> DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> {
            let t0 = $load(device, w0);
            let t1 = $load(device, w1);
            let t2 = $load(device, w2);
            let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
            let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
            let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
            let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![$T, $E]> = x.to_dm(&mut device.tdma);
            let mut ybuf: DmTensor<bf16, Chip, Cl, SlOut, m![$T, Hi3, Rt80]> = DmTensor::new();
            $proc(device, &x_dm, t0, sc0, &mut ybuf, 0);
            $proc(device, &x_dm, t1, sc1, &mut ybuf, 1);
            $proc(device, &x_dm, t2, sc2, &mut ybuf, 2);
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, G32 / 8], m![G32 % 8, Hi3, Rt80]> = ybuf.to_dm(&mut device.tdma);
            let yb: DmTensor<bf16, Chip, $TCl, m![{ $Pre }, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
            normp::p1920::norm_residual::<$TCl, $Pre>(device, &yb, res, post_norm)
        }
    };
}

ao_half_v!(sliding_core_halfv_t128, proc_t128, load_t128, Tk128, E32, m![Tk128 / 64], m![1 # 2, Tk128 % 64]);

// R14 s211ao: port of the teammate's t256r (S2_211_tok, minhnq1402, source commit cdf0c8e: `ao::sliding_core_t256` inside `layer_{sliding,full}_t256r`).
// The o_proj of ALL 256 rows of a hybrid pass in ONE contraction per 80-row weight tile (decode + TRF load once per tile, not once per 128-row half); the attention
// rows of the two halves are DMA'd into the two Rh tiles of ONE DM tensor (the fused K / K' launches keep their two attention-output handles); the post-attention
// RMSNorm + residual run ONCE in the dense 256-row part layout (2 clusters x 256 slices; the 128-row layout fills half the slices of its 2 clusters).
// Per-row arithmetic is that of `ao_half_v!` (same K order per row; the inter-slice part reduce adds the same 2 partial sums).
// `two`: the attention rows of the two halves come from two handles (fused K / K'); `one`: one `[Tk256, C32, E]` view, ONE DMA (non-fused launches).
macro_rules! ao_r256_x {
    (two, $device:ident, $x0:ident $x1:ident, $E:ident) => {{
        let mut x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Rh, Tk128, $E]> = DmTensor::new();
        $x0.to_dm_view(&mut $device.tdma, x_dm.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, $E]>(0));
        $x1.to_dm_view(&mut $device.tdma, x_dm.view_mut().tile::<m![Rh], 1, m![Rh = 1 #{!} 2, Tk128, $E]>(1));
        let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tk256, $E]> = unsafe { x_dm.reshape() };
        x_dm
    }};
    (one, $device:ident, $x0:ident, $E:ident) => {{
        let x_dm: DmTensor<bf16, Chip, Cl, Sl, m![Tk256, $E]> = $x0.to_dm(&mut $device.tdma);
        x_dm
    }};
}
macro_rules! ao_r256 {
    ($core:ident, $proc:ident, $load:ident, $E:ident, $xin:ident, $($x:ident: $X:ty),+) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn $core(
            device: &mut Device,
            $($x: $X,)+
            w0: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w1: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            w2: &HbmTensor<f8e4m3, Chip, m![G32, C32, Rt80, $E]>,
            s0: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s1: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            s2: &HbmTensor<bf16, Chip, m![G32, Rt80]>,
            post_norm: &HbmTensor<bf16, Chip, m![H]>,
            res: &DmTensor<bf16, Chip, m![Tk256 / 128], m![Tk256 % 128, H / 1920], m![H % 1920]>,
        ) -> DmTensor<bf16, Chip, m![Tk256 / 128], m![Tk256 % 128, H / 1920], m![H % 1920]> {
            let t0 = $load(device, w0);
            let t1 = $load(device, w1);
            let t2 = $load(device, w2);
            let sc0: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s0.to_dm(&mut device.tdma);
            let sc1: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s1.to_dm(&mut device.tdma);
            let sc2: DmTensor<bf16, Chip, Cl, SlOut, m![Rt80]> = s2.to_dm(&mut device.tdma);
            let x_dm = ao_r256_x!($xin, device, $($x)+, $E);
            let mut ybuf: DmTensor<bf16, Chip, Cl, SlOut, m![Tk256, Hi3, Rt80]> = DmTensor::new();
            $proc(device, &x_dm, t0, sc0, &mut ybuf, 0);
            $proc(device, &x_dm, t1, sc1, &mut ybuf, 1);
            $proc(device, &x_dm, t2, sc2, &mut ybuf, 2);
            let yb: DmTensor<bf16, Chip, m![Tk256 / 128], m![Tk256 % 128, G32 / 8], m![G32 % 8, Hi3, Rt80]> = ybuf.to_dm(&mut device.tdma);
            let yb: DmTensor<bf16, Chip, m![Tk256 / 128], m![Tk256 % 128, H / 1920], m![H % 1920]> = unsafe { yb.reshape() };
            normp::p1920::norm_residual::<m![Tk256 / 128], m![Tk256 % 128]>(device, &yb, res, post_norm)
        }
    };
}
ao_r256!(sliding_core_r256, proc_t256, load_t256, E32, two, x0: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>, x1: HbmTensorView<'_, bf16, Chip, m![Tk128, C32, E32]>);
ao_r256!(sliding_core_r256_1, proc_t256, load_t256, E32, one, x0: HbmTensorView<'_, bf16, Chip, m![Tk256, C32, E32]>);
ao_r256!(full_core_r256_1, proc_f_t256, load_f_t256, E256, one, x0: HbmTensorView<'_, bf16, Chip, m![Tk256, C32, E256]>);
