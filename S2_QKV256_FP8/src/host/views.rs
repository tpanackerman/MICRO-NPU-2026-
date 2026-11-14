//! Typed views of a byte range of a device tensor: same device allocation, no copy, no launch.
//!
//! Why: a device function is compiled for the exact tensor shapes of its arguments, and the firmware reloads the code of every launch whose image differs from
//! the previous launch's (about 0.2 us per KB of image: 47 us for a QKV image, 65 us for an AO + FFN image, see launch/RESULTS.md). The T = 16 decode and T = 128 chunk
//! passes therefore carried second images (`*_h0` / `*_h1`, which read one half of a 2-half residual through compile-time tile views) and `cat_*` copy kernels
//! (two half-size attention outputs -> one double-size input). With views the plain T = 8 / T = 64 kernels run on the halves directly (the half is just a
//! tensor at another address) and the attention kernels write straight into the halves of the concatenated buffer, so the second launch of an image costs no
//! image switch and the copy launches disappear.
//!
//! Mechanism (public `furiosa-opt-std` / `-rt` API only): `DeviceSend::bind` yields the tensor's `Buffer`, `Buffer::slice` shares its allocation,
//! `impl From<Buffer> for HbmTensor` wraps the slice. The slice keeps the allocation alive; the device addresses it like any other tensor.

use furiosa_opt_std::prelude::*;

use crate::Chip;

/// `S2_VIEWS=0` keeps the `*_h0` / `*_h1` kernels and the `cat_*` copies (A/B switch); default: views (NPU builds only: the CPU emulator cannot alias tensors).
pub fn enabled() -> bool {
    match OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return cfg!(backend = "npu"),
        _ => {}
    }
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| cfg!(backend = "npu") && std::env::var("S2_VIEWS").map(|value| value != "0").unwrap_or(true))
}

/// Tests: force views off / on at run time (`None` = back to the environment default), for A/B runs inside one process.
pub fn set_override(on: Option<bool>) {
    OVERRIDE.store(on.map_or(0, |on| if on { 2 } else { 1 }), std::sync::atomic::Ordering::Relaxed);
}

static OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The tensor of shape `E2` that starts `index * size(E2)` bytes into `tensor`'s allocation (`index` = the half / piece number; offsets stay multiples of the
/// piece size, which for the shapes used here is a multiple of 64 KB).
#[cfg(backend = "npu")]
pub fn piece<D: Scalar, E: M, E2: M>(tensor: &HbmTensor<D, Chip, E>, index: usize) -> Result<HbmTensor<D, Chip, E2>, Error> {
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    let mut buffers = Buffers::new();
    tensor.bind(&mut buffers)?;
    let buffer = buffers.pop().expect("a placed HBM tensor binds one buffer");
    let bytes = HbmTensor::<D, Chip, E2>::size();
    let start = index * bytes;
    assert!(start + bytes <= buffer.size(), "view past the end of the tensor");
    Ok(HbmTensor::from(buffer.slice(start..start + bytes)))
}

/// The CPU emulator keeps tensor bytes on the host, where a view cannot alias them: callers must not enable views there.
#[cfg(not(backend = "npu"))]
pub fn piece<D: Scalar, E: M, E2: M>(_tensor: &HbmTensor<D, Chip, E>, _index: usize) -> Result<HbmTensor<D, Chip, E2>, Error> {
    Err(Error::Unplaced)
}

/// A tensor of shape `E2` and element type `D2` that starts `byte_start` bytes into `tensor`'s allocation (the host packs many small tables into one upload and slices them out
/// as views; `byte_start` must keep the alignment the callers use: a multiple of 8 KB).
#[cfg(backend = "npu")]
pub fn piece_at<D: Scalar, E: M, D2: Scalar, E2: M>(tensor: &HbmTensor<D, Chip, E>, byte_start: usize) -> Result<HbmTensor<D2, Chip, E2>, Error> {
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    let mut buffers = Buffers::new();
    tensor.bind(&mut buffers)?;
    let buffer = buffers.pop().expect("a placed HBM tensor binds one buffer");
    let bytes = HbmTensor::<D2, Chip, E2>::size();
    assert!(byte_start + bytes <= buffer.size(), "view past the end of the tensor");
    Ok(HbmTensor::from(buffer.slice(byte_start..byte_start + bytes)))
}

#[cfg(not(backend = "npu"))]
pub fn piece_at<D: Scalar, E: M, D2: Scalar, E2: M>(_tensor: &HbmTensor<D, Chip, E>, _byte_start: usize) -> Result<HbmTensor<D2, Chip, E2>, Error> {
    Err(Error::Unplaced)
}
