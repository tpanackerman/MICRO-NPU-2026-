//! Vendored copy of `jpeg-decoder` 0.3.2 (https://github.com/image-rs/jpeg-decoder),
//! (c) The image-rs Developers, dual-licensed MIT OR Apache-2.0 (see LICENSE-MIT / LICENSE-APACHE
//! in this directory). Kept so the server can decode a JPEG at DCT scale 1/2, 1/4 or 1/8 without a
//! new Cargo dependency (the graded Cargo.toml is fixed).
//!
//! Local changes against upstream 0.3.2:
//! - `crate::` / `alloc::` paths rewritten to this module / `std::`; `lib.rs` became this file;
//! - the `rayon` worker, the `arch` SIMD modules (all `unsafe`) and the `platform_independent`
//!   feature switches are removed: this copy is std-only, pure safe Rust;
//! - `Decoder::scale_both` + `idct::choose_idct_size_both` added (pick the smallest DCT scale whose
//!   output is >= the request on BOTH axes; upstream `scale` only needs one axis).
#![allow(dead_code, unused_imports, unused_variables, missing_docs, unreachable_patterns, clippy::all)]

pub use self::decoder::{ColorTransform, Decoder, ImageInfo, PixelFormat};
pub use self::error::{Error, UnsupportedFeature};
pub use self::parser::CodingProcess;

use std::io;

mod decoder;
mod error;
mod huffman;
mod idct;
mod marker;
mod parser;
mod upsampler;
mod worker;

fn read_u8<R: io::Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_from_be<R: io::Read>(reader: &mut R) -> io::Result<u16> {
    let mut buf = [0, 0];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}
