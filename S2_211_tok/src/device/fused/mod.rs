//! FUSED SLIDING LAYER: QKV + attention + AO + FFN of one sliding layer of the 256-row hybrid pass (KVD-256, 16 decode riders) in ONE device function.
//!
//! Why: the device keeps only the image of the previous launch, so a pass that alternates QKV / attention / layer launches reloads a 135-352 KB image on every
//! launch (about 0.1 ms per layer kernel, measured: the same layer launched back to back runs 744 us, interleaved 841 us). One function per sliding layer
//! makes the five sliding layers between two full layers the SAME image back to back.
//!
//! A launch takes at most 22 arguments, so the small per-layer weights travel in ONE bf16 blob (`Sw`), the per-pass bf16 tables in ONE region of the
//! packed upload (`Tb`), and same-shape weights in grouped tensors (`[2, ..]` k / v, `[3, ..]` o_proj tiles, `[2, ..]` up / gate). A typed view of a
//! blob is `reshape` of the WHOLE view into `[slots, shape]` and then `tile` of one slot (a tiled view cannot be reshaped): every piece sits at an
//! offset that is a multiple of its own size. Intermediates (normalised x, q, attention output) live in ONE scratch tensor `[Sc3, Tk256, Kc, Ke]`
//! so the compiler orders every write before the reads of the same handle.
use furiosa_opt_std::prelude::*;

pub(crate) mod attn;
pub(crate) mod qkv;

/// Scratch regions: 0 = normalised x, 1 = q, 2 = attention output (each 256 rows x 4096).
axes![Sc3 = 3];
/// Grouped weights: k / v and up / gate.
axes![Wkv = 2, Wug = 2];
/// Small-weight blob (bf16, per layer) and its slot counts per piece size (3840, 4096, 2048, 1280, 256, 8 elements).
axes![Sw = 61440, Sb16 = 16, Sb15 = 15, Sb30 = 30, Sb48 = 48, Sb240 = 240, Sb7680 = 7680];
/// Per-pass table region (bf16) and its slot counts per piece size (2^18, 2^14, 2^12 elements).
axes![Tb = 786432, Tt18 = 3, Tt14 = 48, Tt12 = 192];
/// The three NVFP4 global scales of a layer (up, gate, down), f32.
axes![Gs3 = 3];

/// Slots of the small-weight blob.
pub(crate) mod sw {
    pub(crate) const INPUT_NORM: usize = 0; // Sb16 ([H])
    pub(crate) const POST_ATTN_NORM: usize = 1; // Sb16
    pub(crate) const POST_FF_NORM: usize = 2; // Sb16
    pub(crate) const SQ: usize = 3; // Sb15 ([Rq4, Gg, Rt])
    pub(crate) const PRE_NORM: usize = 5; // Sb15 ([C32, E32])
    pub(crate) const SK: usize = 8; // Sb30 ([Rq2, Gg, Rt])
    pub(crate) const SV: usize = 9; // Sb30
    pub(crate) const AO_S: usize = 20; // Sb48 ([G32, Rt80]), tiles 20, 21, 22
    pub(crate) const Q_NORM: usize = 115; // Sb240 ([Ds])
    pub(crate) const K_NORM: usize = 116; // Sb240
    pub(crate) const LAYER_SCALAR: usize = 3744; // Sb7680 ([1 # 8])
    /// element offset of each slot (the host packs the blob with these)
    pub(crate) const OFFSETS: [(usize, usize); 13] = [
        (0, 3840),
        (3840, 3840),
        (7680, 3840),
        (12288, 4096),
        (20480, 4096),
        (16384, 2048),
        (18432, 2048),
        (25600, 1280),
        (26880, 1280),
        (28160, 1280),
        (29440, 256),
        (29696, 256),
        (29952, 8),
    ];
}

/// Slots of the per-pass table region.
pub(crate) mod tb {
    pub(crate) const MASK: usize = 0; // Tt18 ([Pw, Ts % 32, Tc, Seg, Ts / 32]), waves 0, 1
    pub(crate) const COS: usize = 32; // Tt14 ([Tc, Ds]), units 0..4
    pub(crate) const SIN: usize = 36; // Tt14
    pub(crate) const MASK_D: usize = 40; // Tt14 ([Bd16, Tx % 32, Tx / 32])
    pub(crate) const PEN: usize = 168; // Tt12 ([Pw, Ts % 32, Seg, Ts / 32]), waves 0, 1
}
