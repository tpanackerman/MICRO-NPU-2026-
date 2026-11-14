use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy32, Ns, Ts};

/// 256 slices: head x key lane (ring row r lives in slice (h, r % 32), slot r / 32).
pub(crate) type HeadKb = m![Ns, Ts % 32];
/// Row statistics after an inter-slice all-reduce: the 32 key-block slices of a head all hold the
/// same value (fresh broadcast axis `Dummy32`).
pub(crate) type HeadX = m![Ns, Dummy32];
/// After an inter-slice Add reduce the result lives in key-block slice 0 of each head.
pub(crate) type HeadOne = m![Ns, 1 # 32];
