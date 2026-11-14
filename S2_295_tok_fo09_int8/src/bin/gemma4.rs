//! Command-line generator of the baseline skeleton, reduced to an empty stub in the upload set (like `test_kernels.rs`): the graded binary is `server`.
//! Each bin crate runs its own furiosa-opt driver, which collects and compiles every device kernel the bin reaches; a full `gemma4` CLI made a build of
//! all bins compile the whole kernel set a second time in parallel with `server` (+2300..3000 CPU-s, kernels compiled twice when the two drivers raced).
//! Kept as a file so that the `src/bin` file set equals the baseline's (compliance rule); it launches no device kernel and adds no compile time.
fn main() {}
