//! Raw FFI to the vendored DeepSeek kernels (FlashMLA, DeepGEMM) and their
//! thin C-ABI shims, compiled by `build.rs` into `libdeepseek_kernels.a`.
//! The build also compiles the vendored FA3-hopper instantiation units; their
//! C ABI is the dependent crate's shim, so no Rust declarations live here.
//!
//! Dependents' build scripts read the `links = "deepseek_kernels"` metadata:
//! `DEP_DEEPSEEK_KERNELS_{FLASHMLA,FA3,DEEPGEMM_NATIVE}` (`0`/`1`),
//! `_LIB` (archive path), `_ROOT`, `_FA3_ROOT`, `_CUTLASS_INCLUDE`.

#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::{CUresult, CUstream};

/// bf16/f16 element as raw bits — same layout as CUDA `__nv_bfloat16`/`half`.
pub type Half = u16;

#[cfg(feature = "cuda")]
pub mod deepgemm;
#[cfg(feature = "cuda")]
pub mod flashmla;
