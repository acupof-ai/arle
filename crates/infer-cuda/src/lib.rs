//! CUDA backend executor (consumer NVIDIA + datacenter) — R6 track.
//!
//! Stub: to be filled with `CudaKvPool` (host-indexed page bookkeeping, mirroring
//! `infer_metal::MetalKvPool`) and a `CudaExecutor` implementing the
//! `infer_seam::BackendExecutor` seam. The real device KV buffers + the
//! TileLang/cuda-kernels forward are wired against `crates/cuda-kernels` and
//! verified on V100 (Qwen) / H20 (DSv4) per the verification-targets doc.
//!
//! Depends only on the stable `infer-plan` + `infer-seam` contracts.
