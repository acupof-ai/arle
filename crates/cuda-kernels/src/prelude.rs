//! Proto-API contract for the CUDA kernel crate.
//!
//! This module is the single import point for the seven CUDA tensor / pool /
//! metadata types that 25+ files in `model/` and `ops/` share. It exists for
//! two reasons:
//!
//! 1. **Stable consumer surface.** Model code says
//!    `use cuda_kernels::prelude::{DeviceContext, DeviceVec, …};`
//!    instead of three separate `use cuda_kernels::{tensor::…,
//!    paged_kv::…, tilelang::…}` lines. The consumer's import block is
//!    insulated from the underlying module layout.
//!
//! 2. **Proto-API for the eventual `cuda-kernels` crate extraction.**
//!    See `docs/plans/cuda-kernel-crate-extraction.md`. When one of the
//!    trip wires (FA-3 H100, MLA / DeepSeek-V3, NCCL tensor parallel,
//!    parallel kernel build configs, second consumer of the kernel layer)
//!    fires, this prelude becomes the public API surface of the extracted
//!    crate. Every `pub(crate)` below becomes a real cross-crate `pub` —
//!    and **that is the entire diff for the public surface.** No new
//!    symbols become public, no internal type leaks. The narrower this
//!    list stays, the cheaper the future extraction is.
//!
//! ## Discipline rule
//!
//! Adding a new symbol here requires three justifications, in writing,
//! at the call site of the PR that proposes it:
//!
//! - **It is consumed by ≥3 files outside `backend/cuda/`.** Two-file
//!   helpers stay on direct module paths (e.g. `TokenKVPool`, used by
//!   `model/qwen3/prefill.rs` + `model/qwen35/prefill.rs` + `model.rs`,
//!   does not qualify and lives at `super::paged_kv::TokenKVPool`).
//! - **It is stable.** "Stable" means: its name, layout, and method
//!   signatures will not change in the next 6 months. Internal types
//!   that are still in active design must not be in the prelude.
//! - **Removing it would not break the kernel-crate extraction plan.**
//!   If exporting a symbol forces some currently-private `infer` type
//!   to become `pub` cross-crate (e.g. a model-specific weight struct,
//!   a tokenizer detail, a scheduler state enum), the symbol does not
//!   belong here — it belongs in `infer` proper, not in the kernel
//!   layer.
//!
//! Removing a symbol is encouraged. If a symbol stops meeting all three
//! criteria, kick it out and update the affected consumers.
//!
//! ## What the prelude intentionally does NOT contain
//!
//! - `TokenKVPool` (paged_kv) — too narrow (3 callers).
//! - Anything from `super::ffi::*` — the FFI submodule tree is its own
//!   namespace; consumers that need `extern "C"` symbols use
//!   `cuda_kernels::ffi::xxx` directly.
//! - `EngineOptions` / `InferenceEngineOptions` — these are runtime
//!   configuration types owned by `infer::server_engine`, not kernel types.
//! - Any model-specific state (`Qwen3Model`, etc.) — those are application
//!   types and stay inside `infer::model::*`.

pub use super::paged_kv::PagedKVPool;
pub use super::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, RawDevicePtr};
pub use super::tilelang::TileLangDecodeMetadata;
