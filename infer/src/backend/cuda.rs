//! CUDA backend: kernel FFI, CUDA graph pool, FlashInfer metadata,
//! paged KV cache, device tensors, runtime bootstrap.
//!
//! Everything in this module — except [`graph_pool`] — is gated behind
//! `#[cfg(feature = "cuda")]` so that macOS / `no-cuda` builds skip
//! compilation entirely. [`graph_pool`] is always compiled because its
//! batch-size padding arithmetic, warmup schedule, and pool state
//! tracking are pure Rust and unit-testable on CPU; the GPU capture /
//! replay stubs inside it are individually cfg-gated.

#[cfg(feature = "cuda")]
pub mod bootstrap;
#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod flashinfer;
pub mod graph_pool;
#[cfg(feature = "cuda")]
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod tensor;
