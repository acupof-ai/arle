//! Metal backend executor (Apple Silicon) — the primary AI-PC backend.
//!
//! Scope of this crate at R2:
//! - [`MetalKvPool`] is a **complete, real** host-side page manager: it implements
//!   the host-indexed [`infer_seam::KvPool`] seam (page allocation, slot
//!   growth/truncation, prefix-share retain/release). Because the seam is
//!   host-only, none of this needs a device tensor — it is identical in spirit to
//!   any backend's pool.
//! - [`MetalExecutor`] implements the [`infer_seam::BackendExecutor`] seam
//!   plumbing (submit/poll overlap shape). The actual MLX forward + on-device KV
//!   buffers are wired in R3 (model port via `crates/mlx-sys`); until then
//!   `submit` runs a clearly-marked **placeholder** forward so the seam is
//!   testable.
//!
//! Nothing here references engine-core; this crate depends only on the stable
//! `infer-plan` + `infer-seam` contracts.
//!
//! Internal layout (pure reorganization — same numerics):
//! - [`kv_pool`] — the host page manager [`MetalKvPool`] (feature-free).
//! - [`executor`] — the executor + session machine: [`MetalExecutor`] /
//!   `MetalInflight` placeholder seam (feature-free) plus the real MLX
//!   `RealMetalExecutor` / slot state / page store (`#[cfg(feature = "metal")]`).

// `MetalKvPool` and the `MetalExecutor` placeholder seam must compile WITHOUT
// the `metal` feature — they are exercised by default `infer-server`/agent-bench
// builds and by feature-free unit tests.
mod executor;
mod kv_pool;

#[cfg(feature = "metal")]
mod config;
#[cfg(feature = "metal")]
mod loader;
#[cfg(feature = "metal")]
mod mlx;
#[cfg(feature = "metal")]
mod model_source;
#[cfg(feature = "metal")]
mod qwen35;
#[cfg(feature = "metal")]
mod weights;
#[cfg(feature = "metal")]
mod wired_limit;

pub use executor::{MetalExecutor, MetalInflight};
pub use kv_pool::MetalKvPool;
