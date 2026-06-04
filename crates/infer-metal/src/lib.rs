//! Metal backend executor (Apple Silicon) — the primary AI-PC backend.
//!
//! Implements the host-only [`infer_seam`] contracts: [`MetalKvPool`] is the
//! host page manager and [`MetalExecutor`] the submit/poll seam. The real MLX
//! Qwen3.5 forward + on-device KV live behind `#[cfg(feature = "metal")]`; the
//! feature-free path keeps a CPU placeholder so the seam is testable.

// `kv_pool` and the `MetalExecutor` placeholder seam stay feature-free so default
// `infer-server`/agent-bench builds and unit tests compile without `metal`.
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

#[cfg(feature = "metal")]
pub use executor::pipeline_fast_path_hits;
pub use executor::{MetalExecutor, MetalInflight};
pub use kv_pool::MetalKvPool;
#[cfg(feature = "metal")]
pub use model_source::resolve_model_path;
