//! Metal backend executor (Apple Silicon) — the primary AI-PC backend.
//!
//! Implements the Metal [`infer_seam::BackendExecutor`] contract. [`MetalKvPool`]
//! is a compatibility alias for the backend-neutral host page manager in
//! `infer-seam`; the real MLX Qwen3.5 forward + on-device KV live behind
//! `#[cfg(feature = "metal")]`. The feature-free path keeps a CPU placeholder
//! so the seam is testable.

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
mod resource;
#[cfg(feature = "metal")]
mod weights;
#[cfg(feature = "metal")]
mod wired_limit;

pub use executor::{MetalExecutor, MetalInflight, MetalKvCacheDtype};
#[cfg(feature = "metal")]
pub use executor::{
    default_t2_budget_bytes, paged_kv_read_fallbacks, paged_kv_read_hits, pipeline_fast_path_hits,
};
pub use kv_pool::MetalKvPool;
#[cfg(feature = "metal")]
pub use mlx::{
    AllocatorMemory, allocator_memory, clear_metal_cache, recommended_max_working_set_size_bytes,
};
#[cfg(feature = "metal")]
pub use model_source::resolve_model_path;
#[cfg(feature = "metal")]
pub use resource::{MetalResourcePlan, MetalResourceRequest, plan_resource_budget};
