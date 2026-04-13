// CUDA-only modules — excluded when `no-cuda` feature is active.
#[cfg(feature = "cuda")]
pub mod bootstrap;
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(feature = "cuda")]
pub(crate) mod flashinfer_metadata;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod paged_kv;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod weight_loader;

// Always-available modules (pure Rust, no GPU dependency).
pub mod backend;
pub mod backend_runtime;
pub mod block_manager;
pub mod chat;
pub mod chat_protocol;
#[cfg(feature = "cpu")]
pub mod cpu_backend;
pub mod cuda_graph_pool;
pub mod error;
pub mod gguf;
pub mod hf_hub;
pub mod http_server;
pub mod logging;
pub mod memory_planner;
pub mod metal_backend;
pub mod metal_gdr;
pub mod metal_kv_pool;
pub mod metal_prefix_cache;
pub mod metal_scheduler;
pub mod metrics;
#[cfg(feature = "metal")]
pub mod mlx;
pub mod model_registry;
pub mod prefix_cache;
pub mod quant;
pub mod request_handle;
pub mod sampler;
pub mod scheduler;
pub mod server_engine;
pub mod speculative;
pub mod tensor_parallel;
pub mod tokenizer;
pub mod trace_reporter;

// Atomic workspace crates (phase-1 extraction).
pub use infer_core as core_types;
pub use infer_observability as observability;
pub use infer_policy as policy;

#[cfg(all(test, feature = "metal"))]
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub(crate) fn metal_test_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
