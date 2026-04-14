// CUDA-only modules — excluded when `no-cuda` feature is active.
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod weight_loader;

// Always-available modules (pure Rust, no GPU dependency).
pub mod backend;
pub mod block_manager;
pub mod error;
pub mod events;
pub mod gguf;
pub mod hf_hub;
pub mod http_server;
pub mod kv_tier;
pub mod logging;
pub mod memory_planner;
pub mod metrics;
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
pub mod types;
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
