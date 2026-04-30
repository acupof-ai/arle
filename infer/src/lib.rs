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
pub mod distributed;
pub mod error;
pub mod events;
pub mod gguf;
pub mod hf_hub;
pub mod http_server;
pub mod kv_tier;
pub mod logging;
pub mod metrics;
pub mod model_registry;
pub mod model_source;
pub mod prefix_cache;
pub mod quant;
#[cfg(any(feature = "cuda", feature = "metal"))]
#[path = "model/qwen35/gguf_host.rs"]
pub(crate) mod qwen35_gguf_host;
pub mod request_handle;
pub mod sampler;
pub mod scheduler;
pub mod server_engine;
pub mod session_persistence;
pub mod speculative;
pub mod tensor_parallel;
pub mod tokenizer;
pub mod trace_reporter;
pub mod types;
#[cfg(all(test, feature = "metal"))]
pub(crate) mod test_support {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    pub(crate) struct MetalTestGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl MetalTestGuard {
        fn clear_mlx_cache() {
            // MLX keeps process-global Metal allocator state. Clear it at
            // every test boundary so tiny unit tests do not inherit stale
            // buffers or command-buffer pressure from earlier cases.
            crate::backend::metal::mlx::clear_cache();
        }
    }

    impl Drop for MetalTestGuard {
        fn drop(&mut self) {
            Self::clear_mlx_cache();
        }
    }

    pub(crate) fn metal_test_guard() -> MetalTestGuard {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        MetalTestGuard::clear_mlx_cache();
        MetalTestGuard { _lock: lock }
    }
}
