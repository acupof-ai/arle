//! Vulkan backend host substrate for the AIPC lane (#71/#76/#77).
//!
//! P2 is a seam-correct skeleton: host GGUF/dequant/config modules are
//! re-exported from `infer-hip`, the KV pool is backend-neutral host
//! bookkeeping, and dense Qwen3 forward order is pinned in `model_qwen3`.
//! Device execution stays feature-gated and pending the P1 shader ABI.

pub use infer_hip::{config, dequant, gguf};

pub mod executor;
pub mod kv_pool;
pub mod model_qwen3;

pub use executor::{VulkanExecutor, VulkanInflight, load_qwen3_gguf};
pub use kv_pool::VulkanKvPool;
