//! Vulkan backend host substrate for the AIPC lane (#71/#76/#77).
//!
//! P2 is a seam-correct skeleton: host GGUF/dequant/deepseek4-config
//! modules are re-exported from `infer-gguf`, the KV pool is backend-neutral
//! host bookkeeping, and dense Qwen3 forward order is pinned in
//! `model_qwen3`. Device execution stays feature-gated and pending the P1
//! shader ABI.

#[cfg(test)]
mod utils;

pub use infer_gguf::{deepseek4, dequant, gguf};

pub mod config;
pub mod executor;
#[cfg(feature = "vulkan")]
pub mod forward;
pub mod kv_pool;
pub mod loader;
pub mod model_qwen3;
pub mod model_qwen35;
pub mod model_qwen36;
#[cfg(feature = "vulkan")]
pub mod prefill;

pub use executor::{VulkanExecutor, VulkanInflight, load_qwen3_gguf};
pub use kv_pool::VulkanKvPool;
