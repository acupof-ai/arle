#![allow(warnings)]
//! GPU memory planner: treats memory as a first-class scheduling resource.
//!
//! Tracks GPU memory budget, KV cache allocations, scratch buffer reservations,
//! and provides the scheduler with admission/eviction decisions based on
//! available memory rather than fixed slot counts.
//!
//! # Design (stub — implementation pending)
//!
//! The planner sits between the scheduler and the model:
//! ```text
//! Scheduler → MemoryPlanner.can_admit(prompt_len)? → admit/reject
//! Scheduler → MemoryPlanner.reserve_decode(slot)   → guaranteed memory
//! Scheduler → MemoryPlanner.release(slot)          → reclaim
//! ```
//!
//! Memory is divided into:
//! - **KV pool**: paged token-level KV cache (already exists in `paged_kv.rs`)
//! - **Scratch**: per-step decode/prefill buffers (currently model-owned)
//! - **Headroom**: reserved for CUDA runtime, cuBLAS workspace, fragmentation

/// Stub implementation of memory planner - TODO: implement fully
pub struct MemoryPlanner {
    _placeholder: (),
}

impl MemoryPlanner {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

/// Memory budget for a single GPU device.
pub struct MemoryBudget {
    /// Total GPU memory in bytes.
    pub total_bytes: usize,
    /// Memory reserved for model weights (immutable after loading).
    pub weights_bytes: usize,
    /// Memory reserved for CUDA runtime overhead.
    pub headroom_bytes: usize,
    /// Memory available for KV cache + scratch buffers.
    pub available_bytes: usize,
}

impl MemoryBudget {
    /// Compute budget from GPU info and model weight size.
    pub fn from_gpu(total_bytes: usize, weights_bytes: usize, headroom_bytes: usize) -> Self {
        let available = total_bytes.saturating_sub(weights_bytes + headroom_bytes);
        Self {
            total_bytes,
            weights_bytes,
            headroom_bytes,
            available_bytes: available,
        }
    }

    /// How many tokens of KV cache can fit in the available budget.
    pub fn max_kv_tokens(&self, bytes_per_token: usize) -> usize {
        if bytes_per_token == 0 {
            return 0;
        }
        self.available_bytes / bytes_per_token
    }
}
