//! Shared generation-state helpers used by all model implementations.
//!
//! `GenerationStateBase` bundles the KV cache, prefill logits, and CUDA graph
//! state that every model carries. Model-specific state structs embed it and
//! delegate the common `GenerationState` trait methods through it.

use anyhow::Result;

use super::cuda_graph::CudaGraphState;
use super::kv_cache::KVCache;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

/// Common generation-state fields shared across all model implementations.
///
/// Embed this in model-specific state structs (e.g. `Qwen3State`, `Qwen35State`)
/// and delegate `GenerationState` methods to it.
pub(crate) struct GenerationStateBase {
    pub kv_cache: KVCache,
    pub prefill_logits: Option<DeviceVec>,
    pub graph_state: CudaGraphState,
}

impl GenerationStateBase {
    pub(crate) fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        Self {
            kv_cache: KVCache::new(num_layers, num_kv_heads),
            prefill_logits: None,
            graph_state: CudaGraphState::new(),
        }
    }

    /// Return prefill logits if present, otherwise fall back to the provided
    /// decode-buffer logits.
    pub(crate) fn logits_or<'a>(&'a self, decode_logits: &'a DeviceVec) -> &'a DeviceVec {
        self.prefill_logits.as_ref().unwrap_or(decode_logits)
    }

    /// Reset KV cache, clear prefill logits, and invalidate CUDA graph.
    pub(crate) fn reset(&mut self) -> Result<()> {
        self.kv_cache.reset();
        self.prefill_logits = None;
        self.graph_state = CudaGraphState::new();
        Ok(())
    }

    /// Truncate KV cache to `len` tokens, clear prefill logits, and invalidate
    /// CUDA graph.
    pub(crate) fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.kv_cache.truncate_to(len);
        self.prefill_logits = None;
        self.graph_state = CudaGraphState::new();
        Ok(())
    }

    pub(crate) fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.kv_cache.set_max_gpu_seq_len(max_tokens);
    }

    pub(crate) fn set_max_seq_len(&mut self, max_seq: usize) {
        self.kv_cache.set_max_seq_len(max_seq);
    }

    pub(crate) fn set_kv_dtype(&mut self, dtype: super::kv_cache::KVCacheDtype) {
        self.kv_cache.set_dtype(dtype);
    }

    pub(crate) fn offload_kv_if_needed(&mut self, ctx: &DeviceContext) -> Result<()> {
        self.kv_cache.offload_if_needed(ctx)
    }

    pub(crate) fn migrate_kv_to_paged(
        &self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        use super::kv_cache::KVFormat;

        // Dispatch based on pool format, not contiguous cache dtype.
        // FP8 pool uses bf16 contiguous (quantizes during migration).
        match pool.format {
            KVFormat::BF16 => pool.migrate_from_contiguous(
                ctx,
                slot,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
            ),
            KVFormat::FP8E4M3 => pool.migrate_from_contiguous_fp8(
                ctx,
                slot,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
            ),
            KVFormat::INT8 => pool.migrate_from_contiguous_int8(
                ctx,
                slot,
                self.kv_cache.k_caches_q(),
                self.kv_cache.v_caches_q(),
                self.kv_cache.k_scales(),
                self.kv_cache.v_scales(),
                self.kv_cache.max_seq_len(),
            ),
            KVFormat::TurboQuant { .. } => pool.migrate_from_contiguous_turboquant_range(
                ctx,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
                0,
                pool.token_indices(slot),
            ),
        }
    }

    pub(crate) fn migrate_kv_range_to_paged(
        &self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        use super::kv_cache::KVFormat;

        match pool.format {
            KVFormat::BF16 => pool.migrate_from_contiguous_range(
                ctx,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
                start_pos,
                new_token_indices,
            ),
            KVFormat::FP8E4M3 => pool.migrate_from_contiguous_fp8_range(
                ctx,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
                start_pos,
                new_token_indices,
            ),
            KVFormat::INT8 => pool.migrate_from_contiguous_int8_range(
                ctx,
                self.kv_cache.k_caches_q(),
                self.kv_cache.v_caches_q(),
                self.kv_cache.k_scales(),
                self.kv_cache.v_scales(),
                self.kv_cache.max_seq_len(),
                start_pos,
                new_token_indices,
            ),
            KVFormat::TurboQuant { .. } => pool.migrate_from_contiguous_turboquant_range(
                ctx,
                self.kv_cache.k_caches(),
                self.kv_cache.v_caches(),
                self.kv_cache.max_seq_len(),
                start_pos,
                new_token_indices,
            ),
        }
    }
}
