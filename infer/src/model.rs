//! Model implementations: Qwen3 and Qwen3.5.

use anyhow::Result;
use rand::rngs::StdRng;

use crate::sampler::SamplingParams;
use infer_cuda_kernels::TokenKVPool;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

#[path = "model/common.rs"]
pub(crate) mod common;
#[path = "model/cuda_graph.rs"]
pub(crate) mod cuda_graph;
#[path = "model/generation_state.rs"]
pub(crate) mod generation_state;
#[path = "model/kv_cache.rs"]
pub(crate) mod kv_cache;

#[path = "model/glm4.rs"]
pub mod glm4;
#[path = "model/qwen3.rs"]
pub mod qwen3;
#[path = "model/qwen35.rs"]
pub mod qwen35;

pub use glm4::GLM4Model;
#[cfg(feature = "cuda")]
pub use glm4::GLM4State;
pub use kv_cache::{KVCacheDtype, KVFormat};
pub use qwen3::{ModelRuntimeConfig, Qwen3Model, Qwen3State};
pub use qwen35::{Qwen35Model, Qwen35State};

// ============================================================================
// DecodeContextOps trait — scheduler-level operations on decode buffers
// ============================================================================

/// Operations the scheduler can perform on a model's decode context,
/// independent of the model architecture.
///
/// This decouples scheduler-level work (H2D copies, FlashInfer metadata
/// management) from model-level computation, so new models don't need to
/// duplicate this boilerplate in their `decode_batch()` implementations.
pub trait DecodeContextOps {
    /// Upload token IDs from host to GPU. Called before `forward_decode_batch`.
    fn upload_token_ids(&mut self, ctx: &DeviceContext, tokens: &[u32]) -> Result<()>;

    /// Update FlashInfer paged KV metadata (positions, indptr, indices,
    /// last_page_len) for the given slots.
    ///
    /// Returns `true` if the kv_indices GPU buffer was reallocated (caller
    /// should invalidate any CUDA graph that captured the old pointer).
    fn update_metadata(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot_indices: &[usize],
    ) -> Result<bool>;

    /// Plan FlashInfer attention for the current batch.
    /// Must be called once per decode step after `update_metadata()`.
    /// `kv_format` dispatches between BF16/FP8 FlashInfer plans.
    fn plan_attention(
        &mut self,
        ctx: &DeviceContext,
        batch_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        page_size: usize,
        head_dim: usize,
        kv_format: kv_cache::KVFormat,
    ) -> Result<()>;

    /// Set the active batch size on all internal buffers (must be <= max_batch_size).
    fn set_batch_size(&mut self, bs: usize);

    /// Invalidate the CUDA graph cache entry for the given batch size.
    /// Called by the scheduler when metadata reallocation invalidates captured pointers.
    fn invalidate_graph_cache(&mut self, batch_size: usize);

    /// Access per-request logprobs computed by the last `sample_batch_greedy` call.
    fn logprobs_host(&self) -> &[f32] {
        &[]
    }
}

// ============================================================================
// ModelForward trait — shared by Qwen3 and Qwen3.5
// ============================================================================

/// Per-request mutable state. Separate from model weights for bs > 1 future.
pub trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
    /// Truncate KV cache to `len` tokens, keeping the first `len` tokens.
    fn truncate_to(&mut self, len: usize) -> Result<()>;
    /// Set the maximum contiguous sequence length for the KV cache.
    /// Must be called before the KV cache is first initialized.
    fn set_max_seq_len(&mut self, max_seq: usize);
    /// Set KV cache quantization dtype (BF16 or INT8).
    /// Must be called before the KV cache is first initialized.
    fn set_kv_dtype(&mut self, dtype: kv_cache::KVCacheDtype);

    /// Migrate KV data from contiguous cache to paged pool.
    /// Called after prefill completes, before first decode step.
    fn migrate_kv_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
    ) -> Result<()>;

    /// Migrate only the newly appended contiguous KV range into the paged pool.
    fn migrate_kv_range_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()>;

    // -- Prefix cache support for hybrid models (recurrent + full attention) --

    /// Whether this model supports partial prefix reuse via `truncate_to()`.
    ///
    /// Returns `false` for hybrid models (e.g. Qwen3.5) where accumulated
    /// recurrent state cannot be truncated to an arbitrary prefix length.
    /// The scheduler downgrades partial prefix hits to full MISS for such models.
    fn supports_partial_prefix(&self) -> bool {
        true
    }

    /// Save a snapshot of auxiliary state (recurrent/SSM) after prefill.
    ///
    /// Called by the scheduler after prefill completes successfully. On a
    /// subsequent full prefix hit, `restore_prefix_snapshot()` restores this
    /// clean post-prefill state, avoiding decode-token contamination.
    ///
    /// Default: no-op (pure-attention models have no auxiliary state).
    fn save_prefix_snapshot(&mut self) -> Result<()> {
        Ok(())
    }

    /// Restore auxiliary state from a previously saved snapshot.
    ///
    /// Returns `true` if a snapshot existed and was restored, `false` otherwise.
    /// Called on full prefix cache hit before transitioning to decode.
    fn restore_prefix_snapshot(&mut self) -> Result<bool> {
        Ok(false)
    }
}

/// Deep module interface: explicit prefill/decode phases with typed decode context.
///
/// Phase semantics:
/// - `forward_prefill`: process multiple tokens, populate KV cache
/// - `forward_decode`: process exactly one token, use existing KV cache
/// - `forward_decode_batch`: process B tokens from B requests in one pass
pub trait ModelForward: Send {
    type State: GenerationState + Send;

    /// Pre-allocated buffers for batched decode, owned by the scheduler.
    /// Replaces `Box<dyn Any + Send>` with compile-time type safety.
    ///
    /// Must implement `DecodeContextOps` so the scheduler can perform
    /// model-agnostic pre/post work (H2D copies, FlashInfer metadata).
    type DecodeContext: DecodeContextOps + Send;

    fn create_state(&self) -> Result<Self::State>;

    /// Create decode context for batched decode (lazy-init by scheduler).
    fn create_decode_context(
        &self,
        max_batch_size: usize,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext>;

    /// KV cache memory cost per token in bytes (across all layers, K+V, bf16).
    fn kv_cache_bytes_per_token(&self) -> usize;

    /// Number of KV cache layers (for paged KV pool sizing).
    /// For hybrid models, this is the number of full-attention layers only.
    fn num_kv_layers(&self) -> usize;

    /// Number of KV heads per layer (for paged KV pool sizing).
    fn num_kv_heads(&self) -> usize;

    /// Head dimension (for paged KV pool sizing).
    fn head_dim(&self) -> usize;

    /// Number of query attention heads (for FlashInfer plan scheduling).
    fn num_q_heads(&self) -> usize;

    /// Prefill: process multiple tokens, populate KV cache and produce logits.
    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;

    /// Decode: process exactly one token using existing KV cache.
    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()>;

    /// Convenience: dispatch to prefill or decode based on token count.
    /// Callers that know the phase should use `forward_prefill`/`forward_decode` directly.
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.forward_decode(tokens[0], state)
        } else {
            self.forward_prefill(tokens, state)
        }
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32>;

    /// Select token with logprob. Greedy-capable backends should override this
    /// to return the chosen token's log-probability without forcing callers to
    /// special-case batched vs. non-batched decode.
    fn select_token_with_logprob(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<(u32, Option<f32>)> {
        let token = self.select_token(state, params, rng)?;
        Ok((token, None))
    }

    fn is_stop_token(&self, token_id: u32) -> bool;
    fn device_context(&self) -> &DeviceContext;

    /// Batched sampling: launch all sampling kernels, sync once, readback all.
    /// Returns one token per request. Default falls back to sequential select_token.
    fn select_tokens_batch(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        params: &[&SamplingParams],
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(slot_indices.len());
        for (i, &si) in slot_indices.iter().enumerate() {
            tokens.push(self.select_token(&mut states[si], params[i], rng)?);
        }
        Ok(tokens)
    }

    /// Optional future prefill fast path that scatter-writes K/V to the token pool.
    ///
    /// When `prefill_uses_paged_pool()` returns true, the scheduler pre-allocates
    /// pool pages for the chunk BEFORE the forward call and routes prefill through
    /// this method instead of `forward_prefill()`. The implementation writes K/V
    /// directly into the paged pool via page-table indirection — no contiguous
    /// KV cache is touched and the scheduler skips `migrate_kv_range_to_paged`
    /// afterward.
    ///
    /// `new_token_indices` are the physical pool indices (on GPU) allocated for
    /// this chunk's tokens. The slice has length `tokens.len()`. Implementations
    /// that don't need per-token indices (e.g. paged-prefill variants that read
    /// the page table from the pool itself) may ignore it.
    fn forward_prefill_with_pool(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
        _pool: &TokenKVPool,
        _slot: usize,
        _new_token_indices: &cudarc::driver::CudaSlice<i32>,
    ) -> Result<()> {
        // Default: just call forward_prefill() (no pool write)
        self.forward_prefill(tokens, state)
    }

    /// Returns true when this model's `forward_prefill_with_pool` writes K/V
    /// directly to the paged pool. The scheduler uses this to:
    ///  - route prefill through `forward_prefill_with_pool` instead of
    ///    `forward_prefill`,
    ///  - pre-allocate pool pages BEFORE the forward call (so the forward can
    ///    write into them via page-table indirection),
    ///  - skip the post-forward `migrate_kv_range_to_paged` step,
    ///  - lift the `CONTIGUOUS_KV_TOKENS` chunk-size cap (the contiguous
    ///    scratch is not used by this model's prefill).
    ///
    /// Default: false (contiguous-KV + migrate path, still the majority).
    fn prefill_uses_paged_pool(&self) -> bool {
        false
    }

    /// Fast-path batched greedy sampling on internal contiguous logits.
    ///
    /// Implementations that return `Some(tokens)` should also populate
    /// `DecodeContextOps::logprobs_host()` for the same batch order so the
    /// scheduler/API can surface per-token logprobs without a second pass.
    /// Returns None if fast path unavailable (non-greedy, or model doesn't support it).
    fn sample_batch_greedy(
        &self,
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Launch batched greedy sampling kernels (argmax + logprob) without sync.
    /// GPU work is left in-flight. Call `sample_batch_greedy_readback()` after
    /// CPU overlap completes.
    fn sample_batch_greedy_launch(
        &self,
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Sync + readback after `sample_batch_greedy_launch()`.
    /// Must only be called after launch returned `true`.
    fn sample_batch_greedy_readback(
        &self,
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Prepare per-request sampling buffers when batched greedy sampling needs
    /// to fall back to `select_tokens_batch()`.
    ///
    /// Models that skip per-slot logits scatter on the fast greedy path should
    /// override this to materialize per-request logits before fallback.
    fn prepare_batch_sampling_fallback(
        &self,
        _states: &mut [Self::State],
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<()> {
        Ok(())
    }

    /// Batched decode: process B tokens from B requests in one forward pass.
    ///
    /// `tokens[b]` is decoded using `states[slot_indices[b]]`. Uses GEMM for
    /// linear projections (batched) and per-request attention.
    ///
    /// `paged_kv_pool` is provided when the scheduler owns a paged KV pool.
    /// Implementations may use it for paged attention in batched decode.
    ///
    /// Default implementation falls back to sequential `forward_decode()` calls.
    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
        _skip_logit_scatter: bool,
    ) -> Result<()> {
        for (i, &token) in tokens.iter().enumerate() {
            self.forward_decode(token, &mut states[slot_indices[i]])?;
        }
        Ok(())
    }

    /// Whether this model has a validated eager mixed decode+prefill path.
    fn supports_mixed_batch(&self) -> bool {
        false
    }

    /// Mixed-batch forward: B decode tokens + C prefill tokens in one eager forward pass.
    /// Returns `Ok(true)` if mixed forward was performed, `Ok(false)` if not supported.
    fn forward_mixed_batch(
        &self,
        _decode_tokens: &[u32],
        _prefill_tokens: &[u32],
        _states: &mut [Self::State],
        _decode_slot_indices: &[usize],
        _prefill_slot_idx: usize,
        _prefill_start_pos: usize,
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<bool> {
        Ok(false)
    }
}
