//! Model implementations: Qwen3 and Qwen3.5.

use anyhow::Result;
use rand::rngs::StdRng;

use crate::sampler::SamplingParams;
use cuda_kernels::TokenKVPool;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

#[path = "model/common.rs"]
pub(crate) mod common;
#[path = "model/cuda_graph.rs"]
pub(crate) mod cuda_graph;
#[path = "model/generation_state.rs"]
pub(crate) mod generation_state;
#[path = "model/kv_cache.rs"]
pub(crate) mod kv_cache;
#[path = "model/layer_communicator.rs"]
pub mod layer_communicator;

#[path = "model/deepseek.rs"]
pub mod deepseek;
#[path = "model/qwen3.rs"]
pub mod qwen3;
#[path = "model/qwen35.rs"]
pub mod qwen35;

pub use kv_cache::{KVCacheDtype, KVFormat};
pub use qwen3::{ModelRuntimeConfig, Qwen3Model, Qwen3State};
pub use qwen35::{Qwen35Model, Qwen35RuntimeConfig, Qwen35State};

/// One request worth of prefill work inside a scheduler-planned prefill batch.
#[derive(Clone, Copy, Debug)]
pub struct PrefillBatchRequest<'a> {
    pub slot_idx: usize,
    pub tokens: &'a [u32],
}

/// One scheduler-planned mixed decode + packed-prefill batch.
///
/// `prefills[i]` and `prefill_start_positions[i]` are ordered together; model
/// implementations must treat the two slices as a single row table.
#[derive(Clone, Copy, Debug)]
pub struct MixedBatchRequest<'a> {
    pub decode_tokens: &'a [u32],
    pub decode_slot_indices: &'a [usize],
    pub prefills: &'a [PrefillBatchRequest<'a>],
    pub prefill_start_positions: &'a [usize],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MixedBatchFallbackReason {
    UnsupportedModel,
    InactivePagedPool,
    LoraEnabled,
    UnsupportedKvFormat,
    EmptyDecodeBatch,
    DecodeSlotCountMismatch,
    EmptyPrefillBatch,
    PrefillStartPositionCountMismatch,
    EmptyPrefillTokens,
    PrefillSlotInDecodeBatch,
    DuplicatePrefillSlot,
    PrefillSeqLenMismatch,
    SchedulerPreDispatchFallback,
}

impl MixedBatchFallbackReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedModel => "unsupported_model",
            Self::InactivePagedPool => "inactive_paged_pool",
            Self::LoraEnabled => "lora_enabled",
            Self::UnsupportedKvFormat => "unsupported_kv_format",
            Self::EmptyDecodeBatch => "empty_decode_batch",
            Self::DecodeSlotCountMismatch => "decode_slot_count_mismatch",
            Self::EmptyPrefillBatch => "empty_prefill_batch",
            Self::PrefillStartPositionCountMismatch => "prefill_start_position_count_mismatch",
            Self::EmptyPrefillTokens => "empty_prefill_tokens",
            Self::PrefillSlotInDecodeBatch => "prefill_slot_in_decode_batch",
            Self::DuplicatePrefillSlot => "duplicate_prefill_slot",
            Self::PrefillSeqLenMismatch => "prefill_seq_len_mismatch",
            Self::SchedulerPreDispatchFallback => "scheduler_pre_dispatch_fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedBatchOutcome {
    Executed,
    Fallback(MixedBatchFallbackReason),
}

/// One scheduler-planned speculative verifier row.
///
/// `input_tokens` is `[last_committed_token] + draft_tokens`. Logits row `i`
/// verifies `draft_tokens[i]`; the final row supplies the target bonus token.
#[derive(Clone, Copy, Debug)]
pub struct SpecVerifyRequest<'a> {
    pub slot_idx: usize,
    pub input_tokens: &'a [u32],
    pub draft_tokens: &'a [u32],
}

#[derive(Clone, Debug)]
pub struct SpecVerifyOutput {
    pub slot_idx: usize,
    pub target_argmax_tokens: Vec<u32>,
}

/// One sparse-KV draft attention view for MagicDec-style self speculation.
///
/// `page_ids` are physical paged-KV page IDs selected by the scheduler. Model
/// implementations must treat this as a draft-only approximation; verifier
/// paths continue to use the full per-slot KV page table.
#[derive(Clone, Copy, Debug)]
pub struct SparseKvDraftView<'a> {
    pub slot_idx: usize,
    pub page_ids: &'a [u32],
    pub active_recent_tokens: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerRuntimeWorkspaceBudget {
    pub max_batch_size: usize,
    pub prefill_tokens: usize,
    pub mixed_prefill_tokens: usize,
    pub max_seq_len: Option<usize>,
    pub kv_pool_format: kv_cache::KVFormat,
}

pub(crate) fn decode_metadata_page_capacity(
    max_batch_size: usize,
    max_seq_len: Option<usize>,
    page_size: usize,
    fallback_max_total_pages: usize,
) -> usize {
    max_seq_len.map_or(fallback_max_total_pages.max(1), |max_seq_len| {
        max_batch_size
            .max(1)
            .saturating_mul(max_seq_len.div_ceil(page_size.max(1)).max(1))
    })
}

pub(crate) fn prepare_paged_prefill_batch(
    ctx: &DeviceContext,
    requests: &[PrefillBatchRequest<'_>],
    pool: &mut PagedKVPool,
) -> Result<bool> {
    if requests.is_empty() {
        return Ok(false);
    }

    let mut seen_slots = Vec::with_capacity(requests.len());
    for request in requests {
        if request.tokens.is_empty() || seen_slots.contains(&request.slot_idx) {
            return Ok(false);
        }
        seen_slots.push(request.slot_idx);
    }

    let required_pages: usize = requests
        .iter()
        .map(|request| pool.append_pages_needed(request.slot_idx, request.tokens.len()))
        .sum();
    if required_pages > pool.free_page_count() {
        anyhow::bail!(
            "paged prefill batch needs {required_pages} free pages, only {} available",
            pool.free_page_count()
        );
    }

    for request in requests {
        pool.cow_tail_page_for_append(ctx, request.slot_idx)?;
        pool.alloc_tokens(request.slot_idx, request.tokens.len())?;
    }

    Ok(true)
}

// ============================================================================
// DecodeContextOps trait — scheduler-level operations on decode buffers
// ============================================================================

/// Operations the scheduler can perform on a model's decode context,
/// independent of the model architecture.
///
/// This decouples scheduler-level work (H2D copies, TileLang metadata
/// management) from model-level computation, so new models don't need to
/// duplicate this boilerplate in their `decode_batch()` implementations.
pub trait DecodeContextOps {
    /// Upload token IDs from host to GPU. Called before `forward_decode_batch`.
    fn upload_token_ids(&mut self, ctx: &DeviceContext, tokens: &[u32]) -> Result<()>;

    /// Update TileLang paged KV metadata (positions, indptr, indices,
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

    /// Mark TileLang attention metadata ready for the current batch.
    /// Must be called once per decode step after `update_metadata()`.
    /// TileLang attention is planless; quantized pools use custom decode kernels.
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

    /// Force the next decode call using this context to run eagerly.
    ///
    /// Speculative verification needs a bit-identical target path first; models
    /// with graph replay can opt out for one step while keeping graph capture
    /// enabled for the normal decode path.
    fn force_eager_once(&mut self) {}

    /// Drop model-side GPU sampled-token handoff state for one slot lifecycle
    /// boundary. Default contexts do not pipeline greedy token IDs on device.
    fn invalidate_sampled_token_handoff_for_slot(&mut self, _slot_idx: usize) {}

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
pub trait ModelForward: crate::model_arch::ModelArchInfo + Send {
    type State: GenerationState + Send;

    /// Pre-allocated buffers for batched decode, owned by the scheduler.
    /// Replaces `Box<dyn Any + Send>` with compile-time type safety.
    ///
    /// Must implement `DecodeContextOps` so the scheduler can perform
    /// model-agnostic pre/post work (H2D copies, TileLang metadata).
    type DecodeContext: DecodeContextOps + Send;
    /// Pre-allocated buffers for batched prefill that must outlive queued GPU
    /// work when the scheduler keeps a prefill batch pending across loop turns.
    ///
    /// Models that do not support async batched prefill use `()`.
    type PrefillContext: Send;

    fn create_state(&self) -> Result<Self::State>;

    /// Create decode context for batched decode (lazy-init by scheduler).
    fn create_decode_context(
        &self,
        max_batch_size: usize,
        max_seq_len: Option<usize>,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext>;

    /// Create prefill context for async batched prefill. The scheduler owns
    /// one context for the lifetime of the run, mirroring `DecodeContext`.
    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,
    ) -> Result<Self::PrefillContext>;

    /// Prefill: process multiple tokens, populate KV cache and produce logits.
    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;

    /// Decode: process exactly one token using existing KV cache.
    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()>;

    /// Forward and return the active logits buffer for verifier paths.
    ///
    /// The returned `DeviceVec` is an independent handle so callers can keep
    /// using it after the state advances. Phase 2 verifier code uses this as
    /// the target-model logits source before specialized batched gather kernels
    /// land.
    fn forward_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<u32>, DeviceVec)> {
        self.forward(tokens, state)?;
        Ok((tokens.to_vec(), state.logits().clone()))
    }

    /// Draft-only sparse-KV decode step.
    ///
    /// Implementations may attend to only `sparse_view` plus the active recent
    /// tail. This method is intentionally separate from the verifier API so
    /// sparse approximation cannot contaminate full-KV verification.
    fn forward_sparse_decode_with_logits(
        &self,
        _token: u32,
        _states: &mut [Self::State],
        _slot_idx: usize,
        _pool: &mut PagedKVPool,
        _decode_ctx: &mut Self::DecodeContext,
        _sparse_view: SparseKvDraftView<'_>,
    ) -> Result<u32> {
        anyhow::bail!("model does not support sparse-KV draft decode")
    }

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

    /// Batched prefill: process one or more requests in one scheduler step.
    ///
    /// Default implementation keeps a single semantic path by treating batch
    /// size 1 as the degenerate case and iterating over requests.
    fn forward_prefill_batch(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        match paged_kv_pool {
            Some(pool) if self.prefill_uses_paged_pool() && pool.is_active() => {
                let _ = self.forward_prefill_batch_with_pool(requests, states, pool)?;
            }
            _ => {
                for request in requests {
                    self.forward_prefill(request.tokens, &mut states[request.slot_idx])?;
                }
            }
        }

        Ok(())
    }

    /// Whether this model can keep a batched prefill launch pending across
    /// scheduler loop turns and complete it later via `complete_prefill_batch`.
    fn supports_async_prefill_batch(&self) -> bool {
        false
    }

    /// Launch a batched prefill without synchronizing the device.
    ///
    /// The default path keeps behavior correct by falling back to the
    /// synchronous `forward_prefill_batch`.
    fn launch_prefill_batch(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
        _prefill_ctx: &mut Self::PrefillContext,
    ) -> Result<()> {
        self.forward_prefill_batch(requests, states, paged_kv_pool)
    }

    /// Complete a previously launched async batched prefill if its device work
    /// is ready. Returns `false` when the batch is still in flight, leaving the
    /// context-owned temporary buffers alive for a later poll.
    fn complete_prefill_batch(
        &self,
        _states: &mut [Self::State],
        _prefill_ctx: &mut Self::PrefillContext,
        _slot_indices: &[usize],
    ) -> Result<bool> {
        Ok(true)
    }

    /// Batched paged prefill: process multiple prefill requests in one scheduler step.
    ///
    /// The default implementation keeps behavior correct by falling back to
    /// sequential per-request paged prefill. Models with a real batched paged
    /// prefill path should override this.
    fn forward_prefill_batch_with_pool(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        pool: &mut PagedKVPool,
    ) -> Result<bool> {
        if requests.is_empty() {
            return Ok(false);
        }

        if !prepare_paged_prefill_batch(self.device_context(), requests, pool)? {
            return Ok(false);
        }

        let dummy_indices = self
            .device_context()
            .stream
            .clone_htod(&[0i32])
            .map_err(|e| anyhow::anyhow!("dummy indices H2D failed: {e}"))?;
        for request in requests {
            self.forward_prefill_with_pool(
                request.tokens,
                &mut states[request.slot_idx],
                pool,
                request.slot_idx,
                &dummy_indices,
            )?;
        }
        Ok(true)
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

    /// Returns true when a model can resume a cached prefix on a fresh slot
    /// using only shared paged-KV pages plus the newly supplied suffix tokens.
    ///
    /// Pure-attention models can do this because the shared KV pages fully
    /// capture their prefix state. Hybrid models with auxiliary recurrent state
    /// must return `false` until they can restore or reconstruct that
    /// auxiliary state at the reused prefix length.
    fn supports_cross_slot_prefix_attach(&self) -> bool {
        self.prefill_uses_paged_pool()
    }

    /// GPU workspace the scheduler must reserve before sizing the KV pool.
    ///
    /// This covers model-owned runtime buffers that are allocated after weights
    /// load but before or during serving: decode context, persistent attention
    /// workspaces, logits buffers, and optional mixed prefill/decode scratch.
    /// Returning zero preserves the old behavior for models without a precise
    /// estimate.
    fn scheduler_runtime_workspace_bytes(&self, _budget: SchedulerRuntimeWorkspaceBudget) -> usize {
        0
    }

    /// Optional model-side cap for how many prefill requests may share one
    /// scheduler step. Hybrid models with large per-row prefill scratch can
    /// use this to keep c=16 decode batching without stacking prefill scratch
    /// beyond the reserved workspace.
    fn max_concurrent_prefill_requests(&self) -> Option<usize> {
        None
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
    ) -> Result<Option<usize>> {
        Ok(None)
    }

    /// Poll/read back after `sample_batch_greedy_launch()`.
    /// Must only be called after launch returned an async slot.
    fn sample_batch_greedy_readback(
        &self,
        _slot_indices: &[usize],
        _decode_ctx: &mut Self::DecodeContext,
        _async_slot_idx: Option<usize>,
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

    /// Whether this model has a validated mixed decode + prefill path for the
    /// current paged-KV format. Unsupported formats must return false so the
    /// scheduler uses the separate prefill + decode path instead.
    fn supports_mixed_batch(&self, _kv_pool_format: kv_cache::KVFormat) -> bool {
        false
    }

    /// Mixed-batch forward: decode rows plus packed prefill rows in a single
    /// scheduler-lowered execution unit.
    ///
    /// Returns `Executed` when the model consumed the mixed batch, or
    /// `Fallback(reason)` when the caller should use a non-mixed plan.
    fn forward_mixed_batch(
        &self,
        _batch: MixedBatchRequest<'_>,
        _states: &mut [Self::State],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<MixedBatchOutcome> {
        Ok(MixedBatchOutcome::Fallback(
            MixedBatchFallbackReason::UnsupportedModel,
        ))
    }

    /// Batched speculative verifier: append verifier input to paged KV and
    /// return target argmax tokens for each produced logits row.
    fn forward_spec_verify_batch(
        &self,
        _requests: &[SpecVerifyRequest<'_>],
        _states: &mut [Self::State],
        _pool: &mut PagedKVPool,
    ) -> Result<Vec<SpecVerifyOutput>> {
        anyhow::bail!("model does not support speculative verifier batch")
    }

    /// Whether batched decode for this model can be replayed via a captured
    /// CUDA Graph. Returns `false` when the model forces an eager decode
    /// path (e.g. LoRA adapters allocate per-call temps which stream
    /// capture rejects). Scheduler skips warmup/autotune in that case.
    fn supports_cuda_graph_decode(&self) -> bool {
        true
    }
}
