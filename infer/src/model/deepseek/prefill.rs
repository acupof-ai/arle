//! DeepSeek V4 prefill scaffold.
//!
//! Phase 2A.1 keeps the same CUDA smoke contract as decode: validate the
//! scheduler-visible surface, expose device-resident next-token logits with the
//! expected vocab shape, and advance sequence state. Real V4 hybrid attention,
//! routed MoE, and full-context logits remain behind the Phase 2A kernel work.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};

#[cfg(feature = "cuda")]
use super::state::DeepseekState;
#[cfg(feature = "cuda")]
use super::weights::DeepseekModel;
#[cfg(feature = "cuda")]
use super::weights::dsv4_incremental_kv_enabled;
#[cfg(feature = "cuda")]
use crate::model::PrefillBatchRequest;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::DeviceVec;

/// Pre-allocated scratch for a batched prefill launch. Empty until the kernel
/// surface is real; the scheduler wires `Qwen3PrefillContext` for the same
/// purpose on the Qwen3 path.
///
/// Public so the `ModelForward::PrefillContext` associated type (a `pub`
/// surface on the trait) does not leak a private name.
#[cfg(feature = "cuda")]
pub struct DeepseekPrefillContext;

#[cfg(feature = "cuda")]
impl DeepseekPrefillContext {
    pub(super) fn new() -> Self {
        Self
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Run prefill for a single sequence chunk, then expose next-token logits
    /// via `state.base.prefill_logits`.
    ///
    /// The Phase 2A.1 DeepSeek path uses the scheduler's paged pool for long
    /// context accounting and does not write contiguous KV. Bound the request
    /// by the model context window instead of the small contiguous scratch
    /// allocation that paged-prefill models keep only for decode plumbing.
    pub(super) fn prefill_one(&self, tokens: &[u32], state: &mut DeepseekState) -> Result<()> {
        self.prefill_one_chunk(tokens, state, true)
    }

    pub(super) fn prefill_one_chunk(
        &self,
        tokens: &[u32],
        state: &mut DeepseekState,
        emit_logits: bool,
    ) -> Result<()> {
        self.validate_phase0_sw_decode_scope()?;
        ensure!(
            !tokens.is_empty(),
            "DeepSeek V4 prefill requires at least one token"
        );
        let max_seq_len = self.config.max_position_embeddings;
        ensure!(
            state.base.kv_cache.len().saturating_add(tokens.len()) <= max_seq_len,
            "DeepSeek V4 prefill would exceed max_seq_len: current={} incoming={} max={}",
            state.base.kv_cache.len(),
            tokens.len(),
            max_seq_len
        );
        for &token in tokens {
            ensure!(
                (token as usize) < self.config.vocab_size,
                "DeepSeek V4 token id {token} exceeds vocab_size {}",
                self.config.vocab_size
            );
        }

        if !emit_logits {
            // Non-final chunk of a chunked (>16384-token) prefill. When
            // incremental decode is enabled, the final chunk's logits path
            // (`compute_gpu_logits_after_prefill` →
            // `compute_top_level_logits_incremental`) asserts
            // `incremental.processed_tokens == kv_cache.len()` and reads the
            // per-layer SW/compressed/FP8 caches. Those invariants only hold if
            // EVERY prior chunk also ran the incremental forward: each chunk
            // must advance `processed_tokens` AND write its slice of KV at the
            // correct absolute `start_pos` (= prior chunks' length). Routing
            // only the final chunk through the incremental path (the 8f4db3b6
            // behavior) leaves intermediate chunks at `processed_tokens == 0`
            // with empty KV, so chunk-2 aborts with "incremental state length 0
            // does not match scheduler KV length 16384".
            //
            // `compute_top_level_logits_incremental(emit_logits=false)`
            // processes this chunk's tokens at `start_pos = processed_tokens`
            // (= current `kv_cache.len()`), writes the SW window / compressed /
            // FP8 caches decode reads, advances `processed_tokens`, and returns
            // `None` (no logits for intermediate chunks). RoPE positions and KV
            // offsets use the absolute `start_pos`, so the compressed/SW state
            // stays consistent chunk-to-chunk. Falls back to the stateless
            // accumulate-only path if the incremental path is unavailable
            // (weights not loaded → returns `None`); the final chunk then
            // recomputes the whole prompt statelessly, byte-identical to the
            // pre-8f4db3b6 chunked path.
            //
            // `compute_top_level_logits_incremental` does not touch
            // `reference_tokens`, so accumulate it here exactly as the stateless
            // path and the final-chunk path (`compute_gpu_logits_after_prefill`)
            // do. The incremental call runs BEFORE `advance_seq_len` so its
            // `start_pos == kv_cache.len()` precondition sees the prior chunks'
            // length, mirroring the final-chunk ordering (logits computed before
            // the seq-len advance at the bottom of this fn).
            state.reference_tokens.extend_from_slice(tokens);
            if dsv4_incremental_kv_enabled()? {
                self.compute_top_level_logits_incremental(tokens, state, false)?;
            }
            state.base.prefill_logits = None;
            state.base.kv_cache.advance_seq_len(tokens.len());
            return Ok(());
        }

        if let Some(logits) = self.compute_reference_logits_after_prefill(tokens, state)? {
            state.base.prefill_logits = Some(logits);
            state.base.kv_cache.advance_seq_len(tokens.len());
            return Ok(());
        }

        state.base.prefill_logits = Some(
            if let Some(logits) = self.compute_gpu_logits_after_prefill(tokens, state)? {
                logits
            } else {
                // `from_config` tests still build a shell without weights.
                // Keep that path scheduler-safe while `from_safetensors`
                // exercises the real top-level tensors.
                DeviceVec::zeros(&self.ctx, self.config.vocab_size)?
                    .with_label("dsv4_phase2a1_prefill_logits")
            },
        );
        state.base.kv_cache.advance_seq_len(tokens.len());
        Ok(())
    }

    pub(super) fn prefill_batch_chunks(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [DeepseekState],
    ) -> Result<()> {
        let state_count = states.len();
        for request in requests {
            let state = states.get_mut(request.slot_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 prefill slot {} out of range for {} states",
                    request.slot_idx,
                    state_count
                )
            })?;
            self.prefill_one_chunk(request.tokens, state, request.is_final_chunk())?;
        }
        Ok(())
    }
}
