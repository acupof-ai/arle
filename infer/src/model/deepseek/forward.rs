//! `ModelForward` impl for the DeepSeek V4 scaffold.
//!
//! Phase 2A starts with a CUDA-backed, SW-only one-token decode smoke. It is
//! intentionally shape/finite only: real attention, MoE, and parity work remain
//! separate tranches.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;

#[cfg(feature = "cuda")]
use super::batch_decode::DeepseekBatchDecodeBuffers;
#[cfg(feature = "cuda")]
use super::prefill::DeepseekPrefillContext;
#[cfg(feature = "cuda")]
use super::state::DeepseekState;
#[cfg(feature = "cuda")]
use super::weights::DeepseekModel;
#[cfg(feature = "cuda")]
use crate::model::generation_state::GenerationStateBase;
#[cfg(feature = "cuda")]
use crate::model::{
    GenerationState, MixedBatchFallbackReason, MixedBatchOutcome, MixedBatchRequest, ModelForward,
};
#[cfg(feature = "cuda")]
use crate::model_arch::ModelArchInfo;
#[cfg(feature = "cuda")]
use crate::model_registry::ModelArch;
#[cfg(feature = "cuda")]
use crate::ops;
#[cfg(feature = "cuda")]
use crate::sampler::SamplingParams;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, PagedKVPool};

#[cfg(feature = "cuda")]
impl ModelForward for DeepseekModel {
    type State = DeepseekState;
    type DecodeContext = DeepseekBatchDecodeBuffers;
    type PrefillContext = DeepseekPrefillContext;

    fn create_state(&self) -> Result<Self::State> {
        Ok(DeepseekState {
            base: GenerationStateBase::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
            decode_logits: cuda_kernels::prelude::DeviceVec::zeros(
                &self.ctx,
                self.config.vocab_size,
            )?
            .with_label("dsv4_phase2a0_decode_logits"),
        })
    }

    fn create_decode_context(
        &self,
        max_batch_size: usize,
        _max_seq_len: Option<usize>,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext> {
        DeepseekBatchDecodeBuffers::new(&self.ctx, max_batch_size, pool.max_total_pages)
    }

    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,
    ) -> Result<Self::PrefillContext> {
        Ok(DeepseekPrefillContext::new())
    }

    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        self.prefill_one(tokens, state)
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        self.validate_phase0_sw_decode_scope()?;
        ensure!(
            (token as usize) < self.config.vocab_size,
            "DeepSeek V4 token id {token} exceeds vocab_size {}",
            self.config.vocab_size
        );

        // Phase 2A.0 only licenses the CUDA decode surface: a device-resident
        // logits vector with the correct vocab length and finite values. The
        // buffer is allocated as zeros in `create_state`; real SW attention and
        // shared-expert compute land in later, separately gated tranches.
        state.base.prefill_logits = None;
        state.base.kv_cache.advance_seq_len(1);
        Ok(())
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
        _skip_logit_scatter: bool,
    ) -> Result<()> {
        ensure!(
            tokens.len() == slot_indices.len(),
            "DeepSeek V4 decode token/slot mismatch: tokens={} slots={}",
            tokens.len(),
            slot_indices.len()
        );
        if tokens.is_empty() {
            return Ok(());
        }
        ensure!(
            tokens.len() == 1,
            "DeepSeek V4 Phase 2A.0 supports only B=1 decode, got B={}",
            tokens.len()
        );
        let slot_idx = slot_indices[0];
        ensure!(
            slot_idx < states.len(),
            "DeepSeek V4 decode slot {slot_idx} out of range for {} states",
            states.len()
        );
        self.forward_decode(tokens[0], &mut states[slot_idx])
    }

    fn forward_mixed_batch(
        &self,
        _batch: MixedBatchRequest<'_>,
        _states: &mut [Self::State],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<MixedBatchOutcome> {
        // No mixed-batch support until the V4 prefill + decode kernels share a
        // single varlen launch path. Mirrors qwen3 default.
        Ok(MixedBatchOutcome::Fallback(
            MixedBatchFallbackReason::UnsupportedModel,
        ))
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Result<u32> {
        ensure!(
            params.is_greedy() && !params.has_penalties(),
            "DeepSeek V4 Phase 2A.0 only supports greedy sampling without penalties"
        );
        ops::argmax(&self.ctx, state.logits())
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        // Stop-token resolution mirrors `Qwen3Model::is_stop_token`: BOS / EOS
        // come from the spec config; downstream callers (REPL, HTTP) override
        // via per-request stop sequences.
        self.config.eos_token_id == Some(token_id) || self.config.bos_token_id == Some(token_id)
    }

    fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "cuda")]
impl ModelArchInfo for DeepseekModel {
    fn arch_kind(&self) -> ModelArch {
        ModelArch::DeepSeekV4
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn num_q_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // Phase 0.5 uses the conservative expanded single-KV-head BF16 budget.
        // Phase 2A may replace this once the V4 cache payload for
        // compressor/indexer streams is finalized.
        2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }
}
