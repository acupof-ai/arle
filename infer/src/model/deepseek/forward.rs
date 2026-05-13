//! `ModelForward` impl for the DeepSeek V4 scaffold.
//!
//! Phase 2A starts with a CUDA-backed, SW-only one-token decode smoke. It is
//! intentionally shape/finite only: real attention, MoE, and parity work remain
//! separate tranches.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use rand::{Rng, rngs::StdRng};

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
    PrefillBatchRequest, prepare_paged_prefill_batch,
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
            sample_probs: self
                .ctx
                .stream
                .alloc_zeros(self.config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc DeepSeek V4 sample_probs failed: {e}"))?,
            sample_out: self
                .ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc DeepSeek V4 sample_out failed: {e}"))?,
            reference_tokens: Vec::new(),
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

    fn forward_prefill_batch(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
    ) -> Result<()> {
        if let Some(pool) = paged_kv_pool
            && pool.is_active()
            && !prepare_paged_prefill_batch(self.device_context(), requests, pool)?
        {
            return Ok(());
        }
        self.prefill_batch_chunks(requests, states)
    }

    fn prefill_uses_paged_pool(&self) -> bool {
        true
    }

    fn supports_cross_slot_prefix_attach(&self) -> bool {
        false
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        self.validate_phase0_sw_decode_scope()?;
        ensure!(
            (token as usize) < self.config.vocab_size,
            "DeepSeek V4 token id {token} exceeds vocab_size {}",
            self.config.vocab_size
        );

        if let Some(logits) = self.compute_reference_logits_after_decode(token, state)? {
            state.decode_logits = logits;
            state.base.prefill_logits = None;
            state.base.kv_cache.advance_seq_len(1);
            return Ok(());
        }

        // Phase 2A.1 uses the loaded top-level tensors for non-zero logits when
        // available. Real contextual attention and shared-expert compute land
        // in later, separately gated tranches.
        if let Some(logits) = self.compute_gpu_logits_after_decode(token, state)? {
            state.decode_logits = logits;
        }
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
        rng: &mut StdRng,
    ) -> Result<u32> {
        ensure!(
            !params.has_penalties() && params.min_p <= 0.0,
            "DeepSeek V4 sampler supports greedy and temperature/top_k/top_p sampling; \
             penalties and min_p are not implemented yet"
        );
        let random_val: f32 = rng.random();
        let DeepseekState {
            base,
            decode_logits,
            sample_probs,
            sample_out,
            ..
        } = state;
        let logits = base.logits_or(decode_logits);
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            sample_probs,
            sample_out,
            params,
            random_val,
        )
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        // DeepSeek V4 generation stops on EOS; BOS is a valid emitted special
        // token and the CPU reference path intentionally does not stop on it.
        self.config.eos_token_id == Some(token_id)
    }

    fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    fn scheduler_c128_cache_layers(&self) -> usize {
        self.config
            .compress_ratios
            .iter()
            .copied()
            .filter(|&ratio| {
                self.config.spec.attention_mode_for_compress_ratio(ratio)
                    == deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed
            })
            .count()
            .max(1)
    }

    fn scheduler_c128_cache_head_dim(&self) -> usize {
        let c128_ratio = self
            .config
            .compress_ratios
            .iter()
            .copied()
            .filter(|&ratio| {
                self.config.spec.attention_mode_for_compress_ratio(ratio)
                    == deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed
            })
            .min()
            .unwrap_or(128)
            .max(1);
        self.config.head_dim.div_ceil(c128_ratio).max(1)
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
        self.scheduler_c128_cache_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn num_q_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn head_dim(&self) -> usize {
        self.scheduler_c128_cache_head_dim()
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // Scheduler-visible DSv4 cache profile:
        // - C128/HCA summaries stay hot in the GPU/host-visible TokenKVPool.
        // - C4/CSA entries are sparse and tiered through the offload path.
        // - SWA uses the 128-token local window and is not charged to the
        //   long-context pool.
        //
        // This keeps admission/page accounting aligned with DSv4's compact
        // cache shape instead of the generic expanded MHA K/V envelope.
        2 * self.scheduler_c128_cache_layers()
            * self.config.num_key_value_heads
            * self.scheduler_c128_cache_head_dim()
            * 2
    }
}
