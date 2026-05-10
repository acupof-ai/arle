//! `ModelForward` impl for the DeepSeek V4 scaffold.
//!
//! Every method body is a `todo!()` stub today — the type plumbing is set up
//! so that when the V4 attention + MoE kernels land they can drop into the
//! existing scheduler interface unchanged.

#[cfg(feature = "cuda")]
use anyhow::Result;
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
use crate::model::{MixedBatchFallbackReason, MixedBatchOutcome, MixedBatchRequest, ModelForward};
#[cfg(feature = "cuda")]
use crate::model_arch::ModelArchInfo;
#[cfg(feature = "cuda")]
use crate::model_registry::ModelArch;
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

    fn forward_decode(&self, _token: u32, _state: &mut Self::State) -> Result<()> {
        todo!("DeepSeek V4 attention/MoE kernels — Phase 2A")
    }

    fn forward_decode_batch(
        &self,
        _tokens: &[u32],
        _states: &mut [Self::State],
        _slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
        _skip_logit_scatter: bool,
    ) -> Result<()> {
        todo!("DeepSeek V4 attention/MoE kernels — Phase 2A")
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
        _state: &mut Self::State,
        _params: &SamplingParams,
        _rng: &mut StdRng,
    ) -> Result<u32> {
        todo!("DeepSeek V4 logits/sampling path — Phase 2A")
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
