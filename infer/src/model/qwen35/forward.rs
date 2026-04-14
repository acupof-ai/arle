use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::Qwen35Model;
use crate::backend::cuda::tensor::{DeviceContext, DeviceVec};
use crate::model::generation_state::GenerationStateBase;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;

pub struct Qwen35State {
    pub(super) ctx: DeviceContext,
    pub(super) decode_bufs: DecodeBuffers35,
    pub(super) single_token_bufs: SingleTokenBuffers,
    pub(crate) base: GenerationStateBase,
    pub(super) recurrent_state: RecurrentState,
}

// SAFETY: `Qwen35State` contains CUDA resources (`DeviceContext`, `CudaSlice` inside
// `DecodeBuffers35`, `SingleTokenBuffers`, `GenerationStateBase` wrapping `KVCache`,
// `CudaGraphState`, `DeviceVec`, plus `RecurrentState`) that hold raw CUDA device
// pointers.  These types are `!Send` by default because CUDA contexts and
// allocations must be accessed from the thread that created them.
//
// Invariant upheld: every `Qwen35State` instance is exclusively owned by its
// scheduler slot and only ever accessed from the single blocking inference
// thread that runs `Scheduler::run()`.  No other thread holds a reference to
// or borrows from this state while the inference thread is running.
//
// Violation would mean: concurrent access from multiple threads could cause
// data races on GPU memory or corrupt the CUDA driver state.
unsafe impl Send for Qwen35State {}

impl GenerationState for Qwen35State {
    fn logits(&self) -> &DeviceVec {
        self.base.prefill_logits.as_ref().unwrap_or_else(|| {
            self.decode_bufs
                .current_logits(&self.single_token_bufs.logits)
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.base.reset()?;
        self.recurrent_state.reset(&self.ctx)?;
        Ok(())
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.base.truncate_to(len)?;
        // Recurrent state cannot be partially truncated — reset to zeros.
        // The scheduler should avoid partial prefix hits for hybrid models
        // (supports_partial_prefix() returns false).
        self.recurrent_state.reset(&self.ctx)?;
        Ok(())
    }

    fn supports_partial_prefix(&self) -> bool {
        false
    }

    fn save_prefix_snapshot(&mut self) -> Result<()> {
        self.recurrent_state.save_snapshot(&self.ctx)
    }

    fn restore_prefix_snapshot(&mut self) -> Result<bool> {
        self.recurrent_state.restore_snapshot(&self.ctx)
    }

    fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.base.set_max_gpu_kv(max_tokens);
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.base.set_max_seq_len(max_seq);
    }

    fn set_kv_dtype(&mut self, dtype: crate::model::kv_cache::KVCacheDtype) {
        self.base.set_kv_dtype(dtype);
    }

    fn offload_kv_if_needed(&mut self) -> Result<()> {
        self.base.offload_kv_if_needed(&self.ctx)
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &crate::backend::cuda::tensor::DeviceContext,
        pool: &crate::backend::cuda::paged_kv::PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        self.base.migrate_kv_to_paged(ctx, pool, slot)
    }

    fn migrate_kv_range_to_paged(
        &mut self,
        ctx: &crate::backend::cuda::tensor::DeviceContext,
        pool: &crate::backend::cuda::paged_kv::PagedKVPool,
        start_pos: usize,
        new_token_indices: &[u32],
    ) -> Result<()> {
        self.base
            .migrate_kv_range_to_paged(ctx, pool, start_pos, new_token_indices)
    }
}

#[cfg(feature = "cuda")]
impl ModelForward for Qwen35Model {
    type State = Qwen35State;
    type DecodeContext = super::batch_decode::BatchDecodeBuffers35;

    fn create_state(&self) -> Result<Self::State> {
        let single_token_bufs = SingleTokenBuffers::new(&self.ctx, &self.config)?;
        let decode_bufs = DecodeBuffers35::new(&self.ctx, &self.config, &single_token_bufs.logits)?;
        Ok(Qwen35State {
            ctx: self.ctx.clone(),
            decode_bufs,
            single_token_bufs,
            base: GenerationStateBase::new(
                self.config.num_full_attention_layers(),
                self.config.num_key_value_heads,
            ),
            recurrent_state: RecurrentState::new(&self.ctx, &self.config)?,
        })
    }

    fn create_decode_context(
        &self,
        max_batch_size: usize,
        pool: &crate::backend::cuda::paged_kv::PagedKVPool,
    ) -> Result<Self::DecodeContext> {
        use super::batch_decode::BatchDecodeBuffers35;
        let c = &self.config;
        let q_proj_dim = c.full_attn_q_proj_dim();
        let q_dim = c.full_attn_q_dim();
        let kv_dim = c.full_attn_kv_dim();
        let inter_dim = c.intermediate_size;
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let b_dim = c.linear_num_value_heads;
        let max_pages = pool.max_total_tokens;
        let num_linear_layers = c.num_hidden_layers - c.num_full_attention_layers();
        BatchDecodeBuffers35::new(
            &self.ctx,
            c.hidden_size,
            q_proj_dim,
            q_dim,
            kv_dim,
            inter_dim,
            qkv_dim,
            z_dim,
            b_dim,
            max_batch_size,
            c.num_attention_heads,
            max_pages,
            num_linear_layers,
        )
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // Only full-attention layers have KV cache (linear layers use recurrent state).
        // 2 (K+V) * num_full_attn_layers * num_kv_heads * head_dim * 2 (bf16 = 2 bytes)
        2 * self.config.num_full_attention_layers()
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }

    fn num_kv_layers(&self) -> usize {
        self.config.num_full_attention_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim
    }

    fn num_q_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        // Prefetch offloaded KV before prefill.
        if state.base.kv_cache.has_offloaded() {
            state.base.kv_cache.prefetch_to_gpu(&self.ctx)?;
        }

        let logits =
            self.prefill_forward(tokens, &mut state.base.kv_cache, &mut state.recurrent_state)?;
        state.base.prefill_logits = Some(logits);
        Ok(())
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        self.prefill_forward_single_token(
            token,
            &mut state.base.kv_cache,
            &mut state.recurrent_state,
            &mut state.single_token_bufs,
            &mut state.base.graph_state,
        )?;
        state
            .decode_bufs
            .bind_single_token_logits(&self.ctx, &state.single_token_bufs.logits);
        state.base.prefill_logits = None;
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        if let Some(logits) = state.base.prefill_logits.as_ref() {
            ops::gpu_sample_into(
                &self.ctx,
                logits,
                &mut state.decode_bufs.sample_probs,
                &mut state.decode_bufs.sample_out,
                params,
                random_val,
            )
        } else {
            let (logits, sample_probs, sample_out) = state
                .decode_bufs
                .current_logits_and_sampling_bufs(&state.single_token_bufs.logits);
            ops::gpu_sample_into(
                &self.ctx,
                logits,
                sample_probs,
                sample_out,
                params,
                random_val,
            )
        }
    }

    fn select_token_with_logprob(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<(u32, Option<f32>)> {
        if params.is_greedy() {
            let (token, logprob) = if let Some(logits) = state.base.prefill_logits.as_ref() {
                ops::argmax_with_logprob(
                    &self.ctx,
                    logits,
                    &mut state.decode_bufs.sample_out,
                    &mut state.decode_bufs.sample_probs,
                )?
            } else {
                let (logits, sample_probs, sample_out) = state
                    .decode_bufs
                    .current_logits_and_sampling_bufs(&state.single_token_bufs.logits);
                ops::argmax_with_logprob(&self.ctx, logits, sample_out, sample_probs)?
            };
            Ok((token, Some(logprob)))
        } else {
            let token = self.select_token(state, params, rng)?;
            Ok((token, None))
        }
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }

    fn device_context(&self) -> &crate::backend::cuda::tensor::DeviceContext {
        &self.ctx
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        paged_kv_pool: Option<&mut crate::backend::cuda::paged_kv::PagedKVPool>,
        decode_ctx: &mut Self::DecodeContext,
        skip_logit_scatter: bool,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        match paged_kv_pool {
            Some(pool) if pool.is_active() => self.decode_batch(
                tokens,
                states,
                slot_indices,
                skip_logit_scatter,
                pool,
                decode_ctx,
            ),
            _ => self.decode_batch_contiguous(tokens, states, slot_indices),
        }
    }

    fn sample_batch_greedy(
        &self,
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        let logits = match decode_ctx.logits_batch.as_ref() {
            Some(l) if l.seq_len > 0 => l,
            _ => return Ok(None),
        };
        let batch_size = slot_indices.len();
        crate::ops::argmax_batch_logprob_launch(
            &self.ctx,
            logits,
            &mut decode_ctx.argmax_out,
            &mut decode_ctx.logprobs_gpu,
            batch_size,
        )?;
        self.ctx.sync()?;
        crate::ops::argmax_batch_readback_into(
            &self.ctx,
            &decode_ctx.argmax_out,
            &mut decode_ctx.argmax_host,
            batch_size,
        )?;
        let lp_tmp = self
            .ctx
            .stream
            .clone_dtoh(&decode_ctx.logprobs_gpu)
            .map_err(|e| anyhow::anyhow!("D2H logprobs: {e}"))?;
        decode_ctx.logprobs_host[..batch_size].copy_from_slice(&lp_tmp[..batch_size]);
        Ok(Some(
            decode_ctx.argmax_host[..batch_size]
                .iter()
                .map(|&x| x as u32)
                .collect(),
        ))
    }

    fn prepare_batch_sampling_fallback(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<()> {
        let logits = match decode_ctx.logits_batch.as_ref() {
            Some(logits) if logits.seq_len >= slot_indices.len() => logits,
            _ => return Ok(()),
        };

        for (b, &si) in slot_indices.iter().enumerate() {
            ops::extract_vec_into(
                &self.ctx,
                logits,
                b,
                &mut states[si].decode_bufs.logits_scratch,
            )?;
            states[si].decode_bufs.bind_logits_scratch(&self.ctx);
            states[si].base.prefill_logits = None;
        }

        Ok(())
    }

    fn select_tokens_batch(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        params: &[&crate::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<Vec<u32>> {
        let b = slot_indices.len();

        // Phase 1: Launch all sampling kernels (no sync between requests)
        for i in 0..b {
            let si = slot_indices[i];
            let random_val: f32 = rng.random();
            if states[si].base.prefill_logits.is_some() {
                let logits = states[si].base.prefill_logits.as_ref().unwrap();
                crate::ops::gpu_sample_launch(
                    &self.ctx,
                    logits,
                    &mut states[si].decode_bufs.sample_probs,
                    &mut states[si].decode_bufs.sample_out,
                    params[i],
                    random_val,
                )?;
            } else {
                let ptrs = &states[si].decode_bufs.ptrs;
                crate::ops::gpu_sample_launch_raw(
                    &self.ctx,
                    ptrs.logits_ptr,
                    ptrs.logits_len,
                    ptrs.sample_probs_ptr,
                    ptrs.sample_out_ptr,
                    params[i],
                    random_val,
                )?;
            }
        }

        // Phase 2: Single sync
        self.ctx.sync()?;

        // Phase 3: Readback all results
        let mut tokens = Vec::with_capacity(b);
        for i in 0..b {
            let si = slot_indices[i];
            tokens.push(crate::ops::gpu_sample_readback(
                &self.ctx,
                &states[si].decode_bufs.sample_out,
            )?);
        }
        Ok(tokens)
    }
}
