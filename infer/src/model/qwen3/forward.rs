use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers;
use super::weights::Qwen3Model;
use crate::model::generation_state::GenerationStateBase;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use infer_cuda_kernels::TokenKVPool;
use infer_cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    pub(super) ctx: DeviceContext,
    pub(crate) decode_bufs: DecodeBuffers,
    pub(crate) base: GenerationStateBase,
}

// SAFETY: `Qwen3State` contains CUDA resources (`DeviceContext`, `CudaSlice` inside
// `DecodeBuffers`, `GenerationStateBase` wrapping `KVCache`, `CudaGraphState`,
// `DeviceVec`) that hold raw CUDA device pointers.  These types are `!Send` by
// default because CUDA contexts and allocations must be accessed from the thread
// that created them.
//
// Invariant upheld: every `Qwen3State` instance is exclusively owned by its
// scheduler slot and only ever accessed from the single blocking inference
// thread that runs `Scheduler::run()`.  No other thread holds a reference to
// or borrows from this state while the inference thread is running.
//
// Violation would mean: concurrent access from multiple threads could cause
// data races on GPU memory or corrupt the CUDA driver state.
unsafe impl Send for Qwen3State {}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.base.logits_or(&self.decode_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.base.reset()
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.base.truncate_to(len)
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.base.set_max_seq_len(max_seq);
    }

    fn set_kv_dtype(&mut self, dtype: crate::model::kv_cache::KVCacheDtype) {
        self.base.set_kv_dtype(dtype);
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        self.base.migrate_kv_to_paged(ctx, pool, slot)
    }

    fn migrate_kv_range_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()> {
        self.base
            .migrate_kv_range_to_paged(ctx, pool, slot, start_pos, token_count)
    }
}

#[cfg(feature = "cuda")]
impl ModelForward for Qwen3Model {
    type State = Qwen3State;
    type DecodeContext = super::batch_decode::BatchDecodeBuffers;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
            ctx: self.ctx.clone(),
            decode_bufs: DecodeBuffers::new(&self.ctx, &self.config)?,
            base: GenerationStateBase::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
        })
    }

    fn create_decode_context(
        &self,
        max_batch_size: usize,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter_dim = self.config.intermediate_size;
        let max_pages = pool.max_total_pages;
        super::batch_decode::BatchDecodeBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            max_batch_size,
            num_heads,
            max_pages,
        )
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (bf16 = 2 bytes)
        2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }

    fn num_kv_layers(&self) -> usize {
        self.config.num_hidden_layers
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
        let start_pos = state.base.kv_cache.len();
        let hidden = self.get_embeddings_batch(tokens)?;
        let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.base.kv_cache)?;
        let logits = self.compute_logits_batch(&hidden)?;
        state.base.prefill_logits = Some(logits);
        Ok(())
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        self.decode_one_token(
            token,
            &mut state.base.kv_cache,
            &mut state.decode_bufs,
            &mut state.base.graph_state,
        )?;
        state.base.prefill_logits = None;
        Ok(())
    }

    fn forward_prefill_with_pool(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
        pool: &TokenKVPool,
        _slot: usize,
        new_token_indices: &cudarc::driver::CudaSlice<i32>,
    ) -> Result<()> {
        if tokens.len() == 1 {
            // Single-token: use standard decode path (CUDA Graph).
            // No pool scatter-write needed — decode path writes to pool via
            // decode_prep_paged kernel.
            self.forward_decode(tokens[0], state)?;
        } else {
            let start_pos = state.base.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            // Dual-write: writes K/V to both contiguous cache AND token pool.
            let hidden = self.process_all_layers_batch_with_pool(
                hidden,
                start_pos,
                &mut state.base.kv_cache,
                pool,
                new_token_indices,
            )?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.base.prefill_logits = Some(logits);
        }

        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state.base.logits_or(&state.decode_bufs.logits);
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            &mut state.decode_bufs.sample_probs,
            &mut state.decode_bufs.sample_out,
            params,
            random_val,
        )
    }

    fn select_token_with_logprob(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<(u32, Option<f32>)> {
        if params.is_greedy() {
            let logits = state.base.logits_or(&state.decode_bufs.logits);
            let (token, logprob) = ops::argmax_with_logprob(
                &self.ctx,
                logits,
                &mut state.decode_bufs.sample_out,
                &mut state.decode_bufs.sample_probs, // reuse as f32 scratch
            )?;
            Ok((token, Some(logprob)))
        } else {
            let token = self.select_token(state, params, rng)?;
            Ok((token, None))
        }
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }

    fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }

    fn select_tokens_batch(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        params: &[&crate::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<Vec<u32>> {
        let b = slot_indices.len();

        // Phase 1: Launch all sampling kernels using cached pointers
        for i in 0..b {
            let si = slot_indices[i];
            let random_val: f32 = rng.random();
            // When prefill_logits is set, fall back to the non-cached path
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
        // Use logprob variant — computes argmax + logprob in one kernel (negligible overhead)
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
        // Read back logprobs
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
            ops::extract_vec_into(&self.ctx, logits, b, &mut states[si].decode_bufs.logits)?;
            states[si].base.prefill_logits = None;
        }

        Ok(())
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        paged_kv_pool: Option<&mut PagedKVPool>,
        decode_ctx: &mut Self::DecodeContext,
        skip_logit_scatter: bool,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        // Always use the FlashInfer paged path when the pool is active, even for
        // batch_size=1. Routing B=1 through the contiguous/Triton decode path
        // causes greedy output divergence: (1) K/V is written only to the
        // contiguous cache, not the pool, so later batches read stale pool data;
        // (2) Triton and FlashInfer attention produce numerically different bf16
        // results, making greedy (argmax) output depend on batch composition.
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
}
