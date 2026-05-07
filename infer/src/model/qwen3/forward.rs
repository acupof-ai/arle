use anyhow::Result;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::decode_buffers::DecodeBuffers;
use super::prefill::{Qwen3PagedPrefillRequest, Qwen3PrefillContext};
use super::weights::Qwen3Model;
use crate::model::generation_state::GenerationStateBase;
use crate::model::{
    GenerationState, MixedBatchFallbackReason, MixedBatchOutcome, MixedBatchRequest, ModelForward,
    PrefillBatchRequest, SchedulerRuntimeWorkspaceBudget, SparseKvDraftView, SpecVerifyOutput,
    SpecVerifyRequest, decode_metadata_page_capacity, prepare_paged_prefill_batch,
};
use crate::model_arch::ModelArchInfo;
use crate::model_registry::ModelArch;
use crate::ops::{self, OpsBackend};
use crate::sampler::SamplingParams;
use cuda_kernels::TokenKVPool;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
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
impl Qwen3Model {
    pub fn forward_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Qwen3State,
    ) -> Result<(Vec<u32>, DeviceVec)> {
        if tokens.len() == 1 && state.base.kv_cache.len() > 0 {
            self.forward_decode(tokens[0], state)?;
        } else {
            self.forward_prefill(tokens, state)?;
        }
        Ok((tokens.to_vec(), state.logits().clone()))
    }
}

#[cfg(feature = "cuda")]
impl ModelForward for Qwen3Model {
    type State = Qwen3State;
    type DecodeContext = super::batch_decode::BatchDecodeBuffers;
    type PrefillContext = Qwen3PrefillContext;

    fn forward_with_logits(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<u32>, DeviceVec)> {
        if tokens.len() == 1 && state.base.kv_cache.len() > 0 {
            self.forward_decode(tokens[0], state)?;
        } else {
            self.forward_prefill(tokens, state)?;
        }
        Ok((tokens.to_vec(), state.logits().clone()))
    }

    fn forward_sparse_decode_with_logits(
        &self,
        token: u32,
        states: &mut [Self::State],
        slot_idx: usize,
        pool: &mut PagedKVPool,
        decode_ctx: &mut Self::DecodeContext,
        sparse_view: SparseKvDraftView<'_>,
    ) -> Result<u32> {
        anyhow::ensure!(
            sparse_view.slot_idx == slot_idx,
            "sparse draft view slot mismatch: view={} slot={}",
            sparse_view.slot_idx,
            slot_idx
        );
        anyhow::ensure!(
            pool.is_active(),
            "sparse draft requires active paged KV pool"
        );

        let sparse_pages = sparse_decode_page_indices(pool, slot_idx, sparse_view)?;
        self.prepare_sparse_decode_context(token, slot_idx, &sparse_pages, pool, decode_ctx)?;
        let tokens = [token];
        self.decode_batch(&tokens, states, &[slot_idx], true, pool, decode_ctx)?;

        let logits = decode_ctx
            .logits_batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sparse draft decode did not produce logits"))?;
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);
        ops_backend.argmax_batch_logprob_launch(
            logits,
            &mut decode_ctx.argmax_out,
            &mut decode_ctx.logprobs_gpu,
            1,
        )?;
        self.ctx.sync()?;
        ops_backend.argmax_batch_readback_into(
            &decode_ctx.argmax_out,
            &mut decode_ctx.argmax_host,
            1,
        )?;
        Ok(decode_ctx.argmax_host[0] as u32)
    }

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
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
        max_seq_len: Option<usize>,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let inter_dim = self.config.intermediate_size;
        let max_pages = decode_metadata_page_capacity(
            max_batch_size,
            max_seq_len,
            pool.page_size,
            pool.max_total_pages,
        );
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

    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,
    ) -> Result<Self::PrefillContext> {
        Qwen3PrefillContext::new(&self.ctx)
    }

    fn scheduler_runtime_workspace_bytes(&self, budget: SchedulerRuntimeWorkspaceBudget) -> usize {
        let max_batch_size = budget.max_batch_size;
        let prefill_budget_tokens = budget.prefill_tokens.max(1);
        let mixed_prefill_tokens = budget.mixed_prefill_tokens.max(1);
        let max_seq_len = budget.max_seq_len;
        let kv_pool_format = budget.kv_pool_format;
        let num_heads = self.config.num_attention_heads;
        let q_dim = num_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let page_size = kv_pool_format.default_page_size().max(1);
        let fallback_max_total_pages = max_batch_size
            .max(1)
            .saturating_mul(prefill_budget_tokens.div_ceil(page_size).max(1));
        let metadata_max_pages = decode_metadata_page_capacity(
            max_batch_size,
            max_seq_len,
            page_size,
            fallback_max_total_pages,
        );
        let decode_context = super::batch_decode::BatchDecodeBuffers::device_bytes(
            self.config.hidden_size,
            q_dim,
            kv_dim,
            self.config.intermediate_size,
            max_batch_size,
            num_heads,
            metadata_max_pages,
        );
        let decode_logits = super::batch_decode::BatchDecodeBuffers::logits_device_bytes(
            self.config.vocab_size,
            max_batch_size,
        );
        let mixed_workspace = if self.supports_mixed_batch(kv_pool_format) {
            let mixed_total_tokens = max_batch_size.saturating_add(mixed_prefill_tokens);
            super::batch_decode::BatchDecodeBuffers::mixed_device_bytes(
                self.config.hidden_size,
                q_dim,
                kv_dim,
                self.config.intermediate_size,
                self.config.vocab_size,
                kv_pool_format,
                mixed_total_tokens.max(1),
                max_batch_size,
                num_heads,
                metadata_max_pages,
            )
        } else {
            0
        };

        let prefill_activation_dims = 4usize
            .saturating_mul(self.config.hidden_size)
            .saturating_add(2usize.saturating_mul(q_dim))
            .saturating_add(2usize.saturating_mul(kv_dim))
            .saturating_add(3usize.saturating_mul(self.config.intermediate_size));
        // Activation holds the SUM of all packed prefill rows in one step.
        // `step_mixed_launch` / `step_prefill_batch` build a single
        // `PrefillBuffers` whose row count = Σ per-row chunk sizes, capped
        // by `prefill_budget_tokens` (= `max_prefill_tokens`). Sizing for
        // just `chunked_prefill_size` would OOM under multi-row prefill
        // (codex review caught this on the original Fix 1 attempt).
        let prefill_activation = prefill_activation_dims
            .saturating_mul(prefill_budget_tokens)
            .saturating_mul(2);
        let prefill_plan = cuda_kernels::tilelang::TileLangWorkspace::default_device_bytes(
            prefill_budget_tokens.max(4096),
            num_heads,
        );
        let prefill_workspace = prefill_plan.saturating_add(prefill_activation);

        // Async prefill pending buffers and the lazy persistent mixed buffer
        // can coexist, so both must be subtracted before sizing the KV pool.
        decode_context
            .saturating_add(decode_logits)
            .saturating_add(prefill_workspace)
            .saturating_add(mixed_workspace)
            .saturating_add(128 * 1024 * 1024)
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
        slot: usize,
        _new_token_indices: &cudarc::driver::CudaSlice<i32>,
    ) -> Result<()> {
        let request = [Qwen3PagedPrefillRequest {
            tokens,
            slot,
            state_idx: 0,
        }];
        self.run_prefill_paged_batch_sync(&request, std::slice::from_mut(state), pool)
    }

    fn forward_prefill_batch_with_pool(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        pool: &mut PagedKVPool,
    ) -> Result<bool> {
        if !prepare_paged_prefill_batch(&self.ctx, requests, pool)? {
            return Ok(false);
        }

        let paged_requests: Vec<Qwen3PagedPrefillRequest<'_>> = requests
            .iter()
            .map(|request| Qwen3PagedPrefillRequest {
                tokens: request.tokens,
                slot: request.slot_idx,
                state_idx: request.slot_idx,
            })
            .collect();
        self.run_prefill_paged_batch_sync(&paged_requests, states, pool)?;
        Ok(true)
    }

    fn prefill_uses_paged_pool(&self) -> bool {
        true
    }

    fn supports_async_prefill_batch(&self) -> bool {
        true
    }

    fn launch_prefill_batch(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
        prefill_ctx: &mut Self::PrefillContext,
    ) -> Result<()> {
        match paged_kv_pool {
            Some(pool) if self.prefill_uses_paged_pool() && pool.is_active() => {
                if !prepare_paged_prefill_batch(&self.ctx, requests, pool)? {
                    return Ok(());
                }
                let paged_requests: Vec<Qwen3PagedPrefillRequest<'_>> = requests
                    .iter()
                    .map(|request| Qwen3PagedPrefillRequest {
                        tokens: request.tokens,
                        slot: request.slot_idx,
                        state_idx: request.slot_idx,
                    })
                    .collect();
                self.launch_prefill_paged_batch(&paged_requests, states, pool, prefill_ctx)
            }
            _ => self.forward_prefill_batch(requests, states, paged_kv_pool),
        }
    }

    fn complete_prefill_batch(
        &self,
        _states: &mut [Self::State],
        prefill_ctx: &mut Self::PrefillContext,
        slot_indices: &[usize],
    ) -> Result<bool> {
        prefill_ctx.complete(slot_indices)
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state.base.logits_or(&state.decode_bufs.logits);
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);
        ops_backend.sample_token_into(
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
            let ops_backend = ops::CudaOpsBackend::new(&self.ctx);
            let (token, logprob) = ops_backend.argmax_with_logprob(
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
        for &si in slot_indices {
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
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);
        ops_backend.argmax_batch_logprob_launch(
            logits,
            &mut decode_ctx.argmax_out,
            &mut decode_ctx.logprobs_gpu,
            batch_size,
        )?;
        self.ctx.sync()?;
        ops_backend.argmax_batch_readback_into(
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

    fn sample_batch_greedy_launch(
        &self,
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<bool> {
        let logits = match decode_ctx.logits_batch.as_ref() {
            Some(l) if l.seq_len > 0 => l,
            _ => return Ok(false),
        };
        let batch_size = slot_indices.len();
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);
        ops_backend.argmax_batch_logprob_launch(
            logits,
            &mut decode_ctx.argmax_out,
            &mut decode_ctx.logprobs_gpu,
            batch_size,
        )?;
        decode_ctx.stage_sampled_tokens_for_next_step(&self.ctx, slot_indices)?;
        Ok(true)
    }

    fn sample_batch_greedy_readback(
        &self,
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        let batch_size = slot_indices.len();
        decode_ctx.start_greedy_readback_async(&self.ctx, batch_size)?;
        decode_ctx.poll_greedy_readback(batch_size)
    }

    fn prepare_batch_sampling_fallback(
        &self,
        states: &mut [Self::State],
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<()> {
        decode_ctx.invalidate_sampled_token_handoff();
        let logits = match decode_ctx.logits_batch.as_ref() {
            Some(logits) if logits.seq_len >= slot_indices.len() => logits,
            _ => return Ok(()),
        };
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);

        for (b, &si) in slot_indices.iter().enumerate() {
            ops_backend.extract_vec_into(logits, b, &mut states[si].decode_bufs.logits)?;
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
        // Always use the TileLang paged path when the pool is active, even for
        // batch_size=1. Routing B=1 through the contiguous decode path
        // causes greedy output divergence: (1) K/V is written only to the
        // contiguous cache, not the pool, so later batches read stale pool data;
        // (2) contiguous and paged attention produce numerically different bf16
        // results, making greedy (argmax) output depend on batch composition.
        match paged_kv_pool {
            Some(pool) if pool.is_active() => {
                self.prepare_decode_context(tokens, slot_indices, pool, decode_ctx)?;
                self.decode_batch(
                    tokens,
                    states,
                    slot_indices,
                    skip_logit_scatter,
                    pool,
                    decode_ctx,
                )
            }
            _ => self.decode_batch_contiguous(tokens, states, slot_indices),
        }
    }

    fn supports_mixed_batch(&self, kv_pool_format: crate::model::kv_cache::KVFormat) -> bool {
        self.prefill_uses_paged_pool()
            && self.lora.is_none()
            && matches!(
                kv_pool_format,
                crate::model::kv_cache::KVFormat::BF16
                    | crate::model::kv_cache::KVFormat::FP8E4M3
                    | crate::model::kv_cache::KVFormat::INT8
            )
    }

    fn forward_mixed_batch(
        &self,
        batch: MixedBatchRequest<'_>,
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<MixedBatchOutcome> {
        match paged_kv_pool {
            Some(pool) if pool.is_active() => {
                self.decode_batch_with_prefill(batch, states, pool, decode_ctx)
            }
            _ => Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::InactivePagedPool,
            )),
        }
    }

    fn forward_spec_verify_batch(
        &self,
        requests: &[SpecVerifyRequest<'_>],
        states: &mut [Self::State],
        pool: &mut PagedKVPool,
    ) -> Result<Vec<SpecVerifyOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        for request in requests {
            anyhow::ensure!(
                request.input_tokens.len() == request.draft_tokens.len() + 1,
                "spec verifier input must be last-token + K draft tokens"
            );
        }

        let mut outputs: Vec<SpecVerifyOutput> = requests
            .iter()
            .map(|request| SpecVerifyOutput {
                slot_idx: request.slot_idx,
                target_argmax_tokens: Vec::with_capacity(request.input_tokens.len()),
            })
            .collect();
        let max_steps = requests
            .iter()
            .map(|request| request.input_tokens.len())
            .max()
            .unwrap_or(0);
        let mut decode_ctx = self.create_decode_context(requests.len(), None, pool)?;
        let greedy = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(0x5eec_dec0de);

        for step in 0..max_steps {
            let mut tokens = Vec::new();
            let mut slot_indices = Vec::new();
            let mut output_indices = Vec::new();
            for (idx, request) in requests.iter().enumerate() {
                let Some(&token) = request.input_tokens.get(step) else {
                    continue;
                };
                pool.cow_tail_page_for_append(&self.ctx, request.slot_idx)?;
                pool.alloc_tokens(request.slot_idx, 1)?;
                tokens.push(token);
                slot_indices.push(request.slot_idx);
                output_indices.push(idx);
            }
            self.forward_decode_batch(
                &tokens,
                states,
                &slot_indices,
                Some(pool),
                &mut decode_ctx,
                false,
            )?;
            for (idx, &slot_idx) in output_indices.iter().zip(&slot_indices) {
                let (token, _) =
                    self.select_token_with_logprob(&mut states[slot_idx], &greedy, &mut rng)?;
                outputs[*idx].target_argmax_tokens.push(token);
            }
        }

        Ok(requests
            .iter()
            .zip(outputs)
            .map(|(_request, output)| output)
            .collect())
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        // LoRA decode allocates per-call temp DeviceVecs inside
        // `apply_lora_{gemv,gemm}_add`; CUDA stream capture rejects those.
        // The LoRA-aware batched decode runs eagerly, so skip warmup.
        self.enable_cuda_graph && self.lora.is_none()
    }
}

#[cfg(feature = "cuda")]
impl ModelArchInfo for Qwen3Model {
    fn arch_kind(&self) -> ModelArch {
        ModelArch::Qwen3
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
        // 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (bf16 = 2 bytes)
        2 * self.config.num_hidden_layers
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 2
    }
}

fn sparse_decode_page_indices(
    pool: &PagedKVPool,
    slot_idx: usize,
    sparse_view: SparseKvDraftView<'_>,
) -> Result<Vec<u32>> {
    let slot_pages = pool.page_indices(slot_idx);
    anyhow::ensure!(
        !slot_pages.is_empty(),
        "sparse draft slot {slot_idx} has no KV pages"
    );

    let selected: std::collections::HashSet<u32> = sparse_view.page_ids.iter().copied().collect();
    Ok(sparse_decode_page_indices_from_slot(
        slot_pages,
        pool.page_size,
        &selected,
        sparse_view.active_recent_tokens,
    ))
}

fn sparse_decode_page_indices_from_slot(
    slot_pages: &[u32],
    page_size: usize,
    selected: &std::collections::HashSet<u32>,
    active_recent_tokens: usize,
) -> Vec<u32> {
    let recent_pages = active_recent_tokens.div_ceil(page_size.max(1));
    let recent_start = slot_pages.len().saturating_sub(recent_pages);
    let tail_page = *slot_pages
        .last()
        .expect("slot_pages checked non-empty above");

    let mut pages = Vec::with_capacity(slot_pages.len());
    for (logical_page_idx, &page) in slot_pages.iter().enumerate() {
        if selected.contains(&page) || logical_page_idx >= recent_start || page == tail_page {
            if pages.last().copied() != Some(page) && !pages.contains(&page) {
                pages.push(page);
            }
        }
    }
    if pages.last().copied() != Some(tail_page) {
        pages.push(tail_page);
    }
    pages
}

#[cfg(test)]
mod sparse_tests {
    use super::*;

    #[test]
    fn sparse_decode_pages_preserve_logical_order_and_tail() {
        let selected = [10, 40].into_iter().collect();

        let pages = sparse_decode_page_indices_from_slot(&[10, 20, 30, 40, 50], 16, &selected, 16);

        assert_eq!(pages, vec![10, 40, 50]);
    }

    #[test]
    fn sparse_decode_pages_reduce_full_context_pages() {
        let selected = [10, 20].into_iter().collect();

        let pages = sparse_decode_page_indices_from_slot(
            &[10, 20, 30, 40, 50, 60, 70, 80],
            16,
            &selected,
            32,
        );

        assert_eq!(pages, vec![10, 20, 70, 80]);
        assert!(pages.len() < 8);
    }
}
