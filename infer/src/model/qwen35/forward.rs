use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers35;
use super::prefill_buffers::PagedPrefillBuffers35;
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::Qwen35Model;
use crate::model::generation_state::GenerationStateBase;
use crate::model::{
    GenerationState, ModelForward, PrefillBatchRequest, SchedulerRuntimeWorkspaceBudget,
    decode_metadata_page_capacity, prepare_paged_prefill_batch,
};
use crate::ops;
use crate::sampler::SamplingParams;
use cuda_kernels::TokenKVPool;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};

pub struct Qwen35State {
    pub(super) ctx: DeviceContext,
    pub(super) decode_bufs: DecodeBuffers35,
    pub(super) single_token_bufs: SingleTokenBuffers,
    pub(super) paged_prefill: Option<PagedPrefillBuffers35>,
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

impl Qwen35State {
    fn prefill_logits_from_parts<'a>(
        base: &'a GenerationStateBase,
        paged_prefill: Option<&'a PagedPrefillBuffers35>,
    ) -> Option<&'a DeviceVec> {
        base.prefill_logits.as_ref().or_else(|| {
            paged_prefill
                .filter(|bufs| bufs.logits_valid)
                .map(|bufs| &bufs.logits)
        })
    }

    fn prefill_logits(&self) -> Option<&DeviceVec> {
        Self::prefill_logits_from_parts(&self.base, self.paged_prefill.as_ref())
    }

    fn clear_prefill_logits(&mut self) {
        self.base.prefill_logits = None;
        if let Some(bufs) = self.paged_prefill.as_mut() {
            bufs.clear_logits();
        }
    }

    fn drop_paged_prefill(&mut self) {
        self.clear_prefill_logits();
        self.paged_prefill = None;
    }

    fn ensure_paged_prefill(
        &mut self,
        ctx: &DeviceContext,
        config: &super::config::Config35,
        seq_len: usize,
        page_size: usize,
    ) -> Result<()> {
        let needs_realloc = self
            .paged_prefill
            .as_ref()
            .is_none_or(|bufs| !bufs.matches_shape(seq_len, page_size));
        if needs_realloc {
            self.paged_prefill = Some(PagedPrefillBuffers35::new(ctx, config, seq_len, page_size)?);
        }
        Ok(())
    }

    pub(super) fn prepare_paged_prefill(
        &mut self,
        ctx: &DeviceContext,
        config: &super::config::Config35,
        seq_len: usize,
        page_size: usize,
    ) -> Result<()> {
        self.clear_prefill_logits();
        self.ensure_paged_prefill(ctx, config, seq_len, page_size)
    }
}

impl GenerationState for Qwen35State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits().unwrap_or_else(|| {
            self.decode_bufs
                .current_logits(&self.single_token_bufs.logits)
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.base.reset()?;
        self.paged_prefill = None;
        self.recurrent_state.reset(&self.ctx)?;
        Ok(())
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.base.truncate_to(len)?;
        self.paged_prefill = None;
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
impl ModelForward for Qwen35Model {
    type State = Qwen35State;
    type DecodeContext = super::batch_decode::BatchDecodeBuffers35;
    type PrefillContext = ();

    fn create_state(&self) -> Result<Self::State> {
        let single_token_bufs = SingleTokenBuffers::new(&self.ctx, &self.config)?;
        let decode_bufs = DecodeBuffers35::new(&self.ctx, &self.config, &single_token_bufs.logits)?;
        Ok(Qwen35State {
            ctx: self.ctx.clone(),
            decode_bufs,
            single_token_bufs,
            paged_prefill: None,
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
        max_seq_len: Option<usize>,
        pool: &PagedKVPool,
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
        let max_pages = decode_metadata_page_capacity(
            max_batch_size,
            max_seq_len,
            pool.page_size,
            pool.max_total_pages,
        );
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

    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,
    ) -> Result<Self::PrefillContext> {
        Ok(())
    }

    fn scheduler_runtime_workspace_bytes(&self, budget: SchedulerRuntimeWorkspaceBudget) -> usize {
        let max_batch_size = budget.max_batch_size.max(1);
        let prefill_tokens = budget.prefill_tokens.max(1);
        let page_size = budget.kv_pool_format.default_page_size().max(1);
        let max_seq_len = budget.max_seq_len;
        let c = &self.config;
        let q_proj_dim = c.full_attn_q_proj_dim();
        let q_dim = c.full_attn_q_dim();
        let kv_dim = c.full_attn_kv_dim();
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let b_dim = c.linear_num_value_heads;
        let num_linear_layers = c.num_hidden_layers - c.num_full_attention_layers();
        let fallback_max_total_pages =
            max_batch_size.saturating_mul(prefill_tokens.div_ceil(page_size).max(1));
        let metadata_max_pages = decode_metadata_page_capacity(
            max_batch_size,
            max_seq_len,
            page_size,
            fallback_max_total_pages,
        );

        let bf16 = std::mem::size_of::<half::bf16>();
        let f32_bytes = std::mem::size_of::<f32>();
        let i32_bytes = std::mem::size_of::<i32>();
        let u64_bytes = std::mem::size_of::<u64>();
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;

        let decode_hidden_dims = 6usize
            .saturating_mul(hidden)
            .saturating_add(q_proj_dim)
            .saturating_add(4usize.saturating_mul(q_dim))
            .saturating_add(2usize.saturating_mul(kv_dim))
            .saturating_add(2usize.saturating_mul(qkv_dim))
            .saturating_add(3usize.saturating_mul(z_dim))
            .saturating_add(2usize.saturating_mul(b_dim))
            .saturating_add(3usize.saturating_mul(inter));
        let decode_context = decode_hidden_dims
            .saturating_mul(max_batch_size)
            .saturating_mul(bf16)
            .saturating_add(
                num_linear_layers
                    .saturating_mul(2)
                    .saturating_mul(max_batch_size)
                    .saturating_mul(u64_bytes),
            )
            .saturating_add(
                cuda_kernels::tilelang::TileLangDecodeMetadata::device_bytes(
                    max_batch_size,
                    metadata_max_pages,
                    c.num_attention_heads,
                ),
            )
            .saturating_add(max_batch_size.saturating_mul(4).saturating_mul(i32_bytes))
            .saturating_add(max_batch_size.saturating_mul(f32_bytes));

        let prefill_hidden_dims = 6usize
            .saturating_mul(hidden)
            .saturating_add(q_proj_dim)
            .saturating_add(2usize.saturating_mul(q_dim))
            .saturating_add(2usize.saturating_mul(kv_dim))
            .saturating_add(2usize.saturating_mul(qkv_dim))
            .saturating_add(3usize.saturating_mul(z_dim))
            .saturating_add(2usize.saturating_mul(b_dim))
            .saturating_add(3usize.saturating_mul(inter));
        let prefill_core = prefill_hidden_dims
            .saturating_mul(prefill_tokens)
            .saturating_mul(bf16)
            .saturating_add(3usize.saturating_mul(hidden).saturating_mul(bf16))
            .saturating_add(c.vocab_size.saturating_mul(bf16));

        let num_chunks = prefill_tokens
            .div_ceil(super::prefill_buffers::GdrChunkwiseScratch35::CHUNK_SIZE)
            .max(1);
        let gdr_scratch = 2usize
            .saturating_mul(prefill_tokens)
            .saturating_mul(b_dim)
            .saturating_mul(f32_bytes)
            .saturating_add(
                3usize
                    .saturating_mul(prefill_tokens)
                    .saturating_mul(z_dim)
                    .saturating_mul(bf16),
            )
            .saturating_add(
                prefill_tokens
                    .saturating_mul(b_dim)
                    .saturating_mul(super::prefill_buffers::GdrChunkwiseScratch35::CHUNK_SIZE)
                    .saturating_mul(f32_bytes + bf16),
            )
            .saturating_add(
                num_chunks
                    .saturating_mul(b_dim)
                    .saturating_mul(c.linear_value_head_dim)
                    .saturating_mul(c.linear_key_head_dim)
                    .saturating_mul(f32_bytes),
            );
        let prefill_metadata = prefill_tokens
            .saturating_mul(i32_bytes)
            .saturating_add(
                prefill_tokens
                    .div_ceil(page_size)
                    .max(1)
                    .saturating_mul(i32_bytes),
            )
            .saturating_add(
                (max_batch_size + 1)
                    .saturating_mul(2)
                    .saturating_mul(i32_bytes),
            )
            .saturating_add(max_batch_size.saturating_mul(2).saturating_mul(i32_bytes));
        let prefill_plan = cuda_kernels::tilelang::TileLangWorkspace::device_bytes(
            prefill_tokens.max(4096),
            c.num_attention_heads,
            cuda_kernels::tilelang::TileLangWorkspace::HD256_FLOAT_WORKSPACE_BYTES,
        );
        let prefill_workspace = prefill_core
            .saturating_add(gdr_scratch)
            .saturating_add(prefill_metadata)
            .saturating_add(prefill_plan);

        decode_context
            .saturating_add(prefill_workspace)
            .saturating_add(256 * 1024 * 1024)
    }

    fn max_concurrent_prefill_requests(&self) -> Option<usize> {
        Some(1)
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
        state.drop_paged_prefill();
        let logits =
            self.prefill_forward(tokens, &mut state.base.kv_cache, &mut state.recurrent_state)?;
        state.base.prefill_logits = Some(logits);
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
        state.prepare_paged_prefill(&self.ctx, &self.config, tokens.len(), pool.page_size)?;
        let recurrent = &mut state.recurrent_state;
        let prefill_bufs = state
            .paged_prefill
            .as_mut()
            .expect("paged prefill buffers initialized");
        self.prefill_forward_paged(tokens, pool, slot, recurrent, prefill_bufs)?;
        state.base.prefill_logits = Some(DeviceVec {
            data: self
                .ctx
                .stream
                .clone_dtod(&prefill_bufs.logits.data)
                .map_err(|e| anyhow::anyhow!("clone paged prefill logits D2D failed: {e}"))?,
            len: prefill_bufs.logits.len,
            label: "qwen35_paged_prefill_logits",
        });
        Ok(())
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

        for request in requests {
            states[request.slot_idx].drop_paged_prefill();
        }

        let paged_requests: Vec<super::prefill::Qwen35PagedPrefillRequest<'_>> = requests
            .iter()
            .map(|request| super::prefill::Qwen35PagedPrefillRequest {
                tokens: request.tokens,
                slot: request.slot_idx,
            })
            .collect();
        self.prefill_forward_paged_batch(&paged_requests, states, pool)?;

        Ok(true)
    }

    fn prefill_uses_paged_pool(&self) -> bool {
        true
    }

    fn supports_cross_slot_prefix_attach(&self) -> bool {
        false
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        state.drop_paged_prefill();
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
        state.clear_prefill_logits();
        Ok(())
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        self.enable_cuda_graph
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let Qwen35State {
            decode_bufs,
            single_token_bufs,
            paged_prefill,
            base,
            ..
        } = state;
        if let Some(logits) = Qwen35State::prefill_logits_from_parts(base, paged_prefill.as_ref()) {
            ops::gpu_sample_into(
                &self.ctx,
                logits,
                &mut decode_bufs.sample_probs,
                &mut decode_bufs.sample_out,
                params,
                random_val,
            )
        } else {
            let (logits, sample_probs, sample_out) =
                decode_bufs.current_logits_and_sampling_bufs(&single_token_bufs.logits);
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
        let Qwen35State {
            decode_bufs,
            single_token_bufs,
            paged_prefill,
            base,
            ..
        } = state;
        if params.is_greedy() {
            let (token, logprob) = if let Some(logits) =
                Qwen35State::prefill_logits_from_parts(base, paged_prefill.as_ref())
            {
                ops::argmax_with_logprob(
                    &self.ctx,
                    logits,
                    &mut decode_bufs.sample_out,
                    &mut decode_bufs.sample_probs,
                )?
            } else {
                let (logits, sample_probs, sample_out) =
                    decode_bufs.current_logits_and_sampling_bufs(&single_token_bufs.logits);
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

    fn device_context(&self) -> &DeviceContext {
        &self.ctx
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
        for &slot_idx in slot_indices {
            states[slot_idx].drop_paged_prefill();
        }
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
        crate::ops::argmax_batch_logprob_launch(
            &self.ctx,
            logits,
            &mut decode_ctx.argmax_out,
            &mut decode_ctx.logprobs_gpu,
            batch_size,
        )?;
        Ok(true)
    }

    fn sample_batch_greedy_readback(
        &self,
        slot_indices: &[usize],
        decode_ctx: &mut Self::DecodeContext,
    ) -> Result<Option<Vec<u32>>> {
        let batch_size = slot_indices.len();
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
            states[si].clear_prefill_logits();
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
            let state = &mut states[si];
            let Qwen35State {
                decode_bufs,
                paged_prefill,
                base,
                ..
            } = state;
            if let Some(logits) =
                Qwen35State::prefill_logits_from_parts(base, paged_prefill.as_ref())
            {
                crate::ops::gpu_sample_launch(
                    &self.ctx,
                    logits,
                    &mut decode_bufs.sample_probs,
                    &mut decode_bufs.sample_out,
                    params[i],
                    random_val,
                )?;
            } else {
                let ptrs = &decode_bufs.ptrs;
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
}
