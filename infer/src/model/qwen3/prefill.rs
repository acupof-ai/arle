use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::kv_cache::KVCache;
use crate::ops;
use cuda_kernels::TokenKVPool;
use cuda_kernels::flashinfer::BatchPrefillPagedPlan;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill in `process_all_layers_batch`, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    /// fp32 shadow of the residual stream. Maintained across layers so that
    /// per-layer bf16 outputs accumulate into fp32 without compounding
    /// ~0.4% bf16 rounding noise at each residual add. Norm reads from here
    /// directly to avoid a further bf16 round-trip on the hidden state.
    /// `None` unless `INFER_QWEN3_FP32_RESIDUAL=1` is set.
    residual_f32: Option<CudaSlice<f32>>,
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let residual_f32 = if std::env::var("INFER_QWEN3_FP32_RESIDUAL").is_ok() {
            Some(
                ctx.stream
                    .alloc_zeros::<f32>(hidden_dim * seq_len)
                    .map_err(|e| anyhow::anyhow!("alloc residual_f32: {e}"))?,
            )
        } else {
            None
        };
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            residual_f32,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

pub(super) struct Qwen3PagedPrefillRequest<'a> {
    pub tokens: &'a [u32],
    pub slot: usize,
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        crate::model::common::get_embeddings_batch(
            &self.ctx,
            &self.embed_tokens,
            token_ids,
            self.config.hidden_size,
        )
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    pub(super) fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Allocate all intermediates once — eliminates ~11k cuMemAllocAsync calls.
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        // If fp32 residual shadow is enabled, seed it from the bf16 embedding.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::cast_bf16_to_f32(&self.ctx, &hidden, r)?;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
            )?;
        }

        // If fp32 residual shadow was active, convert back to bf16 for the
        // final norm + LM head which still consume bf16.
        if let Some(ref r) = bufs.residual_f32 {
            ops::cast_f32_to_bf16(&self.ctx, r, &mut hidden)?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    pub(super) fn build_paged_prefill_sequences(
        &self,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        pool: &TokenKVPool,
    ) -> Result<(Vec<ops::PagedPrefillSequence>, Vec<i32>)> {
        anyhow::ensure!(
            !requests.is_empty(),
            "paged prefill batch requires at least one request"
        );

        let mut token_offset = 0usize;
        let mut page_table_offset = 0usize;
        let mut sequences = Vec::with_capacity(requests.len());
        let mut page_indices = Vec::new();

        for req in requests {
            let seq_len = req.tokens.len();
            anyhow::ensure!(
                seq_len > 0,
                "paged prefill request for slot {} must not be empty",
                req.slot
            );

            let pool_seq_len = pool.seq_len(req.slot);
            anyhow::ensure!(
                pool_seq_len >= seq_len,
                "paged prefill: pool seq_len {pool_seq_len} < chunk len {seq_len} for slot {}",
                req.slot
            );
            let start_pos = pool_seq_len - seq_len;
            let num_pages = (start_pos + seq_len).div_ceil(pool.page_size);
            let all_pages = pool.page_indices(req.slot);
            anyhow::ensure!(
                all_pages.len() >= num_pages,
                "paged prefill: slot {} has {} pages, expected at least {num_pages}",
                req.slot,
                all_pages.len()
            );

            page_indices.extend(all_pages[..num_pages].iter().map(|&page| page as i32));
            sequences.push(ops::PagedPrefillSequence {
                token_offset,
                seq_len,
                start_pos,
                page_table_offset,
                num_pages,
            });
            token_offset += seq_len;
            page_table_offset += num_pages;
        }

        Ok((sequences, page_indices))
    }

    fn compute_logits_batch_packed(
        &self,
        hidden: &HiddenStates,
        sequences: &[ops::PagedPrefillSequence],
    ) -> Result<Vec<DeviceVec>> {
        let mut logits = Vec::with_capacity(sequences.len());
        for seq in sequences {
            let last_token = seq.token_offset + seq.seq_len - 1;
            let last_hidden = ops::extract_vec(&self.ctx, hidden, last_token)?;
            let normed = ops::rms_norm(
                &self.ctx,
                &last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
            )?;
            logits.push(ops::linear(&self.ctx, &normed, self.output_projection())?);
        }
        Ok(logits)
    }

    #[fastrace::trace(name = "forward_prefill_paged_batch")]
    pub(super) fn forward_prefill_paged_batch(
        &self,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        pool: &TokenKVPool,
    ) -> Result<Vec<DeviceVec>> {
        let total_tokens = requests.iter().map(|req| req.tokens.len()).sum();
        let mut packed_tokens = Vec::with_capacity(total_tokens);
        for req in requests {
            packed_tokens.extend_from_slice(req.tokens);
        }

        let (sequences, page_indices) = self.build_paged_prefill_sequences(requests, pool)?;
        let page_indices_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&page_indices)
            .map_err(|e| anyhow::anyhow!("page_indices H2D failed: {e}"))?;
        let hidden = self.get_embeddings_batch(&packed_tokens)?;
        let hidden =
            self.process_all_layers_batch_paged(hidden, pool, &sequences, &page_indices_dev)?;
        self.compute_logits_batch_packed(&hidden, &sequences)
    }

    /// Paged-KV prefill over one or more packed requests. Writes K/V directly
    /// to the paged pool via page-table indirection and runs one FlashInfer
    /// `BatchPrefillWithPagedKVCache` call per layer over the packed varlen
    /// batch. No contiguous KV cache is touched; the scheduler must skip
    /// `migrate_kv_range_to_paged` for this forward.
    #[fastrace::trace(name = "process_all_layers_batch_paged")]
    pub(super) fn process_all_layers_batch_paged(
        &self,
        mut hidden: HiddenStates,
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        page_indices: &CudaSlice<i32>,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        anyhow::ensure!(
            !sequences.is_empty(),
            "paged prefill forward requires at least one sequence"
        );
        anyhow::ensure!(
            sequences
                .last()
                .map_or(0, |seq| seq.token_offset + seq.seq_len)
                == seq_len,
            "paged prefill token packing does not cover all hidden rows"
        );

        // Lazy-init the shared paged-prefill plan on first call and reuse
        // across all subsequent prefills. Allocating a fresh 264MB
        // FlashInferWorkspace per forward caused async-free backlog →
        // foreign C++ exception under concurrent load. Matches sglang's
        // single `workspace_buffer` pattern (`flashinfer_backend.py:260-292`).
        let mut plan_guard = self
            .paged_prefill_plan
            .lock()
            .map_err(|_| anyhow::anyhow!("paged_prefill_plan mutex poisoned"))?;
        let required_rows = seq_len.max(4096);
        let needs_realloc = plan_guard
            .as_ref()
            .map(|(max_rows, _)| *max_rows < required_rows)
            .unwrap_or(true);
        if needs_realloc {
            *plan_guard = Some((
                required_rows,
                BatchPrefillPagedPlan::new(&self.ctx, required_rows, num_heads)?,
            ));
        }
        let (_, plan) = plan_guard.as_mut().expect("just initialized");

        // Structural invariant: plan ONCE per forward, not per layer.
        // All 36 layers share the same packed (batch_size, qo_indptr, kv_indptr,
        // num_heads, page_size) shape, so one plan call covers them all.
        // Calling plan per layer overwrites `page_locked_workspace`
        // (host pinned) while a prior layer's `cudaMemcpyAsync` is still
        // queued on the compute stream — classic async-memcpy data race
        // that corrupts FlashInfer's `int_workspace` and poisons the
        // CUDA context under bench concurrency.
        let mut fwd = crate::ops::PagedPrefillForward::new_hd128(
            &self.ctx,
            plan,
            sequences,
            num_heads,
            num_kv_heads,
            pool.page_size,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_paged(
                layer_idx,
                layer,
                &mut hidden,
                &mut bufs,
                pool,
                sequences,
                page_indices,
                &mut fwd,
            )?;
        }

        // This forward owns both the shared FlashInfer plan workspace and
        // per-forward GPU metadata buffers (`page_indices`, qo/kv indptrs).
        // Keep them alive until the compute stream drains; otherwise the next
        // chunk can re-plan into the same host/device workspace while kernels
        // from the prior chunk still consume it.
        self.ctx.sync()?;

        Ok(hidden)
    }

    /// Paged-KV variant of `forward_layer_batch`. Differences vs the contiguous
    /// path:
    ///  - No `kv_cache.init_if_needed` / `prepare_layer` / `commit_layer`.
    ///  - Attention call writes K/V directly into the paged pool through the
    ///    page-table indirection kernel + FlashInfer paged-prefill.
    ///  - No `scatter_write_kv` dual-write step.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut PrefillBuffers,
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        page_indices: &CudaSlice<i32>,
        fwd: &mut crate::ops::PagedPrefillForward,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 2. QKV projections
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }

        // 3. Paged-KV attention: QK norm + RoPE + paged K/V write (page-table
        //    indirection) + FlashInfer BatchPrefillWithPagedKVCache.
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        let meta = ops::PagedPrefillMeta {
            pool,
            layer_idx,
            page_indices,
            sequences,
            page_size: pool.page_size,
        };
        ops::prefill_attention_paged_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            &meta,
            fwd,
            &mut bufs.attn_output,
            &heads,
        )?;

        // 4-8: Same as forward_layer_batch (O proj, residual, MLP)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.gate_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.normed,
                    &mut bufs.gate_out,
                )?;
            }
            if let Some(ad) = ll.up_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.up_out)?;
            }
        }
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.act_out, &mut bufs.o_buf)?;
            }
        }

        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }

    pub(super) fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        crate::model::common::compute_logits_batch(
            &self.ctx,
            hidden,
            &self.norm,
            self.output_projection(),
            self.config.rms_norm_eps,
            false, // standard RMSNorm (not offset)
        )
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm → bufs.normed. When the fp32 residual shadow is active,
        //    read from it directly — skipping the bf16 rounding in `hidden`.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.input_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            );
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.normed,
            &format!("L{layer_idx} after-input-norm"),
            self.config.hidden_size,
        );

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q_proj_out (pre-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.k_batch,
            &format!("L{layer_idx} k_proj_out (pre-norm-rope)"),
            bufs.k_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.v_batch,
            &format!("L{layer_idx} v_proj_out"),
            bufs.v_batch.hidden_dim,
        );

        // 3. FlashAttention-2 (Triton) → bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            &heads,
            start_pos,
        )?;
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q (post-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.attn_output,
            &format!("L{layer_idx} attn_output (pre-o-proj)"),
            bufs.attn_output.hidden_dim,
        );
        // Quantize newly written KV tokens → INT8 storage (no-op for BF16)
        kv_cache.commit_layer(&self.ctx, layer_idx, start_pos, hidden.seq_len)?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.o_buf,
            &format!("L{layer_idx} o_proj_out"),
            bufs.o_buf.hidden_dim,
        );

        // 5. Residual add: hidden + o_buf.
        //    With fp32 shadow: accumulate into residual_f32 (fp32 precision),
        //    then sync hidden for downstream bf16 consumers / debug dumps.
        //    Without shadow: use the classic bf16 add + swap path.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        // 6. MLP RMSNorm → bufs.normed.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.post_attention_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops::rms_norm_batch_into(
                &self.ctx,
                hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            );
        }

        // 7. MLP: gate + up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.gate_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.normed,
                    &mut bufs.gate_out,
                )?;
            }
            if let Some(ad) = ll.up_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.up_out)?;
            }
        }
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.act_out, &mut bufs.o_buf)?;
            }
        }

        // 8. Residual add: attn_residual + mlp_out.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }
}
