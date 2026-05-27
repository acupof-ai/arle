use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::batch_decode::BatchDecodeBuffers35;
use super::forward::Qwen35State;
use super::prefill_buffers::{GdrChunkwiseScratch35, PagedPrefillBuffers35};
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::{KVCache, KVFormat};
use crate::model::{
    MixedBatchFallbackReason, MixedBatchOutcome, MixedBatchRequest, PrefillBatchRequest,
};
use crate::ops;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::{TokenKVPool, ffi, kv_quant};

fn qwen35_cuda_debug_sync_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_CUDA_DEBUG_SYNC")
            .and_then(|value| value.into_string().ok())
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                !value.is_empty() && value != "0" && value != "false" && value != "off"
            })
            .unwrap_or(false)
    })
}

fn qwen35_cuda_debug_sync(ctx: &DeviceContext, label: impl AsRef<str>) -> Result<()> {
    if qwen35_cuda_debug_sync_enabled() {
        let label = label.as_ref();
        ctx.sync()
            .with_context(|| format!("CUDA debug sync after {label}"))?;
    }
    Ok(())
}

fn mixed_prefill_seq_indptr(prefills: &[PrefillBatchRequest<'_>]) -> Result<Vec<i32>> {
    let mut indptr = Vec::with_capacity(prefills.len() + 1);
    let mut offset = 0usize;
    indptr.push(0);
    for prefill in prefills {
        ensure!(
            !prefill.tokens.is_empty(),
            "qwen35 mixed prefill row must not be empty"
        );
        offset += prefill.tokens.len();
        indptr.push(offset as i32);
    }
    Ok(indptr)
}

pub(super) struct Qwen35PagedPrefillRequest<'a> {
    pub tokens: &'a [u32],
    pub slot: usize,
}

impl Qwen35Model {
    pub(super) fn prefill_forward(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "prefill_forward requires at least one token");
        let c = &self.config;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;

        // Get embeddings for all tokens
        let mut hidden_batch = crate::model::common::get_embeddings_batch(
            &self.ctx,
            &self.embed_tokens,
            token_ids,
            c.hidden_size,
        )?;

        // Process layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdr_chunkwise_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv_cache,
                recurrent,
            )?;
        }

        // All layers processed. Advance seq_len counters once for the entire prefill.
        kv_cache.advance_seq_len(seq_len);
        recurrent.seq_len += seq_len;

        // Final norm (1+weight offset) + LM head
        crate::model::common::compute_logits_batch(
            &self.ctx,
            &hidden_batch,
            &self.norm,
            self.output_projection(),
            c.rms_norm_eps,
            true, // offset RMSNorm (1+weight)
        )
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_layer(
        &self,
        _layer_idx: usize,
        layer: &TransformerBlock35,
        hidden_batch: &HiddenStates,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let seq_len = hidden_batch.seq_len;

        // 1. Input layernorm — per-token (no batched offset norm kernel yet)
        // Use standard batched norm and add the offset correction manually
        // Actually we need the (1+w) variant. Process token by token for now.
        let mut normed_batch =
            self.batched_rms_norm_offset(hidden_batch, &layer.input_layernorm, eps)?;

        // 2. Attention / Linear attention — per-token for correctness
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => c.full_attn_q_dim(),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let mut attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &normed_batch,
                full_idx,
                kv_cache,
                attn_out_dim,
                seq_len,
            )?,
            LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                attn,
                &normed_batch,
                linear_idx,
                recurrent,
                gdr_chunkwise_scratch,
                seq_len,
            )?,
        };
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut attn_results)?;

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed_batch)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed_batch)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mut mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut mlp_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    pub(super) fn prefill_full_attention(
        &self,
        attn: &FullAttentionLayer,
        normed_batch: &HiddenStates,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        _attn_out_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let attn_out_dim = c.full_attn_q_dim();
        let eps = c.rms_norm_eps;

        let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
        let k_batch = ops::gemm(&self.ctx, &attn.k_proj, normed_batch)?;
        let v_batch = ops::gemm(&self.ctx, &attn.v_proj, normed_batch)?;
        let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

        let base_pos = kv_cache.len();
        let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, *full_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &attn.q_norm,
            k_norm: &attn.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: eps,
        };
        // `prefill_attention_hd256_batch` takes q_full_batch with per-head
        // concat layout [q|g|q|g|...], extracts Q internally, runs attention,
        // and applies sigmoid(gate) — all in fused kernels.
        ops::prefill_attention_hd256_batch(
            &self.ctx,
            &q_full_batch,
            &k_batch,
            &v_batch,
            &nrp,
            kc,
            vc,
            &mut attn_out_batch,
            c.num_attention_heads,
            c.num_key_value_heads,
            base_pos,
            c.rotary_dim,
        )?;

        *full_idx += 1;

        // O projection (batched)
        ops::gemm(&self.ctx, &attn.o_proj, &attn_out_batch)
    }

    pub(super) fn prefill_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        normed_batch: &HiddenStates,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;

        // Batch projections
        let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, normed_batch)?;
        let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, normed_batch)?;
        let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, normed_batch)?;
        let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, normed_batch)?;

        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let layer_state = &mut recurrent.layers[*linear_idx];

        let mut qkv_conv_batch = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &qkv_batch,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut qkv_conv_batch,
            c.linear_conv_kernel_dim,
        );

        let mut gdr_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::gated_delta_rule_prefill_chunkwise_into(
            &self.ctx,
            &qkv_conv_batch,
            &b_batch,
            &a_batch,
            &ops::GdrWeights {
                dt_bias: &attn.dt_bias,
                a_log: &attn.a_log,
            },
            &mut layer_state.state,
            gdr_chunkwise_scratch,
            &mut gdr_out_batch,
            &ops::GdrHeadConfig {
                num_key_heads: c.linear_num_key_heads,
                num_value_heads: c.linear_num_value_heads,
                key_dim: c.linear_key_head_dim,
                val_dim: c.linear_value_head_dim,
            },
        )?;

        let mut normed_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &gdr_out_batch,
            &attn.norm_weight,
            &z_batch,
            &mut normed_out_batch,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );

        *linear_idx += 1;

        // Output projection (batched)
        ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)
    }

    /// Paged-KV prefill for Qwen3.5. Full-attn layers (8 of 32) write K/V
    /// directly to the paged pool via page-table indirection and run
    /// TileLang paged prefill HD256. Linear-attn layers
    /// (24 of 32) are unchanged — they use the recurrent state, which is
    /// independent of the KV pool.
    ///
    /// Callable only when the scheduler has pre-allocated pool pages for
    /// this chunk (pool.seq_len(slot) already covers `[0, start_pos+seq_len)`).
    pub(super) fn prefill_forward_paged(
        &self,
        token_ids: &[u32],
        pool: &TokenKVPool,
        slot: usize,
        recurrent: &mut RecurrentState,
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "prefill_forward_paged requires ≥1 token");
        anyhow::ensure!(
            bufs.matches_shape(seq_len, pool.page_size),
            "paged prefill buffers expect seq_len={} page_size={}, got seq_len={} page_size={}",
            bufs.seq_len,
            bufs.page_size,
            seq_len,
            pool.page_size
        );

        self.prepare_paged_prefill(token_ids, pool, slot, bufs)?;
        bufs.clear_logits();

        // Replay is shape-based on the canonical paged-prefill path. Metadata,
        // including device-backed `start_pos`, is
        // refreshed before each launch; only pointer-changing reallocations
        // force recapture.
        let use_graph = self.supports_paged_prefill_graph();
        if use_graph {
            let mut graph_state = std::mem::replace(&mut bufs.graph_state, CudaGraphState::new());
            graph_state.run_or_capture(&self.ctx, || {
                self.prefill_forward_paged_kernels(pool, recurrent, bufs, true)
            })?;
            bufs.graph_state = graph_state;
        } else {
            self.prefill_forward_paged_kernels(pool, recurrent, bufs, false)?;
        }

        recurrent.seq_len += seq_len;
        bufs.logits_valid = true;
        Ok(())
    }

    pub(super) fn prefill_forward_paged_batch(
        &self,
        requests: &[Qwen35PagedPrefillRequest<'_>],
        states: &mut [Qwen35State],
        pool: &TokenKVPool,
    ) -> Result<()> {
        anyhow::ensure!(
            !requests.is_empty(),
            "paged prefill batch requires at least one request"
        );

        let request_lens: Vec<usize> = requests
            .iter()
            .map(|request| request.tokens.len())
            .collect();
        let total_tokens = request_lens.iter().sum();
        let mut packed_tokens = Vec::with_capacity(total_tokens);
        for request in requests {
            packed_tokens.extend_from_slice(request.tokens);
        }

        let (sequences, page_indices) = self.build_paged_prefill_sequences(requests, pool)?;
        let mut batch_guard = self.ensure_paged_prefill_batch(total_tokens, pool.page_size)?;
        let bufs = batch_guard
            .as_mut()
            .expect("paged prefill batch buffers initialized");
        let metadata_reallocated = bufs.metadata.update(
            &self.ctx,
            &packed_tokens,
            &page_indices,
            &sequences,
            pool.page_size,
        )?;
        if metadata_reallocated {
            bufs.invalidate_graph();
        }
        self.prefill_forward_paged_batch_kernels(requests, states, pool, &sequences, bufs)?;

        for (request, seq) in requests.iter().zip(sequences.iter()) {
            let state = &mut states[request.slot];
            let last_token_idx = seq.token_offset + seq.seq_len - 1;
            ops::extract_vec_into(
                &self.ctx,
                &bufs.hidden,
                last_token_idx,
                &mut bufs.last_hidden,
            )?;
            ops::rms_norm_offset_into(
                &self.ctx,
                &bufs.last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
                &mut bufs.last_normed,
            )?;
            ops::gemv(
                &self.ctx,
                self.output_projection(),
                &bufs.last_normed,
                &mut bufs.logits,
            )?;
            state.base.prefill_logits =
                Some(DeviceVec {
                    data: self.ctx.stream.clone_dtod(&bufs.logits.data).map_err(|e| {
                        anyhow::anyhow!("clone batch prefill logits D2D failed: {e}")
                    })?,
                    len: bufs.logits.len,
                    label: "qwen35_paged_prefill_logits",
                });
            state.recurrent_state.seq_len += request.tokens.len();
        }

        Ok(())
    }

    pub(super) fn forward_mixed_batch_paged(
        &self,
        batch: MixedBatchRequest<'_>,
        states: &mut [Qwen35State],
        pool: &mut TokenKVPool,
        decode_ctx: &mut BatchDecodeBuffers35,
    ) -> Result<MixedBatchOutcome> {
        let decode_rows = batch.decode_tokens.len();
        let prefill_rows = batch.prefills.len();
        if decode_rows == 0 {
            return Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::EmptyDecodeBatch,
            ));
        }
        if decode_rows != batch.decode_slot_indices.len() {
            return Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::DecodeSlotCountMismatch,
            ));
        }
        if prefill_rows == 0 {
            return Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::EmptyPrefillBatch,
            ));
        }
        if prefill_rows != batch.prefill_start_positions.len() {
            return Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::PrefillStartPositionCountMismatch,
            ));
        }
        if pool.page_size != 16 || pool.format != KVFormat::BF16 {
            return Ok(MixedBatchOutcome::Fallback(
                MixedBatchFallbackReason::UnsupportedKvFormat,
            ));
        }

        let mut prefill_slot_indices = Vec::with_capacity(prefill_rows);
        for (prefill, &start_pos) in batch.prefills.iter().zip(batch.prefill_start_positions) {
            if prefill.tokens.is_empty() {
                return Ok(MixedBatchOutcome::Fallback(
                    MixedBatchFallbackReason::EmptyPrefillTokens,
                ));
            }
            if batch.decode_slot_indices.contains(&prefill.slot_idx) {
                return Ok(MixedBatchOutcome::Fallback(
                    MixedBatchFallbackReason::PrefillSlotInDecodeBatch,
                ));
            }
            if prefill_slot_indices.contains(&prefill.slot_idx) {
                return Ok(MixedBatchOutcome::Fallback(
                    MixedBatchFallbackReason::DuplicatePrefillSlot,
                ));
            }
            if pool.seq_len(prefill.slot_idx) != start_pos {
                return Ok(MixedBatchOutcome::Fallback(
                    MixedBatchFallbackReason::PrefillSeqLenMismatch,
                ));
            }
            prefill_slot_indices.push(prefill.slot_idx);
        }

        let required_pages: usize = batch
            .prefills
            .iter()
            .map(|prefill| pool.append_pages_needed(prefill.slot_idx, prefill.tokens.len()))
            .sum();
        if required_pages > pool.free_page_count() {
            anyhow::bail!(
                "qwen35 mixed prefill needs {required_pages} free pages, only {} available",
                pool.free_page_count()
            );
        }

        for &slot_idx in batch.decode_slot_indices {
            states[slot_idx].drop_paged_prefill();
        }
        for prefill in batch.prefills {
            states[prefill.slot_idx].drop_paged_prefill();
            pool.cow_tail_page_for_append(&self.ctx, prefill.slot_idx)?;
            pool.alloc_tokens(prefill.slot_idx, prefill.tokens.len())?;
        }

        let decode_token_rows: Vec<[u32; 1]> =
            batch.decode_tokens.iter().map(|&token| [token]).collect();
        let total_tokens = decode_rows
            + batch
                .prefills
                .iter()
                .map(|prefill| prefill.tokens.len())
                .sum::<usize>();
        let mut combined_requests = Vec::with_capacity(decode_rows + prefill_rows);
        let mut packed_tokens = Vec::with_capacity(total_tokens);
        for (row, &slot_idx) in batch.decode_slot_indices.iter().enumerate() {
            packed_tokens.push(batch.decode_tokens[row]);
            combined_requests.push(Qwen35PagedPrefillRequest {
                tokens: &decode_token_rows[row],
                slot: slot_idx,
            });
        }
        for prefill in batch.prefills {
            packed_tokens.extend_from_slice(prefill.tokens);
            combined_requests.push(Qwen35PagedPrefillRequest {
                tokens: prefill.tokens,
                slot: prefill.slot_idx,
            });
        }

        let (sequences, page_indices) =
            self.build_paged_prefill_sequences(&combined_requests, pool)?;
        let mut batch_guard = self.ensure_paged_prefill_batch(total_tokens, pool.page_size)?;
        let bufs = batch_guard
            .as_mut()
            .expect("paged prefill batch buffers initialized");
        let metadata_reallocated = bufs.metadata.update(
            &self.ctx,
            &packed_tokens,
            &page_indices,
            &sequences,
            pool.page_size,
        )?;
        if metadata_reallocated {
            bufs.invalidate_graph();
        }
        bufs.set_active_seq_len(total_tokens);
        decode_ctx.upload_mixed_token_ids_with_handoff(
            &self.ctx,
            &mut bufs.metadata.token_ids_gpu,
            batch.decode_tokens,
            batch.decode_slot_indices,
            batch.prefills,
        )?;
        self.prepare_mixed_decode_recurrent_ptrs(batch.decode_slot_indices, states, decode_ctx)?;
        self.forward_mixed_batch_paged_kernels(
            decode_rows,
            batch.prefills,
            states,
            pool,
            &sequences,
            bufs,
            decode_ctx,
        )?;

        let vocab_size = self.output_projection().rows;
        let decode_logits = decode_ctx.ensure_logits_batch(&self.ctx, vocab_size)?;
        let kept_rows = decode_rows + prefill_rows;
        self.compact_mixed_final_rows(&mut bufs.normed, decode_rows, batch.prefills)?;
        bufs.normed.seq_len = kept_rows;
        decode_logits.seq_len = kept_rows;
        ops::gemm_into(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            decode_logits,
        );

        for (i, prefill) in batch.prefills.iter().enumerate() {
            let state = &mut states[prefill.slot_idx];
            if state.base.prefill_logits.is_none() {
                state.base.prefill_logits = Some(DeviceVec::zeros(&self.ctx, vocab_size)?);
            }
            ops::extract_vec_into(
                &self.ctx,
                decode_logits,
                decode_rows + i,
                state
                    .base
                    .prefill_logits
                    .as_mut()
                    .expect("qwen35 mixed prefill logits allocated"),
            )?;
            state.recurrent_state.seq_len += prefill.tokens.len();
        }
        decode_logits.seq_len = decode_rows;
        bufs.set_active_seq_len(total_tokens);

        Ok(MixedBatchOutcome::Executed)
    }

    fn prepare_mixed_decode_recurrent_ptrs(
        &self,
        decode_slot_indices: &[usize],
        states: &mut [Qwen35State],
        decode_ctx: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let decode_rows = decode_slot_indices.len();
        let mut linear_idx = 0usize;
        for layer in &self.layers {
            if matches!(layer.attn, LayerKind::LinearAttention(_)) {
                for (row, &slot_idx) in decode_slot_indices.iter().enumerate() {
                    let layer_state = &mut states[slot_idx].recurrent_state.layers[linear_idx];
                    let (conv_ptr, _) =
                        layer_state.conv_state.data.device_ptr_mut(&self.ctx.stream);
                    let (gdr_ptr, _) = layer_state.state.device_ptr_mut(&self.ctx.stream);
                    decode_ctx.recurrent.conv_state_ptrs_host[row] = conv_ptr;
                    decode_ctx.recurrent.gdr_state_ptrs_host[row] = gdr_ptr;
                }
                self.ctx
                    .stream
                    .memcpy_htod(
                        &decode_ctx.recurrent.conv_state_ptrs_host[..decode_rows],
                        &mut decode_ctx.recurrent.conv_state_ptrs_per_layer[linear_idx],
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("H2D qwen35 mixed conv ptrs layer {linear_idx}: {e}")
                    })?;
                self.ctx
                    .stream
                    .memcpy_htod(
                        &decode_ctx.recurrent.gdr_state_ptrs_host[..decode_rows],
                        &mut decode_ctx.recurrent.gdr_state_ptrs_per_layer[linear_idx],
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("H2D qwen35 mixed gdr ptrs layer {linear_idx}: {e}")
                    })?;
                linear_idx += 1;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mixed_batch_paged_kernels(
        &self,
        decode_rows: usize,
        prefills: &[PrefillBatchRequest<'_>],
        states: &mut [Qwen35State],
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        bufs: &mut PagedPrefillBuffers35,
        decode_ctx: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.metadata.token_ids_gpu,
            &mut bufs.hidden,
        )?;
        let request_lens: Vec<usize> = prefills
            .iter()
            .map(|request| request.tokens.len())
            .collect();
        bufs.ensure_batch_gdr_scratch(&self.ctx, &self.config, &request_lens)?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let eps = self.config.rms_norm_eps;
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => self
                    .prefill_full_attention_paged_batch(attn, &mut full_idx, pool, sequences, bufs)
                    .with_context(|| {
                        format!(
                            "qwen35 mixed full-attention layer_idx={layer_idx} full_idx={full_idx} decode_rows={} prefill_rows={} seq_len={}",
                            decode_rows,
                            prefills.len(),
                            bufs.seq_len
                        )
                    })?,
                LayerKind::LinearAttention(attn) => self
                    .mixed_linear_attention_paged_batch(
                        attn,
                        &mut linear_idx,
                        decode_rows,
                        prefills,
                        states,
                        bufs,
                        decode_ctx,
                    )
                    .with_context(|| {
                        format!(
                            "qwen35 mixed linear-attention layer_idx={layer_idx} linear_idx={linear_idx} decode_rows={} prefill_rows={} seq_len={}",
                            decode_rows,
                            prefills.len(),
                            bufs.seq_len
                        )
                    })?,
            }
            self.layer_communicator
                .post_attn_all_reduce_hidden_states(&mut bufs.attn_results)?;

            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;
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
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );
            self.layer_communicator
                .post_mlp_all_reduce_hidden_states(&mut bufs.mlp_out)?;
            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden_mid,
                &bufs.mlp_out,
                &mut bufs.hidden_next,
            )?;
            std::mem::swap(&mut bufs.hidden, &mut bufs.hidden_next);
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn mixed_linear_attention_paged_batch(
        &self,
        attn: &LinearAttentionLayer,
        linear_idx: &mut usize,
        decode_rows: usize,
        prefills: &[PrefillBatchRequest<'_>],
        states: &mut [Qwen35State],
        bufs: &mut PagedPrefillBuffers35,
        decode_ctx: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let c = &self.config;
        let prefill_tokens: usize = prefills.iter().map(|prefill| prefill.tokens.len()).sum();

        ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

        if decode_rows > 0 {
            ops::conv1d_decode_batch_into(
                &self.ctx,
                &bufs.qkv,
                &attn.conv1d_weight,
                &mut decode_ctx.recurrent.conv_state_ptrs_per_layer[*linear_idx],
                &mut bufs.qkv_conv,
                c.linear_conv_kernel_dim,
                decode_rows,
            );
            ops::gdr_decode_batch_into(
                &self.ctx,
                &bufs.qkv_conv,
                &bufs.b_proj,
                &bufs.a_proj,
                &ops::GdrWeights {
                    dt_bias: &attn.dt_bias,
                    a_log: &attn.a_log,
                },
                &mut decode_ctx.recurrent.gdr_state_ptrs_per_layer[*linear_idx],
                &mut bufs.gdr_out,
                &ops::GdrHeadConfig {
                    num_key_heads: c.linear_num_key_heads,
                    num_value_heads: c.linear_num_value_heads,
                    key_dim: c.linear_key_head_dim,
                    val_dim: c.linear_value_head_dim,
                },
                decode_rows,
            )?;
        }

        if prefill_tokens > 0 {
            self.prepare_mixed_prefill_gdr_launch(*linear_idx, prefills, states, bufs)?;
            let seq_indptr = mixed_prefill_seq_indptr(prefills)?;
            self.conv1d_prefill_packed_tail_into(
                &bufs.qkv,
                &attn.conv1d_weight,
                &bufs.gdr_launch.conv_state_ptrs,
                &seq_indptr,
                &mut bufs.qkv_conv,
                c.linear_conv_kernel_dim,
                decode_rows,
            )?;
            self.gdr_prefill_chunkwise_tail_into(
                &bufs.qkv_conv,
                &bufs.b_proj,
                &bufs.a_proj,
                &ops::GdrWeights {
                    dt_bias: &attn.dt_bias,
                    a_log: &attn.a_log,
                },
                &seq_indptr,
                &mut bufs.gdr_out,
                &ops::GdrHeadConfig {
                    num_key_heads: c.linear_num_key_heads,
                    num_value_heads: c.linear_num_value_heads,
                    key_dim: c.linear_key_head_dim,
                    val_dim: c.linear_value_head_dim,
                },
                decode_rows,
                &bufs.gdr_launch.state_ptrs,
                &bufs.gdr_launch.q_ptrs,
                &bufs.gdr_launch.k_ptrs,
                &bufs.gdr_launch.v_ptrs,
                &bufs.gdr_launch.g_cumsum_ptrs,
                &bufs.gdr_launch.beta_ptrs,
                &bufs.gdr_launch.a_tril_ptrs,
                &bufs.gdr_launch.a_inv_ptrs,
                &bufs.gdr_launch.w_ptrs,
                &bufs.gdr_launch.u_ptrs,
                &bufs.gdr_launch.chunk_state_ptrs,
                &bufs.gdr_launch.v_new_ptrs,
            )?;
        }

        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );
        *linear_idx += 1;
        Ok(())
    }

    fn prepare_mixed_prefill_gdr_launch(
        &self,
        linear_idx: usize,
        prefills: &[PrefillBatchRequest<'_>],
        states: &mut [Qwen35State],
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        for (batch_idx, request) in prefills.iter().enumerate() {
            let state = &mut states[request.slot_idx];
            let layer_state = &mut state.recurrent_state.layers[linear_idx];
            let scratch = &mut bufs.gdr_batch_scratch[batch_idx];

            let (conv_state_ptr, _gconv) =
                layer_state.conv_state.data.device_ptr_mut(&self.ctx.stream);
            let (state_ptr, _gstate) = layer_state.state.device_ptr_mut(&self.ctx.stream);
            let (q_ptr, _gq) = scratch.q_expanded.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _gk) = scratch.k_expanded.data.device_ptr_mut(&self.ctx.stream);
            let (v_ptr, _gv) = scratch.v_raw.data.device_ptr_mut(&self.ctx.stream);
            let (g_cumsum_ptr, _gg) = scratch.g_cumsum.device_ptr_mut(&self.ctx.stream);
            let (beta_ptr, _gbeta) = scratch.beta.device_ptr_mut(&self.ctx.stream);
            let (a_tril_ptr, _ga) = scratch.a_tril.device_ptr_mut(&self.ctx.stream);
            let (a_inv_ptr, _gainv) = scratch.a_inv.device_ptr_mut(&self.ctx.stream);
            let (w_ptr, _gw) = scratch.w.data.device_ptr_mut(&self.ctx.stream);
            let (u_ptr, _gu) = scratch.u.data.device_ptr_mut(&self.ctx.stream);
            let (chunk_state_ptr, _gchunk) = scratch.chunk_state.device_ptr_mut(&self.ctx.stream);
            let (v_new_ptr, _gvnew) = scratch.v_new.data.device_ptr_mut(&self.ctx.stream);

            bufs.gdr_launch.conv_state_ptrs[batch_idx] = conv_state_ptr;
            bufs.gdr_launch.state_ptrs[batch_idx] = state_ptr;
            bufs.gdr_launch.q_ptrs[batch_idx] = q_ptr;
            bufs.gdr_launch.k_ptrs[batch_idx] = k_ptr;
            bufs.gdr_launch.v_ptrs[batch_idx] = v_ptr;
            bufs.gdr_launch.g_cumsum_ptrs[batch_idx] = g_cumsum_ptr;
            bufs.gdr_launch.beta_ptrs[batch_idx] = beta_ptr;
            bufs.gdr_launch.a_tril_ptrs[batch_idx] = a_tril_ptr;
            bufs.gdr_launch.a_inv_ptrs[batch_idx] = a_inv_ptr;
            bufs.gdr_launch.w_ptrs[batch_idx] = w_ptr;
            bufs.gdr_launch.u_ptrs[batch_idx] = u_ptr;
            bufs.gdr_launch.chunk_state_ptrs[batch_idx] = chunk_state_ptr;
            bufs.gdr_launch.v_new_ptrs[batch_idx] = v_new_ptr;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d_prefill_packed_tail_into(
        &self,
        x_batch: &HiddenStates,
        conv_weight: &DeviceVec,
        conv_state_ptrs: &[u64],
        seq_indptr: &[i32],
        out_batch: &mut HiddenStates,
        kernel_size: usize,
        row_offset: usize,
    ) -> Result<()> {
        ensure!(
            conv_state_ptrs.len() + 1 == seq_indptr.len(),
            "qwen35 mixed conv1d seq_indptr len mismatch"
        );
        let num_channels = x_batch.hidden_dim;
        let half_size = std::mem::size_of::<ffi::Half>();
        let (x_ptr, _gx) = x_batch.data.device_ptr(&self.ctx.stream);
        let (w_ptr, _gw) = conv_weight.data.device_ptr(&self.ctx.stream);
        let (out_ptr, _go) = out_batch.data.device_ptr_mut(&self.ctx.stream);
        let x_tail = (x_ptr as usize + row_offset * num_channels * half_size) as *const ffi::Half;
        let out_tail = (out_ptr as usize + row_offset * num_channels * half_size) as *mut ffi::Half;
        unsafe {
            ffi::conv1d_prefill_packed_batch_cuda(
                x_tail,
                w_ptr as *const ffi::Half,
                conv_state_ptrs.as_ptr(),
                seq_indptr.as_ptr(),
                out_tail,
                num_channels as i32,
                kernel_size as i32,
                conv_state_ptrs.len() as i32,
                self.ctx.stream.cu_stream(),
            )
            .result()
            .context("qwen35 mixed conv1d prefill tail failed")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gdr_prefill_chunkwise_tail_into(
        &self,
        qkv_batch: &HiddenStates,
        b_proj_batch: &HiddenStates,
        a_proj_batch: &HiddenStates,
        weights: &ops::GdrWeights<'_>,
        seq_indptr: &[i32],
        output_batch: &mut HiddenStates,
        heads: &ops::GdrHeadConfig,
        row_offset: usize,
        state_ptrs: &[u64],
        q_ptrs: &[u64],
        k_ptrs: &[u64],
        v_ptrs: &[u64],
        g_cumsum_ptrs: &[u64],
        beta_ptrs: &[u64],
        a_tril_ptrs: &[u64],
        a_inv_ptrs: &[u64],
        w_ptrs: &[u64],
        u_ptrs: &[u64],
        chunk_state_ptrs: &[u64],
        v_new_ptrs: &[u64],
    ) -> Result<()> {
        let batch_size = state_ptrs.len();
        ensure!(
            batch_size + 1 == seq_indptr.len(),
            "qwen35 mixed GDR seq_indptr len mismatch"
        );
        ensure!(
            q_ptrs.len() == batch_size
                && k_ptrs.len() == batch_size
                && v_ptrs.len() == batch_size
                && g_cumsum_ptrs.len() == batch_size
                && beta_ptrs.len() == batch_size
                && a_tril_ptrs.len() == batch_size
                && a_inv_ptrs.len() == batch_size
                && w_ptrs.len() == batch_size
                && u_ptrs.len() == batch_size
                && chunk_state_ptrs.len() == batch_size
                && v_new_ptrs.len() == batch_size,
            "qwen35 mixed GDR pointer table length mismatch"
        );
        let half_size = std::mem::size_of::<ffi::Half>();
        let qkv_dim = qkv_batch.hidden_dim;
        let b_dim = b_proj_batch.hidden_dim;
        let out_dim = output_batch.hidden_dim;
        let (qkv_ptr, _gqkv) = qkv_batch.data.device_ptr(&self.ctx.stream);
        let (b_ptr, _gb) = b_proj_batch.data.device_ptr(&self.ctx.stream);
        let (a_ptr, _ga) = a_proj_batch.data.device_ptr(&self.ctx.stream);
        let (dt_ptr, _gdt) = weights.dt_bias.data.device_ptr(&self.ctx.stream);
        let (alog_ptr, _galog) = weights.a_log.device_ptr(&self.ctx.stream);
        let (out_ptr, _go) = output_batch.data.device_ptr_mut(&self.ctx.stream);
        let qkv_tail = (qkv_ptr as usize + row_offset * qkv_dim * half_size) as *const ffi::Half;
        let b_tail = (b_ptr as usize + row_offset * b_dim * half_size) as *const ffi::Half;
        let a_tail = (a_ptr as usize + row_offset * b_dim * half_size) as *const ffi::Half;
        let out_tail = (out_ptr as usize + row_offset * out_dim * half_size) as *mut ffi::Half;
        unsafe {
            ffi::gated_delta_rule_prefill_chunkwise_batch_cuda(
                qkv_tail,
                b_tail,
                a_tail,
                dt_ptr as *const ffi::Half,
                alog_ptr as *const f32,
                state_ptrs.as_ptr(),
                q_ptrs.as_ptr(),
                k_ptrs.as_ptr(),
                v_ptrs.as_ptr(),
                g_cumsum_ptrs.as_ptr(),
                beta_ptrs.as_ptr(),
                a_tril_ptrs.as_ptr(),
                a_inv_ptrs.as_ptr(),
                w_ptrs.as_ptr(),
                u_ptrs.as_ptr(),
                chunk_state_ptrs.as_ptr(),
                v_new_ptrs.as_ptr(),
                seq_indptr.as_ptr(),
                out_tail,
                heads.num_key_heads as i32,
                heads.num_value_heads as i32,
                heads.key_dim as i32,
                heads.val_dim as i32,
                batch_size as i32,
                self.ctx.stream.cu_stream(),
            )
            .result()
            .context("qwen35 mixed GDR prefill tail failed")?;
        }
        Ok(())
    }

    fn compact_mixed_final_rows(
        &self,
        normed: &mut HiddenStates,
        decode_rows: usize,
        prefills: &[PrefillBatchRequest<'_>],
    ) -> Result<()> {
        use cudarc::driver::sys::{cuMemcpyDtoDAsync_v2, cudaError_enum::CUDA_SUCCESS};

        let hidden_dim = normed.hidden_dim;
        let row_bytes = hidden_dim * std::mem::size_of::<ffi::Half>();
        let (base_ptr, _guard) = normed.data.device_ptr_mut(&self.ctx.stream);
        let mut prefill_token_offset = 0usize;
        for (i, prefill) in prefills.iter().enumerate() {
            let src_row = decode_rows + prefill_token_offset + prefill.tokens.len() - 1;
            let dst_row = decode_rows + i;
            if src_row != dst_row {
                let src_ptr = base_ptr + (src_row * row_bytes) as u64;
                let dst_ptr = base_ptr + (dst_row * row_bytes) as u64;
                let result = unsafe {
                    cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, row_bytes, self.ctx.stream.cu_stream())
                };
                if result != CUDA_SUCCESS {
                    anyhow::bail!("qwen35 mixed final-row D2D gather failed: {result:?}");
                }
            }
            prefill_token_offset += prefill.tokens.len();
        }
        Ok(())
    }

    fn prefill_forward_paged_batch_kernels(
        &self,
        requests: &[Qwen35PagedPrefillRequest<'_>],
        states: &mut [Qwen35State],
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.metadata.token_ids_gpu,
            &mut bufs.hidden,
        )?;
        let request_lens: Vec<usize> = requests
            .iter()
            .map(|request| request.tokens.len())
            .collect();
        bufs.ensure_batch_gdr_scratch(&self.ctx, &self.config, &request_lens)?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let eps = self.config.rms_norm_eps;
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => self.prefill_full_attention_paged_batch(
                    attn,
                    &mut full_idx,
                    pool,
                    sequences,
                    bufs,
                )
                .with_context(|| {
                    format!(
                        "qwen35 paged prefill full-attention layer_idx={layer_idx} full_idx={full_idx} batch_size={} seq_len={}",
                        requests.len(),
                        bufs.seq_len
                    )
                })?,
                LayerKind::LinearAttention(attn) => self.prefill_linear_attention_paged_batch(
                    attn,
                    &mut linear_idx,
                    requests,
                    states,
                    bufs,
                )
                .with_context(|| {
                    format!(
                        "qwen35 paged prefill linear-attention layer_idx={layer_idx} linear_idx={linear_idx} batch_size={} seq_len={}",
                        requests.len(),
                        bufs.seq_len
                    )
                })?,
            }
            self.layer_communicator
                .post_attn_all_reduce_hidden_states(&mut bufs.attn_results)?;

            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;
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
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );
            self.layer_communicator
                .post_mlp_all_reduce_hidden_states(&mut bufs.mlp_out)?;
            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden_mid,
                &bufs.mlp_out,
                &mut bufs.hidden_next,
            )?;
            std::mem::swap(&mut bufs.hidden, &mut bufs.hidden_next);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_full_attention_paged_batch(
        &self,
        attn: &FullAttentionLayer,
        full_idx: &mut usize,
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);
        self.refill_fp8_paged_prefill_prefix_if_needed(pool, *full_idx, bufs)?;

        let nrp = ops::NormRopeParams {
            q_norm: &attn.q_norm,
            k_norm: &attn.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: eps,
        };
        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _gqp) = bufs.q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _gk) = bufs.k_attn.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _gv) = bufs.v_attn.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&self.ctx.stream);
            let (pt_ptr, _gpt) = bufs.metadata.page_indices_gpu.device_ptr(&self.ctx.stream);
            let (sp_ptr, _gsp) = bufs.metadata.start_pos_gpu.device_ptr(&self.ctx.stream);
            let kp_ptr = pool.k_ptr(*full_idx, &self.ctx.stream);
            let vp_ptr = pool.v_ptr(*full_idx, &self.ctx.stream);

            let q_full_stride = bufs.q_full.hidden_dim;
            let q_out_stride = bufs.q_prepped.hidden_dim;
            let kv_stride = bufs.k_attn.hidden_dim;
            let half_size = std::mem::size_of::<ffi::Half>();
            let i32_size = std::mem::size_of::<i32>();

            for (batch_idx, seq) in sequences.iter().enumerate() {
                let qf_ptr_offset = (qf_ptr as usize + seq.token_offset * q_full_stride * half_size)
                    as *const ffi::Half;
                let qp_ptr_offset = (qp_ptr as usize + seq.token_offset * q_out_stride * half_size)
                    as *mut ffi::Half;
                let k_ptr_offset =
                    (k_ptr as usize + seq.token_offset * kv_stride * half_size) as *const ffi::Half;
                let v_ptr_offset =
                    (v_ptr as usize + seq.token_offset * kv_stride * half_size) as *const ffi::Half;
                let pt_ptr_offset =
                    (pt_ptr as usize + seq.page_table_offset * i32_size) as *const i32;
                let sp_ptr_offset = (sp_ptr as usize + batch_idx * i32_size) as *const i32;

                ffi::prefill_attention_paged_prep_hd256_cuda(
                    qf_ptr_offset,
                    qp_ptr_offset,
                    k_ptr_offset,
                    v_ptr_offset,
                    qn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pt_ptr_offset,
                    pool.page_size as i32,
                    kp_ptr as *mut ffi::Half,
                    vp_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
                    seq.seq_len as i32,
                    sp_ptr_offset,
                    c.rotary_dim as i32,
                    nrp.rms_eps,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .with_context(|| {
                    format!(
                        "prefill_attention_paged_prep_hd256_cuda layer={full_idx} batch_idx={batch_idx} seq_len={} start_pos={}",
                        seq.seq_len, seq.start_pos
                    )
                })?;
                qwen35_cuda_debug_sync(
                    &self.ctx,
                    format!(
                        "prefill_attention_paged_prep_hd256_cuda layer={full_idx} batch_idx={batch_idx} seq_len={} start_pos={}",
                        seq.seq_len, seq.start_pos
                    ),
                )?;
            }
        }

        {
            let (q_u64, _gq) = bufs.q_prepped.data.device_ptr(&self.ctx.stream);
            let (o_u64, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            let (qoi_u64, _gqoi) = bufs.metadata.qo_indptr_gpu.device_ptr(&self.ctx.stream);
            let (kvi_u64, _gkvi) = bufs.metadata.kv_indptr_gpu.device_ptr(&self.ctx.stream);
            let (kvidx_u64, _gkvidx) = bufs.metadata.page_indices_gpu.device_ptr(&self.ctx.stream);
            let (kvlpl_u64, _gkvlpl) = bufs
                .metadata
                .kv_last_page_len_gpu
                .device_ptr(&self.ctx.stream);
            let max_qlen = bufs
                .metadata
                .qo_indptr_host
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap_or(0);
            let total_pages = bufs.metadata.kv_indptr_host.last().copied().unwrap_or(0);
            ops::prefill_attention_paged_run_hd256(
                &self.ctx,
                q_u64,
                qoi_u64,
                pool.k_ptr(*full_idx, &self.ctx.stream),
                pool.v_ptr(*full_idx, &self.ctx.stream),
                kvi_u64,
                kvidx_u64,
                kvlpl_u64,
                o_u64,
                pool,
                bufs.metadata.batch_size,
                bufs.seq_len,
                c.num_attention_heads,
                c.num_key_value_heads,
                pool.page_size,
                max_qlen,
                total_pages,
            )
            .with_context(|| {
                format!(
                    "tilelang hd256 paged prefill layer={full_idx} batch_size={} seq_len={} max_qlen={max_qlen} total_pages={total_pages}",
                    bufs.metadata.batch_size, bufs.seq_len
                )
            })?;
            qwen35_cuda_debug_sync(
                &self.ctx,
                format!(
                    "tilelang hd256 paged prefill layer={full_idx} batch_size={} seq_len={} max_qlen={max_qlen} total_pages={total_pages}",
                    bufs.metadata.batch_size, bufs.seq_len
                ),
            )?;
        }
        self.commit_fp8_paged_prefill_if_needed(pool, *full_idx, bufs)?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const ffi::Half,
                o_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                bufs.seq_len as i32,
                self.ctx.stream.cu_stream(),
            )
            .result()
            .with_context(|| {
                format!(
                    "attention_gate_batch_hd256_cuda layer={full_idx} seq_len={}",
                    bufs.seq_len
                )
            })?;
            qwen35_cuda_debug_sync(
                &self.ctx,
                format!(
                    "attention_gate_batch_hd256_cuda layer={full_idx} seq_len={}",
                    bufs.seq_len
                ),
            )?;
        }

        *full_idx += 1;
        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        );
        Ok(())
    }

    fn prefill_linear_attention_paged_batch(
        &self,
        attn: &LinearAttentionLayer,
        linear_idx: &mut usize,
        requests: &[Qwen35PagedPrefillRequest<'_>],
        states: &mut [Qwen35State],
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        let c = &self.config;

        ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

        for (batch_idx, request) in requests.iter().enumerate() {
            let state = &mut states[request.slot];
            let layer_state = &mut state.recurrent_state.layers[*linear_idx];
            let scratch = &mut bufs.gdr_batch_scratch[batch_idx];

            let (conv_state_ptr, _gconv) =
                layer_state.conv_state.data.device_ptr_mut(&self.ctx.stream);
            let (state_ptr, _gstate) = layer_state.state.device_ptr_mut(&self.ctx.stream);
            let (q_ptr, _gq) = scratch.q_expanded.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _gk) = scratch.k_expanded.data.device_ptr_mut(&self.ctx.stream);
            let (v_ptr, _gv) = scratch.v_raw.data.device_ptr_mut(&self.ctx.stream);
            let (g_cumsum_ptr, _gg) = scratch.g_cumsum.device_ptr_mut(&self.ctx.stream);
            let (beta_ptr, _gbeta) = scratch.beta.device_ptr_mut(&self.ctx.stream);
            let (a_tril_ptr, _ga) = scratch.a_tril.device_ptr_mut(&self.ctx.stream);
            let (a_inv_ptr, _gainv) = scratch.a_inv.device_ptr_mut(&self.ctx.stream);
            let (w_ptr, _gw) = scratch.w.data.device_ptr_mut(&self.ctx.stream);
            let (u_ptr, _gu) = scratch.u.data.device_ptr_mut(&self.ctx.stream);
            let (chunk_state_ptr, _gchunk) = scratch.chunk_state.device_ptr_mut(&self.ctx.stream);
            let (v_new_ptr, _gvnew) = scratch.v_new.data.device_ptr_mut(&self.ctx.stream);

            bufs.gdr_launch.conv_state_ptrs[batch_idx] = conv_state_ptr;
            bufs.gdr_launch.state_ptrs[batch_idx] = state_ptr;
            bufs.gdr_launch.q_ptrs[batch_idx] = q_ptr;
            bufs.gdr_launch.k_ptrs[batch_idx] = k_ptr;
            bufs.gdr_launch.v_ptrs[batch_idx] = v_ptr;
            bufs.gdr_launch.g_cumsum_ptrs[batch_idx] = g_cumsum_ptr;
            bufs.gdr_launch.beta_ptrs[batch_idx] = beta_ptr;
            bufs.gdr_launch.a_tril_ptrs[batch_idx] = a_tril_ptr;
            bufs.gdr_launch.a_inv_ptrs[batch_idx] = a_inv_ptr;
            bufs.gdr_launch.w_ptrs[batch_idx] = w_ptr;
            bufs.gdr_launch.u_ptrs[batch_idx] = u_ptr;
            bufs.gdr_launch.chunk_state_ptrs[batch_idx] = chunk_state_ptr;
            bufs.gdr_launch.v_new_ptrs[batch_idx] = v_new_ptr;
        }

        ops::conv1d_prefill_packed_batch_into(
            &self.ctx,
            &bufs.qkv,
            &attn.conv1d_weight,
            &ops::Conv1dPrefillBatchLaunch {
                conv_state_ptrs: &bufs.gdr_launch.conv_state_ptrs,
                seq_indptr: &bufs.metadata.qo_indptr_host,
            },
            &mut bufs.qkv_conv,
            c.linear_conv_kernel_dim,
        )?;
        ops::gated_delta_rule_prefill_chunkwise_batch_into(
            &self.ctx,
            &bufs.qkv_conv,
            &bufs.b_proj,
            &bufs.a_proj,
            &ops::GdrWeights {
                dt_bias: &attn.dt_bias,
                a_log: &attn.a_log,
            },
            &ops::GdrPrefillBatchLaunch {
                state_ptrs: &bufs.gdr_launch.state_ptrs,
                q_ptrs: &bufs.gdr_launch.q_ptrs,
                k_ptrs: &bufs.gdr_launch.k_ptrs,
                v_ptrs: &bufs.gdr_launch.v_ptrs,
                g_cumsum_ptrs: &bufs.gdr_launch.g_cumsum_ptrs,
                beta_ptrs: &bufs.gdr_launch.beta_ptrs,
                a_tril_ptrs: &bufs.gdr_launch.a_tril_ptrs,
                a_inv_ptrs: &bufs.gdr_launch.a_inv_ptrs,
                w_ptrs: &bufs.gdr_launch.w_ptrs,
                u_ptrs: &bufs.gdr_launch.u_ptrs,
                chunk_state_ptrs: &bufs.gdr_launch.chunk_state_ptrs,
                v_new_ptrs: &bufs.gdr_launch.v_new_ptrs,
                seq_indptr: &bufs.metadata.qo_indptr_host,
            },
            &mut bufs.gdr_out,
            &ops::GdrHeadConfig {
                num_key_heads: c.linear_num_key_heads,
                num_value_heads: c.linear_num_value_heads,
                key_dim: c.linear_key_head_dim,
                val_dim: c.linear_value_head_dim,
            },
        )
        .with_context(|| {
            format!(
                "gated_delta_rule_prefill_chunkwise_batch_into linear_layer={linear_idx} batch_size={} seq_len={}",
                requests.len(),
                bufs.seq_len
            )
        })?;
        qwen35_cuda_debug_sync(
            &self.ctx,
            format!(
                "gated_delta_rule_prefill_chunkwise_batch_into linear_layer={linear_idx} batch_size={} seq_len={}",
                requests.len(),
                bufs.seq_len
            ),
        )?;
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );

        *linear_idx += 1;
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );
        Ok(())
    }

    pub(super) fn build_paged_prefill_sequences(
        &self,
        requests: &[Qwen35PagedPrefillRequest<'_>],
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

    fn ensure_paged_prefill_batch(
        &self,
        seq_len: usize,
        page_size: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<PagedPrefillBuffers35>>> {
        let mut batch_guard = self
            .paged_prefill_batch
            .lock()
            .map_err(|_| anyhow::anyhow!("paged_prefill_batch mutex poisoned"))?;
        let needs_realloc = batch_guard
            .as_ref()
            .map(|bufs| !bufs.matches_shape(seq_len, page_size))
            .unwrap_or(true);
        if needs_realloc {
            *batch_guard = None;
            self.ctx.sync()?;
            *batch_guard = Some(PagedPrefillBuffers35::new(
                &self.ctx,
                &self.config,
                seq_len,
                page_size,
            )?);
        }
        Ok(batch_guard)
    }

    fn supports_paged_prefill_graph(&self) -> bool {
        self.enable_cuda_graph
            && self.layers.iter().all(|layer| {
                let attn_safe = match &layer.attn {
                    LayerKind::FullAttention(attn) => {
                        Self::graphsafe_batched_weight(&attn.q_proj)
                            && Self::graphsafe_batched_weight(&attn.k_proj)
                            && Self::graphsafe_batched_weight(&attn.v_proj)
                            && Self::graphsafe_batched_weight(&attn.o_proj)
                    }
                    LayerKind::LinearAttention(attn) => {
                        Self::graphsafe_batched_weight(&attn.in_proj_qkv)
                            && Self::graphsafe_batched_weight(&attn.in_proj_z)
                            && Self::graphsafe_batched_weight(&attn.in_proj_b)
                            && Self::graphsafe_batched_weight(&attn.in_proj_a)
                            && Self::graphsafe_batched_weight(&attn.out_proj)
                    }
                };
                attn_safe
                    && Self::graphsafe_batched_weight(&layer.mlp.gate_proj)
                    && Self::graphsafe_batched_weight(&layer.mlp.up_proj)
                    && Self::graphsafe_batched_weight(&layer.mlp.down_proj)
            })
    }

    fn graphsafe_batched_weight(weight: &DeviceMatrix) -> bool {
        weight.is_dense_bf16()
    }

    fn prefill_gemm_into(
        &self,
        weight: &DeviceMatrix,
        x: &HiddenStates,
        out: &mut HiddenStates,
        graphsafe: bool,
    ) -> Result<()> {
        if graphsafe {
            ops::gemm_graphsafe_batched_into(&self.ctx, weight, x, out)
        } else {
            ops::gemm_into(&self.ctx, weight, x, out);
            Ok(())
        }
    }

    fn prepare_paged_prefill(
        &self,
        token_ids: &[u32],
        pool: &TokenKVPool,
        slot: usize,
        bufs: &mut PagedPrefillBuffers35,
    ) -> Result<()> {
        let seq_len = token_ids.len();
        let pool_seq_len = pool.seq_len(slot);
        anyhow::ensure!(
            pool_seq_len >= seq_len,
            "paged prefill: pool seq_len {pool_seq_len} < chunk len {seq_len}"
        );
        let start_pos = pool_seq_len - seq_len;
        let num_pages = (start_pos + seq_len).div_ceil(pool.page_size);
        let all_pages = pool.page_indices(slot);
        anyhow::ensure!(
            all_pages.len() >= num_pages,
            "paged prefill: slot {slot} has {} pages, expected at least {num_pages}",
            all_pages.len()
        );
        let sequences = [ops::PagedPrefillSequence {
            token_offset: 0,
            seq_len,
            start_pos,
            page_table_offset: 0,
            num_pages,
        }];
        let page_indices: Vec<i32> = all_pages[..num_pages]
            .iter()
            .map(|&page| page as i32)
            .collect();
        let page_indices_reallocated = bufs.metadata.update(
            &self.ctx,
            token_ids,
            &page_indices,
            &sequences,
            pool.page_size,
        )?;
        if page_indices_reallocated {
            bufs.invalidate_graph();
        }
        Ok(())
    }

    fn prefill_forward_paged_kernels(
        &self,
        pool: &TokenKVPool,
        recurrent: &mut RecurrentState,
        bufs: &mut PagedPrefillBuffers35,
        graphsafe: bool,
    ) -> Result<()> {
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.metadata.token_ids_gpu,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &self.layers {
            self.prefill_layer_paged(
                layer,
                &mut linear_idx,
                &mut full_idx,
                pool,
                recurrent,
                bufs,
                graphsafe,
            )?;
        }

        ops::extract_vec_into(
            &self.ctx,
            &bufs.hidden,
            bufs.seq_len - 1,
            &mut bufs.last_hidden,
        )?;
        ops::rms_norm_offset_into(
            &self.ctx,
            &bufs.last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.last_normed,
        )?;
        ops::gemv(
            &self.ctx,
            self.output_projection(),
            &bufs.last_normed,
            &mut bufs.logits,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_layer_paged(
        &self,
        layer: &TransformerBlock35,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        pool: &TokenKVPool,
        recurrent: &mut RecurrentState,
        bufs: &mut PagedPrefillBuffers35,
        graphsafe: bool,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        match &layer.attn {
            LayerKind::FullAttention(attn) => {
                self.prefill_full_attention_paged(attn, full_idx, pool, bufs, graphsafe)?;
            }
            LayerKind::LinearAttention(attn) => {
                self.prefill_linear_attention_paged(attn, linear_idx, recurrent, bufs, graphsafe)?;
            }
        }
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.attn_results)?;

        ops::add_batch_into(
            &self.ctx,
            &bufs.hidden,
            &bufs.attn_results,
            &mut bufs.hidden_mid,
        )?;
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        self.prefill_gemm_into(
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
            graphsafe,
        )?;
        self.prefill_gemm_into(
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
            graphsafe,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        self.prefill_gemm_into(
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.mlp_out,
            graphsafe,
        )?;
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.mlp_out)?;
        ops::add_batch_into(
            &self.ctx,
            &bufs.hidden_mid,
            &bufs.mlp_out,
            &mut bufs.hidden_next,
        )?;
        std::mem::swap(&mut bufs.hidden, &mut bufs.hidden_next);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_full_attention_paged(
        &self,
        attn: &FullAttentionLayer,
        full_idx: &mut usize,
        pool: &TokenKVPool,
        bufs: &mut PagedPrefillBuffers35,
        graphsafe: bool,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        self.prefill_gemm_into(&attn.q_proj, &bufs.normed, &mut bufs.q_full, graphsafe)?;
        self.prefill_gemm_into(&attn.k_proj, &bufs.normed, &mut bufs.k_attn, graphsafe)?;
        self.prefill_gemm_into(&attn.v_proj, &bufs.normed, &mut bufs.v_attn, graphsafe)?;
        self.refill_fp8_paged_prefill_prefix_if_needed(pool, *full_idx, bufs)?;

        let nrp = ops::NormRopeParams {
            q_norm: &attn.q_norm,
            k_norm: &attn.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: eps,
        };
        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _gqp) = bufs.q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (k_ptr, _gk) = bufs.k_attn.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _gv) = bufs.v_attn.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _gqn) = nrp.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _gkn) = nrp.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _gc) = nrp.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _gs) = nrp.sin_cache.data.device_ptr(&self.ctx.stream);
            let (pt_ptr, _gpt) = bufs.metadata.page_indices_gpu.device_ptr(&self.ctx.stream);
            let (sp_ptr, _gsp) = bufs.metadata.start_pos_gpu.device_ptr(&self.ctx.stream);
            let kp_ptr = pool.k_ptr(*full_idx, &self.ctx.stream);
            let vp_ptr = pool.v_ptr(*full_idx, &self.ctx.stream);

            ffi::prefill_attention_paged_prep_hd256_cuda(
                qf_ptr as *const ffi::Half,
                qp_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pt_ptr as *const i32,
                pool.page_size as i32,
                kp_ptr as *mut ffi::Half,
                vp_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                c.num_key_value_heads as i32,
                bufs.seq_len as i32,
                sp_ptr as *const i32,
                c.rotary_dim as i32,
                nrp.rms_eps,
                self.ctx.stream.cu_stream(),
            )
            .result()
            .with_context(|| {
                format!(
                    "prefill_attention_paged_prep_hd256_cuda layer={full_idx} seq_len={} max_qlen={}",
                    bufs.seq_len, bufs.seq_len
                )
            })?;
            qwen35_cuda_debug_sync(
                &self.ctx,
                format!(
                    "prefill_attention_paged_prep_hd256_cuda layer={full_idx} seq_len={} max_qlen={}",
                    bufs.seq_len, bufs.seq_len
                ),
            )?;
        }

        {
            let (q_u64, _gq) = bufs.q_prepped.data.device_ptr(&self.ctx.stream);
            let (o_u64, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            let (qoi_u64, _gqoi) = bufs.metadata.qo_indptr_gpu.device_ptr(&self.ctx.stream);
            let (kvi_u64, _gkvi) = bufs.metadata.kv_indptr_gpu.device_ptr(&self.ctx.stream);
            let (kvidx_u64, _gkvidx) = bufs.metadata.page_indices_gpu.device_ptr(&self.ctx.stream);
            let (kvlpl_u64, _gkvlpl) = bufs
                .metadata
                .kv_last_page_len_gpu
                .device_ptr(&self.ctx.stream);
            let max_qlen = bufs
                .metadata
                .qo_indptr_host
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap_or(0);
            let total_pages = bufs.metadata.kv_indptr_host.last().copied().unwrap_or(0);
            ops::prefill_attention_paged_run_hd256(
                &self.ctx,
                q_u64,
                qoi_u64,
                pool.k_ptr(*full_idx, &self.ctx.stream),
                pool.v_ptr(*full_idx, &self.ctx.stream),
                kvi_u64,
                kvidx_u64,
                kvlpl_u64,
                o_u64,
                pool,
                1,
                bufs.seq_len,
                c.num_attention_heads,
                c.num_key_value_heads,
                pool.page_size,
                max_qlen,
                total_pages,
            )
            .with_context(|| {
                format!(
                    "tilelang hd256 paged prefill layer={full_idx} batch_size=1 seq_len={} max_qlen={max_qlen} total_pages={total_pages}",
                    bufs.seq_len
                )
            })?;
            qwen35_cuda_debug_sync(
                &self.ctx,
                format!(
                    "tilelang hd256 paged prefill layer={full_idx} batch_size=1 seq_len={} max_qlen={max_qlen} total_pages={total_pages}",
                    bufs.seq_len
                ),
            )?;
        }
        self.commit_fp8_paged_prefill_if_needed(pool, *full_idx, bufs)?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (o_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const ffi::Half,
                o_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                bufs.seq_len as i32,
                self.ctx.stream.cu_stream(),
            )
            .result()
            .with_context(|| {
                format!(
                    "attention_gate_batch_hd256_cuda layer={full_idx} seq_len={}",
                    bufs.seq_len
                )
            })?;
            qwen35_cuda_debug_sync(
                &self.ctx,
                format!(
                    "attention_gate_batch_hd256_cuda layer={full_idx} seq_len={}",
                    bufs.seq_len
                ),
            )?;
        }

        *full_idx += 1;
        self.prefill_gemm_into(
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
            graphsafe,
        )
    }

    fn commit_fp8_paged_prefill_if_needed(
        &self,
        pool: &TokenKVPool,
        full_idx: usize,
        bufs: &PagedPrefillBuffers35,
    ) -> Result<()> {
        let token_count = bufs.seq_len;
        let stream = &self.ctx.stream;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        match pool.format {
            KVFormat::FP8E4M3 => {
                kv_quant::quantize_paged_kv_fp8(
                    &self.ctx,
                    pool.k_work_ptr(stream),
                    pool.k_data_ptr(full_idx, stream),
                    pool.k_scales_ptr(full_idx, stream),
                    &bufs.metadata.token_rows_gpu,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )?;
                kv_quant::quantize_paged_kv_fp8(
                    &self.ctx,
                    pool.v_work_ptr(stream),
                    pool.v_data_ptr(full_idx, stream),
                    pool.v_scales_ptr(full_idx, stream),
                    &bufs.metadata.token_rows_gpu,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )?;
                Ok(())
            }
            KVFormat::INT8 => {
                // KIVI per-channel K path (mirrors qwen3/prefill.rs INT8 arm).
                // When the pool exposes static per-channel scales, calibrate
                // from this prefill batch's K statistics and quantize through
                // the static table; otherwise fall back to per-(row, head)
                // absmax. V always uses per-(row, head) scales (KIVI's
                // asymmetric choice).
                if let Some(static_scales_ptr) = pool.k_static_scales_ptr(full_idx, stream) {
                    unsafe {
                        cudarc::driver::result::memset_d8_async(
                            cudarc::driver::sys::CUdeviceptr::from(static_scales_ptr),
                            0,
                            num_kv_heads * head_dim * std::mem::size_of::<f32>(),
                            stream.cu_stream(),
                        )
                        .ok();
                    }
                    kv_quant::compute_k_per_channel_absmax(
                        &self.ctx,
                        pool.k_work_ptr(stream),
                        static_scales_ptr,
                        &bufs.metadata.token_rows_gpu,
                        num_kv_heads,
                        head_dim,
                        pool.kv_dim,
                        token_count,
                    )?;
                    kv_quant::finalize_k_per_channel_scales_int8(
                        &self.ctx,
                        static_scales_ptr,
                        num_kv_heads * head_dim,
                    )?;
                    pool.k_kivi_calibrated[full_idx]
                        .store(true, std::sync::atomic::Ordering::Release);
                    kv_quant::quantize_paged_kv_int8_per_channel(
                        &self.ctx,
                        pool.k_work_ptr(stream),
                        pool.k_data_ptr(full_idx, stream),
                        static_scales_ptr,
                        &bufs.metadata.token_rows_gpu,
                        num_kv_heads,
                        head_dim,
                        pool.kv_dim,
                        token_count,
                    )?;
                } else {
                    kv_quant::quantize_paged_kv_single(
                        &self.ctx,
                        pool.k_work_ptr(stream),
                        pool.k_data_ptr(full_idx, stream),
                        pool.k_scales_ptr(full_idx, stream),
                        &bufs.metadata.token_rows_gpu,
                        num_kv_heads,
                        head_dim,
                        pool.kv_dim,
                        token_count,
                    )?;
                }
                kv_quant::quantize_paged_kv_single(
                    &self.ctx,
                    pool.v_work_ptr(stream),
                    pool.v_data_ptr(full_idx, stream),
                    pool.v_scales_ptr(full_idx, stream),
                    &bufs.metadata.token_rows_gpu,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn refill_fp8_paged_prefill_prefix_if_needed(
        &self,
        pool: &TokenKVPool,
        full_idx: usize,
        bufs: &PagedPrefillBuffers35,
    ) -> Result<()> {
        if !matches!(pool.format, KVFormat::FP8E4M3) || bufs.metadata.prefix_token_count == 0 {
            return Ok(());
        }
        kv_quant::dequantize_paged_kv_fp8_to_hnd(
            &self.ctx,
            pool.k_data_ptr(full_idx, &self.ctx.stream),
            pool.k_scales_ptr(full_idx, &self.ctx.stream),
            pool.k_work_ptr(&self.ctx.stream),
            &bufs.metadata.prefix_token_rows_gpu,
            self.config.num_key_value_heads,
            self.config.head_dim,
            pool.kv_dim,
            bufs.metadata.prefix_token_count,
        )?;
        kv_quant::dequantize_paged_kv_fp8_to_hnd(
            &self.ctx,
            pool.v_data_ptr(full_idx, &self.ctx.stream),
            pool.v_scales_ptr(full_idx, &self.ctx.stream),
            pool.v_work_ptr(&self.ctx.stream),
            &bufs.metadata.prefix_token_rows_gpu,
            self.config.num_key_value_heads,
            self.config.head_dim,
            pool.kv_dim,
            bufs.metadata.prefix_token_count,
        )
    }

    fn prefill_linear_attention_paged(
        &self,
        attn: &LinearAttentionLayer,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        bufs: &mut PagedPrefillBuffers35,
        graphsafe: bool,
    ) -> Result<()> {
        let c = &self.config;

        self.prefill_gemm_into(&attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv, graphsafe)?;
        self.prefill_gemm_into(&attn.in_proj_z, &bufs.normed, &mut bufs.z, graphsafe)?;
        self.prefill_gemm_into(&attn.in_proj_b, &bufs.normed, &mut bufs.b_proj, graphsafe)?;
        self.prefill_gemm_into(&attn.in_proj_a, &bufs.normed, &mut bufs.a_proj, graphsafe)?;

        let layer_state = &mut recurrent.layers[*linear_idx];
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &bufs.qkv,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut bufs.qkv_conv,
            c.linear_conv_kernel_dim,
        );
        ops::gated_delta_rule_prefill_chunkwise_into(
            &self.ctx,
            &bufs.qkv_conv,
            &bufs.b_proj,
            &bufs.a_proj,
            &ops::GdrWeights {
                dt_bias: &attn.dt_bias,
                a_log: &attn.a_log,
            },
            &mut layer_state.state,
            &mut bufs.gdr_chunkwise_scratch,
            &mut bufs.gdr_out,
            &ops::GdrHeadConfig {
                num_key_heads: c.linear_num_key_heads,
                num_value_heads: c.linear_num_value_heads,
                key_dim: c.linear_key_head_dim,
                val_dim: c.linear_value_head_dim,
            },
        )
        .with_context(|| {
            format!(
                "gated_delta_rule_prefill_chunkwise_into linear_layer={linear_idx} seq_len={}",
                bufs.seq_len
            )
        })?;
        qwen35_cuda_debug_sync(
            &self.ctx,
            format!(
                "gated_delta_rule_prefill_chunkwise_into linear_layer={linear_idx} seq_len={}",
                bufs.seq_len
            ),
        )?;
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );

        *linear_idx += 1;
        self.prefill_gemm_into(
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
            graphsafe,
        )
    }

    pub(super) fn batched_rms_norm_offset(
        &self,
        x: &HiddenStates,
        weight: &DeviceVec,
        eps: f32,
    ) -> Result<HiddenStates> {
        let mut out = HiddenStates::zeros(&self.ctx, x.hidden_dim, x.seq_len)?;
        ops::rms_norm_batch_offset_into(&self.ctx, x, weight, eps, &mut out)?;
        Ok(out)
    }

    // ── Single-token optimized prefill (zero allocation per step) ───────────

    /// Same numerical result as `prefill_forward(&[token_id], ...)` but uses
    /// pre-allocated buffers, eliminating ~500 alloc/free pairs per decode step.
    /// The kernel sequence is CUDA Graph capturable (all pointers are stable).
    #[allow(clippy::too_many_lines)]
    pub(super) fn prefill_forward_single_token(
        &self,
        token_id: u32,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
        bufs: &mut SingleTokenBuffers,
        graph_state: &mut CudaGraphState,
    ) -> Result<()> {
        let c = &self.config;
        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;

        // H2D copy of token_id and start_pos — BEFORE graph launch
        let start_pos = kv_cache.len() as i32;
        self.ctx
            .stream
            .memcpy_htod(&[token_id as i32], &mut bufs.token_id_gpu)
            .map_err(|e| anyhow::anyhow!("H2D token_id failed: {}", e))?;
        self.ctx
            .stream
            .memcpy_htod(&[start_pos], &mut bufs.start_pos_buf)
            .map_err(|e| anyhow::anyhow!("H2D start_pos failed: {}", e))?;

        // GPU kernel sequence — captured on first call, replayed on subsequent calls
        if <Self as crate::model::ModelForward>::supports_cuda_graph_decode(self) {
            graph_state.run_or_capture(&self.ctx, || {
                self.single_token_kernels(kv_cache, recurrent, bufs)
            })?;
        } else {
            self.single_token_kernels(kv_cache, recurrent, bufs)?;
        }

        // CPU state updates (after graph)
        kv_cache.advance_seq_len(1);
        recurrent.seq_len += 1;

        Ok(())
    }

    /// Pure GPU kernel sequence for single-token prefill. Graph-safe:
    /// no allocation, no CPU-GPU sync, all cuBLAS via graph-safe handle.
    fn single_token_kernels(
        &self,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
        bufs: &mut SingleTokenBuffers,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        // 1. Embedding → hidden_a
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_id_gpu,
            &mut bufs.hidden_a,
        )?;

        // 2. Process all layers (hidden_a is the persistent hidden state)
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for layer in &self.layers {
            // Input layernorm: normed = rms_norm_offset(hidden_a)
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_a,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            // Attention → attn_results [hidden_size, 1]
            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    // QKV projections
                    ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
                    ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
                    ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);

                    let start_pos = kv_cache.len();
                    let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, full_idx)?;
                    let nrp = ops::NormRopeParams {
                        q_norm: &attn.q_norm,
                        k_norm: &attn.k_norm,
                        cos_cache: &self.cos_cache,
                        sin_cache: &self.sin_cache,
                        rms_eps: eps,
                    };
                    // `prefill_attention_hd256_batch_with_scratch` takes q_full
                    // (per-head concat layout), handles Q extraction + q_norm +
                    // RoPE + attention + sigmoid(gate) internally.
                    ops::prefill_attention_hd256_batch_with_scratch(
                        &self.ctx,
                        &bufs.q_full,
                        &bufs.k_attn,
                        &bufs.v_attn,
                        &nrp,
                        kc,
                        vc,
                        &mut bufs.attn_out_full,
                        &mut bufs.q_prepped,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        start_pos,
                        &bufs.start_pos_buf,
                        c.rotary_dim,
                    )?;

                    full_idx += 1;

                    // O projection → attn_results
                    ops::gemm_into(
                        &self.ctx,
                        &attn.o_proj,
                        &bufs.attn_out_full,
                        &mut bufs.attn_results,
                    );
                }
                LayerKind::LinearAttention(attn) => {
                    let layer_state = &mut recurrent.layers[linear_idx];

                    // Projections
                    ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
                    ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
                    ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
                    ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

                    // Conv1d
                    ops::conv1d_prefill_batch_into(
                        &self.ctx,
                        &bufs.qkv,
                        &attn.conv1d_weight,
                        &mut layer_state.conv_state,
                        &mut bufs.qkv_conv,
                        c.linear_conv_kernel_dim,
                    );

                    // GDR decode (fused single-step kernel)
                    ops::gated_delta_rule_decode_into(
                        &self.ctx,
                        &bufs.qkv_conv,
                        &bufs.b_proj,
                        &bufs.a_proj,
                        &ops::GdrWeights {
                            dt_bias: &attn.dt_bias,
                            a_log: &attn.a_log,
                        },
                        &mut layer_state.state,
                        &mut bufs.gdr_out,
                        &ops::GdrHeadConfig {
                            num_key_heads: c.linear_num_key_heads,
                            num_value_heads: c.linear_num_value_heads,
                            key_dim: c.linear_key_head_dim,
                            val_dim: c.linear_value_head_dim,
                        },
                    )?;

                    // Gated RMSNorm
                    ops::rms_norm_gated_batch_into(
                        &self.ctx,
                        &bufs.gdr_out,
                        &attn.norm_weight,
                        &bufs.z,
                        &mut bufs.normed_gated,
                        c.linear_num_value_heads,
                        c.linear_value_head_dim,
                        eps,
                    );
                    linear_idx += 1;

                    // Out projection → attn_results
                    ops::gemm_into(
                        &self.ctx,
                        &attn.out_proj,
                        &bufs.normed_gated,
                        &mut bufs.attn_results,
                    );
                }
            }
            self.layer_communicator
                .post_attn_all_reduce_hidden_states(&mut bufs.attn_results)?;

            // Residual 1: hidden_mid = hidden_a + attn_results
            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden_a,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;

            // Post-attention layernorm
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            // MLP
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
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );
            self.layer_communicator
                .post_mlp_all_reduce_hidden_states(&mut bufs.mlp_out)?;

            // Residual 2: hidden_a = hidden_mid + mlp_out (write back for next layer)
            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden_mid,
                &bufs.mlp_out,
                &mut bufs.hidden_a,
            )?;
        }

        // 3. Extract last hidden → DeviceVec for final norm + LM head
        // For seq_len=1, hidden_a.data has exactly hidden_size elements.
        self.ctx
            .stream
            .memcpy_dtod(&bufs.hidden_a.data, &mut bufs.last_normed.data)
            .map_err(|e| anyhow::anyhow!("D2D copy failed: {}", e))?;

        // Final norm (1+weight offset)
        ops::rms_norm_offset_into(
            &self.ctx,
            &bufs.last_normed,
            &self.norm,
            eps,
            &mut bufs.normed_out,
        )?;

        // LM head → logits
        ops::gemv(
            &self.ctx,
            self.output_projection(),
            &bufs.normed_out,
            &mut bufs.logits,
        )?;

        Ok(())
    }
}
