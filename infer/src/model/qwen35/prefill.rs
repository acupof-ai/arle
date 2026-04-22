use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::prefill_buffers::{GdrChunkwiseScratch35, PagedPrefillBuffers35};
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::KVCache;
use crate::ops;
use cuda_kernels::prelude::{DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::{TokenKVPool, ffi};

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

        // Final norm (1+weight offset) + LM head (tied embeddings)
        crate::model::common::compute_logits_batch(
            &self.ctx,
            &hidden_batch,
            &self.norm,
            &self.embed_tokens,
            c.rms_norm_eps,
            true, // offset RMSNorm (1+weight)
        )
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
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
        let attn_results = match &layer.attn {
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

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed_batch)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed_batch)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    fn prefill_full_attention(
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

    fn prefill_linear_attention(
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
    /// FlashInfer `BatchPrefillWithPagedKVCache` HD256. Linear-attn layers
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

        let start_pos = self.prepare_paged_prefill(token_ids, pool, slot, bufs)?;
        bufs.clear_logits();

        let use_graph = self.supports_paged_prefill_graph() && start_pos == 0;
        if use_graph {
            let mut graph_state = std::mem::replace(&mut bufs.graph_state, CudaGraphState::new());
            graph_state.run_or_capture(&self.ctx, || {
                self.prefill_forward_paged_kernels(pool, start_pos, recurrent, bufs, true)
            })?;
            bufs.graph_state = graph_state;
        } else {
            self.prefill_forward_paged_kernels(pool, start_pos, recurrent, bufs, false)?;
        }

        recurrent.seq_len += seq_len;
        bufs.logits_valid = true;
        Ok(())
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
        !weight.is_quantized() && !weight.has_marlin() && !weight.has_tq()
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
    ) -> Result<usize> {
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
        let page_indices_reallocated = bufs.metadata.update(
            &self.ctx,
            token_ids,
            &all_pages[..num_pages],
            seq_len,
            pool.page_size,
            start_pos,
        )?;
        if page_indices_reallocated {
            bufs.invalidate_graph();
        }

        let qo_indptr = [0_i32, seq_len as i32];
        let kv_indptr = [0_i32, num_pages as i32];
        bufs.plan.plan_hd256(
            &self.ctx,
            &qo_indptr,
            &kv_indptr,
            1,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            pool.page_size,
        )?;
        Ok(start_pos)
    }

    fn prefill_forward_paged_kernels(
        &self,
        pool: &TokenKVPool,
        start_pos: usize,
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
                start_pos,
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
            &self.embed_tokens,
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
        start_pos: usize,
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
                self.prefill_full_attention_paged(attn, full_idx, pool, start_pos, bufs, graphsafe)?
            }
            LayerKind::LinearAttention(attn) => {
                self.prefill_linear_attention_paged(attn, linear_idx, recurrent, bufs, graphsafe)?
            }
        };

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
        start_pos: usize,
        bufs: &mut PagedPrefillBuffers35,
        graphsafe: bool,
    ) -> Result<()> {
        let c = &self.config;
        let eps = c.rms_norm_eps;

        self.prefill_gemm_into(&attn.q_proj, &bufs.normed, &mut bufs.q_full, graphsafe)?;
        self.prefill_gemm_into(&attn.k_proj, &bufs.normed, &mut bufs.k_attn, graphsafe)?;
        self.prefill_gemm_into(&attn.v_proj, &bufs.normed, &mut bufs.v_attn, graphsafe)?;

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
                start_pos as i32,
                c.rotary_dim as i32,
                nrp.rms_eps,
                self.ctx.stream.cu_stream(),
            )
            .result()?;
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
            bufs.plan.run_hd256(
                &self.ctx,
                q_u64,
                qoi_u64,
                pool.k_ptr(*full_idx, &self.ctx.stream),
                pool.v_ptr(*full_idx, &self.ctx.stream),
                kvi_u64,
                kvidx_u64,
                kvlpl_u64,
                o_u64,
                None,
                1,
                c.num_attention_heads,
                c.num_key_value_heads,
                pool.page_size,
            )?;
        }

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
            .result()?;
        }

        *full_idx += 1;
        self.prefill_gemm_into(
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
            graphsafe,
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

    fn batched_rms_norm_offset(
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

        // LM head (tied embeddings) → logits
        ops::gemv(
            &self.ctx,
            &self.embed_tokens,
            &bufs.normed_out,
            &mut bufs.logits,
        )?;

        Ok(())
    }
}
