//! Dense-BF16 Qwen3 CUDA forward.
//!
//! Holds the model state (`CudaModel`/`TransformerBlock`/`Attention`/`Mlp`) and
//! the forward dataflow (`forward_tokens`) driving the layer loop over the
//! `ops`/`attention` kernel wrappers. Weight loading is in `loader`, the step
//! driver + sampling in `executor`.

use anyhow::{Result, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use infer_plan::SamplingParams;
use qwen3_spec::Qwen3Config;

use crate::attention::paged_attention;
use crate::decode_graph::DecodeGraphContext;
use crate::executor::sample_cuda_token;
use crate::loader::{MoeLayerWeights, PageMeta};
use crate::moe::moe_forward;
use crate::ops::{
    add_batch, copy_row_to_vec, embedding_batch, gemm_batch, gemv, rms_norm_batch, rms_norm_vec,
    silu_mul, upload_i32,
};

pub(crate) struct CudaModel {
    pub(crate) ctx: DeviceContext,
    pub(crate) config: Qwen3Config,
    pub(crate) embed_tokens: DeviceMatrix,
    pub(crate) lm_head: Option<DeviceMatrix>,
    pub(crate) layers: Vec<TransformerBlock>,
    pub(crate) norm: DeviceVec,
    pub(crate) cos_cache: DeviceVec,
    pub(crate) sin_cache: DeviceVec,
    /// Tensor-parallel runtime. `world_size == 1` uses the no-op communicator, so
    /// every `all_reduce_sum` returns immediately.
    pub(crate) tp: crate::tp::TpRuntime,
    /// This rank's per-shard head counts (= global config on a single GPU). The
    /// forward sizes its buffers and attention launch from these, not the config.
    pub(crate) local_q_heads: usize,
    pub(crate) local_kv_heads: usize,
    /// MoE router. `None` for dense Qwen3 (no MoE branch taken); `Some` for a
    /// MoE checkpoint, shared by every sparse layer's [`moe_forward`].
    pub(crate) moe_config: Option<infer_moe::MoeConfig>,
}

impl std::fmt::Debug for CudaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaModel")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("kv_heads", &self.config.num_key_value_heads)
            .field("head_dim", &self.config.head_dim)
            .finish()
    }
}

pub(crate) struct TransformerBlock {
    pub(crate) input_layernorm: DeviceVec,
    pub(crate) attention: Attention,
    pub(crate) post_attention_layernorm: DeviceVec,
    /// Dense SwiGLU MLP. Exactly one of `mlp` / `moe` is `Some` per layer.
    pub(crate) mlp: Option<Mlp>,
    /// Per-layer MoE expert weights; `Some` runs [`moe_forward`] instead of the
    /// dense block.
    pub(crate) moe: Option<MoeLayerWeights>,
}

pub(crate) struct Attention {
    pub(crate) q_proj: DeviceMatrix,
    pub(crate) k_proj: DeviceMatrix,
    pub(crate) v_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) q_norm: DeviceVec,
    pub(crate) k_norm: DeviceVec,
}

pub(crate) struct Mlp {
    pub(crate) gate_proj: DeviceMatrix,
    pub(crate) up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
}

impl CudaModel {
    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    /// Logits/vocab dimension (rows of the LM-head projection).
    pub(crate) fn logits_dim(&self) -> usize {
        self.output_projection().rows
    }

    /// This rank's MLP intermediate width (= full `intermediate_size` on a single
    /// GPU; the column shard under TP).
    fn local_intermediate_size(&self) -> usize {
        let tp = self.tp.config();
        if tp.is_single() {
            self.config.intermediate_size
        } else {
            infer_topo::column_shard(self.config.intermediate_size, tp).size
        }
    }

    pub(crate) fn forward_tokens(
        &self,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
        pool: &mut PagedKVPool,
        params: &SamplingParams,
        position: u64,
        penalty: infer_plan::PenaltyHistory<'_>,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "forward_tokens requires at least one token"
        );
        ensure!(
            self.config.head_dim == 128,
            "R6 clean CUDA path only wires TileLang HD128 kernels, got head_dim={}",
            self.config.head_dim
        );
        ensure!(
            self.config.num_key_value_heads == 8,
            "R6 clean CUDA path only wires kv8 TileLang kernels, got kv_heads={}",
            self.config.num_key_value_heads
        );

        let seq_len = tokens.len();
        let hidden_size = self.config.hidden_size;
        // Local per-rank dims (= global config on a single GPU): the buffers and
        // attention launch must match the sharded QKV/O GEMM outputs.
        let q_dim = self.local_q_heads * self.config.head_dim;
        let kv_dim = self.local_kv_heads * self.config.head_dim;
        let inter = self.local_intermediate_size();

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)
        })?;

        let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut q_batch = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut k_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut v_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut attn_output = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut o_buf = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut hidden_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut gate_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut up_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut act_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;

        let meta = PageMeta::for_slot(&self.ctx, pool, slot, start_pos, seq_len)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    &hidden,
                    &layer.input_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "attention", Some(layer_idx), seq_len, || {
                gemm_batch(&self.ctx, &layer.attention.q_proj, &normed, &mut q_batch)?;
                gemm_batch(&self.ctx, &layer.attention.k_proj, &normed, &mut k_batch)?;
                gemm_batch(&self.ctx, &layer.attention.v_proj, &normed, &mut v_batch)?;
                paged_attention(
                    &self.ctx,
                    layer_idx,
                    pool,
                    &mut q_batch,
                    &mut k_batch,
                    &v_batch,
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    self.config.rms_norm_eps,
                    &meta,
                    self.local_q_heads,
                    self.local_kv_heads,
                    self.config.head_dim,
                    &mut attn_output,
                )?;
                gemm_batch(&self.ctx, &layer.attention.o_proj, &attn_output, &mut o_buf)?;
                // Row-parallel o_proj: sum the per-rank partials (no-op single-GPU).
                self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
                add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)
            })?;
            std::mem::swap(&mut hidden, &mut hidden_out);

            crate::profile::profile_op(&self.ctx, "post_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    &hidden,
                    &layer.post_attention_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "mlp", Some(layer_idx), seq_len, || {
                if let Some(moe) = &layer.moe {
                    let cfg = self.moe_config.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("MoE layer present but model has no moe_config")
                    })?;
                    // Dense-path MoE has no EP split (single-rank expert ownership).
                    let split = crate::moe_config::ExpertSplit::single(cfg.num_experts);
                    let moe_out = moe_forward(&self.ctx, moe, &normed, cfg, &split)?;
                    self.ctx
                        .stream
                        .memcpy_dtod(&moe_out.data, &mut o_buf.data)
                        .map_err(|e| anyhow::anyhow!("MoE output D2D into o_buf failed: {e}"))?;
                } else {
                    let mlp = layer.mlp.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("dense layer missing both mlp and moe weights")
                    })?;
                    gemm_batch(&self.ctx, &mlp.gate_proj, &normed, &mut gate_out)?;
                    gemm_batch(&self.ctx, &mlp.up_proj, &normed, &mut up_out)?;
                    silu_mul(&self.ctx, &gate_out, &up_out, &mut act_out)?;
                    gemm_batch(&self.ctx, &mlp.down_proj, &act_out, &mut o_buf)?;
                }
                // Row-parallel down_proj / MoE-combine: sum the partials (no-op single-GPU).
                self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
                add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)
            })?;
            std::mem::swap(&mut hidden, &mut hidden_out);
        }

        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        crate::profile::profile_op(&self.ctx, "lm_head", None, seq_len, || {
            copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
            rms_norm_vec(
                &self.ctx,
                &last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
                &mut last_normed,
            )?;
            let mut logits = DeviceVec::zeros(&self.ctx, self.output_projection().rows)?;
            gemv(
                &self.ctx,
                self.output_projection(),
                &last_normed,
                &mut logits,
            )?;
            Ok(logits)
        })
        .and_then(|logits| {
            crate::profile::profile_op(&self.ctx, "sample", None, seq_len, || {
                sample_cuda_token(&self.ctx, &logits, params, position, penalty)
            })
        })
    }

    /// Batched decode forward (BF16, single-GPU): run `tokens.len() == B`
    /// independent decode rows as ONE forward.
    ///
    /// Mirrors [`Self::forward_tokens`] but over B rows: embed B last-tokens →
    /// (B, hidden); per layer RMSNorm/QKV/RoPE/dense-MLP are plain B-row batched
    /// matmuls (the same kernels prefill uses over seq_len), and the ONE batched
    /// op is the paged attention — `meta` carries per-row `kv_indptr`/page slices
    /// and `positions`, so each row attends ONLY its own slot's KV. Final norm +
    /// LM head + sampling run per row (each row's own params/position), reusing the
    /// single-row tail math so each sampled token is byte-identical to its
    /// equivalent B=1 step. `meta.batch` must equal `B` and `meta.seq_len == 1`.
    pub(crate) fn forward_decode_batch(
        &self,
        tokens: &[u32],
        pool: &mut PagedKVPool,
        meta: &PageMeta,
        params: &[SamplingParams],
        positions: &[u64],
        penalties: &[infer_plan::PenaltyHistory<'_>],
    ) -> Result<Vec<u32>> {
        let batch = tokens.len();
        ensure!(batch >= 1, "forward_decode_batch requires >=1 row");
        ensure!(
            params.len() == batch && positions.len() == batch && penalties.len() == batch,
            "forward_decode_batch length mismatch: {} tokens, {} params, {} positions, {} penalty histories",
            batch,
            params.len(),
            positions.len(),
            penalties.len()
        );
        ensure!(
            meta.batch == batch && meta.seq_len == 1,
            "forward_decode_batch meta (batch {}, seq_len {}) does not match {} decode rows",
            meta.batch,
            meta.seq_len,
            batch
        );
        ensure!(
            self.config.head_dim == 128,
            "R6 clean CUDA path only wires TileLang HD128 kernels, got head_dim={}",
            self.config.head_dim
        );
        ensure!(
            self.config.num_key_value_heads == 8,
            "R6 clean CUDA path only wires kv8 TileLang kernels, got kv_heads={}",
            self.config.num_key_value_heads
        );

        let seq_len = batch; // one query token per decode row
        let hidden_size = self.config.hidden_size;
        let q_dim = self.local_q_heads * self.config.head_dim;
        let kv_dim = self.local_kv_heads * self.config.head_dim;
        let inter = self.local_intermediate_size();

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        crate::profile::profile_op(&self.ctx, "embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)
        })?;

        let mut normed = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut q_batch = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut k_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut v_batch = HiddenStates::zeros(&self.ctx, kv_dim, seq_len)?;
        let mut attn_output = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;
        let mut o_buf = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut hidden_out = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        let mut gate_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut up_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;
        let mut act_out = HiddenStates::zeros(&self.ctx, inter, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    &hidden,
                    &layer.input_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "attention", Some(layer_idx), seq_len, || {
                gemm_batch(&self.ctx, &layer.attention.q_proj, &normed, &mut q_batch)?;
                gemm_batch(&self.ctx, &layer.attention.k_proj, &normed, &mut k_batch)?;
                gemm_batch(&self.ctx, &layer.attention.v_proj, &normed, &mut v_batch)?;
                // Batched paged decode: meta.seq_len == 1 routes to decode_attention,
                // meta.batch == B launches prep + attention over (B, B, 1).
                paged_attention(
                    &self.ctx,
                    layer_idx,
                    pool,
                    &mut q_batch,
                    &mut k_batch,
                    &v_batch,
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    self.config.rms_norm_eps,
                    meta,
                    self.local_q_heads,
                    self.local_kv_heads,
                    self.config.head_dim,
                    &mut attn_output,
                )?;
                gemm_batch(&self.ctx, &layer.attention.o_proj, &attn_output, &mut o_buf)?;
                self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
                add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)
            })?;
            std::mem::swap(&mut hidden, &mut hidden_out);

            crate::profile::profile_op(&self.ctx, "post_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    &hidden,
                    &layer.post_attention_layernorm,
                    self.config.rms_norm_eps,
                    &mut normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "mlp", Some(layer_idx), seq_len, || {
                if let Some(moe) = &layer.moe {
                    let cfg = self.moe_config.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("MoE layer present but model has no moe_config")
                    })?;
                    let split = crate::moe_config::ExpertSplit::single(cfg.num_experts);
                    let moe_out = moe_forward(&self.ctx, moe, &normed, cfg, &split)?;
                    self.ctx
                        .stream
                        .memcpy_dtod(&moe_out.data, &mut o_buf.data)
                        .map_err(|e| anyhow::anyhow!("MoE output D2D into o_buf failed: {e}"))?;
                } else {
                    let mlp = layer.mlp.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("dense layer missing both mlp and moe weights")
                    })?;
                    gemm_batch(&self.ctx, &mlp.gate_proj, &normed, &mut gate_out)?;
                    gemm_batch(&self.ctx, &mlp.up_proj, &normed, &mut up_out)?;
                    silu_mul(&self.ctx, &gate_out, &up_out, &mut act_out)?;
                    gemm_batch(&self.ctx, &mlp.down_proj, &act_out, &mut o_buf)?;
                }
                self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
                add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)
            })?;
            std::mem::swap(&mut hidden, &mut hidden_out);
        }

        // Per-row final norm + LM head + sample. Each row reuses the single-row
        // tail (copy row → rms_norm_vec → gemv → sample), so the sampled token is
        // byte-identical to the equivalent B=1 step.
        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
        let mut out = Vec::with_capacity(batch);
        for row in 0..batch {
            let logits = crate::profile::profile_op(&self.ctx, "lm_head", None, seq_len, || {
                copy_row_to_vec(&self.ctx, &hidden, row, &mut last_hidden)?;
                rms_norm_vec(
                    &self.ctx,
                    &last_hidden,
                    &self.norm,
                    self.config.rms_norm_eps,
                    &mut last_normed,
                )?;
                let mut logits = DeviceVec::zeros(&self.ctx, self.output_projection().rows)?;
                gemv(
                    &self.ctx,
                    self.output_projection(),
                    &last_normed,
                    &mut logits,
                )?;
                Ok(logits)
            })?;
            out.push(crate::profile::profile_op(
                &self.ctx,
                "sample",
                None,
                seq_len,
                || {
                    sample_cuda_token(
                        &self.ctx,
                        &logits,
                        &params[row],
                        positions[row],
                        penalties[row],
                    )
                },
            )?);
        }
        Ok(out)
    }

    /// Captured-graph B=1 decode forward.
    ///
    /// Same per-layer math as [`Self::forward_tokens`] but reads/writes the FIXED
    /// `decode_ctx` buffers (no per-call alloc), so the body is graph-capturable:
    /// only GPU kernels, no host alloc/sync/sampling (the graph ends at
    /// `decode_ctx.logits`). Stage-1 metadata must already be written via
    /// [`DecodeGraphContext::stage1_write`].
    ///
    /// # Errors
    /// Propagates kernel-wrapper errors.
    pub(crate) fn forward_decode_captured(
        &self,
        pool: &mut PagedKVPool,
        decode_ctx: &mut DecodeGraphContext,
    ) -> Result<()> {
        ensure!(
            self.config.head_dim == 128,
            "captured decode only wires TileLang HD128 kernels, got head_dim={}",
            self.config.head_dim
        );
        ensure!(
            self.config.num_key_value_heads == 8,
            "captured decode only wires kv8 TileLang kernels, got kv_heads={}",
            self.config.num_key_value_heads
        );

        ensure!(
            decode_ctx.key().is_some(),
            "forward_decode_captured before stage1_write (metadata not staged)"
        );

        // Borrow the fixed buffers as locals (addresses stable; Stage-1 only
        // overwrote contents). The per-layer hidden/hidden_out swap fixes which
        // baked pointer each kernel uses and is reproduced identically on every
        // capture, so replay matches.
        let DecodeGraphContext {
            ref mut hidden,
            ref mut normed,
            ref mut q_batch,
            ref mut k_batch,
            ref mut v_batch,
            ref mut attn_output,
            ref mut o_buf,
            ref mut hidden_out,
            ref mut gate_out,
            ref mut up_out,
            ref mut act_out,
            ref mut last_hidden,
            ref mut last_normed,
            ref mut logits,
            ref token_ids,
            ref meta,
            ..
        } = *decode_ctx;

        let seq_len = meta.seq_len;

        crate::profile::profile_op(&self.ctx, "embedding", None, seq_len, || {
            embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)
        })?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            crate::profile::profile_op(&self.ctx, "input_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    hidden,
                    &layer.input_layernorm,
                    self.config.rms_norm_eps,
                    normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "attention", Some(layer_idx), seq_len, || {
                gemm_batch(&self.ctx, &layer.attention.q_proj, normed, q_batch)?;
                gemm_batch(&self.ctx, &layer.attention.k_proj, normed, k_batch)?;
                gemm_batch(&self.ctx, &layer.attention.v_proj, normed, v_batch)?;
                paged_attention(
                    &self.ctx,
                    layer_idx,
                    pool,
                    q_batch,
                    k_batch,
                    v_batch,
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    self.config.rms_norm_eps,
                    meta,
                    self.local_q_heads,
                    self.local_kv_heads,
                    self.config.head_dim,
                    attn_output,
                )?;
                gemm_batch(&self.ctx, &layer.attention.o_proj, attn_output, o_buf)?;
                // Always the no-op here (capture only runs single-GPU); kept for
                // parity with the eager forward.
                self.tp.all_reduce_sum(&self.ctx, o_buf)?;
                add_batch(&self.ctx, hidden, o_buf, hidden_out)
            })?;
            std::mem::swap(hidden, hidden_out);

            crate::profile::profile_op(&self.ctx, "post_norm", Some(layer_idx), seq_len, || {
                rms_norm_batch(
                    &self.ctx,
                    hidden,
                    &layer.post_attention_layernorm,
                    self.config.rms_norm_eps,
                    normed,
                )
            })?;
            crate::profile::profile_op(&self.ctx, "mlp", Some(layer_idx), seq_len, || {
                // Dense fast path only: MoE's host-routed `moe_forward` is not
                // graph-capturable, so a MoE layer here is a wiring bug.
                let mlp = layer.mlp.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("captured decode path does not support MoE layers")
                })?;
                gemm_batch(&self.ctx, &mlp.gate_proj, normed, gate_out)?;
                gemm_batch(&self.ctx, &mlp.up_proj, normed, up_out)?;
                silu_mul(&self.ctx, gate_out, up_out, act_out)?;
                gemm_batch(&self.ctx, &mlp.down_proj, act_out, o_buf)?;
                self.tp.all_reduce_sum(&self.ctx, o_buf)?;
                add_batch(&self.ctx, hidden, o_buf, hidden_out)
            })?;
            std::mem::swap(hidden, hidden_out);
        }

        // B=1 decode: the single token row is row 0.
        crate::profile::profile_op(&self.ctx, "lm_head", None, seq_len, || {
            copy_row_to_vec(&self.ctx, hidden, 0, last_hidden)?;
            rms_norm_vec(
                &self.ctx,
                last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
                last_normed,
            )?;
            gemv(&self.ctx, self.output_projection(), last_normed, logits)
        })?;
        Ok(())
    }
}
