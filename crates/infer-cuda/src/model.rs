//! Clean dense-BF16 Qwen3 CUDA forward for the R6 seam port.
//!
//! This is deliberately not a relocation of the legacy `infer/src/model/qwen3`
//! tree. It keeps the tested CUDA kernel calls and the transformer dataflow, but
//! deletes quantized variants, GGUF, TP/NCCL, LoRA, CUDA graphs, spec decode,
//! server/scheduler coupling, and multi-shape dispatch.
//!
//! This file holds the model state (`CudaModel`/`TransformerBlock`/`Attention`/
//! `Mlp`) and the forward dataflow (`forward_tokens`) that drives the layer loop
//! by calling the `ops`/`attention` kernel wrappers. Weight loading lives in
//! `loader`, the op wrappers in `ops`, the attention kernels in `attention`, and
//! the step driver + sampling in `executor`.

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
    /// Tensor-parallel runtime: the resolved rank placement plus its
    /// communicator. For `world_size == 1` this is [`TpRuntime::single`] — the
    /// no-op communicator — so the forward below is byte-identical to the non-TP
    /// path: every `all_reduce_sum` returns immediately and the weights are the
    /// full (unsharded) tensors.
    pub(crate) tp: crate::tp::TpRuntime,
    /// This rank's attention head counts and projection dims, computed at load
    /// time. On a single GPU these equal the global config values; under TP they
    /// are the per-rank shard sizes (Q/KV heads split by [`infer_topo::head_shard`],
    /// dims = `local_heads * head_dim`). The forward sizes its activation buffers
    /// and the attention launch from THESE locals, not the global config.
    pub(crate) local_q_heads: usize,
    pub(crate) local_kv_heads: usize,
    /// MoE router description (MoE-4). `None` for the dense Qwen3 path — the
    /// forward then never touches any MoE branch, so the dense path is
    /// byte-identical. `Some` for a Qwen3.5/3.6 MoE checkpoint, shared by every
    /// sparse layer's [`moe_forward`] router (the per-layer expert weights live
    /// on [`TransformerBlock::moe`]).
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
    /// Dense SwiGLU MLP. `Some` for every dense layer (the byte-identical Qwen3
    /// path always sets this); `None` only on a MoE checkpoint's sparse layer,
    /// where `moe` drives the forward and no dense projections exist in the
    /// checkpoint. Exactly one of `mlp` / `moe` is `Some` per layer.
    pub(crate) mlp: Option<Mlp>,
    /// Per-layer MoE expert weights (MoE-4). `None` ⇒ dense MLP layer (the
    /// byte-identical Qwen3 path and any `mlp_only_layers` on a MoE checkpoint).
    /// `Some` ⇒ sparse layer: the forward runs [`moe_forward`] instead of the
    /// dense gate/up/silu/down block.
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

    /// Logits/vocab dimension (rows of the LM-head projection). Used to size the
    /// fixed `logits` buffer of the captured decode path.
    pub(crate) fn logits_dim(&self) -> usize {
        self.output_projection().rows
    }

    /// This rank's MLP intermediate width. `gate_proj`/`up_proj` are
    /// column-parallel (their output dim is `intermediate_size`), so under TP each
    /// rank holds the column shard; `down_proj` (row-parallel) consumes the same
    /// shard as its input. On a single GPU this is the full `intermediate_size`.
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
        // Per-rank head counts and projection dims (locals equal the global config
        // on a single GPU; under TP they are this rank's head shard). The QKV/O
        // GEMMs run on the sharded weights, so the activation buffers and the
        // attention launch must use the local dims, not the global config.
        let q_dim = self.local_q_heads * self.config.head_dim;
        let kv_dim = self.local_kv_heads * self.config.head_dim;
        let inter = self.local_intermediate_size();

        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let token_ids = upload_i32(&self.ctx, &token_ids)?;
        let mut hidden = HiddenStates::zeros(&self.ctx, hidden_size, seq_len)?;
        embedding_batch(&self.ctx, &self.embed_tokens, &token_ids, &mut hidden)?;

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
            rms_norm_batch(
                &self.ctx,
                &hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut normed,
            )?;
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
            // Row-parallel: each rank produced a partial o_proj output over its
            // attention-head shard; sum across ranks before the residual add.
            // No-op on a single GPU.
            self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
            add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);

            rms_norm_batch(
                &self.ctx,
                &hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut normed,
            )?;
            if let Some(moe) = &layer.moe {
                // Sparse layer (MoE-4): route → grouped experts → combine →
                // shared expert. Produces the block output into `o_buf`.
                let cfg = self.moe_config.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("MoE layer present but model has no moe_config")
                })?;
                let moe_out = moe_forward(&self.ctx, moe, &normed, cfg)?;
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
            // Row-parallel: each rank produced a partial down_proj / MoE-combine
            // output over its intermediate / expert shard; sum across ranks before
            // the residual add. No-op on a single GPU.
            self.tp.all_reduce_sum(&self.ctx, &mut o_buf)?;
            add_batch(&self.ctx, &hidden, &o_buf, &mut hidden_out)?;
            std::mem::swap(&mut hidden, &mut hidden_out);
        }

        let mut last_hidden = DeviceVec::zeros(&self.ctx, hidden_size)?;
        copy_row_to_vec(&self.ctx, &hidden, seq_len - 1, &mut last_hidden)?;
        let mut last_normed = DeviceVec::zeros(&self.ctx, hidden_size)?;
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
        sample_cuda_token(&self.ctx, &logits, params, position)
    }

    /// Captured-graph B=1 decode forward (CG-2, ADDITIVE).
    ///
    /// Runs the **same per-layer math** as [`Self::forward_tokens`] but reads and
    /// writes the FIXED buffers in `decode_ctx` instead of allocating fresh
    /// `HiddenStates`/`DeviceVec`/`PageMeta` each call. This is the body the warmup
    /// path records into a CUDA graph and the submit path replays: it contains
    /// **only GPU kernels** — no host allocation, no host/device sync, no sampling
    /// (sampling stays outside the graph, design §7; the graph ends at
    /// `decode_ctx.logits`).
    ///
    /// Stage-1 metadata (token id, positions, page table) must already be written
    /// into `decode_ctx` via [`DecodeGraphContext::stage1_write`] before calling
    /// this. `forward_tokens` (the numerically-verified eager path) is intentionally
    /// left untouched — this is a parallel fast path that produces the same logits.
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

        // Borrow the fixed buffers as locals so the layer loop reads like the eager
        // path. `meta` is borrowed immutably (its CudaSlice buffers stay at fixed
        // addresses — Stage-1 only overwrote their contents). `hidden`/`hidden_out`
        // are swapped each layer exactly as in the eager forward; that host-side swap
        // fixes which baked pointer each layer's kernel uses, and it is reproduced
        // identically on every capture, so replay matches.
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

        embedding_batch(&self.ctx, &self.embed_tokens, token_ids, hidden)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rms_norm_batch(
                &self.ctx,
                hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                normed,
            )?;
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
            // Row-parallel all-reduce (no-op on a single GPU). The captured-graph
            // path only runs when TP is single (multi-rank disables capture in
            // executor.rs because NCCL is not graph-capturable), so this call is
            // always the no-op here; it stays for parity with the eager forward.
            self.tp.all_reduce_sum(&self.ctx, o_buf)?;
            add_batch(&self.ctx, hidden, o_buf, hidden_out)?;
            std::mem::swap(hidden, hidden_out);

            rms_norm_batch(
                &self.ctx,
                hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                normed,
            )?;
            // The captured B=1 decode graph is the dense Qwen3 fast path only;
            // a MoE checkpoint's host-routed `moe_forward` is not graph-capturable
            // (it syncs + reads logits on the host each step). MoE layers run the
            // eager `forward_tokens` path, so a MoE layer here is a wiring bug.
            let mlp = layer.mlp.as_ref().ok_or_else(|| {
                anyhow::anyhow!("captured decode path does not support MoE layers")
            })?;
            gemm_batch(&self.ctx, &mlp.gate_proj, normed, gate_out)?;
            gemm_batch(&self.ctx, &mlp.up_proj, normed, up_out)?;
            silu_mul(&self.ctx, gate_out, up_out, act_out)?;
            gemm_batch(&self.ctx, &mlp.down_proj, act_out, o_buf)?;
            // Row-parallel all-reduce (no-op on a single GPU; see o_proj note above).
            self.tp.all_reduce_sum(&self.ctx, o_buf)?;
            add_batch(&self.ctx, hidden, o_buf, hidden_out)?;
            std::mem::swap(hidden, hidden_out);
        }

        // B=1 decode: the single token row is row 0.
        copy_row_to_vec(&self.ctx, hidden, 0, last_hidden)?;
        rms_norm_vec(
            &self.ctx,
            last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
            last_normed,
        )?;
        gemv(&self.ctx, self.output_projection(), last_normed, logits)?;
        Ok(())
    }
}
