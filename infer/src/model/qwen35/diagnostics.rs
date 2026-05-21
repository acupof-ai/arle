//! Diagnostic-only Qwen3.5 stage capture for train-vs-infer parity checks.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;

use super::{
    prefill_buffers::GdrChunkwiseScratch35,
    recurrent_state::RecurrentState,
    weights::{LayerKind, Qwen35Model},
};
use crate::{model::common, model::kv_cache::KVCache, ops};
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

#[doc(hidden)]
pub struct Qwen35InferParityStages {
    pub embedding: DeviceVec,
    pub layer0_rmsnorm: DeviceVec,
    pub layer0_attention: DeviceVec,
    pub layer0_ffn: DeviceVec,
    pub layer0_residual: DeviceVec,
    pub final_rmsnorm: DeviceVec,
    pub lm_head: DeviceVec,
}

#[doc(hidden)]
pub struct Qwen35DenseModuleParityOutputs {
    pub embedding: DeviceVec,
    pub final_rmsnorm: DeviceVec,
    pub lm_head: DeviceVec,
}

#[doc(hidden)]
pub struct Qwen35LinearAttentionDiagnosticTensor {
    pub name: &'static str,
    pub values: Vec<f32>,
}

impl Qwen35Model {
    #[doc(hidden)]
    pub fn parity_device_context(&self) -> DeviceContext {
        self.ctx.clone()
    }

    #[doc(hidden)]
    pub fn forward_single_token_parity_stages(
        &self,
        token_id: u32,
    ) -> Result<Qwen35InferParityStages> {
        anyhow::ensure!(
            !self.layers.is_empty(),
            "infer parity diagnostics require at least one layer"
        );

        let c = &self.config;
        let seq_len = 1usize;
        let token_ids = [token_id];
        let mut kv_cache = KVCache::new(c.num_full_attention_layers(), c.num_key_value_heads);
        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;
        let mut recurrent = RecurrentState::new(&self.ctx, c)?;
        let mut scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        let hidden0 =
            common::get_embeddings_batch(&self.ctx, &self.embed_tokens, &token_ids, c.hidden_size)?;
        let embedding = copy_hidden(&self.ctx, &hidden0, "parity_embedding")?;

        let layer0 = &self.layers[0];
        let layer0_norm =
            self.batched_rms_norm_offset(&hidden0, &layer0.input_layernorm, c.rms_norm_eps)?;
        let layer0_rmsnorm = copy_hidden(&self.ctx, &layer0_norm, "parity_layer0_rmsnorm")?;

        let attn_out_dim = match &layer0.attn {
            LayerKind::FullAttention(_) => c.full_attn_q_dim(),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };
        let mut layer0_attention_batch = match &layer0.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &layer0_norm,
                &mut full_idx,
                &mut kv_cache,
                attn_out_dim,
                seq_len,
            )?,
            LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                attn,
                &layer0_norm,
                &mut linear_idx,
                &mut recurrent,
                &mut scratch,
                seq_len,
            )?,
        };
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut layer0_attention_batch)?;
        let layer0_attention = copy_hidden(
            &self.ctx,
            &layer0_attention_batch,
            "parity_layer0_attention",
        )?;

        let hidden_plus_attn = ops::add_batch(&self.ctx, &hidden0, &layer0_attention_batch)?;
        let post_attention_norm = self.batched_rms_norm_offset(
            &hidden_plus_attn,
            &layer0.post_attention_layernorm,
            c.rms_norm_eps,
        )?;
        let gate_out = ops::gemm(&self.ctx, &layer0.mlp.gate_proj, &post_attention_norm)?;
        let up_out = ops::gemm(&self.ctx, &layer0.mlp.up_proj, &post_attention_norm)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mut mlp_out = ops::gemm(&self.ctx, &layer0.mlp.down_proj, &act_out)?;
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut mlp_out)?;
        let layer0_ffn = copy_hidden(&self.ctx, &mlp_out, "parity_layer0_ffn")?;

        let layer0_final = ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)?;
        let layer0_residual = copy_hidden(&self.ctx, &layer0_final, "parity_layer0_residual")?;

        let mut hidden = layer0_final;
        for (layer_idx, layer) in self.layers.iter().enumerate().skip(1) {
            hidden = self.prefill_layer(
                layer_idx,
                layer,
                &hidden,
                &mut scratch,
                &mut linear_idx,
                &mut full_idx,
                &mut kv_cache,
                &mut recurrent,
            )?;
        }

        let last_hidden = ops::extract_vec(&self.ctx, &hidden, hidden.seq_len - 1)?;
        let mut final_norm =
            DeviceVec::zeros(&self.ctx, last_hidden.len)?.with_label("parity_final_rmsnorm");
        ops::rms_norm_offset_into(
            &self.ctx,
            &last_hidden,
            &self.norm,
            c.rms_norm_eps,
            &mut final_norm,
        )?;
        let lm_head = ops::linear(&self.ctx, &final_norm, self.output_projection())?
            .with_label("parity_lm_head");

        Ok(Qwen35InferParityStages {
            embedding,
            layer0_rmsnorm,
            layer0_attention,
            layer0_ffn,
            layer0_residual,
            final_rmsnorm: final_norm,
            lm_head,
        })
    }

    #[doc(hidden)]
    pub fn dense_module_parity_outputs(
        &self,
        token_id: u32,
    ) -> Result<Qwen35DenseModuleParityOutputs> {
        let c = &self.config;
        let token_ids = [token_id];
        let embedding =
            common::get_embeddings_batch(&self.ctx, &self.embed_tokens, &token_ids, c.hidden_size)?;
        let embedding = copy_hidden(&self.ctx, &embedding, "dense_parity_embedding")?;

        let norm_input = deterministic_bf16_vec(c.hidden_size, 17);
        let norm_input = DeviceVec::from_host(&self.ctx, &norm_input)?;
        let mut final_rmsnorm =
            DeviceVec::zeros(&self.ctx, c.hidden_size)?.with_label("dense_parity_final_rmsnorm");
        ops::rms_norm_offset_into(
            &self.ctx,
            &norm_input,
            &self.norm,
            c.rms_norm_eps,
            &mut final_rmsnorm,
        )?;

        let lm_head_input = deterministic_bf16_vec(c.hidden_size, 29);
        let lm_head_input = DeviceVec::from_host(&self.ctx, &lm_head_input)?;
        let lm_head = ops::linear(&self.ctx, &lm_head_input, self.output_projection())?
            .with_label("dense_parity_lm_head");

        Ok(Qwen35DenseModuleParityOutputs {
            embedding,
            final_rmsnorm,
            lm_head,
        })
    }

    #[doc(hidden)]
    pub fn layer0_linear_attention_diagnostic_tensors(
        &self,
        token_id: u32,
    ) -> Result<Vec<Qwen35LinearAttentionDiagnosticTensor>> {
        anyhow::ensure!(
            !self.layers.is_empty(),
            "infer linear-attn diagnostics require at least one layer"
        );

        let c = &self.config;
        let seq_len = 1usize;
        let token_ids = [token_id];
        let mut recurrent = RecurrentState::new(&self.ctx, c)?;
        let mut scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;
        let layer0 = &self.layers[0];
        let attn = match &layer0.attn {
            LayerKind::LinearAttention(attn) => attn,
            LayerKind::FullAttention(_) => anyhow::bail!(
                "layer 0 is full_attention, expected linear_attention for diagnostics"
            ),
        };

        let mut tensors = Vec::new();
        push_hidden_summary(
            &mut tensors,
            "embedding",
            hidden_to_f32(
                &self.ctx,
                &common::get_embeddings_batch(
                    &self.ctx,
                    &self.embed_tokens,
                    &token_ids,
                    c.hidden_size,
                )?,
            )?,
        );

        let hidden0 =
            common::get_embeddings_batch(&self.ctx, &self.embed_tokens, &token_ids, c.hidden_size)?;
        let layer0_norm =
            self.batched_rms_norm_offset(&hidden0, &layer0.input_layernorm, c.rms_norm_eps)?;
        push_hidden_summary(
            &mut tensors,
            "input_layernorm",
            hidden_to_f32(&self.ctx, &layer0_norm)?,
        );

        push_hidden_summary(
            &mut tensors,
            "dt_bias_weight",
            device_vec_to_f32(&self.ctx, &attn.dt_bias)?,
        );
        push_hidden_summary(
            &mut tensors,
            "a_log_weight",
            cuda_f32_to_host(&self.ctx, &attn.a_log)?,
        );
        push_hidden_summary(
            &mut tensors,
            "norm_weight",
            cuda_f32_to_host(&self.ctx, &attn.norm_weight)?,
        );

        let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, &layer0_norm)?;
        push_hidden_summary(
            &mut tensors,
            "in_proj_qkv",
            hidden_to_f32(&self.ctx, &qkv_batch)?,
        );
        let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, &layer0_norm)?;
        push_hidden_summary(
            &mut tensors,
            "in_proj_z",
            hidden_to_f32(&self.ctx, &z_batch)?,
        );
        let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, &layer0_norm)?;
        push_hidden_summary(
            &mut tensors,
            "in_proj_b",
            hidden_to_f32(&self.ctx, &b_batch)?,
        );
        let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, &layer0_norm)?;
        push_hidden_summary(
            &mut tensors,
            "in_proj_a",
            hidden_to_f32(&self.ctx, &a_batch)?,
        );

        let mut qkv_conv_batch = HiddenStates::zeros(&self.ctx, c.linear_attn_qkv_dim(), seq_len)?;
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &qkv_batch,
            &attn.conv1d_weight,
            &mut recurrent.layers[0].conv_state,
            &mut qkv_conv_batch,
            c.linear_conv_kernel_dim,
        );
        push_hidden_summary(
            &mut tensors,
            "conv1d_silu_qkv",
            hidden_to_f32(&self.ctx, &qkv_conv_batch)?,
        );

        let mut gdr_out_batch = HiddenStates::zeros(&self.ctx, c.linear_attn_z_dim(), seq_len)?;
        ops::gated_delta_rule_prefill_chunkwise_into(
            &self.ctx,
            &qkv_conv_batch,
            &b_batch,
            &a_batch,
            &ops::GdrWeights {
                dt_bias: &attn.dt_bias,
                a_log: &attn.a_log,
            },
            &mut recurrent.layers[0].state,
            &mut scratch,
            &mut gdr_out_batch,
            &ops::GdrHeadConfig {
                num_key_heads: c.linear_num_key_heads,
                num_value_heads: c.linear_num_value_heads,
                key_dim: c.linear_key_head_dim,
                val_dim: c.linear_value_head_dim,
            },
        )?;
        push_hidden_summary(
            &mut tensors,
            "gdr_q_expanded",
            hidden_to_f32(&self.ctx, &scratch.q_expanded)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_k_expanded",
            hidden_to_f32(&self.ctx, &scratch.k_expanded)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_v_raw",
            hidden_to_f32(&self.ctx, &scratch.v_raw)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_g_cumsum",
            cuda_f32_to_host(&self.ctx, &scratch.g_cumsum)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_beta",
            cuda_f32_to_host(&self.ctx, &scratch.beta)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_a_tril",
            cuda_f32_to_host(&self.ctx, &scratch.a_tril)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_a_inv",
            cuda_bf16_to_f32(&self.ctx, &scratch.a_inv)?,
        );
        push_hidden_summary(&mut tensors, "gdr_w", hidden_to_f32(&self.ctx, &scratch.w)?);
        push_hidden_summary(&mut tensors, "gdr_u", hidden_to_f32(&self.ctx, &scratch.u)?);
        push_hidden_summary(
            &mut tensors,
            "gdr_chunk_state",
            cuda_f32_to_host(&self.ctx, &scratch.chunk_state)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_v_new",
            hidden_to_f32(&self.ctx, &scratch.v_new)?,
        );
        push_hidden_summary(
            &mut tensors,
            "gdr_output",
            hidden_to_f32(&self.ctx, &gdr_out_batch)?,
        );

        let mut normed_out_batch = HiddenStates::zeros(&self.ctx, c.linear_attn_z_dim(), seq_len)?;
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
        push_hidden_summary(
            &mut tensors,
            "rms_norm_gated",
            hidden_to_f32(&self.ctx, &normed_out_batch)?,
        );

        let output = ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)?;
        push_hidden_summary(&mut tensors, "out_proj", hidden_to_f32(&self.ctx, &output)?);

        Ok(tensors)
    }
}

fn push_hidden_summary(
    out: &mut Vec<Qwen35LinearAttentionDiagnosticTensor>,
    name: &'static str,
    values: Vec<f32>,
) {
    out.push(Qwen35LinearAttentionDiagnosticTensor { name, values });
}

fn copy_hidden(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    label: &'static str,
) -> Result<DeviceVec> {
    let len = hidden.hidden_dim * hidden.seq_len;
    let mut out = DeviceVec::zeros(ctx, len)?.with_label(label);
    ctx.stream
        .memcpy_dtod(&hidden.data, &mut out.data)
        .map_err(|e| anyhow::anyhow!("D2D copy for {label} failed: {e}"))?;
    Ok(out)
}

fn hidden_to_f32(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<f32>> {
    let host = ctx.stream.clone_dtoh(&hidden.data)?;
    Ok(host.into_iter().map(f32::from).collect())
}

fn device_vec_to_f32(ctx: &DeviceContext, vec: &DeviceVec) -> Result<Vec<f32>> {
    let host = ctx.stream.clone_dtoh(&vec.data)?;
    Ok(host.into_iter().map(f32::from).collect())
}

fn cuda_f32_to_host(ctx: &DeviceContext, slice: &CudaSlice<f32>) -> Result<Vec<f32>> {
    Ok(ctx.stream.clone_dtoh(slice)?)
}

fn cuda_bf16_to_f32(ctx: &DeviceContext, slice: &CudaSlice<bf16>) -> Result<Vec<f32>> {
    let host = ctx.stream.clone_dtoh(slice)?;
    Ok(host.into_iter().map(f32::from).collect())
}

fn deterministic_bf16_vec(len: usize, salt: usize) -> Vec<bf16> {
    (0..len)
        .map(|idx| {
            let raw = ((idx.wrapping_mul(37).wrapping_add(salt.wrapping_mul(17))) % 257) as f32;
            bf16::from_f32((raw - 128.0) / 64.0)
        })
        .collect()
}
