//! Diagnostic-only Qwen3.5 stage capture for train-vs-infer parity checks.

use anyhow::Result;

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
        let lm_head =
            ops::linear(&self.ctx, &final_norm, &self.embed_tokens)?.with_label("parity_lm_head");

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
