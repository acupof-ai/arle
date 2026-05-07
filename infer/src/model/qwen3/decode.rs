use anyhow::Result;

use super::decode_buffers::DecodeBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::KVCache;
use crate::ops::{self, OpsBackend};

impl Qwen3Model {
    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    pub(super) fn decode_one_token(
        &self,
        token_id: u32,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
        graph_state: &mut CudaGraphState,
    ) -> Result<()> {
        let pos = kv_cache.len();
        let seq_len = pos + 1;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, seq_len as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        // Respect the shared model-level graph gate so single-token decode,
        // batched decode, and scheduler warmup stay on the same contract.
        let use_graph = <Self as crate::model::ModelForward>::supports_cuda_graph_decode(self);
        if use_graph {
            graph_state.run_or_capture(&self.ctx, || self.decode_kernels(kv_cache, bufs))?;
        } else {
            self.decode_kernels(kv_cache, bufs)?;
        }

        kv_cache.increment_seq_len();
        Ok(())
    }

    fn decode_kernels(&self, kv_cache: &mut KVCache, bufs: &mut DecodeBuffers) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);

        ops_backend.embedding_decode_into(
            &self.embed_tokens,
            &bufs.decode_meta,
            &mut bufs.hidden,
        )?;

        ops_backend.rms_norm_into(
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_layer_inner(layer_idx, layer, kv_cache, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            ops_backend.fused_add_rms_norm_into(
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            )?;
        }

        ops_backend.linear_vec_into(self.output_projection(), &bufs.normed, &mut bufs.logits)?;

        Ok(())
    }

    fn decode_layer_inner(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let ops_backend = ops::CudaOpsBackend::new(&self.ctx);

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        ops_backend.linear_vec_into(&layer.attention.q_proj, &bufs.normed, &mut bufs.q)?;
        ops_backend.linear_vec_into(&layer.attention.k_proj, &bufs.normed, &mut bufs.k)?;
        ops_backend.linear_vec_into(&layer.attention.v_proj, &bufs.normed, &mut bufs.v)?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemv_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemv_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemv_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v)?;
            }
        }

        let pos = kv_cache.len();
        let (k_cache, v_cache) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        ops::fused_attention_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.decode_meta,
            k_cache,
            v_cache,
            &mut bufs.attn_out,
            &mut bufs.partial_out,
            &mut bufs.partial_m,
            &mut bufs.partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;
        // Quantize the newly written decode token → INT8 (no-op for BF16)
        kv_cache.commit_layer(&self.ctx, layer_idx, pos, 1)?;

        ops_backend.linear_vec_into(
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        )?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemv_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_out,
                    &mut bufs.attn_proj,
                )?;
            }
        }
        self.layer_communicator
            .post_attn_all_reduce_device_vec(&mut bufs.attn_proj)?;

        ops_backend.fused_add_rms_norm_into(
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        let mlp_lora = self.layer_lora(layer_idx);
        let has_gate_up_lora =
            mlp_lora.is_some_and(|ll| ll.gate_proj.is_some() || ll.up_proj.is_some());
        if let Some(gate_up_proj) = layer.mlp.fused_gate_up() {
            anyhow::ensure!(
                !has_gate_up_lora,
                "Qwen3 fused gate_up MLP cannot apply gate/up LoRA in decode; \
                 set INFER_QWEN3_FUSED_GATE_UP=0 before loading the model"
            );
            ops::fused_mlp_gate_up_into(
                &self.ctx,
                &bufs.normed,
                gate_up_proj,
                &layer.mlp.down_proj,
                &mut bufs.mlp_act,
                &mut bufs.mlp_out,
            )?;
            if let Some(ad) = mlp_lora.and_then(|ll| ll.down_proj.as_ref()) {
                ops::apply_lora_gemv_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.mlp_act,
                    &mut bufs.mlp_out,
                )?;
            }
        } else if let Some(ll) = mlp_lora
            .filter(|ll| ll.gate_proj.is_some() || ll.up_proj.is_some() || ll.down_proj.is_some())
        {
            let (gate_proj, up_proj) = layer
                .mlp
                .separate_gate_up()
                .expect("separate Qwen3 MLP must carry gate/up weights");
            ops::mlp_decode_with_lora_into(
                &self.ctx,
                &bufs.normed,
                gate_proj,
                up_proj,
                &layer.mlp.down_proj,
                ll.gate_proj.as_ref().map(|ad| (&ad.a, &ad.b)),
                ll.up_proj.as_ref().map(|ad| (&ad.a, &ad.b)),
                ll.down_proj.as_ref().map(|ad| (&ad.a, &ad.b)),
                &mut bufs.mlp_act,
                &mut bufs.mlp_up_scratch,
                &mut bufs.mlp_out,
            )?;
        } else {
            let (gate_proj, up_proj) = layer
                .mlp
                .separate_gate_up()
                .expect("non-fused Qwen3 MLP must carry separate gate/up weights");
            ops_backend.fused_mlp_into(
                &bufs.normed,
                gate_proj,
                up_proj,
                &layer.mlp.down_proj,
                &mut bufs.mlp_act,
                &mut bufs.mlp_out,
            )?;
        }
        self.layer_communicator
            .post_mlp_all_reduce_device_vec(&mut bufs.mlp_out)?;

        Ok(())
    }
}
