use anyhow::Result;
use log::{debug, info};
use std::time::Instant;

use super::config::Config;
use crate::model::common::{self, MLP};
use crate::ops;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use crate::weight_loader::{
    load_tensor_1d, load_tensor_2d, load_tensor_2d_maybe_quantized, precompute_rope,
};

#[derive(Clone, Copy, Debug)]
pub struct ModelRuntimeConfig {
    pub enable_cuda_graph: bool,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

/// Attention layer weights
pub(super) struct Attention {
    pub(super) q_proj: DeviceMatrix,
    pub(super) k_proj: DeviceMatrix,
    pub(super) v_proj: DeviceMatrix,
    /// Merged QKV projection: [q_dim + 2*kv_dim, hidden_dim]
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
}

/// Transformer block
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: common::MLP,
}

/// Qwen3 model — weights and config only. Mutable state lives in `Qwen3State`.
pub struct Qwen3Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
}

impl Qwen3Model {
    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config::from_file(model_path)?;

        let (mmaps, weight_map) = common::load_safetensors(model_path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;

        // Detect weight quantization (quantize_config.json or turboquant_config.json)
        let quant_group_size = {
            let qc_path = std::path::Path::new(model_path).join("quantize_config.json");
            let tq_path = std::path::Path::new(model_path).join("turboquant_config.json");
            if qc_path.exists() {
                let qc: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&qc_path)?)?;
                let gs = qc["group_size"].as_u64().unwrap_or(128) as usize;
                let bits = qc["bits"].as_u64().unwrap_or(8);
                info!("Quantized model detected: W{}A16, group_size={}", bits, gs);
                gs
            } else if tq_path.exists() {
                let tq: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&tq_path)?)?;
                let gs = tq["group_size"].as_u64().unwrap_or(128) as usize;
                let bits = tq["bits"].as_u64().unwrap_or(3);
                info!("TurboQuant model detected: TQ{}, group_size={}", bits, gs);
                gs
            } else {
                0 // not quantized
            }
        };

        // Helper: load linear weight, quantized if available
        let load_linear = |name: &str| -> Result<DeviceMatrix> {
            if quant_group_size > 0 {
                load_tensor_2d_maybe_quantized(&ctx, &shards, &weight_map, name, quant_group_size)
            } else {
                load_tensor_2d(&ctx, &shards, &weight_map, name)
            }
        };

        let t_gpu = Instant::now();
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;
        let lm_head = if config.tie_word_embeddings {
            debug!("Using tied input/output embeddings");
            None
        } else {
            debug!("Loading untied LM head to GPU");
            Some(load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                config.lm_head_tensor_name(),
            )?)
        };

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: {
                    let q_proj = load_linear(&format!("{}.self_attn.q_proj.weight", prefix))?;
                    let k_proj = load_linear(&format!("{}.self_attn.k_proj.weight", prefix))?;
                    let v_proj = load_linear(&format!("{}.self_attn.v_proj.weight", prefix))?;
                    let qkv_proj = if q_proj.is_quantized() {
                        // Can't concat quantized matrices — use q_proj as placeholder.
                        // Batched decode uses individual Q/K/V projections anyway.
                        // TODO: merged quantized QKV for prefill
                        DeviceMatrix::concat_rows(&ctx, &[&q_proj, &k_proj, &v_proj])?
                    } else {
                        DeviceMatrix::concat_rows(&ctx, &[&q_proj, &k_proj, &v_proj])?
                    };
                    Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        qkv_proj,
                        o_proj: load_linear(&format!("{}.self_attn.o_proj.weight", prefix))?,
                        q_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.self_attn.q_norm.weight", prefix),
                        )?,
                        k_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.self_attn.k_norm.weight", prefix),
                        )?,
                    }
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP::load_with_quant(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp", prefix),
                    true, // merge gate+up for batched decode
                    quant_group_size,
                )?,
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "model.norm.weight")?;

        debug!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("GPU model loaded successfully");

        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
        };

        if model.enable_cuda_graph {
            debug!("Preloading decode-path Triton kernels before CUDA Graph capture");
            model.preload_decode_triton_kernels()?;
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    fn preload_decode_triton_kernels(&self) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let cache_len = self.config.num_key_value_heads * 4096 * self.config.head_dim;
        let dummy_token_id = 0_i32;
        let dummy_pos = 0_i32;
        let dummy_seq_len = 1_i32;

        let decode_meta = self
            .ctx
            .stream
            .clone_htod(&[dummy_token_id, dummy_pos, dummy_seq_len])
            .map_err(|e| anyhow::anyhow!("Preload decode_meta H2D failed: {}", e))?;
        let mut embed_out = DeviceVec::zeros(&self.ctx, hidden_size)?;
        ops::embedding_decode_into(&self.ctx, &self.embed_tokens, &decode_meta, &mut embed_out)?;

        let layer0 = &self.layers[0];
        let q = DeviceVec::zeros(&self.ctx, q_dim)?;
        let k = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let v = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let mut k_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut v_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut out = DeviceVec::zeros(&self.ctx, q_dim)?;

        let num_qheads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let num_kv_splits = 4usize;
        let mut partial_out = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits * head_dim)
            .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {}", e))?;
        let mut partial_m = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {}", e))?;
        let mut partial_l = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {}", e))?;

        ops::fused_attention_decode_into(
            &self.ctx,
            &q,
            &k,
            &v,
            &layer0.attention.q_norm,
            &layer0.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &decode_meta,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            &mut partial_out,
            &mut partial_m,
            &mut partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;

        self.ctx.sync()?;
        Ok(())
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        common::output_projection(&self.lm_head, &self.embed_tokens)
    }
}
