use anyhow::Result;
use log::{debug, info};
use std::time::Instant;

use super::config::Config;
use crate::model::common::{self, MLP};
use crate::ops;
use crate::weight_loader::{
    load_tensor_1d, load_tensor_2d, precompute_rope, resolve_rope_cache_len,
};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

/// Attention layer weights for GLM-4.
///
/// Key differences from Qwen3:
/// - Merged QKV weight: single `query_key_value` instead of separate q/k/v
/// - QKV bias present (Qwen3 has no bias)
/// - No Q/K normalization weights
pub(super) struct Attention {
    /// Separate Q projection, split from merged QKV. Used by prefill path.
    pub(super) q_proj: DeviceMatrix,
    /// Separate K projection, split from merged QKV. Used by prefill path.
    pub(super) k_proj: DeviceMatrix,
    /// Separate V projection, split from merged QKV. Used by prefill path.
    pub(super) v_proj: DeviceMatrix,
    /// Merged QKV projection: [q_dim + 2*kv_dim, hidden_dim]. Used by batched decode.
    pub(super) qkv_proj: DeviceMatrix,
    /// Merged QKV bias: [q_dim + 2*kv_dim]. Used by batched decode.
    pub(super) qkv_bias: DeviceVec,
    /// Split Q bias: [q_dim]. Used by single-token decode.
    pub(super) q_bias: DeviceVec,
    /// Split K bias: [kv_dim]. Used by single-token decode.
    pub(super) k_bias: DeviceVec,
    /// Split V bias: [kv_dim]. Used by single-token decode.
    pub(super) v_bias: DeviceVec,
    /// Output projection.
    pub(super) o_proj: DeviceMatrix,
}

/// Transformer block for GLM-4.
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP,
}

/// GLM-4 model -- weights and config only. Mutable state lives in `GLM4State`.
pub struct GLM4Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) output_layer: DeviceMatrix,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
}

impl GLM4Model {
    pub fn from_safetensors(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        info!("Loading GLM-4 model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config::from_file(model_path)?;

        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads();
        let head_dim = config.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let (mmaps, weight_map) = common::load_safetensors(model_path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();

        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            "transformer.embedding.word_embeddings.weight",
        )?;

        debug!("Loading output layer to GPU");
        let output_layer = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            "transformer.output_layer.weight",
        )?;

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers()
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let prefix = format!("transformer.encoder.layers.{}", i);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: {
                    // GLM-4 stores merged QKV: [q_dim + 2*kv_dim, hidden_size]
                    let qkv_proj = load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attention.query_key_value.weight", prefix),
                    )?;
                    let qkv_bias = load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attention.query_key_value.bias", prefix),
                    )?;

                    // Split merged QKV into separate projections for the prefill path.
                    let q_proj = DeviceMatrix::slice_rows(&ctx, &qkv_proj, 0, q_dim)?;
                    let k_proj = DeviceMatrix::slice_rows(&ctx, &qkv_proj, q_dim, q_dim + kv_dim)?;
                    let v_proj = DeviceMatrix::slice_rows(
                        &ctx,
                        &qkv_proj,
                        q_dim + kv_dim,
                        q_dim + 2 * kv_dim,
                    )?;

                    // Split merged QKV bias for decode path.
                    let q_bias = DeviceVec::slice_to_vec(&ctx, &qkv_bias, 0, q_dim)?;
                    let k_bias = DeviceVec::slice_to_vec(&ctx, &qkv_bias, q_dim, q_dim + kv_dim)?;
                    let v_bias = DeviceVec::slice_to_vec(
                        &ctx,
                        &qkv_bias,
                        q_dim + kv_dim,
                        q_dim + 2 * kv_dim,
                    )?;

                    let o_proj = load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attention.dense.weight", prefix),
                    )?;

                    Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        qkv_proj,
                        qkv_bias,
                        q_bias,
                        k_bias,
                        v_bias,
                        o_proj,
                    }
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: {
                    // GLM-4 stores merged gate+up: dense_h_to_4h = [2*inter, hidden]
                    let gate_up_merged = load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.dense_h_to_4h.weight", prefix),
                    )?;
                    let inter = config.intermediate_size();
                    let gate_proj = DeviceMatrix::slice_rows(&ctx, &gate_up_merged, 0, inter)?;
                    let up_proj =
                        DeviceMatrix::slice_rows(&ctx, &gate_up_merged, inter, 2 * inter)?;
                    let down_proj = load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.dense_4h_to_h.weight", prefix),
                    )?;
                    MLP {
                        gate_proj,
                        up_proj,
                        gate_up_proj: Some(gate_up_merged),
                        down_proj,
                    }
                },
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(
            &ctx,
            &shards,
            &weight_map,
            "transformer.encoder.final_layernorm.weight",
        )?;

        debug!("Precomputing RoPE cache on GPU");
        let rope_cache_len = resolve_rope_cache_len(config.rope_cache_len_hint());
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim(), rope_cache_len, config.rope_theta())?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("GLM-4 GPU model loaded successfully");

        let model = Self {
            ctx,
            config,
            embed_tokens,
            output_layer,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph,
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
        let q_dim = self.config.num_attention_heads * self.config.head_dim();
        let kv_dim = self.config.num_key_value_heads() * self.config.head_dim();
        let cache_len = self.config.num_key_value_heads() * 4096 * self.config.head_dim();
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

        let q = DeviceVec::zeros(&self.ctx, q_dim)?;
        let k = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let v = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let mut k_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut v_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut out = DeviceVec::zeros(&self.ctx, q_dim)?;

        let num_qheads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();
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

        // GLM-4 has no Q/K norm — pass dummy norm weights that are all ones.
        // The fused_attention_decode_into kernel will apply identity normalization.
        let dummy_norm = DeviceVec::ones(&self.ctx, head_dim)?;

        ops::fused_attention_decode_into(
            &self.ctx,
            &q,
            &k,
            &v,
            &dummy_norm,
            &dummy_norm,
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
            self.config.num_key_value_heads(),
        )?;

        // Also preload the bias-add kernel path.
        let _dummy_bias_out = DeviceVec::zeros(&self.ctx, q_dim + 2 * kv_dim)?;

        self.ctx.sync()?;
        Ok(())
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        &self.output_layer
    }
}
