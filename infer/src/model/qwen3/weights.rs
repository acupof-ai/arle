use anyhow::{Context, Result};
use log::{debug, info};
use std::time::Instant;

use super::config::Config;
use crate::model::common::{self, MLP};
use crate::model_source::ResolvedModelSource;
use crate::ops;
use crate::weight_loader::{
    QuantLoadConfig, load_tensor_1d, load_tensor_2d, load_tensor_2d_maybe_quantized_with_config,
    precompute_rope, resolve_rope_cache_len,
};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

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
    /// Optional PEFT LoRA bundle. `None` = no adapter, forward uses base
    /// weights verbatim. When `Some`, every projection site in prefill /
    /// decode / batch_decode checks `lora.layers[layer_idx].<module>` and
    /// adds the LoRA delta on top via `ops::apply_lora_{gemv,gemm}_add`.
    pub(super) lora: Option<super::lora::Qwen3LoRA>,
}

impl Qwen3Model {
    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;
        let source = ResolvedModelSource::resolve(model_path)?;
        let resolved_path = source.resolved_path().to_str().unwrap_or(model_path);

        // Try GGUF first — if found, use dequant-at-load path
        if let Some(gguf) = source.gguf() {
            info!("Loading from GGUF: {} tensors", gguf.tensors.len());
            let config = if let Some(config_dir) = source.config_dir() {
                Config::from_file(config_dir.to_str().unwrap_or(resolved_path))?
            } else {
                let gc = gguf.extract_model_config()?;
                info!(
                    "Config from GGUF metadata: {}×{}, {} layers",
                    gc.hidden_size, gc.intermediate_size, gc.num_hidden_layers
                );
                Config::from_parts(
                    qwen3_spec::Qwen3Config {
                        hidden_size: gc.hidden_size,
                        intermediate_size: gc.intermediate_size,
                        num_hidden_layers: gc.num_hidden_layers,
                        num_attention_heads: gc.num_attention_heads,
                        num_key_value_heads: gc.num_key_value_heads,
                        head_dim: gc.head_dim,
                        vocab_size: gc.vocab_size,
                        rms_norm_eps: gc.rms_norm_eps,
                        rope_theta: gc.rope_theta,
                        tie_word_embeddings: true,
                        max_position_embeddings: gc.context_length,
                    },
                    0,
                    0,
                    vec![],
                )
            };
            return Self::from_gguf(&ctx, &config, gguf, runtime);
        }

        let config = Config::from_file(resolved_path)
            .with_context(|| format!("failed to load Qwen3 config from {}", resolved_path))?;

        let (mmaps, weight_map) = common::load_safetensors(resolved_path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;

        let quant = QuantLoadConfig::from_model_path(resolved_path)?;
        if quant.enabled() {
            info!("Weight quantization detected: {:?}", quant);
        }

        // Helper: load linear weight, quantized if available
        let load_linear = |name: &str| -> Result<DeviceMatrix> {
            if quant.enabled() {
                load_tensor_2d_maybe_quantized_with_config(&ctx, &shards, &weight_map, name, quant)
            } else {
                load_tensor_2d(&ctx, &shards, &weight_map, name)
            }
        };

        let t_gpu = Instant::now();
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            config.embed_tokens_tensor_name(),
        )?;
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
            let names = config.layer_tensor_names(i);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &names.input_layernorm,
                )?,
                attention: {
                    let q_proj = load_linear(&names.q_proj)?;
                    let k_proj = load_linear(&names.k_proj)?;
                    let v_proj = load_linear(&names.v_proj)?;
                    Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj: load_linear(&names.o_proj)?,
                        q_norm: load_tensor_1d(&ctx, &shards, &weight_map, &names.q_norm)?,
                        k_norm: load_tensor_1d(&ctx, &shards, &weight_map, &names.k_norm)?,
                    }
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &names.post_attention_layernorm,
                )?,
                mlp: MLP::load_with_quant_config(
                    &ctx,
                    &shards,
                    &weight_map,
                    &names.mlp_prefix,
                    quant,
                )?,
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, config.norm_tensor_name())?;

        debug!("Precomputing RoPE cache on GPU");
        let rope_cache_len = resolve_rope_cache_len(config.rope_cache_len_hint());
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, rope_cache_len, config.rope_theta)?;

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
            lora: None,
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

    /// Attach a loaded `Qwen3LoRA` bundle to this model. Returns `self`
    /// with the adapter set; previous adapter (if any) is dropped.
    #[must_use]
    pub fn with_lora(mut self, lora: super::lora::Qwen3LoRA) -> Self {
        self.lora = Some(lora);
        self
    }

    /// Load a PEFT LoRA adapter directory and attach it. Convenience wrapper
    /// around `lora::load_peft_lora` + `with_lora` for the common CLI path.
    ///
    /// Refuses when the base weights use a format the LoRA path cannot
    /// compose cleanly against: Marlin-packed W4 and TurboQuant formats
    /// lose the row-major BF16 layout `apply_lora_gemm_add` relies on.
    pub fn load_and_attach_lora(self, lora_path: &str) -> Result<Self> {
        if let Some(layer0) = self.layers.first() {
            let qproj = &layer0.attention.q_proj;
            if qproj.has_marlin() {
                anyhow::bail!(
                    "LoRA refuses to attach: base weights are Marlin-packed W4; \
                     LoRA currently requires BF16 or uniform INT{{2,4,8}} base weights"
                );
            }
            if qproj.has_tq() {
                anyhow::bail!(
                    "LoRA refuses to attach: base weights are TurboQuant; \
                     LoRA currently requires BF16 or uniform INT{{2,4,8}} base weights"
                );
            }
        }
        let num_layers = self.config.num_hidden_layers;
        let lora = super::lora::load_peft_lora(&self.ctx, lora_path, num_layers)?;
        Ok(self.with_lora(lora))
    }

    /// Per-layer LoRA slot accessor. Returns `None` when no adapter was
    /// attached or when `layer_idx` is beyond the adapter's coverage.
    pub(super) fn layer_lora(&self, layer_idx: usize) -> Option<&super::lora::LayerLoRA> {
        self.lora.as_ref().and_then(|l| l.layers.get(layer_idx))
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        common::output_projection(self.lm_head.as_ref(), &self.embed_tokens)
    }

    /// Load from a GGUF file — dequantizes all tensors to BF16 at load time.
    fn from_gguf(
        ctx: &DeviceContext,
        config: &Config,
        gguf: &crate::gguf::GgufFile,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        use crate::weight_loader::{
            load_tensor_1d_gguf, load_tensor_2d_gguf, load_tensor_2d_gguf_bf16, precompute_rope,
        };

        let t_gpu = std::time::Instant::now();

        // embed_tokens is read directly via embedding_decode_cuda, which is
        // NOT quant-aware — it would read from the 1-element dummy `.data`
        // buffer of a packed matrix and produce garbage. Force BF16 load.
        let embed_tokens = load_tensor_2d_gguf_bf16(ctx, gguf, config.embed_tokens_tensor_name())?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load_tensor_2d_gguf(
                ctx,
                gguf,
                config.lm_head_tensor_name(),
            )?)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let names = config.layer_tensor_names(i);

            let q_proj = load_tensor_2d_gguf(ctx, gguf, &names.q_proj)?;
            let k_proj = load_tensor_2d_gguf(ctx, gguf, &names.k_proj)?;
            let v_proj = load_tensor_2d_gguf(ctx, gguf, &names.v_proj)?;

            layers.push(TransformerBlock {
                input_layernorm: load_tensor_1d_gguf(ctx, gguf, &names.input_layernorm)?,
                attention: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj: load_tensor_2d_gguf(ctx, gguf, &names.o_proj)?,
                    q_norm: load_tensor_1d_gguf(ctx, gguf, &names.q_norm)?,
                    k_norm: load_tensor_1d_gguf(ctx, gguf, &names.k_norm)?,
                },
                post_attention_layernorm: load_tensor_1d_gguf(
                    ctx,
                    gguf,
                    &names.post_attention_layernorm,
                )?,
                mlp: {
                    let gate = load_tensor_2d_gguf(ctx, gguf, &names.mlp_gate_proj)?;
                    let up = load_tensor_2d_gguf(ctx, gguf, &names.mlp_up_proj)?;
                    let down = load_tensor_2d_gguf(ctx, gguf, &names.mlp_down_proj)?;
                    MLP {
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                    }
                },
            });

            if (i + 1) % 10 == 0 || i + 1 == config.num_hidden_layers {
                info!("GGUF: loaded layer {}/{}", i + 1, config.num_hidden_layers);
            }
        }

        let norm = load_tensor_1d_gguf(ctx, gguf, config.norm_tensor_name())?;
        let rope_cache_len = resolve_rope_cache_len(config.rope_cache_len_hint());
        let (cos_cache, sin_cache) =
            precompute_rope(ctx, config.head_dim, rope_cache_len, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GGUF model loaded in {:.0}ms ({} layers)",
            t_gpu.elapsed().as_secs_f64() * 1e3,
            config.num_hidden_layers
        );

        let model = Self {
            ctx: ctx.clone(),
            config: config.clone(),
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
            lora: None,
        };

        if model.enable_cuda_graph {
            model.preload_decode_triton_kernels()?;
        }
        Ok(model)
    }
}
