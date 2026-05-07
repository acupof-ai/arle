use anyhow::{Context, Result};
use log::{debug, info};
use std::time::Instant;

use super::config::Config;
use crate::model::common::{self, MLP as CommonMLP};
use crate::model::layer_communicator::LayerCommunicator;
use crate::model_source::ResolvedModelSource;
use crate::ops;
#[cfg(test)]
use crate::tensor_parallel::ShardingSpec;
use crate::tensor_parallel::{TpConfig, column_shard};
use crate::tp::TpLoadContext;
use crate::weight_loader::{
    QuantLoadConfig, load_tensor_1d, load_tensor_2d, load_tensor_2d_concat_rows,
    load_tensor_2d_maybe_quantized_with_config, load_tensor_2d_sharded, precompute_rope,
    resolve_rope_cache_len,
};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

#[derive(Clone, Copy, Debug)]
pub struct ModelRuntimeConfig {
    pub enable_cuda_graph: bool,
    pub tp: TpConfig,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tp: TpConfig::single(),
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
    pub(super) mlp: Qwen3Mlp,
}

pub(super) enum Qwen3GateUp {
    Separate {
        gate_proj: DeviceMatrix,
        up_proj: DeviceMatrix,
    },
    Fused {
        gate_up_proj: DeviceMatrix,
    },
}

pub(super) struct Qwen3Mlp {
    pub(super) gate_up: Qwen3GateUp,
    pub(super) down_proj: DeviceMatrix,
}

impl Qwen3Mlp {
    fn from_common(mlp: CommonMLP) -> Self {
        Self {
            gate_up: Qwen3GateUp::Separate {
                gate_proj: mlp.gate_proj,
                up_proj: mlp.up_proj,
            },
            down_proj: mlp.down_proj,
        }
    }

    fn from_separate(
        gate_proj: DeviceMatrix,
        up_proj: DeviceMatrix,
        down_proj: DeviceMatrix,
    ) -> Self {
        Self {
            gate_up: Qwen3GateUp::Separate { gate_proj, up_proj },
            down_proj,
        }
    }

    fn from_fused(gate_up_proj: DeviceMatrix, down_proj: DeviceMatrix) -> Self {
        Self {
            gate_up: Qwen3GateUp::Fused { gate_up_proj },
            down_proj,
        }
    }

    fn load_with_quant_config(
        ctx: &DeviceContext,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        prefix: &str,
        quant: QuantLoadConfig,
    ) -> Result<Self> {
        if quant.enabled() || !qwen3_fused_gate_up_enabled() {
            return Ok(Self::from_common(CommonMLP::load_with_quant_config(
                ctx, shards, weight_map, prefix, quant,
            )?));
        }

        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let down_proj = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &format!("{prefix}.down_proj.weight"),
        )?;
        let gate_up_proj =
            load_tensor_2d_concat_rows(ctx, shards, weight_map, &[&gate_name, &up_name])?;
        Ok(Self::from_fused(gate_up_proj, down_proj))
    }

    pub(super) fn fused_gate_up(&self) -> Option<&DeviceMatrix> {
        match &self.gate_up {
            Qwen3GateUp::Fused { gate_up_proj } => Some(gate_up_proj),
            Qwen3GateUp::Separate { .. } => None,
        }
    }

    pub(super) fn separate_gate_up(&self) -> Option<(&DeviceMatrix, &DeviceMatrix)> {
        match &self.gate_up {
            Qwen3GateUp::Separate { gate_proj, up_proj } => Some((gate_proj, up_proj)),
            Qwen3GateUp::Fused { .. } => None,
        }
    }

    pub(super) fn uses_fused_gate_up(&self) -> bool {
        matches!(self.gate_up, Qwen3GateUp::Fused { .. })
    }
}

fn qwen3_fused_gate_up_enabled() -> bool {
    if std::env::var("INFER_LORA_PATH")
        .ok()
        .is_some_and(|path| !path.trim().is_empty())
    {
        return false;
    }
    match std::env::var("INFER_QWEN3_FUSED_GATE_UP") {
        Ok(value) => matches!(
            value.trim(),
            "1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES"
        ),
        Err(_) => false,
    }
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
    pub(super) layer_communicator: LayerCommunicator,
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
            if !runtime.tp.is_single() {
                anyhow::bail!(
                    "Qwen3 GGUF tensor-parallel sharded load is not wired yet; use BF16 safetensors for TP"
                );
            }
            return Self::from_gguf(&ctx, &config, gguf, runtime);
        }

        let config = Config::from_file(resolved_path)
            .with_context(|| format!("failed to load Qwen3 config from {}", resolved_path))?;
        let mut runtime_config = config.clone();

        let (mmaps, weight_map) = common::load_safetensors(resolved_path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;

        let quant = QuantLoadConfig::from_model_path(resolved_path)?;
        if quant.enabled() {
            info!("Weight quantization detected: {:?}", quant);
        }

        if !runtime.tp.is_single() && !tp_forward_collectives_ready() {
            anyhow::bail!(
                "Qwen3 TP sharded load is staged, but TP forward collectives are not wired yet; keep INFER_TP_SIZE=1 until LayerCommunicator has NCCL all-reduce"
            );
        }

        if !runtime.tp.is_single() {
            TpLoadContext::head(
                runtime.tp.rank,
                runtime.tp.world_size,
                config.num_attention_heads,
                config.num_key_value_heads,
            )?;
            if quant.enabled() {
                anyhow::bail!(
                    "Qwen3 TP sharded load currently requires BF16 safetensors; quantized load config {:?} cannot be sharded safely yet",
                    quant
                );
            }
            info!(
                "Qwen3 TP sharded load enabled: rank={}/{}",
                runtime.tp.rank, runtime.tp.world_size
            );
            runtime_config.spec.num_attention_heads /= runtime.tp.world_size;
            runtime_config.spec.num_key_value_heads /= runtime.tp.world_size;
            runtime_config.spec.intermediate_size =
                column_shard(config.intermediate_size, &runtime.tp).size;
        }

        let load_linear = |name: &str| -> Result<DeviceMatrix> {
            if quant.enabled() {
                load_tensor_2d_maybe_quantized_with_config(&ctx, &shards, &weight_map, name, quant)
            } else {
                load_tensor_2d(&ctx, &shards, &weight_map, name)
            }
        };
        let load_tp_column = |name: &str, total_out_features: usize| -> Result<DeviceMatrix> {
            if runtime.tp.is_single() {
                load_linear(name)
            } else {
                let tp = TpLoadContext::column(
                    runtime.tp.rank,
                    runtime.tp.world_size,
                    total_out_features,
                )?;
                load_tensor_2d_sharded(&ctx, &shards, &weight_map, name, &tp)
            }
        };
        let load_tp_row = |name: &str, total_in_features: usize| -> Result<DeviceMatrix> {
            if runtime.tp.is_single() {
                load_linear(name)
            } else {
                let tp =
                    TpLoadContext::row(runtime.tp.rank, runtime.tp.world_size, total_in_features)?;
                load_tensor_2d_sharded(&ctx, &shards, &weight_map, name, &tp)
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
                    let q_proj = load_tp_column(
                        &names.q_proj,
                        config.num_attention_heads * config.head_dim,
                    )?;
                    let k_proj = load_tp_column(
                        &names.k_proj,
                        config.num_key_value_heads * config.head_dim,
                    )?;
                    let v_proj = load_tp_column(
                        &names.v_proj,
                        config.num_key_value_heads * config.head_dim,
                    )?;
                    Attention {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj: load_tp_row(
                            &names.o_proj,
                            config.num_attention_heads * config.head_dim,
                        )?,
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
                mlp: if runtime.tp.is_single() {
                    Qwen3Mlp::load_with_quant_config(
                        &ctx,
                        &shards,
                        &weight_map,
                        &names.mlp_prefix,
                        quant,
                    )?
                } else {
                    let gate_proj = load_tp_column(&names.mlp_gate_proj, config.intermediate_size)?;
                    let up_proj = load_tp_column(&names.mlp_up_proj, config.intermediate_size)?;
                    let down_proj = load_tp_row(&names.mlp_down_proj, config.intermediate_size)?;
                    Qwen3Mlp::from_separate(gate_proj, up_proj, down_proj)
                },
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
            config: runtime_config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
            layer_communicator: LayerCommunicator::new(
                runtime.tp.rank,
                runtime.tp.world_size,
                0,
                1,
                0,
                1,
            )?,
            lora: None,
        };

        if model.enable_cuda_graph {
            debug!("Preloading decode-path CUDA kernels before CUDA Graph capture");
            model.preload_decode_cuda_kernels()?;
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    fn preload_decode_cuda_kernels(&self) -> Result<()> {
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
        if self
            .layers
            .iter()
            .any(|layer| layer.mlp.uses_fused_gate_up())
            && lora
                .layers
                .iter()
                .any(|layer| layer.gate_proj.is_some() || layer.up_proj.is_some())
        {
            anyhow::bail!(
                "LoRA refuses to attach: fused Qwen3 gate_up MLP weights cannot apply gate/up \
                 adapters. Set INFER_QWEN3_FUSED_GATE_UP=0 before loading the model."
            );
        }
        Ok(self.with_lora(lora))
    }

    /// Per-layer LoRA slot accessor. Returns `None` when no adapter was
    /// attached or when `layer_idx` is beyond the adapter's coverage.
    pub(super) fn layer_lora(&self, layer_idx: usize) -> Option<&super::lora::LayerLoRA> {
        self.lora.as_ref().and_then(|l| l.layers.get(layer_idx))
    }

    pub(super) fn uses_fused_gate_up(&self) -> bool {
        self.layers
            .first()
            .is_some_and(|layer| layer.mlp.uses_fused_gate_up())
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
                    Qwen3Mlp::from_separate(gate, up, down)
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
            layer_communicator: LayerCommunicator::new(
                runtime.tp.rank,
                runtime.tp.world_size,
                0,
                1,
                0,
                1,
            )?,
            lora: None,
        };

        if model.enable_cuda_graph {
            model.preload_decode_cuda_kernels()?;
        }
        Ok(model)
    }
}

fn tp_forward_collectives_ready() -> bool {
    false
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen3TpLayerShards {
    pub q_proj: ShardingSpec,
    pub k_proj: ShardingSpec,
    pub v_proj: ShardingSpec,
    pub o_proj: ShardingSpec,
    pub gate_proj: ShardingSpec,
    pub up_proj: ShardingSpec,
    pub down_proj: ShardingSpec,
}

#[cfg(test)]
pub(crate) fn qwen3_tp_layer_shards(config: &Config, tp: TpConfig) -> Result<Qwen3TpLayerShards> {
    TpLoadContext::head(
        tp.rank,
        tp.world_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    )?;
    Ok(Qwen3TpLayerShards {
        q_proj: TpLoadContext::column(
            tp.rank,
            tp.world_size,
            config.num_attention_heads * config.head_dim,
        )?
        .sharding,
        k_proj: TpLoadContext::column(
            tp.rank,
            tp.world_size,
            config.num_key_value_heads * config.head_dim,
        )?
        .sharding,
        v_proj: TpLoadContext::column(
            tp.rank,
            tp.world_size,
            config.num_key_value_heads * config.head_dim,
        )?
        .sharding,
        o_proj: TpLoadContext::row(
            tp.rank,
            tp.world_size,
            config.num_attention_heads * config.head_dim,
        )?
        .sharding,
        gate_proj: TpLoadContext::column(tp.rank, tp.world_size, config.intermediate_size)?
            .sharding,
        up_proj: TpLoadContext::column(tp.rank, tp.world_size, config.intermediate_size)?.sharding,
        down_proj: TpLoadContext::row(tp.rank, tp.world_size, config.intermediate_size)?.sharding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        Config::from_parts(
            qwen3_spec::Qwen3Config {
                hidden_size: 16,
                intermediate_size: 32,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                head_dim: 4,
                vocab_size: 128,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                tie_word_embeddings: true,
                max_position_embeddings: 4096,
            },
            0,
            0,
            vec![],
        )
    }

    #[test]
    fn qwen3_tp1_layer_shards_are_full() {
        let config = tiny_config();
        let shards = qwen3_tp_layer_shards(&config, TpConfig::single()).unwrap();

        assert!(shards.q_proj.is_full());
        assert!(shards.k_proj.is_full());
        assert!(shards.v_proj.is_full());
        assert!(shards.o_proj.is_full());
        assert!(shards.gate_proj.is_full());
        assert!(shards.up_proj.is_full());
        assert!(shards.down_proj.is_full());
    }

    #[test]
    fn qwen3_tp2_layer_shards_cover_attention_and_mlp_dimensions() {
        let config = tiny_config();
        let rank0 = qwen3_tp_layer_shards(&config, TpConfig::new(2, 0).unwrap()).unwrap();
        let rank1 = qwen3_tp_layer_shards(&config, TpConfig::new(2, 1).unwrap()).unwrap();

        assert_eq!((rank0.q_proj.offset, rank0.q_proj.size), (0, 8));
        assert_eq!((rank1.q_proj.offset, rank1.q_proj.size), (8, 8));
        assert_eq!((rank0.k_proj.offset, rank0.k_proj.size), (0, 4));
        assert_eq!((rank1.k_proj.offset, rank1.k_proj.size), (4, 4));
        assert_eq!((rank0.v_proj.offset, rank0.v_proj.size), (0, 4));
        assert_eq!((rank1.v_proj.offset, rank1.v_proj.size), (4, 4));
        assert_eq!((rank0.o_proj.offset, rank0.o_proj.size), (0, 8));
        assert_eq!((rank1.o_proj.offset, rank1.o_proj.size), (8, 8));
        assert_eq!((rank0.gate_proj.size + rank1.gate_proj.size), 32);
        assert_eq!((rank0.up_proj.size + rank1.up_proj.size), 32);
        assert_eq!((rank0.down_proj.size + rank1.down_proj.size), 32);
    }
}
