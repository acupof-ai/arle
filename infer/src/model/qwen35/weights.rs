use anyhow::Result;
use cudarc::driver::CudaSlice;
use log::{debug, info};
use std::time::Instant;

use super::config::{Config35, LayerType};
use crate::model::common::{self, MLP};
use crate::model::layer_communicator::LayerCommunicator;
use crate::model::qwen35::prefill_buffers::PagedPrefillBuffers35;
use crate::model_source::ResolvedModelSource;
#[cfg(test)]
use crate::tensor_parallel::ShardingSpec;
use crate::tensor_parallel::{TpConfig, column_shard};
use crate::tp::TpLoadContext;
use crate::weight_loader::{
    QuantLoadConfig, load_tensor_1d, load_tensor_1d_f32, load_tensor_1d_f32_sharded,
    load_tensor_1d_fused_segments_sharded, load_tensor_1d_sharded, load_tensor_2d,
    load_tensor_2d_fused_column_segments_sharded, load_tensor_2d_maybe_quantized_with_config,
    load_tensor_2d_sharded, precompute_rope, resolve_rope_cache_len,
};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub(super) struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub(super) q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub(super) o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub(super) struct LinearAttentionLayer {
    /// Fused QKV projection: [q_dim + k_dim + v_dim, hidden_size]
    pub(super) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [z_dim, hidden_size]
    pub(super) in_proj_z: DeviceMatrix,
    /// Beta projection: [num_value_heads, hidden_size]
    pub(super) in_proj_b: DeviceMatrix,
    /// Alpha projection: [num_value_heads, hidden_size]
    pub(super) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [qkv_dim * conv_kernel_dim] (flattened from [qkv_dim, 1, 4])
    pub(super) conv1d_weight: DeviceVec,
    /// dt_bias: [num_value_heads] bf16
    pub(super) dt_bias: DeviceVec,
    /// A_log: [num_value_heads] f32
    pub(super) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32
    pub(super) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, z_dim]
    pub(super) out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub(super) enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

/// Transformer block for Qwen3.5.
pub(super) struct TransformerBlock35 {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attn: LayerKind,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: common::MLP,
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen35RuntimeConfig {
    pub enable_cuda_graph: bool,
    pub tp: TpConfig,
}

impl Default for Qwen35RuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            tp: TpConfig::single(),
        }
    }
}

/// Qwen3.5 model (text-only).
pub struct Qwen35Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config35,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) layers: Vec<TransformerBlock35>,
    pub(super) norm: DeviceVec,
    // Partial RoPE cache: [max_seq_len * rotary_dim]
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
    pub(super) layer_communicator: LayerCommunicator,
    pub(super) paged_prefill_batch: std::sync::Mutex<Option<PagedPrefillBuffers35>>,
}

impl Qwen35Model {
    #[cfg(test)]
    fn from_safetensors(model_path: &str) -> Result<Self> {
        Self::from_safetensors_with_options(model_path, true)
    }

    pub fn from_safetensors_with_options(
        model_path: &str,
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        Self::from_safetensors_with_runtime(
            model_path,
            Qwen35RuntimeConfig {
                enable_cuda_graph,
                ..Qwen35RuntimeConfig::default()
            },
        )
    }

    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: Qwen35RuntimeConfig,
    ) -> Result<Self> {
        info!("Loading Qwen3.5 model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;
        let source = ResolvedModelSource::resolve(model_path)?;
        let resolved_path = source.resolved_path().to_str().unwrap_or(model_path);

        let config = if let Some(config_dir) = source.config_dir() {
            Config35::from_file(config_dir.to_str().unwrap_or(resolved_path))?
        } else if let Some(gguf) = source.gguf() {
            gguf.extract_qwen35_config()?
        } else {
            Config35::from_file(resolved_path)?
        };
        let mut runtime_config = config.clone();
        debug!(
            "Config: hidden_size={}, num_layers={}, full_attn={}, linear_attn={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_full_attention_layers(),
            config.num_hidden_layers - config.num_full_attention_layers()
        );

        // Try GGUF first
        if let Some(gguf) = source.gguf() {
            if !runtime.tp.is_single() {
                anyhow::bail!(
                    "Qwen3.5 GGUF tensor-parallel sharded load is not wired yet; use BF16 safetensors for TP"
                );
            }
            info!("Loading Qwen3.5 from GGUF: {} tensors", gguf.tensors.len());
            return Self::from_gguf(&ctx, &config, gguf, runtime);
        }

        let (mmaps, weight_map) = common::load_safetensors(resolved_path, true)?;
        let shards = common::deserialize_shards(&mmaps)?;
        let quant = QuantLoadConfig::from_model_path(resolved_path)?;
        if quant.enabled() {
            info!("Weight quantization detected: {:?}", quant);
        }
        if !runtime.tp.is_single() && !tp_forward_collectives_ready() {
            anyhow::bail!(
                "Qwen3.5 TP sharded load is staged, but TP forward collectives are not wired yet; keep INFER_TP_SIZE=1 until LayerCommunicator has NCCL all-reduce"
            );
        }
        if !runtime.tp.is_single() {
            TpLoadContext::head(
                runtime.tp.rank,
                runtime.tp.world_size,
                config.num_attention_heads,
                config.num_key_value_heads,
            )?;
            TpLoadContext::head(
                runtime.tp.rank,
                runtime.tp.world_size,
                config.linear_num_key_heads,
                config.linear_num_key_heads,
            )?;
            anyhow::ensure!(
                config
                    .linear_num_value_heads
                    .is_multiple_of(runtime.tp.world_size),
                "Qwen3.5 linear_num_value_heads ({}) must be divisible by TP world size ({})",
                config.linear_num_value_heads,
                runtime.tp.world_size
            );
            if quant.enabled() {
                anyhow::bail!(
                    "Qwen3.5 TP sharded load currently requires BF16 safetensors; quantized load config {:?} cannot be sharded safely yet",
                    quant
                );
            }
            info!(
                "Qwen3.5 TP sharded load enabled: rank={}/{}",
                runtime.tp.rank, runtime.tp.world_size
            );
            runtime_config.num_attention_heads /= runtime.tp.world_size;
            runtime_config.num_key_value_heads /= runtime.tp.world_size;
            runtime_config.linear_num_key_heads /= runtime.tp.world_size;
            runtime_config.linear_num_value_heads /= runtime.tp.world_size;
            runtime_config.intermediate_size =
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
        let load_tp_vec = |name: &str, total_len: usize| -> Result<DeviceVec> {
            if runtime.tp.is_single() {
                load_tensor_1d(&ctx, &shards, &weight_map, name)
            } else {
                let tp = TpLoadContext::column(runtime.tp.rank, runtime.tp.world_size, total_len)?;
                load_tensor_1d_sharded(&ctx, &shards, &weight_map, name, &tp)
            }
        };
        let load_tp_f32 = |name: &str, total_len: usize| -> Result<CudaSlice<f32>> {
            if runtime.tp.is_single() {
                load_tensor_1d_f32(&ctx, &shards, &weight_map, name)
            } else {
                let tp = TpLoadContext::column(runtime.tp.rank, runtime.tp.world_size, total_len)?;
                load_tensor_1d_f32_sharded(&ctx, &shards, &weight_map, name, &tp)
            }
        };
        let load_linear_qkv = |name: &str| -> Result<DeviceMatrix> {
            load_tensor_2d_fused_column_segments_sharded(
                &ctx,
                &shards,
                &weight_map,
                name,
                &[
                    config.linear_num_key_heads * config.linear_key_head_dim,
                    config.linear_num_key_heads * config.linear_key_head_dim,
                    config.linear_num_value_heads * config.linear_value_head_dim,
                ],
                runtime.tp.rank,
                runtime.tp.world_size,
            )
        };
        let load_linear_qkv_vec = |name: &str| -> Result<DeviceVec> {
            load_tensor_1d_fused_segments_sharded(
                &ctx,
                &shards,
                &weight_map,
                name,
                &[
                    config.linear_num_key_heads
                        * config.linear_key_head_dim
                        * config.linear_conv_kernel_dim,
                    config.linear_num_key_heads
                        * config.linear_key_head_dim
                        * config.linear_conv_kernel_dim,
                    config.linear_num_value_heads
                        * config.linear_value_head_dim
                        * config.linear_conv_kernel_dim,
                ],
                runtime.tp.rank,
                runtime.tp.world_size,
            )
        };

        let t_gpu = Instant::now();
        // Weight prefix for Qwen3.5 text model
        let wp = "model.language_model";

        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            &format!("{}.embed_tokens.weight", wp),
        )?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("{}.layers.{}", wp, i);
            let layer_type = config.layer_types[i];

            let attn = match layer_type {
                LayerType::FullAttention => {
                    let attn_prefix = format!("{}.self_attn", prefix);
                    LayerKind::FullAttention(FullAttentionLayer {
                        q_proj: load_tp_column(
                            &format!("{}.q_proj.weight", attn_prefix),
                            config.full_attn_q_proj_dim(),
                        )?,
                        k_proj: load_tp_column(
                            &format!("{}.k_proj.weight", attn_prefix),
                            config.full_attn_kv_dim(),
                        )?,
                        v_proj: load_tp_column(
                            &format!("{}.v_proj.weight", attn_prefix),
                            config.full_attn_kv_dim(),
                        )?,
                        o_proj: load_tp_row(
                            &format!("{}.o_proj.weight", attn_prefix),
                            config.full_attn_q_dim(),
                        )?,
                        q_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_norm.weight", attn_prefix),
                        )?,
                        k_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_norm.weight", attn_prefix),
                        )?,
                    })
                }
                LayerType::LinearAttention => {
                    let attn_prefix = format!("{}.linear_attn", prefix);
                    LayerKind::LinearAttention(LinearAttentionLayer {
                        in_proj_qkv: load_linear_qkv(&format!(
                            "{}.in_proj_qkv.weight",
                            attn_prefix
                        ))?,
                        in_proj_z: load_tp_column(
                            &format!("{}.in_proj_z.weight", attn_prefix),
                            config.linear_attn_z_dim(),
                        )?,
                        in_proj_b: load_tp_column(
                            &format!("{}.in_proj_b.weight", attn_prefix),
                            config.linear_num_value_heads,
                        )?,
                        in_proj_a: load_tp_column(
                            &format!("{}.in_proj_a.weight", attn_prefix),
                            config.linear_num_value_heads,
                        )?,
                        conv1d_weight: load_linear_qkv_vec(&format!(
                            "{}.conv1d.weight",
                            attn_prefix
                        ))?,
                        dt_bias: load_tp_vec(
                            &format!("{}.dt_bias", attn_prefix),
                            config.linear_num_value_heads,
                        )?,
                        a_log: load_tp_f32(
                            &format!("{}.A_log", attn_prefix),
                            config.linear_num_value_heads,
                        )?,
                        norm_weight: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.norm.weight", attn_prefix),
                        )?,
                        out_proj: load_tp_row(
                            &format!("{}.out_proj.weight", attn_prefix),
                            config.linear_attn_z_dim(),
                        )?,
                    })
                }
            };

            let block = TransformerBlock35 {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attn,
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: if runtime.tp.is_single() {
                    MLP::load_with_quant_config(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp", prefix),
                        quant,
                    )?
                } else {
                    MLP {
                        gate_proj: load_tp_column(
                            &format!("{}.mlp.gate_proj.weight", prefix),
                            config.intermediate_size,
                        )?,
                        up_proj: load_tp_column(
                            &format!("{}.mlp.up_proj.weight", prefix),
                            config.intermediate_size,
                        )?,
                        down_proj: load_tp_row(
                            &format!("{}.mlp.down_proj.weight", prefix),
                            config.intermediate_size,
                        )?,
                    }
                },
            };

            debug!(
                "Loaded layer {}/{}: {:?}",
                i + 1,
                config.num_hidden_layers,
                layer_type
            );
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, &format!("{}.norm.weight", wp))?;

        debug!(
            "Precomputing partial RoPE cache (rotary_dim={})",
            config.rotary_dim
        );
        let rope_cache_len = resolve_rope_cache_len(config.rope_cache_len_hint());
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.rotary_dim, rope_cache_len, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("Qwen3.5 GPU model loaded successfully");
        if runtime.enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(Self {
            ctx,
            config: runtime_config,
            embed_tokens,
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
            paged_prefill_batch: std::sync::Mutex::new(None),
        })
    }

    #[cfg(test)]
    fn verify_shapes(&self) -> Result<()> {
        let c = &self.config;

        assert_shape(
            "embed_tokens",
            &self.embed_tokens,
            c.vocab_size,
            c.hidden_size,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer.{}", i);

            assert_vec_len(
                &format!("{}.input_layernorm", prefix),
                &layer.input_layernorm,
                c.hidden_size,
            )?;
            assert_vec_len(
                &format!("{}.post_attn_layernorm", prefix),
                &layer.post_attention_layernorm,
                c.hidden_size,
            )?;

            assert_shape(
                &format!("{}.mlp.gate_proj", prefix),
                &layer.mlp.gate_proj,
                c.intermediate_size,
                c.hidden_size,
            )?;
            assert_shape(
                &format!("{}.mlp.up_proj", prefix),
                &layer.mlp.up_proj,
                c.intermediate_size,
                c.hidden_size,
            )?;
            assert_shape(
                &format!("{}.mlp.down_proj", prefix),
                &layer.mlp.down_proj,
                c.hidden_size,
                c.intermediate_size,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    let q_proj_dim = c.full_attn_q_proj_dim();
                    let kv_dim = c.full_attn_kv_dim();
                    let q_dim = c.full_attn_q_dim();

                    assert_shape(
                        &format!("{}.q_proj", prefix),
                        &attn.q_proj,
                        q_proj_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.k_proj", prefix),
                        &attn.k_proj,
                        kv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.v_proj", prefix),
                        &attn.v_proj,
                        kv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.o_proj", prefix),
                        &attn.o_proj,
                        c.hidden_size,
                        q_dim,
                    )?;
                    assert_vec_len(&format!("{}.q_norm", prefix), &attn.q_norm, c.head_dim)?;
                    assert_vec_len(&format!("{}.k_norm", prefix), &attn.k_norm, c.head_dim)?;
                }
                LayerKind::LinearAttention(attn) => {
                    let qkv_dim = c.linear_attn_qkv_dim();
                    let z_dim = c.linear_attn_z_dim();
                    let num_v_heads = c.linear_num_value_heads;

                    assert_shape(
                        &format!("{}.in_proj_qkv", prefix),
                        &attn.in_proj_qkv,
                        qkv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_z", prefix),
                        &attn.in_proj_z,
                        z_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_b", prefix),
                        &attn.in_proj_b,
                        num_v_heads,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_a", prefix),
                        &attn.in_proj_a,
                        num_v_heads,
                        c.hidden_size,
                    )?;
                    assert_vec_len(
                        &format!("{}.conv1d_weight", prefix),
                        &attn.conv1d_weight,
                        qkv_dim * c.linear_conv_kernel_dim,
                    )?;
                    assert_vec_len(&format!("{}.dt_bias", prefix), &attn.dt_bias, num_v_heads)?;
                    assert_shape(
                        &format!("{}.out_proj", prefix),
                        &attn.out_proj,
                        c.hidden_size,
                        z_dim,
                    )?;
                }
            }
        }

        assert_vec_len("norm", &self.norm, c.hidden_size)?;

        info!("All weight shapes verified successfully");
        Ok(())
    }

    /// Load Qwen3.5 from GGUF — dequant all tensors to BF16 at load time.
    fn from_gguf(
        ctx: &DeviceContext,
        config: &Config35,
        gguf: &crate::gguf::GgufFile,
        runtime: Qwen35RuntimeConfig,
    ) -> Result<Self> {
        use crate::gguf::{
            load_matrix_v_reorder_cols_bf16_host, load_qwen35_a_log_f32_host,
            load_qwen35_conv1d_bf16_host, load_qwen35_qkv_matrix_bf16_host, load_vector_f32_host,
            load_vector_v_reorder_bf16_host,
        };
        use crate::qwen35_gguf_host::Qwen35LinearGgufLayout;
        use crate::weight_loader::{
            load_tensor_1d_gguf_offset_norm, load_tensor_2d_gguf,
            load_tensor_2d_gguf_v_reorder_rows, precompute_rope,
        };

        let linear_layout = Qwen35LinearGgufLayout::from_config(config)?;
        let num_k = linear_layout.num_key_heads;
        let num_v = linear_layout.num_value_heads;
        let vpk = linear_layout.num_value_heads_per_key();
        let hd_k = linear_layout.key_head_dim;
        let hd_v = linear_layout.value_head_dim;
        // GGUF stores norm weights with +1 offset baked in (e.g., w_gguf = 1 + w_hf).
        // Use load_tensor_1d_gguf_offset_norm for all RMSNorm/QK-norm weights.
        let load_norm =
            |ctx: &DeviceContext, gguf: &crate::gguf::GgufFile, name: &str| -> Result<DeviceVec> {
                load_tensor_1d_gguf_offset_norm(ctx, gguf, name)
            };

        // Qwen3.5 GGUF uses standard blk.N prefix — map_gguf_name handles
        // SSM tensors (ssm_a, ssm_conv1d, etc.) → linear_attn.* HF names.
        // The weight_loader's find_gguf_tensor_name does reverse lookup.
        //
        // Note: Qwen3.5 HF uses "model.language_model" prefix, but GGUF
        // uses flat "blk.N" — the reverse mapping in find_gguf_tensor_name
        // handles this by trying map_gguf_name_with_prefix for both prefixes.

        let t_gpu = std::time::Instant::now();
        let wp = "model.language_model";

        let embed_tokens = load_tensor_2d_gguf(ctx, gguf, &format!("{wp}.embed_tokens.weight"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("{wp}.layers.{i}");
            let layer_type = config.layer_types[i];

            let attn = match layer_type {
                LayerType::FullAttention => {
                    let ap = format!("{p}.self_attn");
                    LayerKind::FullAttention(FullAttentionLayer {
                        q_proj: load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.q_proj.weight"))?,
                        k_proj: load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.k_proj.weight"))?,
                        v_proj: load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.v_proj.weight"))?,
                        o_proj: load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.o_proj.weight"))?,
                        q_norm: load_norm(ctx, gguf, &format!("{ap}.q_norm.weight"))?,
                        k_norm: load_norm(ctx, gguf, &format!("{ap}.k_norm.weight"))?,
                    })
                }
                LayerType::LinearAttention => {
                    let ap = format!("{p}.linear_attn");
                    LayerKind::LinearAttention(LinearAttentionLayer {
                        in_proj_qkv: {
                            if vpk <= 1 {
                                load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.in_proj_qkv.weight"))?
                            } else {
                                let tensor = load_qwen35_qkv_matrix_bf16_host(
                                    gguf,
                                    &format!("{ap}.in_proj_qkv.weight"),
                                    num_k,
                                    vpk,
                                    hd_k,
                                    hd_v,
                                )?;
                                DeviceMatrix::from_host(
                                    ctx,
                                    &tensor.data,
                                    tensor.shape[0],
                                    tensor.shape[1],
                                )?
                            }
                        },
                        in_proj_z: load_tensor_2d_gguf_v_reorder_rows(
                            ctx,
                            gguf,
                            &format!("{ap}.in_proj_z.weight"),
                            num_k,
                            vpk,
                            hd_v,
                        )?,
                        in_proj_b: load_tensor_2d_gguf_v_reorder_rows(
                            ctx,
                            gguf,
                            &format!("{ap}.in_proj_b.weight"),
                            num_k,
                            vpk,
                            1,
                        )?,
                        in_proj_a: load_tensor_2d_gguf_v_reorder_rows(
                            ctx,
                            gguf,
                            &format!("{ap}.in_proj_a.weight"),
                            num_k,
                            vpk,
                            1,
                        )?,
                        conv1d_weight: {
                            let tensor = load_qwen35_conv1d_bf16_host(
                                gguf,
                                &format!("{ap}.conv1d.weight"),
                                num_k,
                                hd_k,
                                num_v,
                                hd_v,
                                linear_layout.conv_kernel_dim,
                            )?;
                            DeviceVec::from_host(ctx, &tensor.data)?
                        },
                        dt_bias: {
                            let tensor = load_vector_v_reorder_bf16_host(
                                gguf,
                                &format!("{ap}.dt_bias"),
                                num_k,
                                vpk,
                                1,
                            )?;
                            DeviceVec::from_host(ctx, &tensor.data)?
                        },
                        a_log: {
                            let tensor = load_qwen35_a_log_f32_host(
                                gguf,
                                &format!("{ap}.a_log"),
                                num_k,
                                vpk,
                            )?;
                            ctx.stream.clone_htod(&tensor.data)?
                        },
                        norm_weight: {
                            let tensor = load_vector_f32_host(gguf, &format!("{ap}.norm.weight"))?;
                            ctx.stream.clone_htod(&tensor.data)?
                        },
                        out_proj: {
                            if vpk <= 1 {
                                load_tensor_2d_gguf(ctx, gguf, &format!("{ap}.out_proj.weight"))?
                            } else {
                                let tensor = load_matrix_v_reorder_cols_bf16_host(
                                    gguf,
                                    &format!("{ap}.out_proj.weight"),
                                    num_k,
                                    vpk,
                                    hd_v,
                                )?;
                                DeviceMatrix::from_host(
                                    ctx,
                                    &tensor.data,
                                    tensor.shape[0],
                                    tensor.shape[1],
                                )?
                            }
                        },
                    })
                }
            };

            layers.push(TransformerBlock35 {
                input_layernorm: load_norm(ctx, gguf, &format!("{p}.input_layernorm.weight"))?,
                attn,
                post_attention_layernorm: load_norm(
                    ctx,
                    gguf,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                mlp: {
                    let gate =
                        load_tensor_2d_gguf(ctx, gguf, &format!("{p}.mlp.gate_proj.weight"))?;
                    let up = load_tensor_2d_gguf(ctx, gguf, &format!("{p}.mlp.up_proj.weight"))?;
                    let down =
                        load_tensor_2d_gguf(ctx, gguf, &format!("{p}.mlp.down_proj.weight"))?;
                    common::MLP {
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                    }
                },
            });

            if (i + 1) % 8 == 0 || i + 1 == config.num_hidden_layers {
                info!("GGUF: loaded layer {}/{}", i + 1, config.num_hidden_layers);
            }
        }

        let norm = load_norm(ctx, gguf, &format!("{wp}.norm.weight"))?;
        // Qwen3.5 uses partial RoPE: only the first `rotary_dim` elements of
        // each head are rotated. The safetensors loader passes `rotary_dim`
        // here; the GGUF loader used to pass the full `head_dim`, which made
        // cos_cache have stride=256 while the CUDA kernel indexes with
        // stride=rotary_dim=64 → every position > 0 read garbage trig values,
        // collapsing attention to prompt-independent degenerate output.
        let rope_cache_len = resolve_rope_cache_len(config.rope_cache_len_hint());
        let (cos_cache, sin_cache) =
            precompute_rope(ctx, config.rotary_dim, rope_cache_len, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "Qwen3.5 GGUF loaded in {:.0}ms ({} layers)",
            t_gpu.elapsed().as_secs_f64() * 1e3,
            config.num_hidden_layers
        );

        Ok(Self {
            ctx: ctx.clone(),
            config: config.clone(),
            embed_tokens,
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
            paged_prefill_batch: std::sync::Mutex::new(None),
        })
    }
}

#[cfg(test)]
fn assert_shape(name: &str, m: &DeviceMatrix, rows: usize, cols: usize) -> Result<()> {
    anyhow::ensure!(
        m.rows == rows && m.cols == cols,
        "{}: expected [{}, {}], got [{}, {}]",
        name,
        rows,
        cols,
        m.rows,
        m.cols
    );
    Ok(())
}

#[cfg(test)]
fn assert_vec_len(name: &str, v: &DeviceVec, expected: usize) -> Result<()> {
    anyhow::ensure!(
        v.len == expected,
        "{}: expected len {}, got {}",
        name,
        expected,
        v.len
    );
    Ok(())
}

fn tp_forward_collectives_ready() -> bool {
    false
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35TpLayerShards {
    pub full_q_proj: ShardingSpec,
    pub full_k_proj: ShardingSpec,
    pub full_v_proj: ShardingSpec,
    pub full_o_proj: ShardingSpec,
    pub linear_qkv: ShardingSpec,
    pub linear_z: ShardingSpec,
    pub linear_b: ShardingSpec,
    pub linear_a: ShardingSpec,
    pub linear_conv1d: ShardingSpec,
    pub linear_dt_bias: ShardingSpec,
    pub linear_a_log: ShardingSpec,
    pub linear_out: ShardingSpec,
    pub mlp_gate: ShardingSpec,
    pub mlp_up: ShardingSpec,
    pub mlp_down: ShardingSpec,
}

#[cfg(test)]
pub(crate) fn qwen35_tp_layer_shards(
    config: &Config35,
    tp: TpConfig,
) -> Result<Qwen35TpLayerShards> {
    TpLoadContext::head(
        tp.rank,
        tp.world_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    )?;
    TpLoadContext::head(
        tp.rank,
        tp.world_size,
        config.linear_num_key_heads,
        config.linear_num_key_heads,
    )?;
    Ok(Qwen35TpLayerShards {
        full_q_proj: TpLoadContext::column(tp.rank, tp.world_size, config.full_attn_q_proj_dim())?
            .sharding,
        full_k_proj: TpLoadContext::column(tp.rank, tp.world_size, config.full_attn_kv_dim())?
            .sharding,
        full_v_proj: TpLoadContext::column(tp.rank, tp.world_size, config.full_attn_kv_dim())?
            .sharding,
        full_o_proj: TpLoadContext::row(tp.rank, tp.world_size, config.full_attn_q_dim())?.sharding,
        linear_qkv: TpLoadContext::column(tp.rank, tp.world_size, config.linear_attn_qkv_dim())?
            .sharding,
        linear_z: TpLoadContext::column(tp.rank, tp.world_size, config.linear_attn_z_dim())?
            .sharding,
        linear_b: TpLoadContext::column(tp.rank, tp.world_size, config.linear_num_value_heads)?
            .sharding,
        linear_a: TpLoadContext::column(tp.rank, tp.world_size, config.linear_num_value_heads)?
            .sharding,
        linear_conv1d: TpLoadContext::column(
            tp.rank,
            tp.world_size,
            config.linear_attn_qkv_dim() * config.linear_conv_kernel_dim,
        )?
        .sharding,
        linear_dt_bias: TpLoadContext::column(
            tp.rank,
            tp.world_size,
            config.linear_num_value_heads,
        )?
        .sharding,
        linear_a_log: TpLoadContext::column(tp.rank, tp.world_size, config.linear_num_value_heads)?
            .sharding,
        linear_out: TpLoadContext::row(tp.rank, tp.world_size, config.linear_attn_z_dim())?
            .sharding,
        mlp_gate: TpLoadContext::column(tp.rank, tp.world_size, config.intermediate_size)?.sharding,
        mlp_up: TpLoadContext::column(tp.rank, tp.world_size, config.intermediate_size)?.sharding,
        mlp_down: TpLoadContext::row(tp.rank, tp.world_size, config.intermediate_size)?.sharding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    fn hybrid_config() -> Config35 {
        Config35::from_json_str(
            r#"{
                "text_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "vocab_size": 128,
                    "rms_norm_eps": 1e-6,
                    "eos_token_id": 127,
                    "bos_token_id": 0,
                    "tie_word_embeddings": true,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "layer_types": ["full_attention", "linear_attention"],
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_num_key_heads": 4,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 4,
                    "rope_parameters": {
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.5
                    },
                    "max_position_embeddings": 4096
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_load_qwen35_model() {
        let model = Qwen35Model::from_safetensors(MODEL_PATH).unwrap();

        assert_eq!(model.layers.len(), 32);
        assert_eq!(model.config.num_hidden_layers, 32);

        let full_count = model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, LayerKind::FullAttention(_)))
            .count();
        let linear_count = model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, LayerKind::LinearAttention(_)))
            .count();
        assert_eq!(full_count, 8);
        assert_eq!(linear_count, 24);

        model.verify_shapes().unwrap();
    }

    #[test]
    fn qwen35_tp1_layer_shards_are_full_for_hybrid_shapes() {
        let config = hybrid_config();
        let shards = qwen35_tp_layer_shards(&config, TpConfig::single()).unwrap();

        assert!(shards.full_q_proj.is_full());
        assert!(shards.full_k_proj.is_full());
        assert!(shards.full_v_proj.is_full());
        assert!(shards.full_o_proj.is_full());
        assert!(shards.linear_qkv.is_full());
        assert!(shards.linear_z.is_full());
        assert!(shards.linear_b.is_full());
        assert!(shards.linear_a.is_full());
        assert!(shards.linear_conv1d.is_full());
        assert!(shards.linear_dt_bias.is_full());
        assert!(shards.linear_a_log.is_full());
        assert!(shards.linear_out.is_full());
        assert!(shards.mlp_gate.is_full());
        assert!(shards.mlp_up.is_full());
        assert!(shards.mlp_down.is_full());
    }

    #[test]
    fn qwen35_tp2_layer_shards_cover_full_and_linear_attention_dimensions() {
        let config = hybrid_config();
        let rank0 = qwen35_tp_layer_shards(&config, TpConfig::new(2, 0).unwrap()).unwrap();
        let rank1 = qwen35_tp_layer_shards(&config, TpConfig::new(2, 1).unwrap()).unwrap();

        assert_eq!(
            rank0.full_q_proj.size + rank1.full_q_proj.size,
            config.full_attn_q_proj_dim()
        );
        assert_eq!(
            rank0.full_k_proj.size + rank1.full_k_proj.size,
            config.full_attn_kv_dim()
        );
        assert_eq!(
            rank0.full_v_proj.size + rank1.full_v_proj.size,
            config.full_attn_kv_dim()
        );
        assert_eq!(
            rank0.full_o_proj.size + rank1.full_o_proj.size,
            config.full_attn_q_dim()
        );
        assert_eq!(
            rank0.linear_qkv.size + rank1.linear_qkv.size,
            config.linear_attn_qkv_dim()
        );
        assert_eq!(
            rank0.linear_z.size + rank1.linear_z.size,
            config.linear_attn_z_dim()
        );
        assert_eq!(
            rank0.linear_b.size + rank1.linear_b.size,
            config.linear_num_value_heads
        );
        assert_eq!(
            rank0.linear_a.size + rank1.linear_a.size,
            config.linear_num_value_heads
        );
        assert_eq!(
            rank0.linear_conv1d.size + rank1.linear_conv1d.size,
            config.linear_attn_qkv_dim() * config.linear_conv_kernel_dim
        );
        assert_eq!(
            rank0.linear_dt_bias.size + rank1.linear_dt_bias.size,
            config.linear_num_value_heads
        );
        assert_eq!(
            rank0.linear_a_log.size + rank1.linear_a_log.size,
            config.linear_num_value_heads
        );
        assert_eq!(
            rank0.linear_out.size + rank1.linear_out.size,
            config.linear_attn_z_dim()
        );
        assert_eq!(
            rank0.mlp_gate.size + rank1.mlp_gate.size,
            config.intermediate_size
        );
        assert_eq!(
            rank0.mlp_up.size + rank1.mlp_up.size,
            config.intermediate_size
        );
        assert_eq!(
            rank0.mlp_down.size + rank1.mlp_down.size,
            config.intermediate_size
        );
    }
}
