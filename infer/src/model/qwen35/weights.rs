use anyhow::Result;
use cudarc::driver::CudaSlice;
use log::{debug, info};
use std::time::Instant;

use super::config::{Config35, LayerType};
use crate::model::common::{self, MLP};
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use crate::weight_loader::{load_tensor_1d, load_tensor_1d_f32, load_tensor_2d, precompute_rope};

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
        info!("Loading Qwen3.5 model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config35::from_file(model_path)?;
        debug!(
            "Config: hidden_size={}, num_layers={}, full_attn={}, linear_attn={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_full_attention_layers(),
            config.num_hidden_layers - config.num_full_attention_layers()
        );

        // Try GGUF first
        if let Some(gguf) = crate::weight_loader::try_open_gguf(model_path) {
            info!("Loading Qwen3.5 from GGUF: {} tensors", gguf.tensors.len());
            return Self::from_gguf(&ctx, &config, &gguf, enable_cuda_graph);
        }

        let (mmaps, weight_map) = common::load_safetensors(model_path, true)?;
        let shards = common::deserialize_shards(&mmaps)?;

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
                        q_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_proj.weight", attn_prefix),
                        )?,
                        k_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_proj.weight", attn_prefix),
                        )?,
                        v_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.v_proj.weight", attn_prefix),
                        )?,
                        o_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.o_proj.weight", attn_prefix),
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
                        in_proj_qkv: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_qkv.weight", attn_prefix),
                        )?,
                        in_proj_z: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_z.weight", attn_prefix),
                        )?,
                        in_proj_b: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_b.weight", attn_prefix),
                        )?,
                        in_proj_a: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_a.weight", attn_prefix),
                        )?,
                        conv1d_weight: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.conv1d.weight", attn_prefix),
                        )?,
                        dt_bias: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.dt_bias", attn_prefix),
                        )?,
                        a_log: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.A_log", attn_prefix),
                        )?,
                        norm_weight: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.norm.weight", attn_prefix),
                        )?,
                        out_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.out_proj.weight", attn_prefix),
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
                mlp: MLP::load(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.mlp", prefix),
                    false, // Qwen3.5 doesn't use merged gate+up
                )?,
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
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.rotary_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("Qwen3.5 GPU model loaded successfully");
        if enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph,
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
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        use crate::weight_loader::{
            load_tensor_1d_gguf, load_tensor_1d_gguf_offset_norm, load_tensor_1d_gguf_v_reorder,
            load_tensor_2d_gguf, load_tensor_2d_gguf_v_reorder_cols,
            load_tensor_2d_gguf_v_reorder_rows, precompute_rope, reverse_v_reorder_f32,
            reverse_v_reorder_rows,
        };

        // SSM V-head reorder config (from llama.cpp _reorder_v_heads)
        let num_k = config.linear_num_key_heads;
        let num_v = config.linear_num_value_heads;
        let vpk = num_v / num_k; // num_v_per_k
        let hd_k = config.linear_key_head_dim;
        let hd_v = config.linear_value_head_dim;
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
                        // QKV: only V rows are reordered (Q/K stay in place)
                        // Ref: llama.cpp _LinearAttentionVReorderBase.modify_tensors
                        in_proj_qkv: {
                            let q_rows = num_k * hd_k;
                            let k_rows = num_k * hd_k;
                            let v_rows = num_v * hd_v;
                            let gguf_name = crate::weight_loader::find_gguf_tensor_name_pub(
                                gguf,
                                &format!("{ap}.in_proj_qkv.weight"),
                            )?;
                            let info = &gguf.tensors[&gguf_name];
                            let mut data = gguf.read_tensor_bf16(&gguf_name)?;
                            let cols = info.shape[0] as usize;
                            // Reorder only V rows
                            let v_start = (q_rows + k_rows) * cols;
                            reverse_v_reorder_rows(
                                &mut data[v_start..v_start + v_rows * cols],
                                v_rows,
                                cols,
                                num_k,
                                vpk,
                                hd_v,
                            );
                            let rows = q_rows + k_rows + v_rows;
                            DeviceMatrix::from_host(ctx, &data, rows, cols)?
                        },
                        // in_proj_z: V-reorder rows (head_dim = hd_v)
                        in_proj_z: load_tensor_2d_gguf_v_reorder_rows(
                            ctx,
                            gguf,
                            &format!("{ap}.in_proj_z.weight"),
                            num_k,
                            vpk,
                            hd_v,
                        )?,
                        // in_proj_a/b: V-reorder rows (head_dim = 1)
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
                        // conv1d: [qkv_channels=8192, kernel=4], reorder only V rows.
                        // Each channel has kernel_dim consecutive weights.
                        conv1d_weight: {
                            let gguf_name = crate::weight_loader::find_gguf_tensor_name_pub(
                                gguf,
                                &format!("{ap}.conv1d.weight"),
                            )?;
                            let mut data = gguf.read_tensor_bf16(&gguf_name)?;
                            let kernel_dim = config.linear_conv_kernel_dim;
                            let qk_channels = hd_k * num_k * 2;
                            let v_channels = num_v * hd_v;
                            assert_eq!(data.len(), (qk_channels + v_channels) * kernel_dim);
                            // Reorder only V portion: treat as [v_channels, kernel] matrix
                            let v_start = qk_channels * kernel_dim;
                            let v_size = v_channels * kernel_dim;
                            reverse_v_reorder_rows(
                                &mut data[v_start..v_start + v_size],
                                v_channels,
                                kernel_dim,
                                num_k,
                                vpk,
                                hd_v,
                            );
                            DeviceVec::from_host(ctx, &data)?
                        },
                        // dt_bias: 1D V-reorder (head_dim = 1)
                        dt_bias: load_tensor_1d_gguf_v_reorder(
                            ctx,
                            gguf,
                            &format!("{ap}.dt_bias"),
                            num_k,
                            vpk,
                        )?,
                        a_log: {
                            // GGUF `ssm_a` stores the raw (negative) A parameter per V-head,
                            // following llama.cpp's `A = -exp(A_log)` convention — so
                            //   |A| = exp(A_log)   ⇒   A_log = log(|A|).
                            // Values we see in Carnice-27b are A ≈ -0.04..-0.3 → A_log ≈ -3..−1.
                            //
                            // The GDR kernel (`gated_delta_rule.cu`) computes
                            //   g = -exp(a_log) * softplus(a_proj + dt_bias)
                            // so a_log must carry the log-magnitude with the correct sign.
                            // Earlier code used `-log(|A|)` which flipped the sign and made
                            // every forward step produce garbage generation on any model
                            // that exercises this GGUF path (Qwen3.5-4B passed the e2e test
                            // only because it loads via safetensors, not via this branch).
                            let gguf_name = crate::weight_loader::find_gguf_tensor_name_pub(
                                gguf,
                                &format!("{ap}.a_log"),
                            )?;
                            let raw = gguf.read_tensor_raw(&gguf_name)?;
                            let info = &gguf.tensors[&gguf_name];
                            let raw_f32: Vec<f32> = if info.dtype == crate::gguf::GgmlType::F32 {
                                unsafe {
                                    std::slice::from_raw_parts(
                                        raw.as_ptr().cast::<f32>(),
                                        info.numel(),
                                    )
                                }
                                .to_vec()
                            } else {
                                gguf.read_tensor_bf16(&gguf_name)?
                                    .iter()
                                    .map(|v| v.to_f32())
                                    .collect()
                            };
                            // A → A_log = log(|A|)  (matches llama.cpp convention)
                            let mut a_log: Vec<f32> = raw_f32
                                .iter()
                                .map(|&a| {
                                    let abs_a = a.abs().max(1e-10);
                                    abs_a.ln()
                                })
                                .collect();
                            // Reverse llama.cpp V-head reorder (head_dim = 1 for A_log)
                            reverse_v_reorder_f32(&mut a_log, num_k, vpk, 1);
                            ctx.stream.clone_htod(&a_log)?
                        },
                        norm_weight: {
                            let gguf_name = crate::weight_loader::find_gguf_tensor_name_pub(
                                gguf,
                                &format!("{ap}.norm.weight"),
                            )?;
                            let raw = gguf.read_tensor_raw(&gguf_name)?;
                            let info = &gguf.tensors[&gguf_name];
                            let mut f32s: Vec<f32> = if info.dtype == crate::gguf::GgmlType::F32 {
                                unsafe {
                                    std::slice::from_raw_parts(
                                        raw.as_ptr().cast::<f32>(),
                                        info.numel(),
                                    )
                                }
                                .to_vec()
                            } else {
                                gguf.read_tensor_bf16(&gguf_name)?
                                    .iter()
                                    .map(|v| v.to_f32())
                                    .collect()
                            };
                            // SSM norm does NOT have +1 offset in GGUF
                            // (verified: GGUF[0]=0.884 == HF[0]=0.884).
                            // Only RMSNorm layer norms have the offset.
                            ctx.stream.clone_htod(&f32s)?
                        },
                        // out_proj: V-reorder columns (input dim = num_v × hd_v)
                        out_proj: load_tensor_2d_gguf_v_reorder_cols(
                            ctx,
                            gguf,
                            &format!("{ap}.out_proj.weight"),
                            num_k,
                            vpk,
                            hd_v,
                        )?,
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
                    let gate_up = DeviceMatrix::concat_rows(ctx, &[&gate, &up])?;
                    common::MLP {
                        gate_proj: gate,
                        up_proj: up,
                        down_proj: down,
                        gate_up_proj: Some(gate_up),
                    }
                },
            });

            if (i + 1) % 8 == 0 || i + 1 == config.num_hidden_layers {
                info!("GGUF: loaded layer {}/{}", i + 1, config.num_hidden_layers);
            }
        }

        let norm = load_norm(ctx, gguf, &format!("{wp}.norm.weight"))?;
        let (cos_cache, sin_cache) =
            precompute_rope(ctx, config.head_dim, 4096, config.rope_theta)?;

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
            enable_cuda_graph,
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

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

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
}
