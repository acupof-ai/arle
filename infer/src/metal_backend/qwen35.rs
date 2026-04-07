use std::{path::Path, time::Instant};

use anyhow::{Context, Result};
use mlx_rs::{
    Array, Dtype, StreamOrDevice,
    ops::{
        self, indexing::TryIndexMutOp, indexing::TryIndexOp, indexing::take_axis, reshape,
        split_sections, transpose_axes, zeros_dtype_device,
    },
};

use super::{
    KV_CACHE_CHUNK, MetalModelArch, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
    MlpInputProjection, WeightTensor, clear_metal_cache, extend_kv_cache, gpu_sample_token, linear,
    load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map,
    merge_quantized_projection_rows, tensor_get, tie_lm_head_from_embed_tokens,
};
use crate::{
    metal_gdr::{MetalLinearAttnWeights, MetalRecurrentState, metal_gdr_decode_step},
    sampler::SamplingParams,
};

pub(super) struct MetalQwen35FullAttentionWeights {
    pub(super) q_proj: WeightTensor,
    pub(super) k_proj: WeightTensor,
    pub(super) v_proj: WeightTensor,
    pub(super) o_proj: WeightTensor,
    pub(super) q_norm: Array,
    pub(super) k_norm: Array,
}

pub(super) enum MetalQwen35Attention {
    Full(MetalQwen35FullAttentionWeights),
    Linear(MetalLinearAttnWeights),
}

pub(super) struct MetalQwen35BlockWeights {
    pub(super) input_layernorm: Array,
    pub(super) attention: MetalQwen35Attention,
    pub(super) post_attention_layernorm: Array,
    pub(super) mlp_inputs: MlpInputProjection,
    pub(super) down_proj: WeightTensor,
}

pub(super) struct Qwen35MetalWeights {
    pub(super) embed_tokens: Array,
    pub(super) layers: Vec<MetalQwen35BlockWeights>,
    pub(super) norm: Array,
    pub(super) lm_head: WeightTensor,
}

pub(super) fn metal_generate_qwen35(
    input_ids: &[u32],
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<super::MetalGenerateOutput> {
    if max_new_tokens == 0 {
        return Ok(super::MetalGenerateOutput {
            tokens: Vec::new(),
            finish_reason: "length",
            ttft_ms: 0.0,
            total_time_ms: 0.0,
        });
    }
    anyhow::ensure!(
        !input_ids.is_empty(),
        "Qwen3.5 Metal generation requires at least one prompt token"
    );

    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("Qwen3.5 Metal path requires a Qwen3.5 config");
    };

    log::info!("Metal fused path: Qwen3.5 Hybrid (Rust/MLX)");

    let num_full_layers = arch.num_full_attention_layers();
    let prefill_len = input_ids.len() as i32;
    let initial_cap = ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
    let cache_shape = [
        1i32,
        config.num_key_value_heads as i32,
        initial_cap,
        config.head_dim as i32,
    ];
    let mut kv_capacity = initial_cap;
    let mut k_caches: Vec<Array> = (0..num_full_layers)
        .map(|_| zeros_dtype_device(&cache_shape, Dtype::Bfloat16, StreamOrDevice::default()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("pre-alloc qwen3.5 k_caches")?;
    let mut v_caches: Vec<Array> = (0..num_full_layers)
        .map(|_| zeros_dtype_device(&cache_shape, Dtype::Bfloat16, StreamOrDevice::default()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("pre-alloc qwen3.5 v_caches")?;

    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut cache_len = 0i32;
    let mut logits = None;
    for &token in input_ids {
        logits = Some(qwen35_forward_step(
            token,
            weights,
            config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        )?);
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
    }

    let mut logits = logits.context("Qwen3.5 prompt produced no logits")?;
    let mut generated = Vec::new();
    let mut ttft_ms = 0.0;

    let finish_reason = loop {
        let next_token = gpu_sample_token(&logits, params)
            .context("sample qwen3.5 token")?
            .item::<i32>() as u32;

        if generated.is_empty() {
            ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "  TTFT: {ttft_ms:.1}ms (prefill {} tokens)",
                input_ids.len()
            );
        }

        let stop = (!params.ignore_eos && config.is_stop_token(next_token))
            || params.stop_token_ids.contains(&next_token);
        generated.push(next_token);
        on_token(next_token)?;

        if stop {
            break "stop";
        }
        if generated.len() >= max_new_tokens {
            break "length";
        }

        if cache_len + 1 > kv_capacity {
            let new_cap = kv_capacity + KV_CACHE_CHUNK;
            for li in 0..num_full_layers {
                extend_kv_cache(
                    &mut k_caches[li],
                    config.num_key_value_heads as i32,
                    config.head_dim as i32,
                    new_cap,
                )?;
                extend_kv_cache(
                    &mut v_caches[li],
                    config.num_key_value_heads as i32,
                    config.head_dim as i32,
                    new_cap,
                )?;
            }
            kv_capacity = new_cap;
        }

        if !generated.is_empty() && generated.len().is_multiple_of(256) {
            clear_metal_cache();
        }

        logits = qwen35_forward_step(
            next_token,
            weights,
            config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        )?;
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
    };

    let elapsed = t0.elapsed().as_secs_f64();
    let total_time_ms = elapsed * 1000.0;
    let decode_elapsed = (elapsed - ttft_ms / 1000.0).max(1e-9);
    let tps = generated.len() as f64 / decode_elapsed;
    log::info!("  generated {} tokens  ({tps:.1} tok/s)", generated.len());

    Ok(super::MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen35_forward_step(
    token_id: u32,
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_caches: &mut [Array],
    v_caches: &mut [Array],
    recurrent: &mut MetalRecurrentState,
    cache_len: i32,
) -> Result<Array> {
    let token_arr = Array::from_slice(&[token_id as i32], &[1]);
    let mut x = take_axis(&weights.embed_tokens, &token_arr, 0).context("embedding take_axis")?;
    let mut full_idx = 0usize;
    let mut linear_idx = 0usize;

    for layer in &weights.layers {
        let residual = x.clone();
        let normed = rms_norm_last_dim(
            &x,
            &layer.input_layernorm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        )
        .context("qwen3.5 input_layernorm")?;

        let attn_out = match &layer.attention {
            MetalQwen35Attention::Full(attn) => {
                let out = qwen35_full_attention_step(
                    &normed,
                    attn,
                    config,
                    arch,
                    &mut k_caches[full_idx],
                    &mut v_caches[full_idx],
                    cache_len,
                )?;
                full_idx += 1;
                out
            }
            MetalQwen35Attention::Linear(attn) => {
                let out = metal_gdr_decode_step(&normed, attn, recurrent, linear_idx, &arch.linear)
                    .context("qwen3.5 linear attention")?;
                linear_idx += 1;
                out
            }
        };

        x = ops::add(&residual, &attn_out).context("qwen3.5 residual + attention")?;

        let residual2 = x.clone();
        let xn = rms_norm_last_dim(
            &x,
            &layer.post_attention_layernorm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        )
        .context("qwen3.5 post_attention_layernorm")?;
        let (gate_raw, up) = layer.mlp_inputs.project(&xn).context("qwen3.5 mlp proj")?;
        let gate_raw = gate_raw
            .as_dtype(Dtype::Bfloat16)
            .context("gate_raw to bf16")?;
        let up = up.as_dtype(Dtype::Bfloat16).context("up to bf16")?;
        let gate = mlx_rs::nn::silu(&gate_raw).context("qwen3.5 silu gate")?;
        let fused = ops::multiply(&gate, &up)
            .context("qwen3.5 gate*up")?
            .as_dtype(Dtype::Bfloat16)
            .context("gate*up to bf16")?;
        let mlp = linear(&fused, &layer.down_proj)
            .context("qwen3.5 down_proj")?
            .as_dtype(Dtype::Bfloat16)
            .context("mlp to bf16")?;
        x = ops::add(&residual2, &mlp).context("qwen3.5 residual + mlp")?;
    }

    let final_norm = rms_norm_last_dim(
        &x,
        &weights.norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    )
    .context("qwen3.5 final norm")?;
    linear(&final_norm, &weights.lm_head).context("qwen3.5 lm_head")
}

fn qwen35_full_attention_step(
    x: &Array,
    attn: &MetalQwen35FullAttentionWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_cache: &mut Array,
    v_cache: &mut Array,
    cache_len: i32,
) -> Result<Array> {
    use mlx_rs::fast;

    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let q_dim = n_heads * head_dim;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

    let q_full = linear(x, &attn.q_proj)
        .context("qwen3.5 q_proj")?
        .as_dtype(Dtype::Bfloat16)
        .context("q_full to bf16")?;
    let q_full = reshape(&q_full, &[1, 1, n_heads, head_dim * 2]).context("reshape q_full")?;
    let [q_heads, gate_heads] = <[Array; 2]>::try_from(
        split_sections(&q_full, &[head_dim], -1).context("split q/gate per head")?,
    )
    .map_err(|parts| anyhow::anyhow!("expected q+gate split, got {}", parts.len()))?;
    let k_raw = linear(x, &attn.k_proj)
        .context("qwen3.5 k_proj")?
        .as_dtype(Dtype::Bfloat16)
        .context("k_raw to bf16")?;
    let v_raw = linear(x, &attn.v_proj)
        .context("qwen3.5 v_proj")?
        .as_dtype(Dtype::Bfloat16)
        .context("v_raw to bf16")?;

    let q = rms_norm_last_dim(
        &q_heads,
        &attn.q_norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    )
    .context("q norm")?;
    let q = transpose_axes(&q, &[0, 2, 1, 3]).context("transpose q")?;
    let q = fast::rope(
        &q,
        arch.rotary_dim as i32,
        false,
        config.rope_theta as f32,
        1.0f32,
        cache_len,
        None,
    )
    .context("rope q")?;

    let k = reshape(&k_raw, &[1, 1, n_kv_heads, head_dim]).context("reshape k")?;
    let k = rms_norm_last_dim(
        &k,
        &attn.k_norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    )
    .context("k norm")?;
    let k = transpose_axes(&k, &[0, 2, 1, 3]).context("transpose k")?;
    let k = fast::rope(
        &k,
        arch.rotary_dim as i32,
        false,
        config.rope_theta as f32,
        1.0f32,
        cache_len,
        None,
    )
    .context("rope k")?;

    let v = reshape(&v_raw, &[1, 1, n_kv_heads, head_dim]).context("reshape v")?;
    let v = transpose_axes(&v, &[0, 2, 1, 3]).context("transpose v")?;

    let end_pos = cache_len + 1;
    k_cache
        .try_index_mut((.., .., cache_len..end_pos, ..), &k)
        .context("write qwen3.5 k_cache")?;
    v_cache
        .try_index_mut((.., .., cache_len..end_pos, ..), &v)
        .context("write qwen3.5 v_cache")?;
    let k_full = k_cache
        .try_index((.., .., 0i32..end_pos, ..))
        .context("slice qwen3.5 k_cache")?;
    let v_full = v_cache
        .try_index((.., .., 0i32..end_pos, ..))
        .context("slice qwen3.5 v_cache")?;

    let attn_out = fast::scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None)
        .context("qwen3.5 sdpa")?;
    let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]).context("transpose attn_out")?;
    let attn_out = reshape(&attn_out, &[1, q_dim]).context("reshape attn_out")?;
    let gate = reshape(&gate_heads, &[1, q_dim]).context("reshape gate")?;
    let gate = mlx_rs::nn::sigmoid(&gate.as_dtype(Dtype::Float32)?).context("sigmoid gate")?;
    let gated = ops::multiply(&attn_out.as_dtype(Dtype::Float32)?, &gate)
        .context("apply full-attn gate")?
        .as_dtype(Dtype::Bfloat16)
        .context("gated attn to bf16")?;
    linear(&gated, &attn.o_proj)
        .context("qwen3.5 o_proj")?
        .as_dtype(Dtype::Bfloat16)
        .context("o_proj to bf16")
}

fn rms_norm_last_dim(x: &Array, weight: &Array, eps: f32, offset: bool) -> Result<Array> {
    let last_dim = x
        .shape()
        .last()
        .copied()
        .context("rms_norm_last_dim missing last dimension")? as f32;
    let x = x.as_dtype(Dtype::Float32).context("x to f32")?;
    let weight = weight.as_dtype(Dtype::Float32).context("weight to f32")?;
    let inv_dim = Array::from_slice(&[1.0f32 / last_dim], &[1]);
    let eps_arr = Array::from_slice(&[eps], &[1]);
    let one = Array::from_slice(&[1.0f32], &[1]);

    let sq = ops::multiply(&x, &x).context("square x")?;
    let sum_sq = sq.sum_axis(-1, true).context("sum square")?;
    let mean_sq = ops::multiply(&sum_sq, &inv_dim).context("mean square")?;
    let inv_rms = ops::add(&mean_sq, &eps_arr)
        .context("add eps")?
        .sqrt()
        .context("sqrt rms")?
        .reciprocal()
        .context("reciprocal rms")?;
    let normed = ops::multiply(&x, &inv_rms).context("apply rms")?;
    let scale = if offset {
        ops::add(&weight, &one).context("offset norm scale")?
    } else {
        weight
    };
    ops::multiply(&normed, &scale)
        .context("apply norm scale")?
        .as_dtype(Dtype::Bfloat16)
        .context("norm output to bf16")
}

pub(super) fn load_qwen35_metal_weights(
    model_dir: &Path,
    config: &MetalModelConfig,
) -> Result<Qwen35MetalWeights> {
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("Qwen3.5 Metal loader requires a Qwen3.5 config");
    };
    let tensors = load_tensor_map(model_dir)?;

    let prefix = ["language_model.model", "model.language_model", "model"]
        .into_iter()
        .find(|candidate| {
            tensors.contains_key(&format!("{candidate}.embed_tokens.weight"))
                && tensors.contains_key(&format!("{candidate}.norm.weight"))
        })
        .context("could not detect Qwen3.5 text weight prefix")?;

    let get = |name: &str| tensor_get(&tensors, name);
    let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, config.quantization);

    let embed_base = format!("{prefix}.embed_tokens");
    let embed_tokens = load_embed_tokens_from_tensors(&tensors, &embed_base, config.quantization)?;
    let norm = get(&format!("{prefix}.norm.weight"))?;
    let lm_head = load_lm_head(
        &tensors,
        &[
            "lm_head".to_string(),
            "language_model.lm_head".to_string(),
            format!("{prefix}.lm_head"),
        ],
        &embed_tokens,
        &load_proj,
    )?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let layer_prefix = format!("{prefix}.layers.{i}");
        let attention = match arch.layer_types[i] {
            MetalQwen35LayerType::FullAttention => {
                let attn_prefix = format!("{layer_prefix}.self_attn");
                MetalQwen35Attention::Full(MetalQwen35FullAttentionWeights {
                    q_proj: load_proj(&format!("{attn_prefix}.q_proj"))?,
                    k_proj: load_proj(&format!("{attn_prefix}.k_proj"))?,
                    v_proj: load_proj(&format!("{attn_prefix}.v_proj"))?,
                    o_proj: load_proj(&format!("{attn_prefix}.o_proj"))?,
                    q_norm: get(&format!("{attn_prefix}.q_norm.weight"))?,
                    k_norm: get(&format!("{attn_prefix}.k_norm.weight"))?,
                })
            }
            MetalQwen35LayerType::LinearAttention => {
                let attn_prefix = format!("{layer_prefix}.linear_attn");
                MetalQwen35Attention::Linear(MetalLinearAttnWeights {
                    in_proj_qkv: load_proj(&format!("{attn_prefix}.in_proj_qkv"))?,
                    in_proj_z: load_proj(&format!("{attn_prefix}.in_proj_z"))?,
                    in_proj_beta: load_proj(&format!("{attn_prefix}.in_proj_b"))?,
                    in_proj_alpha: load_proj(&format!("{attn_prefix}.in_proj_a"))?,
                    conv1d_weight: load_conv1d_weight(
                        &get(&format!("{attn_prefix}.conv1d.weight"))?,
                        &arch.linear,
                    )?,
                    dt_bias: get(&format!("{attn_prefix}.dt_bias"))?
                        .as_dtype(Dtype::Float32)
                        .context("dt_bias to f32")?,
                    a_log: get(&format!("{attn_prefix}.A_log"))?
                        .as_dtype(Dtype::Float32)
                        .context("A_log to f32")?,
                    norm_weight: get(&format!("{attn_prefix}.norm.weight"))?
                        .as_dtype(Dtype::Float32)
                        .context("linear norm.weight to f32")?,
                    out_proj: load_proj(&format!("{attn_prefix}.out_proj"))?,
                })
            }
        };

        let gate_proj = load_proj(&format!("{layer_prefix}.mlp.gate_proj"))?;
        let up_proj = load_proj(&format!("{layer_prefix}.mlp.up_proj"))?;
        let gate_dim = gate_proj.output_dim()?;
        let up_dim = up_proj.output_dim()?;
        let mlp_inputs =
            if let Some(gate_up_proj) = merge_quantized_projection_rows(&[&gate_proj, &up_proj])? {
                MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                    up_dim,
                }
            } else {
                MlpInputProjection::Split { gate_proj, up_proj }
            };

        layers.push(MetalQwen35BlockWeights {
            input_layernorm: get(&format!("{layer_prefix}.input_layernorm.weight"))?,
            attention,
            post_attention_layernorm: get(&format!(
                "{layer_prefix}.post_attention_layernorm.weight"
            ))?,
            mlp_inputs,
            down_proj: load_proj(&format!("{layer_prefix}.mlp.down_proj"))?,
        });
    }

    Ok(Qwen35MetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
    })
}

fn load_lm_head(
    tensors: &super::TensorMap,
    candidates: &[String],
    embed_tokens: &Array,
    load_proj: &impl Fn(&str) -> Result<WeightTensor>,
) -> Result<WeightTensor> {
    for candidate in candidates {
        if tensors.contains_key(&format!("{candidate}.weight"))
            || tensors.contains_key(&format!("{candidate}.scales"))
        {
            return load_proj(candidate);
        }
    }

    tie_lm_head_from_embed_tokens(embed_tokens)
}

fn load_conv1d_weight(weight: &Array, linear: &crate::metal_gdr::MetalGdrConfig) -> Result<Array> {
    match weight.shape() {
        [qkv_dim, kernel] => {
            anyhow::ensure!(
                *qkv_dim == linear.qkv_dim() as i32 && *kernel == linear.conv_kernel as i32,
                "unexpected conv1d weight shape {:?}, expected [{}, {}]",
                weight.shape(),
                linear.qkv_dim(),
                linear.conv_kernel
            );
            weight
                .as_dtype(Dtype::Float32)
                .context("conv1d weight to f32")
        }
        [qkv_dim, dim1, dim2]
            if *qkv_dim == linear.qkv_dim() as i32
                && ((*dim1 == 1 && *dim2 == linear.conv_kernel as i32)
                    || (*dim1 == linear.conv_kernel as i32 && *dim2 == 1)) =>
        {
            weight
                .reshape(&[linear.qkv_dim() as i32, linear.conv_kernel as i32])
                .context("reshape conv1d weight")?
                .as_dtype(Dtype::Float32)
                .context("conv1d weight to f32")
        }
        shape => anyhow::bail!("unsupported conv1d weight shape {:?}", shape),
    }
}
