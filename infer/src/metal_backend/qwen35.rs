use std::{path::Path, time::Instant};

use anyhow::{Context, Result};

use crate::mlx::{
    self, Dtype, MlxArray, add, as_dtype, concatenate_axis, multiply, reshape, rms_norm, rope,
    scaled_dot_product_attention, sigmoid, silu, slice, slice_update, take_axis, transpose_axes,
    zeros,
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
    pub(super) q_norm: MlxArray,
    pub(super) k_norm: MlxArray,
}

pub(super) enum MetalQwen35Attention {
    Full(MetalQwen35FullAttentionWeights),
    Linear(MetalLinearAttnWeights),
}

pub(super) struct MetalQwen35BlockWeights {
    pub(super) input_layernorm: MlxArray,
    pub(super) attention: MetalQwen35Attention,
    pub(super) post_attention_layernorm: MlxArray,
    pub(super) mlp_inputs: MlpInputProjection,
    pub(super) down_proj: WeightTensor,
}

pub(super) struct Qwen35MetalWeights {
    pub(super) embed_tokens: MlxArray,
    pub(super) layers: Vec<MetalQwen35BlockWeights>,
    pub(super) norm: MlxArray,
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
    let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();

    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut cache_len = 0i32;
    let mut logits = None;
    for &token in input_ids {
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        logits = Some(qwen35_forward_step(
            &token_arr,
            weights,
            config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        ));
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
    }

    let logits = logits.context("Qwen3.5 prompt produced no logits")?;
    let mut generated = Vec::new();
    let mut ttft_ms = 0.0;

    // ── mlx_lm-style double-buffered decode loop ────────────────────────
    //
    // Pass the LAZY sample token `y` directly to forward_step (not the
    // materialized u32). This lets CPU build the next graph while GPU
    // executes the current step, overlapping ~1.5ms of CPU work.
    //
    //   y = sample(prefill)     async_eval(y)
    //   loop:
    //     next_y = sample(fwd(y))   ← CPU builds graph (y still lazy)
    //     async_eval(next_y)        ← submit to GPU
    //     eval(y)                   ← wait for current
    //     token = y.item()          ← read result
    //     y = next_y

    let mut y = gpu_sample_token(&logits, params)?;
    crate::mlx::async_eval(&[&y]);

    let finish_reason = 'decode: loop {
        // Build NEXT step's graph while GPU computes CURRENT y.
        // y is lazy — forward_step builds a graph that depends on it.
        let next_logits = qwen35_forward_step(
            &y,
            weights,
            config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        );
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
        let next_y = gpu_sample_token(&next_logits, params)?;
        crate::mlx::async_eval(&[&next_y]);

        // Now wait for CURRENT y and process the token.
        crate::mlx::eval(&[&y]);
        let next_token = y.item_i32() as u32;

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
            break 'decode "stop";
        }
        if generated.len() >= max_new_tokens {
            break 'decode "length";
        }

        // Grow KV cache if needed (rare — only every 256 tokens)
        if cache_len + 1 > kv_capacity {
            let new_cap = kv_capacity + KV_CACHE_CHUNK;
            for li in 0..num_full_layers {
                extend_kv_cache(
                    &mut k_caches[li],
                    config.num_key_value_heads as i32,
                    config.head_dim as i32,
                    new_cap,
                );
                extend_kv_cache(
                    &mut v_caches[li],
                    config.num_key_value_heads as i32,
                    config.head_dim as i32,
                    new_cap,
                );
            }
            kv_capacity = new_cap;
        }
        if generated.len().is_multiple_of(256) {
            clear_metal_cache();
        }

        y = next_y;
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
    token: &MlxArray,
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    recurrent: &mut MetalRecurrentState,
    cache_len: i32,
) -> MlxArray {
    let mut x = take_axis(&weights.embed_tokens, token, 0);
    let mut full_idx = 0usize;
    let mut linear_idx = 0usize;

    for layer in &weights.layers {
        let residual = x.clone();

        // C++ fused blocks do norm internally; Rust fallback needs explicit norm.
        let attn_out = match &layer.attention {
            MetalQwen35Attention::Full(attn) => {
                let out = fused_full_attn_step(
                    &x,
                    &layer.input_layernorm,
                    attn,
                    config,
                    arch,
                    &mut k_caches[full_idx],
                    &mut v_caches[full_idx],
                    cache_len,
                );
                full_idx += 1;
                out
            }
            MetalQwen35Attention::Linear(attn) => {
                let out = fused_gdr_step(
                    &x,
                    &layer.input_layernorm,
                    attn,
                    recurrent,
                    linear_idx,
                    &arch.linear,
                    config,
                );
                linear_idx += 1;
                out
            }
        };

        x = add(&residual, &attn_out);

        let residual2 = x.clone();
        let xn = rms_norm_last_dim(
            &x,
            &layer.post_attention_layernorm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        );
        let (gate_raw, up) = mlp_project(&layer.mlp_inputs, &xn);
        // SwiGLU: silu(gate) * up → compiled kernel. No casts needed.
        let fused_val = multiply(&silu(&gate_raw), &up);
        let mlp = linear(&fused_val, &layer.down_proj);
        x = add(&residual2, &mlp);
    }

    let final_norm = rms_norm_last_dim(
        &x,
        &weights.norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    linear(&final_norm, &weights.lm_head)
}

// ── Fused C++ wrappers (dispatch to C++ FFI or Rust fallback) ────────────────

/// Helper: extract raw mlx_array + quant params from a WeightTensor.
/// For Dense: returns (w_t, empty, empty, 0, 0, false).
/// For Quantized: returns (w, scales, biases, group_size, bits, true).
#[cfg(metal_qwen35_fused_ops)]
fn wt_parts(
    wt: &WeightTensor,
) -> (
    mlx_sys::mlx_array,
    mlx_sys::mlx_array,
    mlx_sys::mlx_array,
    i32,
    i32,
    bool,
) {
    match wt {
        WeightTensor::Dense(w_t) => {
            let empty = unsafe { mlx_sys::mlx_array_new() };
            (w_t.as_raw(), empty, empty, 0, 0, false)
        }
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => (
            w.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            *group_size,
            *bits,
            true,
        ),
    }
}

#[cfg(metal_qwen35_fused_ops)]
#[allow(clippy::too_many_arguments)]
fn fused_full_attn_step(
    x: &MlxArray,
    input_norm_w: &MlxArray,
    attn: &MetalQwen35FullAttentionWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_cache: &mut MlxArray,
    v_cache: &mut MlxArray,
    cache_len: i32,
) -> MlxArray {
    let (q_w, q_s, q_b, gs, bits, is_q) = wt_parts(&attn.q_proj);
    let (k_w, k_s, k_b, _, _, _) = wt_parts(&attn.k_proj);
    let (v_w, v_s, v_b, _, _, _) = wt_parts(&attn.v_proj);
    let (o_w, o_s, o_b, _, _, _) = wt_parts(&attn.o_proj);
    let n_heads = config.num_attention_heads as i32;
    let n_kv = config.num_key_value_heads as i32;
    let hd = config.head_dim as i32;
    let attn_scale = 1.0f32 / (hd as f32).sqrt();

    let result_raw: mlx_sys::mlx_array = unsafe {
        let mut r = std::mem::MaybeUninit::<mlx_sys::mlx_array>::uninit();
        super::metal_ffi::metal_qwen35_full_attn_block(
            x.as_raw(),
            input_norm_w.as_raw(),
            q_w,
            q_s,
            q_b,
            k_w,
            k_s,
            k_b,
            v_w,
            v_s,
            v_b,
            o_w,
            o_s,
            o_b,
            attn.q_norm.as_raw(),
            attn.k_norm.as_raw(),
            n_heads,
            n_kv,
            hd,
            attn_scale,
            config.rope_theta as f32,
            arch.rotary_dim as i32,
            config.rms_norm_eps as f32,
            gs,
            bits,
            is_q,
            config.norm_weight_mode.uses_offset(),
            k_cache.as_raw_mut(),
            v_cache.as_raw_mut(),
            cache_len,
            r.as_mut_ptr(),
        );
        r.assume_init()
    };
    unsafe { MlxArray::from_raw(result_raw) }
}

#[cfg(not(metal_qwen35_fused_ops))]
#[allow(clippy::too_many_arguments)]
fn fused_full_attn_step(
    x: &MlxArray,
    input_norm_w: &MlxArray,
    attn: &MetalQwen35FullAttentionWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_cache: &mut MlxArray,
    v_cache: &mut MlxArray,
    cache_len: i32,
) -> MlxArray {
    let normed = rms_norm_last_dim(
        x,
        input_norm_w,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    qwen35_full_attention_step(&normed, attn, config, arch, k_cache, v_cache, cache_len)
}

// GDR step uses the Rust path (compiled ops + Metal kernel).
#[allow(clippy::too_many_arguments)]
fn fused_gdr_step(
    x: &MlxArray,
    input_norm_w: &MlxArray,
    attn: &MetalLinearAttnWeights,
    recurrent: &mut MetalRecurrentState,
    layer_idx: usize,
    gdr_cfg: &crate::metal_gdr::MetalGdrConfig,
    config: &MetalModelConfig,
) -> MlxArray {
    let normed = rms_norm_last_dim(
        x,
        input_norm_w,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    metal_gdr_decode_step(&normed, attn, recurrent, layer_idx, gdr_cfg)
}

// ── Rust fallback implementations (used when metal_fused_ops not compiled) ───

fn qwen35_full_attention_step(
    x: &MlxArray,
    attn: &MetalQwen35FullAttentionWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_cache: &mut MlxArray,
    v_cache: &mut MlxArray,
    cache_len: i32,
) -> MlxArray {
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let q_dim = n_heads * head_dim;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();

    let q_full = linear(x, &attn.q_proj);
    let q_full = reshape(&q_full, &[1, 1, n_heads, head_dim * 2]);
    // Split q and gate: q_full is [1, 1, n_heads, head_dim*2], split at head_dim on last axis
    let q_heads = slice(
        &q_full,
        &[0, 0, 0, 0],
        &[1, 1, n_heads, head_dim],
        &[1, 1, 1, 1],
    );
    let gate_heads = slice(
        &q_full,
        &[0, 0, 0, head_dim],
        &[1, 1, n_heads, head_dim * 2],
        &[1, 1, 1, 1],
    );

    let k_raw = linear(x, &attn.k_proj);
    let v_raw = linear(x, &attn.v_proj);

    let q = rms_norm_last_dim(
        &q_heads,
        &attn.q_norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    let q = transpose_axes(&q, &[0, 2, 1, 3]);
    let q = rope(
        &q,
        arch.rotary_dim as i32,
        false,
        config.rope_theta as f32,
        1.0f32,
        cache_len,
    );

    let k = reshape(&k_raw, &[1, 1, n_kv_heads, head_dim]);
    let k = rms_norm_last_dim(
        &k,
        &attn.k_norm,
        config.rms_norm_eps as f32,
        config.norm_weight_mode.uses_offset(),
    );
    let k = transpose_axes(&k, &[0, 2, 1, 3]);
    let k = rope(
        &k,
        arch.rotary_dim as i32,
        false,
        config.rope_theta as f32,
        1.0f32,
        cache_len,
    );

    let v = reshape(&v_raw, &[1, 1, n_kv_heads, head_dim]);
    let v = transpose_axes(&v, &[0, 2, 1, 3]);

    // KV cache update
    let end_pos = cache_len + 1;
    *k_cache = slice_update(
        k_cache,
        &k,
        &[0, 0, cache_len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );
    *v_cache = slice_update(
        v_cache,
        &v,
        &[0, 0, cache_len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );
    let k_full = slice(
        k_cache,
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );
    let v_full = slice(
        v_cache,
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );

    let attn_out = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None);
    let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
    let attn_out = reshape(&attn_out, &[1, q_dim]);
    let gate = reshape(&gate_heads, &[1, q_dim]);
    let gate = sigmoid(&as_dtype(&gate, Dtype::Float32));
    let gated = as_dtype(
        &multiply(&as_dtype(&attn_out, Dtype::Float32), &gate),
        Dtype::Bfloat16,
    );
    linear(&gated, &attn.o_proj)
}

fn rms_norm_last_dim(x: &MlxArray, weight: &MlxArray, eps: f32, offset: bool) -> MlxArray {
    if !offset {
        // Use MLX's fused fast.rms_norm — single op instead of 10 manual ops.
        // This is the same as mlx_lm's nn.RMSNorm.__call__.
        return rms_norm(x, weight, eps);
    }
    // Offset mode: weight = weight + 1, then manual norm.
    use crate::mlx::{reciprocal, sqrt, sum_axis};
    let last_dim = *x.shape().last().expect("rms_norm_last_dim: empty shape") as f32;
    let x = as_dtype(x, Dtype::Float32);
    let weight = as_dtype(weight, Dtype::Float32);
    let inv_dim = MlxArray::from_slice_f32(&[1.0f32 / last_dim], &[1]);
    let eps_arr = MlxArray::from_slice_f32(&[eps], &[1]);
    let one = MlxArray::from_slice_f32(&[1.0f32], &[1]);
    let sq = multiply(&x, &x);
    let sum_sq = sum_axis(&sq, -1, true);
    let mean_sq = multiply(&sum_sq, &inv_dim);
    let inv_rms = reciprocal(&sqrt(&add(&mean_sq, &eps_arr)));
    let normed = multiply(&x, &inv_rms);
    let scale = add(&weight, &one);
    as_dtype(&multiply(&normed, &scale), Dtype::Bfloat16)
}

/// MLP projection helper — replaces the mlx_rs method on MlpInputProjection.
fn mlp_project(mlp: &MlpInputProjection, x: &MlxArray) -> (MlxArray, MlxArray) {
    match mlp {
        MlpInputProjection::Split { gate_proj, up_proj } => {
            (linear(x, gate_proj), linear(x, up_proj))
        }
        MlpInputProjection::MergedQuantized {
            gate_up_proj,
            gate_dim,
            up_dim,
        } => {
            let gate_up = linear(x, gate_up_proj);
            let gate = slice(&gate_up, &[0, 0], &[1, *gate_dim], &[1, 1]);
            let up = slice(
                &gate_up,
                &[0, *gate_dim],
                &[1, *gate_dim + *up_dim],
                &[1, 1],
            );
            (gate, up)
        }
    }
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

    log::info!(
        "  {} layers ({} full attention, {} GDR)",
        config.num_hidden_layers,
        arch.num_full_attention_layers(),
        arch.num_linear_attention_layers(),
    );
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
                let qkv_proj = load_proj(&format!("{attn_prefix}.in_proj_qkv"))?;
                let z_proj = load_proj(&format!("{attn_prefix}.in_proj_z"))?;
                let beta_proj = load_proj(&format!("{attn_prefix}.in_proj_b"))?;
                let alpha_proj = load_proj(&format!("{attn_prefix}.in_proj_a"))?;
                let qkv_dim = qkv_proj.output_dim()?;
                let z_dim = z_proj.output_dim()?;
                let beta_dim = beta_proj.output_dim()?;

                // Merge QKV+Z → single matmul (saves 1 dispatch per layer)
                let in_proj_qkvz = match merge_quantized_projection_rows(&[&qkv_proj, &z_proj])? {
                    Some(merged) => merged,
                    None => match (&qkv_proj, &z_proj) {
                        (WeightTensor::Dense(qkv_t), WeightTensor::Dense(z_t)) => {
                            WeightTensor::Dense(concatenate_axis(&[qkv_t.clone(), z_t.clone()], 1))
                        }
                        _ => anyhow::bail!(
                            "layer {i}: mixed dense/quantized QKV+Z projections not supported"
                        ),
                    },
                };

                // Merge Beta+Alpha → single matmul (saves 1 dispatch per layer)
                let in_proj_ba = match merge_quantized_projection_rows(&[&beta_proj, &alpha_proj])?
                {
                    Some(merged) => merged,
                    None => match (&beta_proj, &alpha_proj) {
                        (WeightTensor::Dense(b_t), WeightTensor::Dense(a_t)) => {
                            WeightTensor::Dense(concatenate_axis(&[b_t.clone(), a_t.clone()], 1))
                        }
                        _ => anyhow::bail!(
                            "layer {i}: mixed dense/quantized beta+alpha projections not supported"
                        ),
                    },
                };

                let inv_scale = 1.0 / (arch.linear.key_dim as f32).sqrt();
                MetalQwen35Attention::Linear(MetalLinearAttnWeights {
                    in_proj_qkvz,
                    in_proj_ba,
                    qkvz_split: (qkv_dim, z_dim),
                    ba_num_heads: beta_dim,
                    conv1d_weight: load_conv1d_weight(
                        &get(&format!("{attn_prefix}.conv1d.weight"))?,
                        &arch.linear,
                    )?,
                    dt_bias: get(&format!("{attn_prefix}.dt_bias"))?,
                    a_log: as_dtype(&get(&format!("{attn_prefix}.A_log"))?, Dtype::Float32),
                    norm_weight: get(&format!("{attn_prefix}.norm.weight"))?,
                    out_proj: load_proj(&format!("{attn_prefix}.out_proj"))?,
                    q_scale: MlxArray::scalar_f32(inv_scale * inv_scale),
                    k_scale: MlxArray::scalar_f32(inv_scale),
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
    embed_tokens: &MlxArray,
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

/// Load conv1d weight in nn.Conv1d format: [out_channels, kernel_size, in_channels/groups].
/// For depthwise conv (groups=C), shape is [C, K, 1]. Keep native dtype (bf16).
fn load_conv1d_weight(
    weight: &MlxArray,
    linear_cfg: &crate::metal_gdr::MetalGdrConfig,
) -> Result<MlxArray> {
    let c = linear_cfg.qkv_dim() as i32;
    let k = linear_cfg.conv_kernel as i32;
    match weight.shape() {
        // Already [C, K, 1] — nn.Conv1d format
        [ch, ks, 1] if *ch == c && *ks == k => Ok(weight.clone()),
        // [C, 1, K] — transpose to [C, K, 1]
        [ch, 1, ks] if *ch == c && *ks == k => Ok(reshape(weight, &[c, k, 1])),
        // [C, K] — reshape to [C, K, 1]
        [ch, ks] if *ch == c && *ks == k => Ok(reshape(weight, &[c, k, 1])),
        shape => anyhow::bail!(
            "unsupported conv1d weight shape {:?}, expected [{c}, {k}, 1]",
            shape
        ),
    }
}
