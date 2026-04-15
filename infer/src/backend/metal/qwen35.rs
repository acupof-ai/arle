use std::{path::Path, time::Instant};

use anyhow::{Context, Result};

use super::mlx::{
    Dtype, MlxArray, add, as_dtype, concatenate_axis, multiply, reshape, rms_norm, rope,
    scaled_dot_product_attention, sigmoid, silu, slice, slice_update, take_axis, transpose_axes,
    zeros,
};

use super::gdr::{MetalLinearAttnWeights, MetalRecurrentState, metal_gdr_decode_step};
use super::{
    KV_CACHE_CHUNK, MetalModelArch, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
    MlpInputProjection, WeightTensor, clear_metal_cache, extend_kv_cache, gpu_sample_token, linear,
    load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map,
    merge_quantized_projection_rows, tensor_get, tie_lm_head_from_embed_tokens,
};
use crate::sampler::SamplingParams;

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
    /// Individual gate/up projections used by the optional C++ step path.
    pub(super) gate_proj: WeightTensor,
    pub(super) up_proj: WeightTensor,
}

pub(super) struct Qwen35MetalWeights {
    pub(super) embed_tokens: MlxArray,
    pub(super) layers: Vec<MetalQwen35BlockWeights>,
    pub(super) norm: MlxArray,
    pub(super) lm_head: WeightTensor,
    /// Quantized embed weights for as_linear lm_head (when tied).
    /// Avoids 1.2GB dense matmul — uses 0.3GB quantized_matmul instead.
    pub(super) embed_quantized: Option<WeightTensor>,
    /// Optional C++ forward model handle. Eliminates most per-op FFI overhead.
    pub(super) cpp_model: Option<CppQwen35Model>,
}

/// RAII wrapper for the C++ Qwen35 forward model.
pub(crate) struct CppQwen35Model(*mut std::ffi::c_void);

impl Drop for CppQwen35Model {
    fn drop(&mut self) {
        unsafe { mlx_sys::qwen35_compiled_free(self.0) }
    }
}
unsafe impl Send for CppQwen35Model {}

fn use_qwen35_cpp_separate_proj() -> bool {
    std::env::var("AGENT_INFER_QWEN35_CPP_SEPARATE")
        .map(|value| value != "0")
        .unwrap_or(true)
}

impl CppQwen35Model {
    /// Wrap a raw C++ model pointer (takes ownership).
    pub(crate) fn from_raw(ptr: *mut std::ffi::c_void) -> Self {
        Self(ptr)
    }
    /// Build a C++ step model from loaded Rust weights. Returns None if weights
    /// are not fully supported by the C++ route.
    fn build(
        weights: &Qwen35MetalWeights,
        config: &MetalModelConfig,
        arch: &MetalQwen35ArchConfig,
    ) -> Option<Self> {
        let model = unsafe { mlx_sys::qwen35_compiled_new() };
        if model.is_null() {
            return None;
        }

        // Config
        unsafe {
            mlx_sys::qwen35_compiled_set_config(
                model,
                config.rope_theta as f32,
                config.rms_norm_eps as f32,
                config.num_attention_heads as i32,
                config.num_key_value_heads as i32,
                config.head_dim as i32,
                arch.rotary_dim as i32,
                config.hidden_size as i32,
            );
        }

        // Embed + norm + lm_head (supports both Dense and Quantized)
        let (lm_w, lm_s, lm_b, lm_gs, lm_bits) = match &weights.lm_head {
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
            ),
            WeightTensor::Dense(w_t) => (
                w_t.as_raw(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                0, // bits=0 signals Dense to C++
            ),
        };
        unsafe {
            mlx_sys::qwen35_compiled_set_embed(
                model,
                weights.embed_tokens.as_raw(),
                weights.norm.as_raw(),
                lm_w,
                lm_s,
                lm_b,
                lm_gs,
                lm_bits,
            );
            // Set quantized embed for as_linear lm_head (avoids 1.2GB dense matmul)
            // Only use embed_as_linear when lm_head is Dense (tied to embed_tokens).
            // If lm_head is already Quantized (independent), don't override it.
            if lm_bits == 0 {
                if let Some(WeightTensor::Quantized {
                    w,
                    scales,
                    biases,
                    group_size,
                    bits,
                }) = &weights.embed_quantized
                {
                    mlx_sys::qwen35_compiled_set_embed_as_linear(
                        model,
                        w.as_raw(),
                        scales.as_raw(),
                        biases.as_raw(),
                        *group_size,
                        *bits,
                    );
                    log::info!("  using quantized lm_head (as_linear, tied weights)");
                }
            }
        }

        // Layers
        for layer in &weights.layers {
            let (input_ln, post_ln) = (
                layer.input_layernorm.as_raw(),
                layer.post_attention_layernorm.as_raw(),
            );

            // MLP weights (shared by both attention types)
            let (gu_w, gu_s, gu_b, gu_gs, gu_bits, gate_dim) =
                if let MlpInputProjection::MergedQuantized {
                    gate_up_proj:
                        WeightTensor::Quantized {
                            w,
                            scales,
                            biases,
                            group_size,
                            bits,
                        },
                    gate_dim,
                    ..
                } = &layer.mlp_inputs
                {
                    (
                        w.as_raw(),
                        scales.as_raw(),
                        biases.as_raw(),
                        *group_size,
                        *bits,
                        *gate_dim,
                    )
                } else {
                    log::warn!("C++ forward model requires quantized MLP — falling back to Rust");
                    unsafe { mlx_sys::qwen35_compiled_free(model) };
                    return None;
                };
            let (dw_w, dw_s, dw_b) = if let WeightTensor::Quantized {
                w, scales, biases, ..
            } = &layer.down_proj
            {
                (w.as_raw(), scales.as_raw(), biases.as_raw())
            } else {
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            };

            match &layer.attention {
                MetalQwen35Attention::Full(attn) => {
                    let q = extract_qw(&attn.q_proj)?;
                    let k = extract_qw(&attn.k_proj)?;
                    let v = extract_qw(&attn.v_proj)?;
                    let o = extract_qw(&attn.o_proj)?;
                    unsafe {
                        mlx_sys::qwen35_compiled_push_full_attn(
                            model,
                            input_ln,
                            post_ln,
                            q.0,
                            q.1,
                            q.2,
                            q.3,
                            q.4,
                            k.0,
                            k.1,
                            k.2,
                            v.0,
                            v.1,
                            v.2,
                            o.0,
                            o.1,
                            o.2,
                            attn.q_norm.as_raw(),
                            attn.k_norm.as_raw(),
                            gu_w,
                            gu_s,
                            gu_b,
                            gu_gs,
                            gu_bits,
                            gate_dim,
                            dw_w,
                            dw_s,
                            dw_b,
                        );
                    }
                }
                MetalQwen35Attention::Linear(attn) => {
                    let qkvz = extract_qw(&attn.in_proj_qkvz)?;
                    let ba = extract_qw(&attn.in_proj_ba)?;
                    let out = extract_qw(&attn.out_proj)?;
                    unsafe {
                        mlx_sys::qwen35_compiled_push_gdr(
                            model,
                            input_ln,
                            post_ln,
                            qkvz.0,
                            qkvz.1,
                            qkvz.2,
                            qkvz.3,
                            qkvz.4,
                            attn.qkvz_split.0,
                            attn.qkvz_split.1,
                            ba.0,
                            ba.1,
                            ba.2,
                            ba.3,
                            ba.4,
                            attn.ba_num_heads,
                            attn.conv1d_weight.as_raw(),
                            arch.linear.conv_kernel as i32,
                            attn.a_log.as_raw(),
                            attn.dt_bias.as_raw(),
                            attn.norm_weight.as_raw(),
                            arch.linear.rms_norm_eps,
                            out.0,
                            out.1,
                            out.2,
                            out.3,
                            out.4,
                            arch.linear.num_key_heads as i32,
                            arch.linear.key_dim as i32,
                            arch.linear.num_value_heads as i32,
                            arch.linear.value_dim as i32,
                            gu_w,
                            gu_s,
                            gu_b,
                            gu_gs,
                            gu_bits,
                            gate_dim,
                            dw_w,
                            dw_s,
                            dw_b,
                        );
                    }
                    let separate_proj = use_qwen35_cpp_separate_proj();
                    if let (Some(qkv), Some(z), Some(b), Some(a), Some(gp), Some(up)) = (
                        extract_qw(&attn.in_proj_qkv),
                        extract_qw(&attn.in_proj_z),
                        extract_qw(&attn.in_proj_b),
                        extract_qw(&attn.in_proj_a),
                        extract_qw(&layer.gate_proj),
                        extract_qw(&layer.up_proj),
                    ) {
                        if separate_proj {
                            unsafe {
                                mlx_sys::qwen35_compiled_set_separate_proj(
                                    model, qkv.0, qkv.1, qkv.2, qkv.3, qkv.4, z.0, z.1, z.2, b.0,
                                    b.1, b.2, a.0, a.1, a.2, gp.0, gp.1, gp.2, gp.3, gp.4, up.0,
                                    up.1, up.2,
                                );
                            }
                        }
                        unsafe {
                            mlx_sys::qwen35_compiled_set_separate_mlp(
                                model, gp.0, gp.1, gp.2, gp.3, gp.4, up.0, up.1, up.2,
                            );
                        }
                    }
                }
            }
        }

        // Finalize (compile)
        let rc = unsafe { mlx_sys::qwen35_compiled_finalize(model) };
        if rc != 0 {
            log::warn!("C++ forward model finalize failed — falling back to Rust");
            unsafe { mlx_sys::qwen35_compiled_free(model) };
            return None;
        }

        log::info!(
            "  C++ forward model ready (all {} layers wired through one step call)",
            weights.layers.len()
        );
        Some(Self(model))
    }

    /// Run one decode step. Returns logits. Updates caches in place.
    pub(super) fn step(
        &self,
        token: &MlxArray,
        cache_pos: i32,
        kv_caches: &mut [MlxArray], // [k0, v0, k1, v1, ...] for full-attn layers
        gdr_states: &mut [MlxArray], // [gdr0, conv0, gdr1, conv1, ...] for GDR layers
    ) -> Result<MlxArray> {
        let n_kv = kv_caches.len() as i32;
        let n_gdr = gdr_states.len() as i32;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv as usize];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr as usize];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_step(
                self.0,
                token.as_raw(),
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                &raw mut out_logits,
                out_kv.as_mut_ptr(),
                out_gdr.as_mut_ptr(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        // Update caches in place
        for (i, ptr) in out_kv.into_iter().enumerate() {
            let old = std::mem::replace(&mut kv_caches[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        for (i, ptr) in out_gdr.into_iter().enumerate() {
            let old = std::mem::replace(&mut gdr_states[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(super) fn prefill(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
        kv_caches: &mut [MlxArray],
        gdr_states: &mut [MlxArray],
    ) -> Result<MlxArray> {
        let n_kv = kv_caches.len() as i32;
        let n_gdr = gdr_states.len() as i32;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv as usize];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr as usize];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_prefill(
                self.0,
                tokens.as_raw(),
                prompt_len,
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                &raw mut out_logits,
                out_kv.as_mut_ptr(),
                out_gdr.as_mut_ptr(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        for (i, ptr) in out_kv.into_iter().enumerate() {
            let old = std::mem::replace(&mut kv_caches[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        for (i, ptr) in out_gdr.into_iter().enumerate() {
            let old = std::mem::replace(&mut gdr_states[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    /// Full decode loop in C++ — all intermediates stay alive within the loop.
    #[allow(clippy::items_after_statements)]
    pub(crate) fn generate(
        &self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        stop_token_ids: &[u32],
        on_token: &mut impl FnMut(u32) -> Result<()>,
    ) -> Result<(Vec<u32>, f64, f64)> {
        // Returns (tokens, prefill_ms, decode_ms)
        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
        let stop_i32: Vec<i32> = stop_token_ids.iter().map(|&id| id as i32).collect();
        let mut out_tokens = vec![0i32; max_new_tokens];
        let mut out_count: i32 = 0;

        // Callback wrapper
        struct CallbackCtx<'a> {
            on_token: &'a mut dyn FnMut(u32) -> Result<()>,
            error: Option<anyhow::Error>,
        }
        let mut ctx = CallbackCtx {
            on_token,
            error: None,
        };

        unsafe extern "C" fn token_callback(token_id: i32, ctx_ptr: *mut std::ffi::c_void) -> i32 {
            let ctx = unsafe { &mut *ctx_ptr.cast::<CallbackCtx<'_>>() };
            match (ctx.on_token)(token_id as u32) {
                Ok(()) => 0,
                Err(e) => {
                    ctx.error = Some(e);
                    -1
                }
            }
        }

        let mut prefill_ms: f64 = 0.0;
        let mut decode_ms: f64 = 0.0;

        let rc = unsafe {
            mlx_sys::qwen35_compiled_generate(
                self.0,
                prompt_i32.as_ptr(),
                prompt_i32.len() as i32,
                max_new_tokens as i32,
                temperature,
                out_tokens.as_mut_ptr(),
                &raw mut out_count,
                &raw mut prefill_ms,
                &raw mut decode_ms,
                Some(token_callback),
                (&raw mut ctx).cast::<std::ffi::c_void>(),
                stop_i32.as_ptr(),
                stop_i32.len() as i32,
            )
        };

        if let Some(e) = ctx.error {
            return Err(e);
        }
        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        Ok((
            out_tokens[..out_count as usize]
                .iter()
                .map(|&id| id as u32)
                .collect(),
            prefill_ms,
            decode_ms,
        ))
    }
}

/// Extract quantized weight raw pointers. Returns None for Dense weights.
fn extract_qw(
    wt: &WeightTensor,
) -> Option<(
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
    i32,
    i32,
)> {
    match wt {
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => Some((
            w.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            *group_size,
            *bits,
        )),
        WeightTensor::Dense(_) => None,
    }
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

    // C++ full generate path — entire decode loop in C++ for maximum GPU buffer reuse.
    if let Some(ref cpp_model) = weights.cpp_model {
        log::info!("Metal forward path: C++ full generate (all in C++)");
        let mut stop_ids: Vec<u32> = params.stop_token_ids.clone();
        if !params.ignore_eos {
            stop_ids.push(config.eos_token_id);
        }

        let (tokens, prefill_ms, decode_ms) = cpp_model.generate(
            input_ids,
            max_new_tokens,
            params.temperature,
            &stop_ids,
            on_token,
        )?;

        let total_time_ms = prefill_ms + decode_ms;
        let decode_tps = if decode_ms > 0.0 {
            tokens.len() as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let prompt_tps = if prefill_ms > 0.0 {
            input_ids.len() as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        log::info!(
            "  prefill {} tokens ({prompt_tps:.1} tok/s, {prefill_ms:.1}ms) decode {} tokens ({decode_tps:.1} tok/s, {decode_ms:.1}ms)",
            input_ids.len(),
            tokens.len(),
        );

        let finish_reason = if tokens.last().is_some_and(|t| stop_ids.contains(t)) {
            "stop"
        } else {
            "length"
        };

        return Ok(super::MetalGenerateOutput {
            tokens,
            finish_reason,
            ttft_ms: prefill_ms,
            total_time_ms,
        });
    }

    log::info!("Metal forward path: Qwen3.5 hybrid (Rust/MLX)");

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

    // Use the C++ step model if available (1 FFI call per step vs ~1600).
    if weights.cpp_model.is_some() {
        log::info!("  using C++ step model (1 FFI call/step)");
    }

    // Build flat cache arrays for C++ path: [k0, v0, k1, v1, ...] and [gdr0, conv0, ...]
    let mut kv_flat: Vec<MlxArray> = k_caches
        .iter()
        .zip(v_caches.iter())
        .flat_map(|(k, v)| [k.clone(), v.clone()])
        .collect();
    let mut gdr_flat: Vec<MlxArray> = recurrent
        .states
        .iter()
        .zip(recurrent.conv_states.iter())
        .flat_map(|(s, c)| [s.clone(), c.clone()])
        .collect();

    // Helper: run forward step dispatching to C++ or Rust path.
    let do_step = |token: &MlxArray,
                   cpp: &Option<CppQwen35Model>,
                   kv_flat: &mut [MlxArray],
                   gdr_flat: &mut [MlxArray],
                   k_caches: &mut [MlxArray],
                   v_caches: &mut [MlxArray],
                   recurrent: &mut MetalRecurrentState,
                   cache_len: i32|
     -> Result<MlxArray> {
        if let Some(m) = cpp {
            m.step(token, cache_len, kv_flat, gdr_flat)
        } else {
            Ok(qwen35_forward_step(
                token, weights, config, arch, k_caches, v_caches, recurrent, cache_len,
            ))
        }
    };

    let mut logits = None;
    for (idx, &token) in input_ids.iter().enumerate() {
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        let step_logits = do_step(
            &token_arr,
            &weights.cpp_model,
            &mut kv_flat,
            &mut gdr_flat,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        )?;
        if weights.cpp_model.is_some() && idx + 1 != input_ids.len() {
            let mut prompt_outputs: Vec<&MlxArray> =
                Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
            prompt_outputs.push(&step_logits);
            prompt_outputs.extend(kv_flat.iter());
            prompt_outputs.extend(gdr_flat.iter());
            super::mlx::eval(&prompt_outputs);
        }
        logits = Some(step_logits);
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
    }

    let logits = logits.context("Qwen3.5 prompt produced no logits")?;
    let mut generated = Vec::new();
    let mut ttft_ms = 0.0;

    // ── mlx_lm-style double-buffered decode loop ────────────────────────
    let mut y = gpu_sample_token(&logits, params);
    super::mlx::async_eval(&[&y]);

    let finish_reason = 'decode: loop {
        // Build NEXT graph while GPU computes CURRENT y.
        let next_logits = do_step(
            &y,
            &weights.cpp_model,
            &mut kv_flat,
            &mut gdr_flat,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            cache_len,
        )?;
        cache_len += 1;
        recurrent.seq_len = cache_len as usize;
        let next_y = gpu_sample_token(&next_logits, params);
        super::mlx::async_eval(&[&next_y]);

        // Now wait for CURRENT y and process the token.
        super::mlx::eval(&[&y]);
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
            if weights.cpp_model.is_some() {
                for li in 0..num_full_layers {
                    extend_kv_cache(
                        &mut kv_flat[2 * li],
                        config.num_key_value_heads as i32,
                        config.head_dim as i32,
                        new_cap,
                    );
                    extend_kv_cache(
                        &mut kv_flat[2 * li + 1],
                        config.num_key_value_heads as i32,
                        config.head_dim as i32,
                        new_cap,
                    );
                }
            } else {
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
pub(super) fn qwen35_forward_step(
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

        // Full-attention and GDR steps both operate on the normalized input.
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

// ── Qwen3.5 attention wrappers ───────────────────────────────────────────────

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
    gdr_cfg: &super::gdr::MetalGdrConfig,
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

// ── Rust/MLX implementations ──────────────────────────────────────────────────

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
    use super::mlx::{reciprocal, sqrt, sum_axis};

    if !offset {
        // Use MLX's fused fast.rms_norm — single op instead of 10 manual ops.
        // This is the same as mlx_lm's nn.RMSNorm.__call__.
        return rms_norm(x, weight, eps);
    }
    // Offset mode: weight = weight + 1, then manual norm.
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
    // Also load quantized embed for as_linear lm_head (avoids 1.2GB dense matmul)
    let embed_quantized = if config.quantization.is_some() {
        load_proj_from_tensors(&tensors, &embed_base, config.quantization).ok()
    } else {
        None
    };
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

                // Reload individual projections for the optional C++ step path.
                let in_proj_qkv_ind = load_proj(&format!("{attn_prefix}.in_proj_qkv"))?;
                let in_proj_z_ind = load_proj(&format!("{attn_prefix}.in_proj_z"))?;
                let in_proj_b_ind = load_proj(&format!("{attn_prefix}.in_proj_b"))?;
                let in_proj_a_ind = load_proj(&format!("{attn_prefix}.in_proj_a"))?;

                let inv_scale = 1.0 / (arch.linear.key_dim as f32).sqrt();
                MetalQwen35Attention::Linear(MetalLinearAttnWeights {
                    in_proj_qkvz,
                    in_proj_ba,
                    in_proj_qkv: in_proj_qkv_ind,
                    in_proj_z: in_proj_z_ind,
                    in_proj_b: in_proj_b_ind,
                    in_proj_a: in_proj_a_ind,
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

        // Store individual gate/up projections for the optional C++ step path.
        let gate_proj_individual = load_proj(&format!("{layer_prefix}.mlp.gate_proj"))?;
        let up_proj_individual = load_proj(&format!("{layer_prefix}.mlp.up_proj"))?;

        layers.push(MetalQwen35BlockWeights {
            input_layernorm: get(&format!("{layer_prefix}.input_layernorm.weight"))?,
            attention,
            post_attention_layernorm: get(&format!(
                "{layer_prefix}.post_attention_layernorm.weight"
            ))?,
            mlp_inputs,
            gate_proj: gate_proj_individual,
            up_proj: up_proj_individual,
            down_proj: load_proj(&format!("{layer_prefix}.mlp.down_proj"))?,
        });
    }

    let mut weights = Qwen35MetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
        embed_quantized,
        cpp_model: None,
    };

    // Try to build the optional C++ step model.
    if std::env::var("METAL_NO_CPP").is_err() {
        weights.cpp_model = CppQwen35Model::build(&weights, config, arch);
    }

    Ok(weights)
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

    Ok(tie_lm_head_from_embed_tokens(embed_tokens))
}

/// Load conv1d weight in nn.Conv1d format: [out_channels, kernel_size, in_channels/groups].
/// For depthwise conv (groups=C), shape is [C, K, 1]. Keep native dtype (bf16).
fn load_conv1d_weight(
    weight: &MlxArray,
    linear_cfg: &super::gdr::MetalGdrConfig,
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
