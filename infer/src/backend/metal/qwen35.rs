use std::{path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use half::bf16;

use super::mlx::{
    Dtype, MlxArray, add, as_dtype, concatenate_axis, multiply, reshape, rms_norm, rope,
    scaled_dot_product_attention, sigmoid, silu, slice, slice_update, take_axis, transpose_axes,
    zeros,
};

use super::gdr::{MetalLinearAttnWeights, MetalRecurrentState, metal_gdr_decode_step};
use super::weights::{StackedQuantized, load_quantized_with_bits, load_stacked_quantized};
use super::{
    KV_CACHE_CHUNK, MetalModelArch, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
    MlpInputProjection, WeightTensor, clear_metal_cache, dflash, extend_kv_cache, gpu_sample_token,
    linear, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map,
    merge_quantized_projection_rows, tensor_get, tie_lm_head_from_embed_tokens,
};
use crate::backend::is_stream_stop_matched;
use crate::backend::metal::dflash::MetalDflashRuntime;
use crate::gguf::{
    GgufFile, HostTensor, find_tensor_name, load_matrix_bf16_host,
    load_matrix_v_reorder_cols_bf16_host, load_matrix_v_reorder_rows_bf16_host,
    load_qwen35_a_log_f32_host, load_qwen35_conv1d_bf16_host, load_qwen35_qkv_matrix_bf16_host,
    load_vector_bf16_host, load_vector_v_reorder_bf16_host,
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

/// MoE sparse-block weights for one Qwen3.5/3.6 transformer layer.
///
/// Shapes follow the mlx-lm `qwen3_5_moe.py sanitize()` output (the MLX
/// community checkpoints already ship in this layout — no runtime splitting
/// of `experts.gate_up_proj` is required):
///
/// | Weight | Shape (packed) | Purpose |
/// |---|---|---|
/// | `router`               | `[E, H/pack]` 8-bit             | token → expert logits |
/// | `switch_gate`          | `[E, Hmoe, H/pack]` 4-bit       | per-expert SwiGLU gate |
/// | `switch_up`            | `[E, Hmoe, H/pack]` 4-bit       | per-expert SwiGLU up   |
/// | `switch_down`          | `[E, H, Hmoe/pack]` 4-bit       | per-expert out projection |
/// | `shared_gate`          | `[Hshared, H/pack]` 4-bit       | always-on SwiGLU gate  |
/// | `shared_up`            | `[Hshared, H/pack]` 4-bit       | always-on SwiGLU up    |
/// | `shared_down`          | `[H, Hshared/pack]` 4-bit       | always-on SwiGLU out   |
/// | `shared_expert_gate`   | `[1, H/pack]` 8-bit             | scalar shared-expert gate |
///
/// All scalars (`num_experts`, `top_k`, `norm_topk_prob`, bits/group_size)
/// are snapshotted from [`super::config::MetalQwen35MoeConfig`] at load time
/// so the hot path stays free of config lookups.
#[cfg(feature = "metal")]
pub(super) struct MetalQwen35MoeWeights {
    pub(super) router: WeightTensor,
    pub(super) switch_gate: StackedQuantized,
    pub(super) switch_up: StackedQuantized,
    pub(super) switch_down: StackedQuantized,
    pub(super) shared_gate: WeightTensor,
    pub(super) shared_up: WeightTensor,
    pub(super) shared_down: WeightTensor,
    pub(super) shared_expert_gate: WeightTensor,
    pub(super) num_experts: i32,
    pub(super) top_k: i32,
    pub(super) norm_topk_prob: bool,
    pub(super) router_bits: i32,
    pub(super) router_group_size: i32,
    pub(super) expert_bits: i32,
    pub(super) expert_group_size: i32,
}

/// Dense SwiGLU MLP weights for one Qwen3.5 transformer layer (original
/// Qwen3.5 path, plus the `mlp_only_layers` escape hatch for future MoE
/// configs that mix dense layers).
#[cfg(feature = "metal")]
pub(super) struct MetalQwen35DenseMlpWeights {
    pub(super) inputs: MlpInputProjection,
    pub(super) down_proj: WeightTensor,
    /// Individual gate/up projections used by the optional C++ step path.
    /// Kept alongside the (possibly merged) `inputs` because the C++ route
    /// wants a separate gate_proj/up_proj pair per layer.
    pub(super) gate_proj: WeightTensor,
    pub(super) up_proj: WeightTensor,
}

/// MLP kind for a single Qwen3.5/3.6 transformer layer.
///
/// Dense = classic Qwen3.5 SwiGLU. Moe = Qwen3.6 `SparseMoeBlock`. Per-layer
/// selection follows [`super::config::MetalQwen35MoeConfig::is_moe_layer`].
#[cfg(feature = "metal")]
pub(super) enum MlpKind {
    Dense(MetalQwen35DenseMlpWeights),
    Moe(MetalQwen35MoeWeights),
}

pub(super) struct MetalQwen35BlockWeights {
    pub(super) input_layernorm: MlxArray,
    pub(super) attention: MetalQwen35Attention,
    pub(super) post_attention_layernorm: MlxArray,
    pub(super) mlp: MlpKind,
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

fn mlx_bf16_array(data: &[bf16], shape: &[i32]) -> MlxArray {
    unsafe { MlxArray::from_raw_data(data.as_ptr().cast(), shape, Dtype::Bfloat16) }
}

fn mlx_tensor_shape(shape: &[usize]) -> Vec<i32> {
    shape
        .iter()
        .map(|&dim| i32::try_from(dim).expect("GGUF tensor dim fits in i32"))
        .collect()
}

fn mlx_bf16_tensor(tensor: HostTensor<bf16>) -> MlxArray {
    let shape = mlx_tensor_shape(&tensor.shape);
    mlx_bf16_array(&tensor.data, &shape)
}

fn mlx_f32_tensor(tensor: HostTensor<f32>) -> MlxArray {
    let shape = mlx_tensor_shape(&tensor.shape);
    MlxArray::from_slice_f32(&tensor.data, &shape)
}

fn qwen35_norm_needs_offset_correction(weight: &MlxArray) -> bool {
    let weight_f32 = as_dtype(weight, Dtype::Float32);
    super::mlx::eval(&[&weight_f32]);
    let slice = weight_f32.as_slice_f32();
    let mean_abs = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len().max(1) as f32;
    mean_abs < 0.75
}

fn qwen35_normalize_direct_norm_weight(
    weight: &MlxArray,
    needs_offset_correction: bool,
) -> MlxArray {
    if !needs_offset_correction {
        return weight.clone();
    }
    let one = as_dtype(&MlxArray::scalar_f32(1.0), weight.dtype());
    add(weight, &one)
}

fn dense_weight_from_matrix(matrix: &MlxArray) -> WeightTensor {
    let w_t = super::mlx::transpose_all(matrix);
    super::mlx::eval(&[&w_t]);
    WeightTensor::Dense(w_t)
}

fn concat_dense_weights(lhs: &WeightTensor, rhs: &WeightTensor) -> Result<WeightTensor> {
    match (lhs, rhs) {
        (WeightTensor::Dense(lhs), WeightTensor::Dense(rhs)) => Ok(WeightTensor::Dense(
            concatenate_axis(&[lhs.clone(), rhs.clone()], 1),
        )),
        _ => anyhow::bail!("expected dense GGUF weights during Metal GGUF load"),
    }
}

/// RAII wrapper for the C++ Qwen35 forward model.
pub(crate) struct CppQwen35Model(*mut std::ffi::c_void);

fn metal_qwen35_trace_enabled() -> bool {
    std::env::var("AGENT_INFER_METAL_QWEN35_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

impl Drop for CppQwen35Model {
    fn drop(&mut self) {
        unsafe { mlx_sys::qwen35_compiled_free(self.0) }
    }
}
unsafe impl Send for CppQwen35Model {}

pub(crate) fn capture_qwen35_hidden_from_cpp_outputs(
    cpp_model_raw: *mut std::ffi::c_void,
    expected_layers: usize,
) -> Result<Option<MlxArray>> {
    let n_cap = unsafe { mlx_sys::qwen35_get_captured_hidden_count(cpp_model_raw) };
    if n_cap <= 0 {
        return Ok(None);
    }
    anyhow::ensure!(
        n_cap as usize == expected_layers,
        "Qwen3.5 DFlash captured hidden mismatch: expected {expected_layers}, got {n_cap}"
    );

    let mut layers: Vec<MlxArray> = Vec::with_capacity(expected_layers);
    for ci in 0..n_cap {
        let mut hidden_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc =
            unsafe { mlx_sys::qwen35_get_captured_hidden(cpp_model_raw, ci, &raw mut hidden_ptr) };
        anyhow::ensure!(
            rc == 0 && !hidden_ptr.is_null(),
            "Qwen3.5 DFlash failed to capture hidden state {ci}"
        );
        layers.push(unsafe { MlxArray::from_raw(hidden_ptr) });
    }

    let squeezed: Vec<MlxArray> = layers
        .iter()
        .map(|hidden| {
            let shape = hidden.shape();
            if shape.len() == 3 {
                super::mlx::reshape(hidden, &[shape[1], shape[2]])
            } else {
                hidden.clone()
            }
        })
        .collect();
    Ok(Some(concatenate_axis(&squeezed, 1)))
}

pub(crate) fn append_qwen35_captured_hidden_chunk(
    accumulated: &mut Option<MlxArray>,
    captured_chunk: Option<MlxArray>,
) {
    let Some(chunk) = captured_chunk else {
        return;
    };
    let combined = if let Some(existing) = accumulated.take() {
        concatenate_axis(&[existing, chunk], 0)
    } else {
        chunk
    };
    *accumulated = Some(combined);
}

pub(crate) fn with_qwen35_capture_layers<T>(
    cpp_model_raw: *mut std::ffi::c_void,
    target_layer_ids: &[usize],
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let capture_layer_ids: Vec<i32> = target_layer_ids.iter().map(|&idx| idx as i32).collect();
    unsafe {
        mlx_sys::qwen35_set_capture_layers(
            cpp_model_raw,
            capture_layer_ids.as_ptr(),
            capture_layer_ids.len() as i32,
        );
    }
    let result = f();
    unsafe {
        mlx_sys::qwen35_set_capture_layers(cpp_model_raw, std::ptr::null(), 0);
    }
    result
}

fn use_qwen35_cpp_separate_proj() -> bool {
    std::env::var("AGENT_INFER_QWEN35_CPP_SEPARATE").map_or(true, |value| value != "0")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Qwen35VerifySummary {
    pub(super) matched_prefix_len: usize,
    pub(super) next_token: u32,
}

impl CppQwen35Model {
    /// Wrap a raw C++ model pointer (takes ownership).
    pub(crate) fn from_raw(ptr: *mut std::ffi::c_void) -> Self {
        Self(ptr)
    }
    /// Raw pointer to the underlying C++ model (for FFI calls).
    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.0
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

            let (dense, moe) = match &layer.mlp {
                MlpKind::Dense(d) => (Some(d), None),
                MlpKind::Moe(m) => (None, Some(m)),
            };
            // Dense MLP weights are only needed for non-MoE layers. MoE layers
            // pass null dense-MLP pointers to the compiled attention push and
            // receive a separate MoE push below.
            let (gu_w, gu_s, gu_b, gu_gs, gu_bits, gate_dim) = if let Some(dense) = dense {
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
                } = &dense.inputs
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
                }
            } else {
                (
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    0,
                    0,
                )
            };
            let (dw_w, dw_s, dw_b) = if let Some(dense) = dense {
                if let WeightTensor::Quantized {
                    w, scales, biases, ..
                } = &dense.down_proj
                {
                    (w.as_raw(), scales.as_raw(), biases.as_raw())
                } else {
                    unsafe { mlx_sys::qwen35_compiled_free(model) };
                    return None;
                }
            } else {
                (
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
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
                    if let Some(dense) = dense {
                        let separate_proj = use_qwen35_cpp_separate_proj();
                        if let (Some(qkv), Some(z), Some(b), Some(a), Some(gp), Some(up)) = (
                            extract_qw(&attn.in_proj_qkv),
                            extract_qw(&attn.in_proj_z),
                            extract_qw(&attn.in_proj_b),
                            extract_qw(&attn.in_proj_a),
                            extract_qw(&dense.gate_proj),
                            extract_qw(&dense.up_proj),
                        ) {
                            if separate_proj {
                                unsafe {
                                    mlx_sys::qwen35_compiled_set_separate_proj(
                                        model, qkv.0, qkv.1, qkv.2, qkv.3, qkv.4, z.0, z.1, z.2,
                                        b.0, b.1, b.2, a.0, a.1, a.2, gp.0, gp.1, gp.2, gp.3, gp.4,
                                        up.0, up.1, up.2,
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
            if let Some(moe) = moe {
                let router = extract_qw(&moe.router)?;
                let switch_gate = extract_stacked_qw(&moe.switch_gate);
                let switch_up = extract_stacked_qw(&moe.switch_up);
                let switch_down = extract_stacked_qw(&moe.switch_down);
                let shared_gate = extract_qw(&moe.shared_gate)?;
                let shared_up = extract_qw(&moe.shared_up)?;
                let shared_down = extract_qw(&moe.shared_down)?;
                let shared_expert_gate = extract_qw(&moe.shared_expert_gate)?;
                unsafe {
                    mlx_sys::qwen35_compiled_set_last_moe_mlp(
                        model,
                        router.0,
                        router.1,
                        router.2,
                        moe.router_group_size,
                        moe.router_bits,
                        switch_gate.0,
                        switch_gate.1,
                        switch_gate.2,
                        switch_up.0,
                        switch_up.1,
                        switch_up.2,
                        switch_down.0,
                        switch_down.1,
                        switch_down.2,
                        moe.expert_group_size,
                        moe.expert_bits,
                        shared_gate.0,
                        shared_gate.1,
                        shared_gate.2,
                        shared_up.0,
                        shared_up.1,
                        shared_up.2,
                        shared_down.0,
                        shared_down.1,
                        shared_down.2,
                        shared_expert_gate.0,
                        shared_expert_gate.1,
                        shared_expert_gate.2,
                        moe.num_experts,
                        moe.top_k,
                        moe.norm_topk_prob,
                    );
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
    pub(super) fn begin_session(
        &self,
        kv_caches: &[MlxArray],
        gdr_states: &[MlxArray],
    ) -> Result<()> {
        let n_kv = kv_caches.len() as i32;
        let n_gdr = gdr_states.len() as i32;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();

        let rc = unsafe {
            mlx_sys::qwen35_session_begin(
                self.0,
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        Ok(())
    }

    pub(super) fn end_session(
        &self,
        n_kv: usize,
        n_gdr: usize,
    ) -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr];

        let rc = unsafe {
            mlx_sys::qwen35_session_end(
                self.0,
                out_kv.as_mut_ptr(),
                n_kv as i32,
                out_gdr.as_mut_ptr(),
                n_gdr as i32,
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        let kv_caches = out_kv
            .into_iter()
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        let gdr_states = out_gdr
            .into_iter()
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        Ok((kv_caches, gdr_states))
    }

    pub(super) fn step_session(&self, token: &MlxArray, cache_pos: i32) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_session(
                self.0,
                token.as_raw(),
                cache_pos,
                &raw mut out_logits,
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(super) fn prefill_session(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
    ) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let rc = unsafe {
            mlx_sys::qwen35_compiled_prefill_session(
                self.0,
                tokens.as_raw(),
                prompt_len,
                cache_pos,
                &raw mut out_logits,
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
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

    pub(super) fn step_batch(
        &self,
        tokens: &MlxArray,
        batch_size: i32,
        cache_pos: i32,
        kv_caches: &mut [MlxArray],
        n_kv_per_request: i32,
        gdr_states: &mut [MlxArray],
        n_gdr_per_request: i32,
        attn_mask: Option<&MlxArray>,
        rope_offsets: Option<&MlxArray>,
    ) -> Result<MlxArray> {
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); kv_caches.len()];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); gdr_states.len()];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_batch(
                self.0,
                tokens.as_raw(),
                batch_size,
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                n_kv_per_request,
                gdr_ptrs.as_mut_ptr(),
                n_gdr_per_request,
                attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw),
                rope_offsets.map_or(std::ptr::null_mut(), MlxArray::as_raw),
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

    pub(super) fn step_batch_packed(
        &self,
        tokens: &MlxArray,
        batch_size: i32,
        cache_pos: i32,
        packed_kv_caches: &mut [MlxArray],
        n_kv: i32,
        packed_gdr_states: &mut [MlxArray],
        n_gdr: i32,
        attn_mask: Option<&MlxArray>,
        rope_offsets: Option<&MlxArray>,
    ) -> Result<MlxArray> {
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); packed_kv_caches.len()];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); packed_gdr_states.len()];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_batch_packed(
                self.0,
                tokens.as_raw(),
                batch_size,
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw),
                rope_offsets.map_or(std::ptr::null_mut(), MlxArray::as_raw),
                &raw mut out_logits,
                out_kv.as_mut_ptr(),
                out_gdr.as_mut_ptr(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        for (i, ptr) in out_kv.into_iter().enumerate() {
            let old =
                std::mem::replace(&mut packed_kv_caches[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        for (i, ptr) in out_gdr.into_iter().enumerate() {
            let old = std::mem::replace(&mut packed_gdr_states[i], unsafe {
                MlxArray::from_raw(ptr)
            });
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

    #[cfg(test)]
    pub(super) fn prefill_full_attention(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
        k_caches: &mut [MlxArray],
        v_caches: &mut [MlxArray],
    ) -> Result<MlxArray> {
        anyhow::ensure!(
            k_caches.len() == v_caches.len(),
            "Qwen3 compiled prefill requires matching k/v cache counts"
        );

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> = Vec::with_capacity(k_caches.len() * 2);
        for (k_cache, v_cache) in k_caches.iter().zip(v_caches.iter()) {
            kv_ptrs.push(k_cache.as_raw());
            kv_ptrs.push(v_cache.as_raw());
        }

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); kv_ptrs.len()];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_prefill(
                self.0,
                tokens.as_raw(),
                prompt_len,
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                kv_ptrs.len() as i32,
                std::ptr::null_mut(),
                0,
                &raw mut out_logits,
                out_kv.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        for ((k_cache, v_cache), out_pair) in k_caches
            .iter_mut()
            .zip(v_caches.iter_mut())
            .zip(out_kv.chunks_exact(2))
        {
            let old_k = std::mem::replace(k_cache, unsafe { MlxArray::from_raw(out_pair[0]) });
            drop(old_k);
            let old_v = std::mem::replace(v_cache, unsafe { MlxArray::from_raw(out_pair[1]) });
            drop(old_v);
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    /// DFlash verify: parallel forward over a draft block, returning all-position
    /// logits `[1, block_size, vocab]`. Mirrors `prefill` but forces
    /// `last_logits_only = false` so the caller can sample every position in the
    /// draft in a single pass. Tape/hidden capture flags on the model are
    /// respected — one call emits the full per-step GDR innovation tape and the
    /// full hidden-state capture for the block.
    #[cfg(test)]
    pub(super) fn verify_block(
        &self,
        tokens: &MlxArray,
        block_size: i32,
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
            mlx_sys::qwen35_compiled_verify_block(
                self.0,
                tokens.as_raw(),
                block_size,
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

    /// Single-row DFlash verify fast path: samples the posterior inside C++
    /// and returns only the acceptance summary needed by Rust.
    pub(super) fn verify_block_summary(
        &self,
        tokens: &MlxArray,
        block_size: i32,
        cache_pos: i32,
        kv_caches: &mut [MlxArray],
        gdr_states: &mut [MlxArray],
        params: &SamplingParams,
        suppress_token_id: Option<u32>,
    ) -> Result<Qwen35VerifySummary> {
        let n_kv = kv_caches.len() as i32;
        let n_gdr = gdr_states.len() as i32;
        let greedy = params.temperature <= 1e-6 || params.top_k == 1;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut matched_prefix_len = 0i32;
        let mut next_token = 0i32;
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv as usize];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr as usize];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_verify_block_summary(
                self.0,
                tokens.as_raw(),
                block_size,
                cache_pos,
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                params.temperature,
                greedy,
                suppress_token_id.map_or(-1, |token_id| token_id as i32),
                &raw mut matched_prefix_len,
                &raw mut next_token,
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

        anyhow::ensure!(
            (0..block_size).contains(&matched_prefix_len),
            "Qwen3.5 DFlash verify summary returned invalid matched_prefix_len={matched_prefix_len} for block_size={block_size}"
        );
        anyhow::ensure!(
            next_token >= 0,
            "Qwen3.5 DFlash verify summary returned negative next_token={next_token}"
        );

        Ok(Qwen35VerifySummary {
            matched_prefix_len: matched_prefix_len as usize,
            next_token: next_token as u32,
        })
    }

    /// Batched DFlash verify: run `block_size` draft tokens for `batch_size`
    /// rows in a single forward. Mirrors `verify_block` but feeds the packed
    /// KV/GDR states and per-row `cache_pos_arr` / `rope_offsets` that
    /// `step_batch_packed` already uses for plain-decode.
    ///
    /// Shapes:
    /// - `tokens`: int32 `[B, block_size]`.
    /// - `cache_pos_arr`: host int32 `[B]` — per-row physical KV write start.
    /// - `rope_offsets`: int32 `[B]` — per-row RoPE base offset (typically
    ///   equal to `cache_pos_arr[b]` when left-padding is not used).
    /// - `attn_mask`: optional additive `[B, 1, block_size, key_len]`. Pass
    ///   `None` when every row's left-pad is zero (e.g. fresh DFlash slot
    ///   with equal cache lengths).
    /// - `packed_kv_caches`: slice of layer tensors `[B, n_kv_heads, kv_cap,
    ///   head_dim]` — updated in place.
    /// - `packed_gdr_states`: `[state_0, conv_0, state_1, conv_1, …]` with
    ///   `[B, Hv, Dv, Dk]` state and `[B, conv_kernel-1, …]` conv slabs —
    ///   updated in place.
    ///
    /// Returns logits `[B, block_size, vocab]`. Tape and hidden-state
    /// capture settings on the underlying C++ model are respected; each
    /// tape entry becomes `[B, block_size, …]`.
    #[cfg(test)]
    pub(super) fn verify_block_batched(
        &self,
        tokens: &MlxArray,
        batch_size: i32,
        block_size: i32,
        cache_pos_arr: &[i32],
        packed_kv_caches: &mut [MlxArray],
        packed_gdr_states: &mut [MlxArray],
        attn_mask: Option<&MlxArray>,
        rope_offsets: &MlxArray,
    ) -> Result<MlxArray> {
        ensure!(
            cache_pos_arr.len() == batch_size as usize,
            "verify_block_batched cache_pos_arr len {} != batch_size {}",
            cache_pos_arr.len(),
            batch_size
        );
        let n_kv = packed_kv_caches.len() as i32;
        let n_gdr = packed_gdr_states.len() as i32;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv as usize];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr as usize];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_verify_block_batched(
                self.0,
                tokens.as_raw(),
                batch_size,
                block_size,
                cache_pos_arr.as_ptr(),
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw),
                rope_offsets.as_raw(),
                &raw mut out_logits,
                out_kv.as_mut_ptr(),
                out_gdr.as_mut_ptr(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        for (i, ptr) in out_kv.into_iter().enumerate() {
            let old =
                std::mem::replace(&mut packed_kv_caches[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        for (i, ptr) in out_gdr.into_iter().enumerate() {
            let old = std::mem::replace(&mut packed_gdr_states[i], unsafe {
                MlxArray::from_raw(ptr)
            });
            drop(old);
        }

        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    /// Batched DFlash verify fast path: same packed-KV/GDR update as
    /// `verify_block_batched`, but samples the posterior inside C++ and
    /// returns token ids `[B, block_size]` instead of logits.
    #[allow(dead_code)]
    pub(super) fn verify_block_batched_sampled(
        &self,
        tokens: &MlxArray,
        batch_size: i32,
        block_size: i32,
        cache_pos_arr: &[i32],
        packed_kv_caches: &mut [MlxArray],
        packed_gdr_states: &mut [MlxArray],
        attn_mask: Option<&MlxArray>,
        rope_offsets: &MlxArray,
        params: &SamplingParams,
        suppress_token_id: Option<u32>,
    ) -> Result<MlxArray> {
        ensure!(
            cache_pos_arr.len() == batch_size as usize,
            "verify_block_batched_sampled cache_pos_arr len {} != batch_size {}",
            cache_pos_arr.len(),
            batch_size
        );
        let n_kv = packed_kv_caches.len() as i32;
        let n_gdr = packed_gdr_states.len() as i32;
        let greedy = params.temperature <= 1e-6 || params.top_k == 1;

        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            packed_gdr_states.iter().map(MlxArray::as_raw).collect();

        let mut out_sampled: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv as usize];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr as usize];

        let rc = unsafe {
            mlx_sys::qwen35_compiled_verify_block_batched_sampled(
                self.0,
                tokens.as_raw(),
                batch_size,
                block_size,
                cache_pos_arr.as_ptr(),
                kv_ptrs.as_mut_ptr(),
                n_kv,
                gdr_ptrs.as_mut_ptr(),
                n_gdr,
                attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw),
                rope_offsets.as_raw(),
                params.temperature,
                greedy,
                suppress_token_id.map_or(-1, |token_id| token_id as i32),
                &raw mut out_sampled,
                out_kv.as_mut_ptr(),
                out_gdr.as_mut_ptr(),
            )
        };

        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }

        for (i, ptr) in out_kv.into_iter().enumerate() {
            let old =
                std::mem::replace(&mut packed_kv_caches[i], unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        for (i, ptr) in out_gdr.into_iter().enumerate() {
            let old = std::mem::replace(&mut packed_gdr_states[i], unsafe {
                MlxArray::from_raw(ptr)
            });
            drop(old);
        }

        Ok(unsafe { MlxArray::from_raw(out_sampled) })
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
            stop_requested: bool,
        }
        let mut ctx = CallbackCtx {
            on_token,
            error: None,
            stop_requested: false,
        };

        unsafe extern "C" fn token_callback(token_id: i32, ctx_ptr: *mut std::ffi::c_void) -> i32 {
            let ctx = unsafe { &mut *ctx_ptr.cast::<CallbackCtx<'_>>() };
            match (ctx.on_token)(token_id as u32) {
                Ok(()) => 0,
                Err(e) => {
                    ctx.stop_requested = is_stream_stop_matched(&e);
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

        if ctx.stop_requested {
            return Ok((
                out_tokens[..out_count as usize]
                    .iter()
                    .map(|&id| id as u32)
                    .collect(),
                prefill_ms,
                decode_ms,
            ));
        }
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

fn extract_stacked_qw(
    wt: &StackedQuantized,
) -> (
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
) {
    (wt.weight.as_raw(), wt.scales.as_raw(), wt.biases.as_raw())
}

pub(super) fn metal_generate_qwen35(
    input_ids: &[u32],
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    dflash_runtime: Option<&MetalDflashRuntime>,
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

    if let Some(runtime) = dflash_runtime {
        return metal_generate_qwen35_dflash(
            runtime,
            input_ids,
            weights,
            config,
            arch,
            params,
            max_new_tokens,
            t0,
            on_token,
        );
    }

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
    let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut kv_capacity = cache_shape[2];

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
    let trace_rust_prefill = weights.cpp_model.is_none() && metal_qwen35_trace_enabled();
    let prefill_started = trace_rust_prefill.then(Instant::now);
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
    if let Some(prefill_started) = prefill_started {
        eprintln!(
            "metal_trace[qwen35_direct_prefill]: mode=rust_scalar_prefill tokens={} cache_len={} elapsed_ms={:.1}",
            input_ids.len(),
            cache_len,
            prefill_started.elapsed().as_secs_f64() * 1000.0,
        );
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
        if let Err(err) = on_token(next_token) {
            if is_stream_stop_matched(&err) {
                break 'decode "stop";
            }
            return Err(err);
        }

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
fn metal_generate_qwen35_dflash(
    runtime: &MetalDflashRuntime,
    input_ids: &[u32],
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<super::MetalGenerateOutput> {
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("Qwen3.5/Qwen3.6 DFlash requires the compiled C++ model")?;

    let num_full_layers = arch.num_full_attention_layers();
    let prefill_len = input_ids.len() as i32;
    let initial_cap = ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
    let cache_shape = [
        1i32,
        config.num_key_value_heads as i32,
        initial_cap.max(KV_CACHE_CHUNK),
        config.head_dim as i32,
    ];
    let k_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let v_caches: Vec<MlxArray> = (0..num_full_layers)
        .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
        .collect();
    let mut kv_capacity = cache_shape[2];
    let mut recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
    let mut cache_len = 0i32;
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

    let prompt_values: Vec<i32> = input_ids.iter().map(|&token| token as i32).collect();
    let prompt_arr = MlxArray::from_slice_i32(&prompt_values, &[input_ids.len() as i32]);
    let logits =
        with_qwen35_capture_layers(cpp_model.as_raw(), runtime.target_layer_ids(), || {
            cpp_model.prefill(
                &prompt_arr,
                input_ids.len() as i32,
                cache_len,
                &mut kv_flat,
                &mut gdr_flat,
            )
        })?;
    let target_hidden = capture_qwen35_hidden_from_cpp_outputs(
        cpp_model.as_raw(),
        runtime.target_layer_ids().len(),
    )?;
    cache_len += input_ids.len() as i32;
    recurrent.seq_len = cache_len as usize;
    let mut current_token =
        dflash::sample_last_token_suppress(&logits, params, Some(runtime.mask_token_id()))?;
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut generated = vec![current_token];
    on_token(current_token)?;
    if config.is_stop_token(current_token) || generated.len() >= max_new_tokens {
        return Ok(super::MetalGenerateOutput {
            tokens: generated,
            finish_reason: if config.is_stop_token(current_token) {
                "stop"
            } else {
                "length"
            },
            ttft_ms,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        });
    }

    let mut draft_state = dflash::ContiguousKvState::new(
        runtime.draft_num_hidden_layers(),
        runtime.draft_n_kv_heads(),
        runtime.draft_head_dim(),
        input_ids.len() + max_new_tokens,
    );
    let mut prefetched_draft = None;
    let mut target_hidden =
        target_hidden.context("Qwen3.5/Qwen3.6 DFlash prefill did not capture target_hidden")?;

    let finish_reason = 'decode: loop {
        let needed_cap = cache_len
            + i32::try_from(runtime.block_size()).context("Qwen3.5 DFlash block_size overflow")?;
        if needed_cap > kv_capacity {
            let new_cap = ((needed_cap + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
            for cache in &mut kv_flat {
                extend_kv_cache(
                    cache,
                    config.num_key_value_heads as i32,
                    config.head_dim as i32,
                    new_cap,
                );
            }
            kv_capacity = new_cap;
        }

        let block = dflash::qwen35_dflash_speculative_block(
            runtime,
            current_token,
            &target_hidden,
            &weights.embed_tokens,
            &weights.lm_head,
            config,
            cpp_model,
            params,
            &mut kv_flat,
            &mut gdr_flat,
            &mut cache_len,
            &mut draft_state,
            prefetched_draft.take(),
        )?;
        prefetched_draft = block.prefetched_next_draft;
        target_hidden = block.updated_target_hidden;
        for token in block.accepted_tokens {
            current_token = token;
            generated.push(token);
            on_token(token)?;
            if config.is_stop_token(token) {
                break 'decode "stop";
            }
            if generated.len() >= max_new_tokens {
                break 'decode "length";
            }
        }
    };

    Ok(super::MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
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
        let mlp = mlp_forward(&layer.mlp, &xn);
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

/// Like `qwen35_forward_step` but captures hidden states at specified layer indices.
/// Used by DFlash to extract target context features for the draft model.
pub(super) fn qwen35_forward_with_hidden_states(
    input_ids: &[u32],
    weights: &Qwen35MetalWeights,
    config: &MetalModelConfig,
    arch: &MetalQwen35ArchConfig,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    recurrent: &mut MetalRecurrentState,
    cache_len: i32,
    target_layer_ids: &[usize],
) -> (MlxArray, MlxArray) {
    use std::collections::HashSet;
    let selected: HashSet<usize> = target_layer_ids.iter().copied().collect();
    let _selected_hidden: Vec<MlxArray> = Vec::with_capacity(target_layer_ids.len());

    // Process tokens one at a time (the attention/GDR helpers expect single-token input).
    // Capture hidden states after each layer for selected layers.
    let mut all_per_token_hidden: Vec<Vec<MlxArray>> = Vec::new();
    let mut last_logits = MlxArray::scalar_f32(0.0);
    for (pos, &token) in (cache_len..).zip(input_ids.iter()) {
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        let mut x = take_axis(&weights.embed_tokens, &token_arr, 0);
        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut token_hidden: Vec<MlxArray> = Vec::new();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let residual = x.clone();
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
                        pos,
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
            let mlp = mlp_forward(&layer.mlp, &xn);
            x = add(&residual2, &mlp);

            if selected.contains(&layer_idx) {
                token_hidden.push(x.clone());
            }
        }

        let final_norm = rms_norm_last_dim(
            &x,
            &weights.norm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        );
        last_logits = linear(&final_norm, &weights.lm_head);
        all_per_token_hidden.push(token_hidden);
    }

    // Concatenate: for each target layer, stack all tokens along axis 0,
    // then concatenate layers along axis 1.
    let num_captured = target_layer_ids.len();
    let mut layer_stacks: Vec<MlxArray> = Vec::with_capacity(num_captured);
    for li in 0..num_captured {
        let per_tok: Vec<MlxArray> = all_per_token_hidden
            .iter()
            .map(|th| th[li].clone())
            .collect();
        layer_stacks.push(concatenate_axis(&per_tok, 0));
    }
    let combined = concatenate_axis(&layer_stacks, 1);
    (last_logits, combined)
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

fn dense_mlp_forward(mlp: &MetalQwen35DenseMlpWeights, x: &MlxArray) -> MlxArray {
    let (gate_raw, up) = mlp_project(&mlp.inputs, x);
    let fused_val = multiply(&silu(&gate_raw), &up);
    linear(&fused_val, &mlp.down_proj)
}

fn moe_mlp_forward(x: &MlxArray, moe: &MetalQwen35MoeWeights) -> MlxArray {
    let router = extract_qw(&moe.router).expect("Qwen3.6 MoE router must be quantized");
    let shared_gate =
        extract_qw(&moe.shared_gate).expect("Qwen3.6 shared expert gate_proj must be quantized");
    let shared_up =
        extract_qw(&moe.shared_up).expect("Qwen3.6 shared expert up_proj must be quantized");
    let shared_down =
        extract_qw(&moe.shared_down).expect("Qwen3.6 shared expert down_proj must be quantized");
    let shared_expert_gate =
        extract_qw(&moe.shared_expert_gate).expect("Qwen3.6 shared_expert_gate must be quantized");

    let raw = unsafe {
        mlx_sys::qwen35_moe_block_forward(
            x.as_raw(),
            router.0,
            router.1,
            router.2,
            moe.router_bits,
            moe.router_group_size,
            moe.switch_gate.weight.as_raw(),
            moe.switch_gate.scales.as_raw(),
            moe.switch_gate.biases.as_raw(),
            moe.switch_up.weight.as_raw(),
            moe.switch_up.scales.as_raw(),
            moe.switch_up.biases.as_raw(),
            moe.switch_down.weight.as_raw(),
            moe.switch_down.scales.as_raw(),
            moe.switch_down.biases.as_raw(),
            moe.expert_bits,
            moe.expert_group_size,
            shared_gate.0,
            shared_gate.1,
            shared_gate.2,
            shared_up.0,
            shared_up.1,
            shared_up.2,
            shared_down.0,
            shared_down.1,
            shared_down.2,
            shared_expert_gate.0,
            shared_expert_gate.1,
            shared_expert_gate.2,
            moe.num_experts,
            moe.top_k,
            moe.norm_topk_prob,
        )
    };
    unsafe { MlxArray::from_raw(raw) }
}

fn mlp_forward(mlp: &MlpKind, x: &MlxArray) -> MlxArray {
    match mlp {
        MlpKind::Dense(dense) => dense_mlp_forward(dense, x),
        MlpKind::Moe(moe) => moe_mlp_forward(x, moe),
    }
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

pub(super) fn load_qwen35_metal_weights_from_gguf(
    gguf: &GgufFile,
    config: &MetalModelConfig,
) -> Result<Qwen35MetalWeights> {
    let MetalModelArch::Qwen35(arch) = &config.arch else {
        anyhow::bail!("Qwen3.5 Metal GGUF loader requires a Qwen3.5 config");
    };
    ensure!(
        arch.moe.is_none(),
        "Metal GGUF loading currently supports dense Qwen3.5 only"
    );

    let num_k = arch.linear.num_key_heads;
    let num_v = arch.linear.num_value_heads;
    ensure!(
        num_k > 0 && num_v.is_multiple_of(num_k),
        "invalid Qwen3.5 linear-attention dimensions: num_key_heads={num_k}, num_value_heads={num_v}"
    );
    let num_v_per_k = num_v / num_k;
    let key_head_dim = arch.linear.key_dim;
    let value_head_dim = arch.linear.value_dim;
    let prefix = "model";

    log::info!("  loading Qwen3.5 GGUF on Metal — dequantizing to BF16 during load");

    let embed_tokens = mlx_bf16_tensor(load_matrix_bf16_host(
        gguf,
        &format!("{prefix}.embed_tokens.weight"),
    )?);
    let norm = mlx_bf16_tensor(load_vector_bf16_host(
        gguf,
        &format!("{prefix}.norm.weight"),
    )?);
    let lm_head = if find_tensor_name(gguf, "lm_head.weight").is_ok() {
        dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
            gguf,
            "lm_head.weight",
        )?))
    } else {
        tie_lm_head_from_embed_tokens(&embed_tokens)
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let layer_prefix = format!("{prefix}.layers.{i}");
        let attention = match arch.layer_types[i] {
            MetalQwen35LayerType::FullAttention => {
                let attn_prefix = format!("{layer_prefix}.self_attn");
                MetalQwen35Attention::Full(MetalQwen35FullAttentionWeights {
                    q_proj: dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.q_proj.weight"),
                    )?)),
                    k_proj: dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.k_proj.weight"),
                    )?)),
                    v_proj: dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.v_proj.weight"),
                    )?)),
                    o_proj: dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.o_proj.weight"),
                    )?)),
                    q_norm: mlx_bf16_tensor(load_vector_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.q_norm.weight"),
                    )?),
                    k_norm: mlx_bf16_tensor(load_vector_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.k_norm.weight"),
                    )?),
                })
            }
            MetalQwen35LayerType::LinearAttention => {
                let attn_prefix = format!("{layer_prefix}.linear_attn");
                let qkv_proj =
                    dense_weight_from_matrix(&mlx_bf16_tensor(load_qwen35_qkv_matrix_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.in_proj_qkv.weight"),
                        num_k,
                        num_v_per_k,
                        key_head_dim,
                        value_head_dim,
                    )?));
                let z_proj = dense_weight_from_matrix(&mlx_bf16_tensor(
                    load_matrix_v_reorder_rows_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.in_proj_z.weight"),
                        num_k,
                        num_v_per_k,
                        value_head_dim,
                    )?,
                ));
                let beta_proj = dense_weight_from_matrix(&mlx_bf16_tensor(
                    load_matrix_v_reorder_rows_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.in_proj_b.weight"),
                        num_k,
                        num_v_per_k,
                        1,
                    )?,
                ));
                let alpha_proj = dense_weight_from_matrix(&mlx_bf16_tensor(
                    load_matrix_v_reorder_rows_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.in_proj_a.weight"),
                        num_k,
                        num_v_per_k,
                        1,
                    )?,
                ));
                let qkv_dim = qkv_proj.output_dim()?;
                let z_dim = z_proj.output_dim()?;
                let beta_dim = beta_proj.output_dim()?;
                let in_proj_qkvz = concat_dense_weights(&qkv_proj, &z_proj)?;
                let in_proj_ba = concat_dense_weights(&beta_proj, &alpha_proj)?;
                let inv_scale = 1.0 / (arch.linear.key_dim as f32).sqrt();
                MetalQwen35Attention::Linear(MetalLinearAttnWeights {
                    in_proj_qkvz,
                    in_proj_ba,
                    in_proj_qkv: qkv_proj,
                    in_proj_z: z_proj,
                    in_proj_b: beta_proj,
                    in_proj_a: alpha_proj,
                    qkvz_split: (qkv_dim, z_dim),
                    ba_num_heads: beta_dim,
                    conv1d_weight: mlx_bf16_tensor(load_qwen35_conv1d_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.conv1d.weight"),
                        arch.linear.num_key_heads,
                        arch.linear.key_dim,
                        arch.linear.num_value_heads,
                        arch.linear.value_dim,
                        arch.linear.conv_kernel,
                    )?),
                    dt_bias: mlx_bf16_tensor(load_vector_v_reorder_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.dt_bias"),
                        num_k,
                        num_v_per_k,
                        1,
                    )?),
                    a_log: mlx_f32_tensor(load_qwen35_a_log_f32_host(
                        gguf,
                        &format!("{attn_prefix}.a_log"),
                        num_k,
                        num_v_per_k,
                    )?),
                    norm_weight: mlx_bf16_tensor(load_vector_bf16_host(
                        gguf,
                        &format!("{attn_prefix}.norm.weight"),
                    )?),
                    out_proj: dense_weight_from_matrix(&mlx_bf16_tensor(
                        load_matrix_v_reorder_cols_bf16_host(
                            gguf,
                            &format!("{attn_prefix}.out_proj.weight"),
                            num_k,
                            num_v_per_k,
                            value_head_dim,
                        )?,
                    )),
                    q_scale: MlxArray::scalar_f32(inv_scale * inv_scale),
                    k_scale: MlxArray::scalar_f32(inv_scale),
                })
            }
        };

        let gate_proj = dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
            gguf,
            &format!("{layer_prefix}.mlp.gate_proj.weight"),
        )?));
        let up_proj = dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
            gguf,
            &format!("{layer_prefix}.mlp.up_proj.weight"),
        )?));
        let mlp = MlpKind::Dense(MetalQwen35DenseMlpWeights {
            inputs: MlpInputProjection::Split {
                gate_proj: gate_proj.clone(),
                up_proj: up_proj.clone(),
            },
            down_proj: dense_weight_from_matrix(&mlx_bf16_tensor(load_matrix_bf16_host(
                gguf,
                &format!("{layer_prefix}.mlp.down_proj.weight"),
            )?)),
            gate_proj,
            up_proj,
        });

        layers.push(MetalQwen35BlockWeights {
            input_layernorm: mlx_bf16_tensor(load_vector_bf16_host(
                gguf,
                &format!("{layer_prefix}.input_layernorm.weight"),
            )?),
            attention,
            post_attention_layernorm: mlx_bf16_tensor(load_vector_bf16_host(
                gguf,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
            )?),
            mlp,
        });
    }

    let mut weights = Qwen35MetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
        embed_quantized: None,
        cpp_model: None,
    };

    if std::env::var("METAL_NO_CPP").is_err() {
        weights.cpp_model = CppQwen35Model::build(&weights, config, arch);
    }

    Ok(weights)
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
    let norms_need_offset_correction = {
        let sample = get(&format!("{prefix}.layers.0.input_layernorm.weight"))?;
        qwen35_norm_needs_offset_correction(&sample)
    };
    if norms_need_offset_correction {
        log::info!(
            "  Qwen3.5 safetensors use HF offset RMSNorm weights — normalizing to direct form at load"
        );
    }
    let load_norm = |name: &str| -> Result<MlxArray> {
        let weight = get(name)?;
        Ok(qwen35_normalize_direct_norm_weight(
            &weight,
            norms_need_offset_correction,
        ))
    };

    let embed_base = format!("{prefix}.embed_tokens");
    let embed_tokens = load_embed_tokens_from_tensors(&tensors, &embed_base, config.quantization)?;
    // Also load quantized embed for as_linear lm_head (avoids 1.2GB dense matmul)
    let embed_quantized = if config.quantization.is_some() {
        load_proj_from_tensors(&tensors, &embed_base, config.quantization).ok()
    } else {
        None
    };
    let norm = load_norm(&format!("{prefix}.norm.weight"))?;
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
        "  {} layers ({} full attention, {} GDR, {} MoE)",
        config.num_hidden_layers,
        arch.num_full_attention_layers(),
        arch.num_linear_attention_layers(),
        (0..config.num_hidden_layers)
            .filter(|&idx| arch.moe.as_ref().is_some_and(|moe| moe.is_moe_layer(idx)))
            .count(),
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
                    q_norm: load_norm(&format!("{attn_prefix}.q_norm.weight"))?,
                    k_norm: load_norm(&format!("{attn_prefix}.k_norm.weight"))?,
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

        let mlp = if let Some(moe_cfg) = arch.moe.as_ref().filter(|moe| moe.is_moe_layer(i)) {
            MlpKind::Moe(load_qwen35_moe_layer_weights(
                &tensors,
                &layer_prefix,
                moe_cfg,
            )?)
        } else {
            let gate_proj = load_proj(&format!("{layer_prefix}.mlp.gate_proj"))?;
            let up_proj = load_proj(&format!("{layer_prefix}.mlp.up_proj"))?;
            let gate_dim = gate_proj.output_dim()?;
            let up_dim = up_proj.output_dim()?;
            let inputs = if let Some(gate_up_proj) =
                merge_quantized_projection_rows(&[&gate_proj, &up_proj])?
            {
                MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                    up_dim,
                }
            } else {
                MlpInputProjection::Split { gate_proj, up_proj }
            };
            let gate_proj_individual = load_proj(&format!("{layer_prefix}.mlp.gate_proj"))?;
            let up_proj_individual = load_proj(&format!("{layer_prefix}.mlp.up_proj"))?;
            MlpKind::Dense(MetalQwen35DenseMlpWeights {
                inputs,
                down_proj: load_proj(&format!("{layer_prefix}.mlp.down_proj"))?,
                gate_proj: gate_proj_individual,
                up_proj: up_proj_individual,
            })
        };

        layers.push(MetalQwen35BlockWeights {
            input_layernorm: load_norm(&format!("{layer_prefix}.input_layernorm.weight"))?,
            attention,
            post_attention_layernorm: load_norm(&format!(
                "{layer_prefix}.post_attention_layernorm.weight"
            ))?,
            mlp,
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

fn load_qwen35_moe_layer_weights(
    tensors: &super::TensorMap,
    layer_prefix: &str,
    moe_cfg: &super::config::MetalQwen35MoeConfig,
) -> Result<MetalQwen35MoeWeights> {
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let num_experts =
        i32::try_from(moe_cfg.num_experts).context("Qwen3.6 num_experts does not fit in i32")?;
    let top_k = i32::try_from(moe_cfg.num_experts_per_tok)
        .context("Qwen3.6 num_experts_per_tok does not fit in i32")?;
    anyhow::ensure!(
        num_experts > 0 && top_k > 0 && top_k <= num_experts,
        "invalid Qwen3.6 MoE config: num_experts={num_experts}, top_k={top_k}"
    );

    Ok(MetalQwen35MoeWeights {
        router: load_quantized_with_bits(
            tensors,
            &format!("{mlp_prefix}.gate"),
            moe_cfg.router_group_size,
            moe_cfg.router_bits,
        )?,
        switch_gate: load_stacked_quantized(
            tensors,
            &format!("{mlp_prefix}.switch_mlp.gate_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        switch_up: load_stacked_quantized(
            tensors,
            &format!("{mlp_prefix}.switch_mlp.up_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        switch_down: load_stacked_quantized(
            tensors,
            &format!("{mlp_prefix}.switch_mlp.down_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        shared_gate: load_quantized_with_bits(
            tensors,
            &format!("{mlp_prefix}.shared_expert.gate_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        shared_up: load_quantized_with_bits(
            tensors,
            &format!("{mlp_prefix}.shared_expert.up_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        shared_down: load_quantized_with_bits(
            tensors,
            &format!("{mlp_prefix}.shared_expert.down_proj"),
            moe_cfg.expert_group_size,
            moe_cfg.expert_bits,
        )?,
        shared_expert_gate: load_quantized_with_bits(
            tensors,
            &format!("{mlp_prefix}.shared_expert_gate"),
            moe_cfg.router_group_size,
            moe_cfg.router_bits,
        )?,
        num_experts,
        top_k,
        norm_topk_prob: moe_cfg.norm_topk_prob,
        router_bits: moe_cfg.router_bits,
        router_group_size: moe_cfg.router_group_size,
        expert_bits: moe_cfg.expert_bits,
        expert_group_size: moe_cfg.expert_group_size,
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

    Ok(tie_lm_head_from_embed_tokens(embed_tokens))
}

/// Load conv1d weight in nn.Conv1d format: [out_channels, kernel_size, in_channels/groups].
/// For depthwise conv (groups=C), shape is [C, K, 1]. Keep native dtype (bf16).
fn load_conv1d_weight(
    weight: &MlxArray,
    linear_cfg: &super::gdr::MetalGdrConfig,
) -> Result<MlxArray> {
    use super::mlx::transpose_axes;

    let c = linear_cfg.qkv_dim() as i32;
    let k = linear_cfg.conv_kernel as i32;
    match weight.shape() {
        // Already [C, K, 1] — nn.Conv1d format
        [ch, ks, 1] if *ch == c && *ks == k => Ok(weight.clone()),
        // HF safetensors store Conv1d kernels in PyTorch layout [C, 1, K].
        // This must be a real axis swap, not a reshape, or the time axis is scrambled.
        [ch, 1, ks] if *ch == c && *ks == k => Ok(transpose_axes(weight, &[0, 2, 1])),
        // [C, K] — reshape to [C, K, 1]
        [ch, ks] if *ch == c && *ks == k => Ok(reshape(weight, &[c, k, 1])),
        shape => anyhow::bail!(
            "unsupported conv1d weight shape {:?}, expected [{c}, {k}, 1]",
            shape
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use super::*;
    use crate::backend::metal::forward::build_forward_graph;
    use crate::backend::metal::sampling::{gpu_sample_token, gpu_sample_token_batched};
    use crate::backend::metal::{
        config::load_metal_config,
        gdr::MetalRecurrentState,
        mlx::{Dtype, as_dtype, concatenate_axis, eval, reshape, slice, slice_update, zeros},
        weights::load_qwen3_metal_weights,
    };
    use crate::test_support::metal_test_guard;
    use crate::tokenizer::Tokenizer;

    fn slice_row_for_sampling(array: &MlxArray, row: i32) -> MlxArray {
        let mut start = vec![0; array.shape().len()];
        let mut end = array.shape().to_vec();
        let strides = vec![1; array.shape().len()];
        start[0] = row;
        end[0] = row + 1;
        slice(array, &start, &end, &strides)
    }

    #[test]
    fn append_qwen35_captured_hidden_chunk_concatenates_context_rows() {
        let _guard = metal_test_guard();
        let mut accumulated = None;
        append_qwen35_captured_hidden_chunk(
            &mut accumulated,
            Some(MlxArray::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])),
        );
        append_qwen35_captured_hidden_chunk(
            &mut accumulated,
            Some(MlxArray::from_slice_f32(&[5.0, 6.0], &[1, 2])),
        );

        let combined = accumulated.expect("captured hidden");
        let combined = as_dtype(&combined, Dtype::Float32);
        eval(&[&combined]);
        assert_eq!(combined.shape(), &[3, 2]);
        assert_eq!(combined.as_slice_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    fn left_pad_kv_cache_row_for_test(
        array: &MlxArray,
        left_pad: i32,
        cache_len: i32,
        target_kv_capacity: i32,
    ) -> MlxArray {
        let shape = array.shape();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0], 1);

        let n_kv = shape[1];
        let head_dim = shape[3];
        let mut padded = zeros(&[1, n_kv, target_kv_capacity, head_dim], array.dtype());
        if cache_len == 0 {
            return padded;
        }

        let valid = slice(
            array,
            &[0, 0, 0, 0],
            &[1, n_kv, cache_len, head_dim],
            &[1, 1, 1, 1],
        );
        padded = slice_update(
            &mut padded,
            &valid,
            &[0, 0, left_pad, 0],
            &[1, n_kv, left_pad + cache_len, head_dim],
        );
        padded
    }

    fn qwen3_model_path() -> Option<PathBuf> {
        env::var_os("QWEN3_MODEL_PATH")
            .map(PathBuf::from)
            .or_else(|| {
                let fallback =
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3-0.6B");
                fallback.exists().then_some(fallback)
            })
    }

    fn qwen35_safetensors_model_path() -> Option<PathBuf> {
        env::var_os("QWEN35_MODEL_PATH")
            .map(PathBuf::from)
            .or_else(|| {
                let fallback =
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3.5-0.8B");
                fallback.exists().then_some(fallback)
            })
    }

    fn qwen35_gguf_model_path() -> Option<PathBuf> {
        env::var_os("QWEN35_GGUF_PATH")
            .map(PathBuf::from)
            .or_else(|| {
                let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
                fallback.exists().then_some(fallback)
            })
    }

    fn dense_weight_f32(weight: &WeightTensor) -> Result<MlxArray> {
        match weight {
            WeightTensor::Dense(tensor) => {
                let tensor = as_dtype(tensor, Dtype::Float32);
                eval(&[&tensor]);
                Ok(tensor)
            }
            WeightTensor::Quantized { .. } => anyhow::bail!("expected dense weight tensor"),
        }
    }

    fn tensor_diff_stats(lhs: &MlxArray, rhs: &MlxArray) -> (f32, usize, f32, f32) {
        let lhs = as_dtype(lhs, Dtype::Float32);
        let rhs = as_dtype(rhs, Dtype::Float32);
        eval(&[&lhs, &rhs]);
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");

        let lhs_slice = lhs.as_slice_f32();
        let rhs_slice = rhs.as_slice_f32();
        let mut max_abs = 0.0_f32;
        let mut max_idx = 0_usize;
        let mut diff_sq = 0.0_f32;
        let mut rhs_sq = 0.0_f32;
        for (idx, (&lhs_value, &rhs_value)) in lhs_slice.iter().zip(rhs_slice.iter()).enumerate() {
            let diff = lhs_value - rhs_value;
            let abs = diff.abs();
            if abs > max_abs {
                max_abs = abs;
                max_idx = idx;
            }
            diff_sq += diff * diff;
            rhs_sq += rhs_value * rhs_value;
        }

        let rms = (diff_sq / lhs_slice.len().max(1) as f32).sqrt();
        let rel_rms = rms / rhs_sq.max(1e-12).sqrt();
        (max_abs, max_idx, rms, rel_rms)
    }

    fn print_tensor_diff(name: &str, lhs: &MlxArray, rhs: &MlxArray) {
        let lhs = as_dtype(lhs, Dtype::Float32);
        let rhs = as_dtype(rhs, Dtype::Float32);
        eval(&[&lhs, &rhs]);
        assert_eq!(lhs.shape(), rhs.shape(), "{name}: shape mismatch");

        let lhs_slice = lhs.as_slice_f32();
        let rhs_slice = rhs.as_slice_f32();
        let (max_abs, max_idx, rms, rel_rms) = tensor_diff_stats(&lhs, &rhs);
        let head_len = lhs_slice.len().min(8);
        eprintln!(
            "{name}: shape={:?} max_abs={max_abs:.6} @ {max_idx} rms={rms:.6} rel_rms={rel_rms:.6} lhs_head={:?} rhs_head={:?}",
            lhs.shape(),
            &lhs_slice[..head_len],
            &rhs_slice[..head_len],
        );
    }

    fn print_tensor_diff_with_transpose_hint(name: &str, lhs: &MlxArray, rhs: &MlxArray) {
        print_tensor_diff(name, lhs, rhs);
        if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
            return;
        }
        let rhs_t = transpose_axes(rhs, &[1, 0]);
        eval(&[&rhs_t]);
        if lhs.shape() != rhs_t.shape() {
            eprintln!(
                "{name}: transpose_hint skipped lhs_shape={:?} rhs_t_shape={:?}",
                lhs.shape(),
                rhs_t.shape(),
            );
            return;
        }
        let (max_abs, max_idx, rms, rel_rms) = tensor_diff_stats(lhs, &rhs_t);
        eprintln!(
            "{name}: transpose_hint max_abs={max_abs:.6} @ {max_idx} rms={rms:.6} rel_rms={rel_rms:.6}",
        );
    }

    fn print_embed_row_diff(name: &str, lhs: &MlxArray, rhs: &MlxArray, row: i32, width: i32) {
        let lhs_row = slice(lhs, &[row, 0], &[row + 1, width], &[1, 1]);
        let rhs_row = slice(rhs, &[row, 0], &[row + 1, width], &[1, 1]);
        print_tensor_diff(name, &lhs_row, &rhs_row);
    }

    fn print_gguf_tensor_info(gguf: &GgufFile, hf_name: &str) -> Result<()> {
        let Ok(gguf_name) = find_tensor_name(gguf, hf_name) else {
            eprintln!("{hf_name}: gguf lookup missing");
            return Ok(());
        };
        let info = &gguf.tensors[&gguf_name];
        let raw_len = gguf.read_tensor_raw(&gguf_name)?.len();
        eprintln!(
            "{hf_name}: gguf_name={gguf_name} dtype={:?} shape={:?} raw_bytes={} size_bytes={}",
            info.dtype,
            info.shape,
            raw_len,
            info.size_bytes(),
        );
        Ok(())
    }

    fn topk_logits(logits: &MlxArray, k: usize) -> Vec<(usize, f32)> {
        let logits = as_dtype(logits, Dtype::Float32);
        eval(&[&logits]);
        let shape = logits.shape();
        let slice = logits.as_slice_f32();
        let row = match shape {
            [vocab] => &slice[..*vocab as usize],
            [batch, vocab] => {
                let batch = *batch as usize;
                let vocab = *vocab as usize;
                let start = batch.saturating_sub(1) * vocab;
                &slice[start..start + vocab]
            }
            other => panic!("unexpected logits shape: {other:?}"),
        };

        let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        pairs.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        pairs.truncate(k);
        pairs
    }

    fn clone_linear_attn_weights(weights: &MetalLinearAttnWeights) -> MetalLinearAttnWeights {
        MetalLinearAttnWeights {
            in_proj_qkvz: weights.in_proj_qkvz.clone(),
            in_proj_ba: weights.in_proj_ba.clone(),
            in_proj_qkv: weights.in_proj_qkv.clone(),
            in_proj_z: weights.in_proj_z.clone(),
            in_proj_b: weights.in_proj_b.clone(),
            in_proj_a: weights.in_proj_a.clone(),
            qkvz_split: weights.qkvz_split,
            ba_num_heads: weights.ba_num_heads,
            conv1d_weight: weights.conv1d_weight.clone(),
            dt_bias: weights.dt_bias.clone(),
            a_log: weights.a_log.clone(),
            norm_weight: weights.norm_weight.clone(),
            out_proj: weights.out_proj.clone(),
            q_scale: weights.q_scale.clone(),
            k_scale: weights.k_scale.clone(),
        }
    }

    fn clone_dense_mlp_weights(weights: &MetalQwen35DenseMlpWeights) -> MetalQwen35DenseMlpWeights {
        MetalQwen35DenseMlpWeights {
            inputs: MlpInputProjection::Split {
                gate_proj: weights.gate_proj.clone(),
                up_proj: weights.up_proj.clone(),
            },
            down_proj: weights.down_proj.clone(),
            gate_proj: weights.gate_proj.clone(),
            up_proj: weights.up_proj.clone(),
        }
    }

    fn clone_dense_mlp_kind(mlp: &MlpKind) -> Result<MlpKind> {
        match mlp {
            MlpKind::Dense(weights) => Ok(MlpKind::Dense(clone_dense_mlp_weights(weights))),
            MlpKind::Moe(_) => anyhow::bail!("expected dense MLP"),
        }
    }

    fn clone_linear_block(block: &MetalQwen35BlockWeights) -> Result<MetalQwen35BlockWeights> {
        let attention = match &block.attention {
            MetalQwen35Attention::Linear(attn) => {
                MetalQwen35Attention::Linear(clone_linear_attn_weights(attn))
            }
            MetalQwen35Attention::Full(_) => anyhow::bail!("expected linear-attention block"),
        };
        Ok(MetalQwen35BlockWeights {
            input_layernorm: block.input_layernorm.clone(),
            attention,
            post_attention_layernorm: block.post_attention_layernorm.clone(),
            mlp: clone_dense_mlp_kind(&block.mlp)?,
        })
    }

    fn forward_linear_block_outputs(
        x: &MlxArray,
        block: &MetalQwen35BlockWeights,
        arch: &MetalQwen35ArchConfig,
        config: &MetalModelConfig,
    ) -> Result<(MlxArray, MlxArray)> {
        let MetalQwen35Attention::Linear(attn) = &block.attention else {
            anyhow::bail!("expected linear-attention block");
        };
        let mut recurrent = MetalRecurrentState::new(1, &arch.linear);
        let attn_out = fused_gdr_step(
            x,
            &block.input_layernorm,
            attn,
            &mut recurrent,
            0,
            &arch.linear,
            config,
        );
        let after_attn = add(x, &attn_out);
        let xn = rms_norm_last_dim(
            &after_attn,
            &block.post_attention_layernorm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        );
        let mlp = mlp_forward(&block.mlp, &xn);
        let after_block = add(&after_attn, &mlp);
        Ok((after_attn, after_block))
    }

    fn print_linear_block_mix_diff(
        name: &str,
        x: &MlxArray,
        block: &MetalQwen35BlockWeights,
        st_after_attn: &MlxArray,
        st_after_block: &MlxArray,
        arch: &MetalQwen35ArchConfig,
        config: &MetalModelConfig,
    ) -> Result<()> {
        let (mix_after_attn, mix_after_block) =
            forward_linear_block_outputs(x, block, arch, config)?;
        print_tensor_diff(
            &format!("{name}.after_attn"),
            st_after_attn,
            &mix_after_attn,
        );
        print_tensor_diff(
            &format!("{name}.after_block"),
            st_after_block,
            &mix_after_block,
        );
        Ok(())
    }

    #[test]
    fn qwen3_compiled_prefill_matches_rust_prefill_for_long_prompt() -> Result<()> {
        let Some(model_path) = qwen3_model_path() else {
            eprintln!(
                "QWEN3_MODEL_PATH unset and ../models/Qwen3-0.6B missing; skipping Qwen3 compiled prefill equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let weights = load_qwen3_metal_weights(&model_path, &config)?;
        let Some(cpp_model) = weights.cpp_model.as_ref() else {
            eprintln!(
                "Qwen3 compiled C++ model unavailable for {}; skipping equivalence test",
                model_path.display()
            );
            return Ok(());
        };

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let prompt_tokens: Vec<u32> = (1..=64).collect();
        let prompt_tokens_i32: Vec<i32> = prompt_tokens.iter().map(|&token| token as i32).collect();
        let prompt_len = i32::try_from(prompt_tokens.len()).expect("prompt len fits in i32");
        let kv_capacity = prompt_len + KV_CACHE_CHUNK;
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_key_value_heads as i32;
        let head_dim = config.head_dim as i32;
        let cache_shape = [1_i32, n_kv_heads, kv_capacity, head_dim];
        let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
        let init_caches = || {
            (
                (0..config.num_hidden_layers)
                    .map(|_| zeros(&cache_shape, kv_dtype))
                    .collect::<Vec<_>>(),
                (0..config.num_hidden_layers)
                    .map(|_| zeros(&cache_shape, kv_dtype))
                    .collect::<Vec<_>>(),
            )
        };

        let (mut rust_k, mut rust_v) = init_caches();
        let rust_sampled = build_forward_graph(
            &prompt_tokens,
            &weights,
            &mut rust_k,
            &mut rust_v,
            0,
            n_heads,
            n_kv_heads,
            head_dim,
            1.0f32 / (head_dim as f32).sqrt(),
            config.rope_theta as f32,
            config.rms_norm_eps as f32,
            None,
            0,
            &params,
        )?;
        let mut rust_refs: Vec<&MlxArray> = Vec::with_capacity(1 + rust_k.len() + rust_v.len());
        rust_refs.push(&rust_sampled);
        rust_refs.extend(rust_k.iter());
        rust_refs.extend(rust_v.iter());
        eval(&rust_refs);
        let rust_token = rust_sampled.item_i32();

        let (mut cpp_k, mut cpp_v) = init_caches();
        let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens_i32, &[prompt_len]);
        let cpp_logits =
            cpp_model.prefill_full_attention(&prompt_arr, prompt_len, 0, &mut cpp_k, &mut cpp_v)?;
        let cpp_sampled = gpu_sample_token(&cpp_logits, &params);
        let mut cpp_refs: Vec<&MlxArray> = Vec::with_capacity(2 + cpp_k.len() + cpp_v.len());
        cpp_refs.push(&cpp_logits);
        cpp_refs.push(&cpp_sampled);
        cpp_refs.extend(cpp_k.iter());
        cpp_refs.extend(cpp_v.iter());
        eval(&cpp_refs);
        let cpp_token = cpp_sampled.item_i32();

        let (session_k, session_v) = init_caches();
        let mut session_kv_flat: Vec<MlxArray> = session_k
            .iter()
            .zip(session_v.iter())
            .flat_map(|(k, v)| [k.clone(), v.clone()])
            .collect();
        cpp_model.begin_session(&session_kv_flat, &[])?;
        let session_logits = cpp_model.prefill_session(&prompt_arr, prompt_len, 0)?;
        let session_sampled = gpu_sample_token(&session_logits, &params);
        let (session_kv_flat_out, session_gdr_flat_out) =
            cpp_model.end_session(session_kv_flat.len(), 0)?;
        anyhow::ensure!(
            session_gdr_flat_out.is_empty(),
            "Qwen3 full-attention session prefill unexpectedly returned GDR state"
        );
        session_kv_flat = session_kv_flat_out;
        let mut session_k = Vec::with_capacity(config.num_hidden_layers);
        let mut session_v = Vec::with_capacity(config.num_hidden_layers);
        let mut session_iter = session_kv_flat.into_iter();
        for _ in 0..config.num_hidden_layers {
            session_k.push(
                session_iter
                    .next()
                    .context("session prefill missing Qwen3 K cache")?,
            );
            session_v.push(
                session_iter
                    .next()
                    .context("session prefill missing Qwen3 V cache")?,
            );
        }
        anyhow::ensure!(
            session_iter.next().is_none(),
            "session prefill returned unexpected extra Qwen3 KV caches"
        );
        let mut session_refs: Vec<&MlxArray> =
            Vec::with_capacity(2 + session_k.len() + session_v.len());
        session_refs.push(&session_logits);
        session_refs.push(&session_sampled);
        session_refs.extend(session_k.iter());
        session_refs.extend(session_v.iter());
        eval(&session_refs);
        let session_token = session_sampled.item_i32();

        assert_eq!(rust_token, cpp_token);
        assert_eq!(cpp_token, session_token);

        for (
            layer_idx,
            (
                ((rust_k_layer, rust_v_layer), (cpp_k_layer, cpp_v_layer)),
                (session_k_layer, session_v_layer),
            ),
        ) in rust_k
            .iter()
            .zip(rust_v.iter())
            .zip(cpp_k.iter().zip(cpp_v.iter()))
            .zip(session_k.iter().zip(session_v.iter()))
            .enumerate()
        {
            let rust_k_valid = slice(
                rust_k_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );
            let rust_v_valid = slice(
                rust_v_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );
            let cpp_k_valid = slice(
                cpp_k_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );
            let cpp_v_valid = slice(
                cpp_v_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );
            let session_k_valid = slice(
                session_k_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );
            let session_v_valid = slice(
                session_v_layer,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, prompt_len, head_dim],
                &[1, 1, 1, 1],
            );

            let rust_k_f32 = as_dtype(&rust_k_valid, Dtype::Float32);
            let rust_v_f32 = as_dtype(&rust_v_valid, Dtype::Float32);
            let cpp_k_f32 = as_dtype(&cpp_k_valid, Dtype::Float32);
            let cpp_v_f32 = as_dtype(&cpp_v_valid, Dtype::Float32);
            let session_k_f32 = as_dtype(&session_k_valid, Dtype::Float32);
            let session_v_f32 = as_dtype(&session_v_valid, Dtype::Float32);
            eval(&[
                &rust_k_f32,
                &rust_v_f32,
                &cpp_k_f32,
                &cpp_v_f32,
                &session_k_f32,
                &session_v_f32,
            ]);

            for (idx, (lhs, rhs)) in rust_k_f32
                .as_slice_f32()
                .iter()
                .zip(cpp_k_f32.as_slice_f32().iter())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "layer {layer_idx} k[{idx}] mismatch: rust={lhs} cpp={rhs}"
                );
            }
            for (idx, (lhs, rhs)) in cpp_k_f32
                .as_slice_f32()
                .iter()
                .zip(session_k_f32.as_slice_f32().iter())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "layer {layer_idx} k[{idx}] mismatch: cpp={lhs} session={rhs}"
                );
            }
            for (idx, (lhs, rhs)) in rust_v_f32
                .as_slice_f32()
                .iter()
                .zip(cpp_v_f32.as_slice_f32().iter())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "layer {layer_idx} v[{idx}] mismatch: rust={lhs} cpp={rhs}"
                );
            }
            for (idx, (lhs, rhs)) in cpp_v_f32
                .as_slice_f32()
                .iter()
                .zip(session_v_f32.as_slice_f32().iter())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "layer {layer_idx} v[{idx}] mismatch: cpp={lhs} session={rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn load_conv1d_weight_transposes_pytorch_depthwise_layout() -> Result<()> {
        let _guard = metal_test_guard();

        let cfg = crate::backend::metal::gdr::MetalGdrConfig {
            num_key_heads: 1,
            key_dim: 1,
            num_value_heads: 1,
            value_dim: 2,
            conv_kernel: 4,
            hidden_size: 1,
            rms_norm_eps: 1e-6,
        };

        let raw = MlxArray::from_slice_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0,
                2000.0, 3000.0, 4000.0,
            ],
            &[4, 1, 4],
        );
        let weight = load_conv1d_weight(&raw, &cfg)?;
        let weight_f32 = as_dtype(&weight, Dtype::Float32);
        eval(&[&weight_f32]);

        assert_eq!(weight_f32.shape(), &[4, 4, 1]);
        assert_eq!(
            weight_f32.as_slice_f32(),
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0,
                2000.0, 3000.0, 4000.0,
            ]
        );

        let reshaped = reshape(&weight_f32, &[4, 4]);
        eval(&[&reshaped]);
        assert_eq!(
            reshaped.as_slice_f32(),
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 1000.0,
                2000.0, 3000.0, 4000.0,
            ]
        );

        Ok(())
    }

    #[test]
    #[ignore = "debug helper for local Metal Qwen3.5 layer norm inspection"]
    fn debug_qwen35_single_token_hidden_norms() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!("QWEN35_MODEL_PATH unset; skipping debug_qwen35_single_token_hidden_norms");
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let num_full_layers = arch.num_full_attention_layers();
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            8,
            config.head_dim as i32,
        ];
        let mut k_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
            .collect();
        let mut v_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
            .collect();
        let mut recurrent =
            MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let target_layers: Vec<usize> = (0..config.num_hidden_layers).collect();
        let input_ids: Vec<u32> = env::var("QWEN35_DEBUG_IDS")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .filter_map(|part| {
                        let trimmed = part.trim();
                        (!trimmed.is_empty())
                            .then(|| trimmed.parse::<u32>().ok())
                            .flatten()
                    })
                    .collect()
            })
            .filter(|ids: &Vec<u32>| !ids.is_empty())
            .unwrap_or_else(|| vec![9419]);

        let (logits, hidden) = qwen35_forward_with_hidden_states(
            &input_ids,
            &weights,
            &config,
            arch,
            &mut k_caches,
            &mut v_caches,
            &mut recurrent,
            0,
            &target_layers,
        );
        let logits = as_dtype(&logits, Dtype::Float32);
        let hidden = as_dtype(&hidden, Dtype::Float32);
        eval(&[&logits, &hidden]);

        let sampled = gpu_sample_token(
            &logits,
            &crate::sampler::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
        );
        eval(&[&sampled]);
        eprintln!(
            "input_ids={input_ids:?} sampled token={}",
            sampled.item_i32()
        );

        let hidden_slice = hidden.as_slice_f32();
        let row_stride = config.num_hidden_layers * config.hidden_size;
        let last_row_start = (input_ids.len() - 1) * row_stride;
        for layer_idx in 0..config.num_hidden_layers {
            let start = last_row_start + layer_idx * config.hidden_size;
            let end = start + config.hidden_size;
            let norm = hidden_slice[start..end]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            eprintln!("layer {layer_idx:02} norm={norm:.6}");
        }

        Ok(())
    }

    #[test]
    #[ignore = "debug helper for comparing loaded Qwen3.5 safetensors vs GGUF tensors"]
    fn compare_qwen35_0p8b_loaded_tensors_safetensors_vs_gguf() -> Result<()> {
        let Some(model_path) = qwen35_safetensors_model_path() else {
            eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
            return Ok(());
        };
        let Some(gguf_path) = qwen35_gguf_model_path() else {
            eprintln!(
                "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
        let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
        let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

        print_gguf_tensor_info(&gguf, "model.norm.weight")?;
        print_tensor_diff("final_norm", &safetensors.norm, &gguf_weights.norm);
        print_gguf_tensor_info(&gguf, "model.embed_tokens.weight")?;
        print_embed_row_diff(
            "embed_tokens[row0, :64]",
            &safetensors.embed_tokens,
            &gguf_weights.embed_tokens,
            0,
            64,
        );
        print_embed_row_diff(
            "embed_tokens[row9419, :64]",
            &safetensors.embed_tokens,
            &gguf_weights.embed_tokens,
            9419,
            64,
        );

        let full_idx = arch
            .layer_types
            .iter()
            .position(|layer| *layer == MetalQwen35LayerType::FullAttention)
            .context("missing full-attention layer")?;
        let linear_idx = arch
            .layer_types
            .iter()
            .position(|layer| *layer == MetalQwen35LayerType::LinearAttention)
            .context("missing linear-attention layer")?;

        let full_s = &safetensors.layers[full_idx];
        let full_g = &gguf_weights.layers[full_idx];
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{full_idx}.input_layernorm.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{full_idx}.input_layernorm"),
            &full_s.input_layernorm,
            &full_g.input_layernorm,
        );
        if let (MetalQwen35Attention::Full(full_s), MetalQwen35Attention::Full(full_g)) =
            (&full_s.attention, &full_g.attention)
        {
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.q_norm.weight"),
            )?;
            print_tensor_diff(
                &format!("layer{full_idx}.self_attn.q_norm"),
                &full_s.q_norm,
                &full_g.q_norm,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.k_norm.weight"),
            )?;
            print_tensor_diff(
                &format!("layer{full_idx}.self_attn.k_norm"),
                &full_s.k_norm,
                &full_g.k_norm,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.q_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{full_idx}.self_attn.q_proj"),
                &dense_weight_f32(&full_s.q_proj)?,
                &dense_weight_f32(&full_g.q_proj)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.k_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{full_idx}.self_attn.k_proj"),
                &dense_weight_f32(&full_s.k_proj)?,
                &dense_weight_f32(&full_g.k_proj)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.v_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{full_idx}.self_attn.v_proj"),
                &dense_weight_f32(&full_s.v_proj)?,
                &dense_weight_f32(&full_g.v_proj)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{full_idx}.self_attn.o_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{full_idx}.self_attn.o_proj"),
                &dense_weight_f32(&full_s.o_proj)?,
                &dense_weight_f32(&full_g.o_proj)?,
            );
        }

        let linear_s = &safetensors.layers[linear_idx];
        let linear_g = &gguf_weights.layers[linear_idx];
        print_gguf_tensor_info(
            &gguf,
            &format!("model.layers.{linear_idx}.input_layernorm.weight"),
        )?;
        print_tensor_diff(
            &format!("layer{linear_idx}.input_layernorm"),
            &linear_s.input_layernorm,
            &linear_g.input_layernorm,
        );
        if let (MetalQwen35Attention::Linear(linear_s), MetalQwen35Attention::Linear(linear_g)) =
            (&linear_s.attention, &linear_g.attention)
        {
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.in_proj_qkv.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.linear_attn.in_proj_qkv"),
                &dense_weight_f32(&linear_s.in_proj_qkv)?,
                &dense_weight_f32(&linear_g.in_proj_qkv)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.in_proj_z.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.linear_attn.in_proj_z"),
                &dense_weight_f32(&linear_s.in_proj_z)?,
                &dense_weight_f32(&linear_g.in_proj_z)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.in_proj_b.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.linear_attn.in_proj_b"),
                &dense_weight_f32(&linear_s.in_proj_b)?,
                &dense_weight_f32(&linear_g.in_proj_b)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.in_proj_a.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.linear_attn.in_proj_a"),
                &dense_weight_f32(&linear_s.in_proj_a)?,
                &dense_weight_f32(&linear_g.in_proj_a)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.conv1d.weight"),
            )?;
            print_tensor_diff(
                &format!("layer{linear_idx}.linear_attn.conv1d_weight"),
                &linear_s.conv1d_weight,
                &linear_g.conv1d_weight,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.dt_bias"),
            )?;
            print_tensor_diff(
                &format!("layer{linear_idx}.linear_attn.dt_bias"),
                &linear_s.dt_bias,
                &linear_g.dt_bias,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.a_log"),
            )?;
            print_tensor_diff(
                &format!("layer{linear_idx}.linear_attn.a_log"),
                &linear_s.a_log,
                &linear_g.a_log,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.norm.weight"),
            )?;
            print_tensor_diff(
                &format!("layer{linear_idx}.linear_attn.norm_weight"),
                &linear_s.norm_weight,
                &linear_g.norm_weight,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.linear_attn.out_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.linear_attn.out_proj"),
                &dense_weight_f32(&linear_s.out_proj)?,
                &dense_weight_f32(&linear_g.out_proj)?,
            );
        }

        if let (MlpKind::Dense(mlp_s), MlpKind::Dense(mlp_g)) = (&linear_s.mlp, &linear_g.mlp) {
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.mlp.gate_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.mlp.gate_proj"),
                &dense_weight_f32(&mlp_s.gate_proj)?,
                &dense_weight_f32(&mlp_g.gate_proj)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.mlp.up_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.mlp.up_proj"),
                &dense_weight_f32(&mlp_s.up_proj)?,
                &dense_weight_f32(&mlp_g.up_proj)?,
            );
            print_gguf_tensor_info(
                &gguf,
                &format!("model.layers.{linear_idx}.mlp.down_proj.weight"),
            )?;
            print_tensor_diff_with_transpose_hint(
                &format!("layer{linear_idx}.mlp.down_proj"),
                &dense_weight_f32(&mlp_s.down_proj)?,
                &dense_weight_f32(&mlp_g.down_proj)?,
            );
        }

        Ok(())
    }

    #[test]
    #[ignore = "debug helper for comparing Qwen3.5 safetensors vs GGUF first-token logits"]
    fn compare_qwen35_0p8b_first_token_logits_safetensors_vs_gguf() -> Result<()> {
        let Some(model_path) = qwen35_safetensors_model_path() else {
            eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
            return Ok(());
        };
        let Some(gguf_path) = qwen35_gguf_model_path() else {
            eprintln!(
                "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
        let input_ids = tokenizer.encode("Hello")?;
        eprintln!("input_ids={input_ids:?}");

        let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
        let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
        let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

        let num_full_layers = arch.num_full_attention_layers();
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            input_ids.len() as i32 + 8,
            config.head_dim as i32,
        ];
        let init_kv = || {
            (
                (0..num_full_layers)
                    .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                    .collect::<Vec<_>>(),
                (0..num_full_layers)
                    .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                    .collect::<Vec<_>>(),
            )
        };

        let (mut st_k, mut st_v) = init_kv();
        let (mut gg_k, mut gg_v) = init_kv();
        let mut st_recurrent =
            MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gg_recurrent =
            MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

        let target_layer = config.num_hidden_layers - 1;
        let (st_logits, st_hidden) = qwen35_forward_with_hidden_states(
            &input_ids,
            &safetensors,
            &config,
            arch,
            &mut st_k,
            &mut st_v,
            &mut st_recurrent,
            0,
            &[target_layer],
        );
        let (gg_logits, gg_hidden) = qwen35_forward_with_hidden_states(
            &input_ids,
            &gguf_weights,
            &config,
            arch,
            &mut gg_k,
            &mut gg_v,
            &mut gg_recurrent,
            0,
            &[target_layer],
        );

        print_tensor_diff("final_hidden", &st_hidden, &gg_hidden);

        let st_final_norm = rms_norm_last_dim(
            &st_hidden,
            &safetensors.norm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        );
        let gg_final_norm = rms_norm_last_dim(
            &gg_hidden,
            &gguf_weights.norm,
            config.rms_norm_eps as f32,
            config.norm_weight_mode.uses_offset(),
        );
        let st_hidden_st_lm = linear(&st_final_norm, &safetensors.lm_head);
        let st_hidden_gg_lm = linear(&st_final_norm, &gguf_weights.lm_head);
        let gg_hidden_st_lm = linear(&gg_final_norm, &safetensors.lm_head);
        let gg_hidden_gg_lm = linear(&gg_final_norm, &gguf_weights.lm_head);

        let st_top = topk_logits(&st_logits, 8);
        let gg_top = topk_logits(&gg_logits, 8);
        let st_hidden_st_lm_top = topk_logits(&st_hidden_st_lm, 4);
        let st_hidden_gg_lm_top = topk_logits(&st_hidden_gg_lm, 4);
        let gg_hidden_st_lm_top = topk_logits(&gg_hidden_st_lm, 4);
        let gg_hidden_gg_lm_top = topk_logits(&gg_hidden_gg_lm, 4);
        let decode = |id: usize| {
            tokenizer
                .decode(&[id as u32])
                .unwrap_or_else(|_| "<decode err>".into())
        };

        eprintln!("safetensors_top8:");
        for (id, logit) in &st_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }
        eprintln!("gguf_top8:");
        for (id, logit) in &gg_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }
        eprintln!("cross_top4 st_hidden + st_lm_head:");
        for (id, logit) in &st_hidden_st_lm_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }
        eprintln!("cross_top4 st_hidden + gg_lm_head:");
        for (id, logit) in &st_hidden_gg_lm_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }
        eprintln!("cross_top4 gg_hidden + st_lm_head:");
        for (id, logit) in &gg_hidden_st_lm_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }
        eprintln!("cross_top4 gg_hidden + gg_lm_head:");
        for (id, logit) in &gg_hidden_gg_lm_top {
            eprintln!("  id={id} logit={logit:.6} text={:?}", decode(*id));
        }

        let st_sampled = gpu_sample_token(
            &st_logits,
            &crate::sampler::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
        );
        let gg_sampled = gpu_sample_token(
            &gg_logits,
            &crate::sampler::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
        );
        eval(&[&st_sampled, &gg_sampled]);
        let st_id = st_sampled.item_i32() as usize;
        let gg_id = gg_sampled.item_i32() as usize;
        eprintln!(
            "sampled: safetensors id={st_id} text={:?} | gguf id={gg_id} text={:?}",
            decode(st_id),
            decode(gg_id),
        );

        Ok(())
    }

    #[test]
    #[ignore = "debug helper for comparing Qwen3.5 safetensors vs GGUF layer-wise hidden states"]
    fn compare_qwen35_0p8b_layerwise_hidden_safetensors_vs_gguf() -> Result<()> {
        let Some(model_path) = qwen35_safetensors_model_path() else {
            eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
            return Ok(());
        };
        let Some(gguf_path) = qwen35_gguf_model_path() else {
            eprintln!(
                "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
        let input_ids = tokenizer.encode("Hello")?;
        eprintln!("input_ids={input_ids:?}");

        let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
        let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
        let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

        let num_full_layers = arch.num_full_attention_layers();
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            input_ids.len() as i32 + 8,
            config.head_dim as i32,
        ];
        let init_kv = || {
            (
                (0..num_full_layers)
                    .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                    .collect::<Vec<_>>(),
                (0..num_full_layers)
                    .map(|_| zeros(&cache_shape, Dtype::Bfloat16))
                    .collect::<Vec<_>>(),
            )
        };
        let target_layers: Vec<usize> = (0..config.num_hidden_layers).collect();

        let (mut st_k, mut st_v) = init_kv();
        let (mut gg_k, mut gg_v) = init_kv();
        let mut st_recurrent =
            MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gg_recurrent =
            MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

        let (_, st_hidden) = qwen35_forward_with_hidden_states(
            &input_ids,
            &safetensors,
            &config,
            arch,
            &mut st_k,
            &mut st_v,
            &mut st_recurrent,
            0,
            &target_layers,
        );
        let (_, gg_hidden) = qwen35_forward_with_hidden_states(
            &input_ids,
            &gguf_weights,
            &config,
            arch,
            &mut gg_k,
            &mut gg_v,
            &mut gg_recurrent,
            0,
            &target_layers,
        );

        let st_hidden = as_dtype(&st_hidden, Dtype::Float32);
        let gg_hidden = as_dtype(&gg_hidden, Dtype::Float32);
        eval(&[&st_hidden, &gg_hidden]);
        let st_slice = st_hidden.as_slice_f32();
        let gg_slice = gg_hidden.as_slice_f32();
        let hidden_size = config.hidden_size;
        for layer_idx in 0..config.num_hidden_layers {
            let start = layer_idx * hidden_size;
            let end = start + hidden_size;
            let st_layer =
                MlxArray::from_slice_f32(&st_slice[start..end], &[1, hidden_size as i32]);
            let gg_layer =
                MlxArray::from_slice_f32(&gg_slice[start..end], &[1, hidden_size as i32]);
            print_tensor_diff(&format!("layer{layer_idx}.hidden"), &st_layer, &gg_layer);
        }

        Ok(())
    }

    #[test]
    #[ignore = "debug helper for isolating the first divergent Qwen3.5 GGUF layer0 subgroup"]
    fn compare_qwen35_0p8b_layer0_linear_subgroup_ablation() -> Result<()> {
        let Some(model_path) = qwen35_safetensors_model_path() else {
            eprintln!("QWEN35_MODEL_PATH unset and ../models/Qwen3.5-0.8B missing; skipping");
            return Ok(());
        };
        let Some(gguf_path) = qwen35_gguf_model_path() else {
            eprintln!(
                "QWEN35_GGUF_PATH unset and ../models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf missing; skipping"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        anyhow::ensure!(
            arch.layer_types.first() == Some(&MetalQwen35LayerType::LinearAttention),
            "expected Qwen3.5-0.8B layer0 to be linear attention"
        );

        let tokenizer = Tokenizer::from_file(model_path.to_str().context("invalid model path")?)?;
        let input_ids = tokenizer.encode("Hello")?;
        anyhow::ensure!(
            !input_ids.is_empty(),
            "expected non-empty tokenized prompt for layer0 ablation"
        );
        let token = MlxArray::from_slice_i32(&[input_ids[0] as i32], &[1]);

        let gguf = GgufFile::open(gguf_path.to_str().context("invalid GGUF path")?)?;
        let safetensors = load_qwen35_metal_weights(&model_path, &config)?;
        let gguf_weights = load_qwen35_metal_weights_from_gguf(&gguf, &config)?;

        let x = take_axis(&safetensors.embed_tokens, &token, 0);
        let st_block = clone_linear_block(&safetensors.layers[0])?;
        let gg_block = clone_linear_block(&gguf_weights.layers[0])?;

        let (st_after_attn, st_after_block) =
            forward_linear_block_outputs(&x, &st_block, arch, &config)?;
        let (gg_after_attn, gg_after_block) =
            forward_linear_block_outputs(&x, &gg_block, arch, &config)?;
        print_tensor_diff("layer0.gguf.after_attn", &st_after_attn, &gg_after_attn);
        print_tensor_diff("layer0.gguf.after_block", &st_after_block, &gg_after_block);

        let (
            MetalQwen35Attention::Linear(st_attn),
            MetalQwen35Attention::Linear(gg_attn),
            MlpKind::Dense(st_mlp),
            MlpKind::Dense(gg_mlp),
        ) = (
            &st_block.attention,
            &gg_block.attention,
            &st_block.mlp,
            &gg_block.mlp,
        )
        else {
            anyhow::bail!("expected dense linear-attention layer0");
        };

        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.input_layernorm = gg_block.input_layernorm.clone();
            print_linear_block_mix_diff(
                "layer0.swap_input_layernorm",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.in_proj_qkvz = gg_attn.in_proj_qkvz.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_in_proj_qkvz",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.in_proj_qkv = gg_attn.in_proj_qkv.clone();
                attn.in_proj_qkvz = concat_dense_weights(&gg_attn.in_proj_qkv, &st_attn.in_proj_z)?;
            }
            print_linear_block_mix_diff(
                "layer0.swap_in_proj_qkv_only",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.in_proj_z = gg_attn.in_proj_z.clone();
                attn.in_proj_qkvz = concat_dense_weights(&st_attn.in_proj_qkv, &gg_attn.in_proj_z)?;
            }
            print_linear_block_mix_diff(
                "layer0.swap_in_proj_z_only",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.in_proj_ba = gg_attn.in_proj_ba.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_in_proj_ba",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.conv1d_weight = gg_attn.conv1d_weight.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_conv1d",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.dt_bias = gg_attn.dt_bias.clone();
                attn.a_log = gg_attn.a_log.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_gate_params",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.norm_weight = gg_attn.norm_weight.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_linear_norm",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            if let MetalQwen35Attention::Linear(attn) = &mut mix.attention {
                attn.out_proj = gg_attn.out_proj.clone();
            }
            print_linear_block_mix_diff(
                "layer0.swap_out_proj",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.post_attention_layernorm = gg_block.post_attention_layernorm.clone();
            print_linear_block_mix_diff(
                "layer0.swap_post_attention_layernorm",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.mlp = MlpKind::Dense(clone_dense_mlp_weights(gg_mlp));
            print_linear_block_mix_diff(
                "layer0.swap_mlp_all",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.mlp = MlpKind::Dense(MetalQwen35DenseMlpWeights {
                inputs: MlpInputProjection::Split {
                    gate_proj: gg_mlp.gate_proj.clone(),
                    up_proj: gg_mlp.up_proj.clone(),
                },
                down_proj: st_mlp.down_proj.clone(),
                gate_proj: gg_mlp.gate_proj.clone(),
                up_proj: gg_mlp.up_proj.clone(),
            });
            print_linear_block_mix_diff(
                "layer0.swap_mlp_gate_up",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.mlp = MlpKind::Dense(MetalQwen35DenseMlpWeights {
                inputs: MlpInputProjection::Split {
                    gate_proj: st_mlp.gate_proj.clone(),
                    up_proj: st_mlp.up_proj.clone(),
                },
                down_proj: gg_mlp.down_proj.clone(),
                gate_proj: st_mlp.gate_proj.clone(),
                up_proj: st_mlp.up_proj.clone(),
            });
            print_linear_block_mix_diff(
                "layer0.swap_mlp_down_proj",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }
        {
            let mut mix = clone_linear_block(&st_block)?;
            mix.attention = MetalQwen35Attention::Linear(clone_linear_attn_weights(gg_attn));
            print_linear_block_mix_diff(
                "layer0.swap_attention_all",
                &x,
                &mix,
                &st_after_attn,
                &st_after_block,
                arch,
                &config,
            )?;
        }

        Ok(())
    }

    #[test]
    fn verify_block_batched_matches_verify_block_for_b1() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping verify_block_batched B=1 equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let prompt_tokens = [1_i32, 2, 3, 4];
        let block_tokens = [5_i32, 6];
        let prompt_len = prompt_tokens.len() as i32;
        let block_size = block_tokens.len() as i32;
        let kv_capacity = prompt_len + block_size + 4;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);

        let cache_pos = prompt_len;
        let mut verify_kv = kv_flat.clone();
        let mut verify_gdr = gdr_flat.clone();
        let verify_tokens = MlxArray::from_slice_i32(&block_tokens, &[block_size]);
        let verify_logits = cpp_model.verify_block(
            &verify_tokens,
            block_size,
            cache_pos,
            &mut verify_kv,
            &mut verify_gdr,
        )?;

        let mut batched_kv = kv_flat.clone();
        let mut batched_gdr = gdr_flat.clone();
        let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
        let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let batched_logits = cpp_model.verify_block_batched(
            &batched_tokens,
            1,
            block_size,
            &[cache_pos],
            &mut batched_kv,
            &mut batched_gdr,
            None,
            &rope_offsets,
        )?;

        let verify_logits_f32 = as_dtype(&verify_logits, Dtype::Float32);
        let batched_logits_f32 = as_dtype(&batched_logits, Dtype::Float32);
        eval(&[&verify_logits_f32, &batched_logits_f32]);

        assert_eq!(verify_logits_f32.shape(), batched_logits_f32.shape());
        for (idx, (lhs, rhs)) in verify_logits_f32
            .as_slice_f32()
            .iter()
            .zip(batched_logits_f32.as_slice_f32().iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "logit[{idx}] mismatch: {lhs} vs {rhs}"
            );
        }

        Ok(())
    }

    #[test]
    fn verify_block_summary_matches_batched_sampled_for_b1() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping verify_block_summary B=1 equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let prompt_tokens = [1_i32, 2, 3, 4];
        let block_tokens = [5_i32, 6];
        let prompt_len = prompt_tokens.len() as i32;
        let block_size = block_tokens.len() as i32;
        let kv_capacity = prompt_len + block_size + 4;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);

        let cache_pos = prompt_len;
        let mut scalar_kv = kv_flat.clone();
        let mut scalar_gdr = gdr_flat.clone();
        let verify_tokens = MlxArray::from_slice_i32(&block_tokens, &[block_size]);
        let scalar_summary = cpp_model.verify_block_summary(
            &verify_tokens,
            block_size,
            cache_pos,
            &mut scalar_kv,
            &mut scalar_gdr,
            &params,
            None,
        )?;

        let mut batched_kv = kv_flat.clone();
        let mut batched_gdr = gdr_flat.clone();
        let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
        let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let batched_sampled = cpp_model.verify_block_batched_sampled(
            &batched_tokens,
            1,
            block_size,
            &[cache_pos],
            &mut batched_kv,
            &mut batched_gdr,
            None,
            &rope_offsets,
            &params,
            None,
        )?;
        let batched_sampled = reshape(&batched_sampled, &[block_size]);

        eval(&[&batched_sampled]);
        let batched_sampled = batched_sampled.as_slice_i32();
        let expected_matched_prefix_len = block_tokens
            .iter()
            .skip(1)
            .zip(batched_sampled.iter())
            .take_while(|(draft, sampled)| draft == sampled)
            .count();
        assert_eq!(
            scalar_summary.matched_prefix_len,
            expected_matched_prefix_len
        );
        assert_eq!(
            scalar_summary.next_token,
            batched_sampled[expected_matched_prefix_len] as u32
        );

        Ok(())
    }

    #[test]
    fn verify_block_batched_matches_independent_verify_block_for_b2() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping verify_block_batched B=2 equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        // Two independent rows with DIFFERENT prompts but the SAME prompt
        // length (so both sit at `cache_pos = prompt_len` after prefill and
        // no left-padding / attention-mask plumbing is exercised at this
        // layer — that's the 2c.4 scheduler's job). Different tokens ensure
        // the per-row GDR state actually diverges, catching any accidental
        // cross-row leak in the batched verify.
        let row_prompts: [[i32; 4]; 2] = [[1, 2, 3, 4], [5, 6, 7, 8]];
        let row_blocks: [[i32; 2]; 2] = [[11, 12], [13, 14]];
        let prompt_len = row_prompts[0].len() as i32;
        let block_size = row_blocks[0].len() as i32;
        let kv_capacity = prompt_len + block_size + 4;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut per_row_kv_after_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
        let mut per_row_gdr_after_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
        let mut per_row_verify_logits: Vec<MlxArray> = Vec::with_capacity(2);

        // Also capture the post-prefill (pre-verify) state per row — we'll
        // stack these as the starting point for the batched verify call.
        let mut per_row_kv_pre_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
        let mut per_row_gdr_pre_verify: Vec<Vec<MlxArray>> = Vec::with_capacity(2);

        for (prompt_tokens, block_tokens) in row_prompts.iter().zip(row_blocks.iter()) {
            let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
                .flat_map(|_| {
                    [
                        zeros(&cache_shape, Dtype::Bfloat16),
                        zeros(&cache_shape, Dtype::Bfloat16),
                    ]
                })
                .collect();
            let recurrent =
                MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
            let mut gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(state, conv)| [state.clone(), conv.clone()])
                .collect();

            let prompt_arr = MlxArray::from_slice_i32(prompt_tokens, &[prompt_len]);
            let prompt_logits =
                cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
            let mut prompt_refs: Vec<&MlxArray> =
                Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
            prompt_refs.push(&prompt_logits);
            prompt_refs.extend(kv_flat.iter());
            prompt_refs.extend(gdr_flat.iter());
            eval(&prompt_refs);

            per_row_kv_pre_verify.push(kv_flat.clone());
            per_row_gdr_pre_verify.push(gdr_flat.clone());

            let verify_tokens = MlxArray::from_slice_i32(block_tokens, &[block_size]);
            let mut verify_kv = kv_flat;
            let mut verify_gdr = gdr_flat;
            let verify_logits = cpp_model.verify_block(
                &verify_tokens,
                block_size,
                prompt_len,
                &mut verify_kv,
                &mut verify_gdr,
            )?;
            let mut verify_refs: Vec<&MlxArray> =
                Vec::with_capacity(1 + verify_kv.len() + verify_gdr.len());
            verify_refs.push(&verify_logits);
            verify_refs.extend(verify_kv.iter());
            verify_refs.extend(verify_gdr.iter());
            eval(&verify_refs);

            per_row_verify_logits.push(verify_logits);
            per_row_kv_after_verify.push(verify_kv);
            per_row_gdr_after_verify.push(verify_gdr);
        }

        // Stack the pre-verify per-row states along batch axis for the
        // batched verify call.
        let n_kv = per_row_kv_pre_verify[0].len();
        let n_gdr = per_row_gdr_pre_verify[0].len();
        let mut packed_kv: Vec<MlxArray> = Vec::with_capacity(n_kv);
        for kv_idx in 0..n_kv {
            let stacked: Vec<MlxArray> = per_row_kv_pre_verify
                .iter()
                .map(|row_kv| row_kv[kv_idx].clone())
                .collect();
            packed_kv.push(concatenate_axis(&stacked, 0));
        }
        let mut packed_gdr: Vec<MlxArray> = Vec::with_capacity(n_gdr);
        for gdr_idx in 0..n_gdr {
            let stacked: Vec<MlxArray> = per_row_gdr_pre_verify
                .iter()
                .map(|row_gdr| row_gdr[gdr_idx].clone())
                .collect();
            packed_gdr.push(concatenate_axis(&stacked, 0));
        }

        let batched_block_tokens: Vec<i32> = row_blocks.iter().flatten().copied().collect();
        let batched_tokens = MlxArray::from_slice_i32(&batched_block_tokens, &[2, block_size]);
        let rope_offsets = MlxArray::from_slice_i32(&[prompt_len, prompt_len], &[2]);

        let batched_logits = cpp_model.verify_block_batched(
            &batched_tokens,
            2,
            block_size,
            &[prompt_len, prompt_len],
            &mut packed_kv,
            &mut packed_gdr,
            None,
            &rope_offsets,
        )?;

        let batched_logits_f32 = as_dtype(&batched_logits, Dtype::Float32);
        let mut eval_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
        eval_refs.push(&batched_logits_f32);
        eval_refs.extend(packed_kv.iter());
        eval_refs.extend(packed_gdr.iter());
        eval(&eval_refs);

        assert_eq!(
            batched_logits_f32.shape(),
            &[2, block_size, config.vocab_size as i32]
        );

        let vocab = config.vocab_size;
        let batched_slice = batched_logits_f32.as_slice_f32();
        assert_eq!(
            batched_slice.len(),
            2 * (block_size as usize) * vocab,
            "batched logits elem count mismatch"
        );

        for (row_idx, scalar_logits) in per_row_verify_logits.iter().enumerate() {
            let scalar_f32 = as_dtype(scalar_logits, Dtype::Float32);
            eval(&[&scalar_f32]);
            let scalar_slice = scalar_f32.as_slice_f32();
            let expected_len = (block_size as usize) * vocab;
            assert_eq!(
                scalar_slice.len(),
                expected_len,
                "scalar row {row_idx} logits len mismatch"
            );
            let offset = row_idx * expected_len;
            for (pos, (lhs, rhs)) in scalar_slice
                .iter()
                .zip(batched_slice[offset..offset + expected_len].iter())
                .enumerate()
            {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "row {row_idx} logit[{pos}] mismatch: scalar={lhs} batched={rhs}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn packed_decode_batched_sampling_matches_scalar_sampling_for_b4() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping packed decode batched sampling B=4 equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let prompt_rows = [
            [1_i32, 2, 3, 4],
            [5_i32, 6, 7, 8],
            [9_i32, 10, 11, 12],
            [13_i32, 14, 15, 16],
        ];
        let batch_size = i32::try_from(prompt_rows.len()).expect("batch size fits in i32");
        let prompt_len = i32::try_from(prompt_rows[0].len()).expect("prompt len fits in i32");
        let kv_capacity = prompt_len + KV_CACHE_CHUNK;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut per_row_kv = Vec::with_capacity(prompt_rows.len());
        let mut per_row_gdr = Vec::with_capacity(prompt_rows.len());
        let mut decode_inputs = Vec::with_capacity(prompt_rows.len());

        for prompt_tokens in prompt_rows {
            let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
                .flat_map(|_| {
                    [
                        zeros(&cache_shape, Dtype::Bfloat16),
                        zeros(&cache_shape, Dtype::Bfloat16),
                    ]
                })
                .collect();
            let recurrent =
                MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
            let mut gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(state, conv)| [state.clone(), conv.clone()])
                .collect();

            let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
            let prompt_logits =
                cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
            let mut prompt_refs: Vec<&MlxArray> =
                Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
            prompt_refs.push(&prompt_logits);
            prompt_refs.extend(kv_flat.iter());
            prompt_refs.extend(gdr_flat.iter());
            eval(&prompt_refs);

            let decode_input = gpu_sample_token(&prompt_logits, &params);
            eval(&[&decode_input]);
            decode_inputs.push(decode_input.item_i32());
            per_row_kv.push(kv_flat);
            per_row_gdr.push(gdr_flat);
        }

        let n_kv = i32::try_from(per_row_kv[0].len()).expect("kv len fits in i32");
        let n_gdr = i32::try_from(per_row_gdr[0].len()).expect("gdr len fits in i32");
        let mut packed_kv = Vec::with_capacity(n_kv as usize);
        for kv_idx in 0..n_kv as usize {
            let stacked: Vec<MlxArray> = per_row_kv
                .iter()
                .map(|row_kv| row_kv[kv_idx].clone())
                .collect();
            packed_kv.push(concatenate_axis(&stacked, 0));
        }
        let mut packed_gdr = Vec::with_capacity(n_gdr as usize);
        for gdr_idx in 0..n_gdr as usize {
            let stacked: Vec<MlxArray> = per_row_gdr
                .iter()
                .map(|row_gdr| row_gdr[gdr_idx].clone())
                .collect();
            packed_gdr.push(concatenate_axis(&stacked, 0));
        }

        let decode_tokens = MlxArray::from_slice_i32(&decode_inputs, &[batch_size]);
        let rope_offsets = MlxArray::from_slice_i32(
            &vec![prompt_len; usize::try_from(batch_size).expect("batch size fits in usize")],
            &[batch_size],
        );
        let batched_logits = cpp_model.step_batch_packed(
            &decode_tokens,
            batch_size,
            prompt_len,
            &mut packed_kv,
            n_kv,
            &mut packed_gdr,
            n_gdr,
            None,
            Some(&rope_offsets),
        )?;
        let mut decode_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
        decode_refs.push(&batched_logits);
        decode_refs.extend(packed_kv.iter());
        decode_refs.extend(packed_gdr.iter());
        eval(&decode_refs);

        let batched_sampled = gpu_sample_token_batched(&batched_logits, &params);
        eval(&[&batched_sampled]);
        let batched_tokens = batched_sampled.as_slice_i32();

        let mut scalar_sampled = Vec::with_capacity(prompt_rows.len());
        for row_idx in 0..batch_size {
            let row_logits = slice_row_for_sampling(&batched_logits, row_idx);
            scalar_sampled.push(gpu_sample_token(&row_logits, &params));
        }
        let scalar_refs: Vec<&MlxArray> = scalar_sampled.iter().collect();
        eval(&scalar_refs);
        let scalar_tokens: Vec<i32> = scalar_sampled
            .iter()
            .map(|sampled| sampled.item_i32())
            .collect();

        assert_eq!(batched_tokens, scalar_tokens);

        Ok(())
    }

    #[test]
    fn packed_decode_varlen_matches_independent_scalar_decode_for_b2() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping packed decode varlen B=2 equivalence test"
            );
            return Ok(());
        };
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let prompt_rows: [&[i32]; 2] = [&[1, 2, 3, 4], &[5, 6]];
        let prompt_lens: Vec<i32> = prompt_rows
            .iter()
            .map(|tokens| i32::try_from(tokens.len()).expect("prompt len fits in i32"))
            .collect();
        let batch_size = i32::try_from(prompt_rows.len()).expect("batch size fits in i32");
        let batch_cache_len = *prompt_lens
            .iter()
            .max()
            .expect("packed decode varlen test requires rows");
        let kv_capacity = batch_cache_len + KV_CACHE_CHUNK;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut per_row_prefill_kv = Vec::with_capacity(prompt_rows.len());
        let mut per_row_prefill_gdr = Vec::with_capacity(prompt_rows.len());
        let mut decode_inputs = Vec::with_capacity(prompt_rows.len());
        let mut scalar_logits_per_row = Vec::with_capacity(prompt_rows.len());

        for (prompt_tokens, &prompt_len) in prompt_rows.iter().zip(prompt_lens.iter()) {
            let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
                .flat_map(|_| {
                    [
                        zeros(&cache_shape, Dtype::Bfloat16),
                        zeros(&cache_shape, Dtype::Bfloat16),
                    ]
                })
                .collect();
            let recurrent =
                MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
            let mut gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(state, conv)| [state.clone(), conv.clone()])
                .collect();

            let prompt_arr = MlxArray::from_slice_i32(prompt_tokens, &[prompt_len]);
            let prompt_logits =
                cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
            let mut prompt_refs: Vec<&MlxArray> =
                Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
            prompt_refs.push(&prompt_logits);
            prompt_refs.extend(kv_flat.iter());
            prompt_refs.extend(gdr_flat.iter());
            eval(&prompt_refs);

            let decode_input = gpu_sample_token(&prompt_logits, &params);
            eval(&[&decode_input]);
            let decode_input = decode_input.item_i32();
            decode_inputs.push(decode_input);

            let decode_token_arr = MlxArray::from_slice_i32(&[decode_input], &[1]);
            let mut scalar_kv = kv_flat.clone();
            let mut scalar_gdr = gdr_flat.clone();
            let scalar_logits = cpp_model.step(
                &decode_token_arr,
                prompt_len,
                &mut scalar_kv,
                &mut scalar_gdr,
            )?;
            let mut scalar_refs: Vec<&MlxArray> =
                Vec::with_capacity(1 + scalar_kv.len() + scalar_gdr.len());
            scalar_refs.push(&scalar_logits);
            scalar_refs.extend(scalar_kv.iter());
            scalar_refs.extend(scalar_gdr.iter());
            eval(&scalar_refs);

            scalar_logits_per_row.push(as_dtype(&scalar_logits, Dtype::Float32));
            per_row_prefill_kv.push(kv_flat);
            per_row_prefill_gdr.push(gdr_flat);
        }

        let left_padding: Vec<i32> = prompt_lens
            .iter()
            .map(|&prompt_len| batch_cache_len - prompt_len)
            .collect();
        let n_kv = per_row_prefill_kv[0].len();
        let n_gdr = per_row_prefill_gdr[0].len();

        let mut packed_kv = Vec::with_capacity(n_kv);
        for kv_idx in 0..n_kv {
            let stacked: Vec<MlxArray> = per_row_prefill_kv
                .iter()
                .zip(left_padding.iter())
                .zip(prompt_lens.iter())
                .map(|((row_kv, &left_pad), &prompt_len)| {
                    if left_pad == 0 {
                        row_kv[kv_idx].clone()
                    } else {
                        left_pad_kv_cache_row_for_test(
                            &row_kv[kv_idx],
                            left_pad,
                            prompt_len,
                            kv_capacity,
                        )
                    }
                })
                .collect();
            packed_kv.push(concatenate_axis(&stacked, 0));
        }

        let mut packed_gdr = Vec::with_capacity(n_gdr);
        for gdr_idx in 0..n_gdr {
            let stacked: Vec<MlxArray> = per_row_prefill_gdr
                .iter()
                .map(|row_gdr| row_gdr[gdr_idx].clone())
                .collect();
            packed_gdr.push(concatenate_axis(&stacked, 0));
        }

        let decode_tokens = MlxArray::from_slice_i32(&decode_inputs, &[batch_size]);
        let attn_mask =
            crate::backend::metal::mlx::build_varlen_decode_mask(&left_padding, batch_cache_len);
        let rope_offsets = MlxArray::from_slice_i32(&prompt_lens, &[batch_size]);
        let packed_logits = cpp_model.step_batch_packed(
            &decode_tokens,
            batch_size,
            batch_cache_len,
            &mut packed_kv,
            i32::try_from(n_kv).expect("kv len fits in i32"),
            &mut packed_gdr,
            i32::try_from(n_gdr).expect("gdr len fits in i32"),
            Some(&attn_mask),
            Some(&rope_offsets),
        )?;
        let packed_logits = as_dtype(&packed_logits, Dtype::Float32);
        let mut packed_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + packed_kv.len() + packed_gdr.len());
        packed_refs.push(&packed_logits);
        packed_refs.extend(packed_kv.iter());
        packed_refs.extend(packed_gdr.iter());
        eval(&packed_refs);

        assert_eq!(
            packed_logits.shape(),
            &[batch_size, 1, config.vocab_size as i32]
        );

        for (row_idx, scalar_logits) in scalar_logits_per_row.iter().enumerate() {
            eval(&[scalar_logits]);
            let packed_row = slice_row_for_sampling(
                &packed_logits,
                i32::try_from(row_idx).expect("row index fits in i32"),
            );
            let scalar_slice = scalar_logits.as_slice_f32();
            let packed_slice = packed_row.as_slice_f32();
            assert_eq!(
                scalar_slice.len(),
                packed_slice.len(),
                "row {row_idx} logits len mismatch"
            );
            for (pos, (lhs, rhs)) in scalar_slice.iter().zip(packed_slice.iter()).enumerate() {
                assert!(
                    (lhs - rhs).abs() < 1e-3,
                    "row {row_idx} logit[{pos}] mismatch: scalar={lhs} packed={rhs}"
                );
            }
        }

        Ok(())
    }
}
