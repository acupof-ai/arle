use std::{path::Path, time::Instant};

use anyhow::{Context, Result};

use super::mlx::{
    Dtype, MlxArray, add, as_dtype, concatenate_axis, multiply, reshape, rms_norm, rope,
    scaled_dot_product_attention, sigmoid, silu, slice, slice_update, take_axis, transpose_axes,
    zeros,
};

use super::gdr::{MetalLinearAttnWeights, MetalRecurrentState, metal_gdr_decode_step};
use super::weights::{StackedQuantized, load_quantized_with_bits, load_stacked_quantized};
use super::{
    KV_CACHE_CHUNK, MetalModelArch, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
    MlpInputProjection, WeightTensor, clear_metal_cache, extend_kv_cache, gpu_sample_token, linear,
    load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map,
    merge_quantized_projection_rows, tensor_get, tie_lm_head_from_embed_tokens,
};
use crate::backend::is_stream_stop_matched;
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

    /// DFlash verify: parallel forward over a draft block, returning all-position
    /// logits `[1, block_size, vocab]`. Mirrors `prefill` but forces
    /// `last_logits_only = false` so the caller can sample every position in the
    /// draft in a single pass. Tape/hidden capture flags on the model are
    /// respected — one call emits the full per-step GDR innovation tape and the
    /// full hidden-state capture for the block.
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

    /// Batched DFlash verify: run `block_size` draft tokens for `batch_size`
    /// rows in a single forward. Mirrors `verify_block` but feeds the packed
    /// KV/GDR states and per-row `cache_pos_arr` / `rope_offsets` that
    /// `step_batch_packed` already uses for plain-decode.
    ///
    /// Shapes:
    /// - `tokens`: int32 `[B, block_size]`.
    /// - `cache_pos_arr`: int32 `[B]` — per-row physical KV write start.
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
    pub(super) fn verify_block_batched(
        &self,
        tokens: &MlxArray,
        batch_size: i32,
        block_size: i32,
        cache_pos_arr: &MlxArray,
        packed_kv_caches: &mut [MlxArray],
        packed_gdr_states: &mut [MlxArray],
        attn_mask: Option<&MlxArray>,
        rope_offsets: &MlxArray,
    ) -> Result<MlxArray> {
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
                cache_pos_arr.as_raw(),
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
    let mut pos = cache_len;

    for &token in input_ids {
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
        pos += 1;
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
            input_layernorm: get(&format!("{layer_prefix}.input_layernorm.weight"))?,
            attention,
            post_attention_layernorm: get(&format!(
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

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use super::*;
    use crate::backend::metal::sampling::gpu_sample_token_batched;
    use crate::backend::metal::{
        config::load_metal_config,
        gdr::MetalRecurrentState,
        mlx::{Dtype, as_dtype, concatenate_axis, eval, slice, slice_update, zeros},
    };
    use crate::test_support::metal_test_guard;

    fn slice_row_for_sampling(array: &MlxArray, row: i32) -> MlxArray {
        let mut start = vec![0; array.shape().len()];
        let mut end = array.shape().to_vec();
        let strides = vec![1; array.shape().len()];
        start[0] = row;
        end[0] = row + 1;
        slice(array, &start, &end, &strides)
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
        let cache_pos_arr = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let batched_logits = cpp_model.verify_block_batched(
            &batched_tokens,
            1,
            block_size,
            &cache_pos_arr,
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
        let cache_pos_arr = MlxArray::from_slice_i32(&[prompt_len, prompt_len], &[2]);
        let rope_offsets = MlxArray::from_slice_i32(&[prompt_len, prompt_len], &[2]);

        let batched_logits = cpp_model.verify_block_batched(
            &batched_tokens,
            2,
            block_size,
            &cache_pos_arr,
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
