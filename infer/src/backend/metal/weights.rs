use std::path::Path;

use anyhow::{Context, Result};

use super::config::MetalModelConfig;
use super::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
use super::mlx::{Dtype, MlxArray, concatenate_axis, eval};
use super::qwen35;

/// A weight matrix that is either full-precision BF16 or MLX affine 4-bit quantized.
///
/// MLX quantization format (affine per-group):
/// - `w`: packed uint32 — shape `[out, in / (32/bits)]`
/// - `scales`: f16/bf16 — shape `[out, in / group_size]`
/// - `biases`: f16/bf16 — shape `[out, in / group_size]`
///
/// `Dense(w_t)` stores the **transposed** weight `w.T` — shape `[in, out]`.
/// Pre-transposed at load time so `linear()` calls `matmul(x, w_t)` directly,
/// eliminating one `transpose()` view creation per forward-pass call.
// GPU required: MlxArray is backed by Metal buffers.
#[cfg(feature = "metal")]
#[derive(Clone)]
pub enum WeightTensor {
    /// Pre-transposed weight: shape [in, out] = w.T. Ready for `x @ w_t`.
    Dense(MlxArray),
    Quantized {
        w: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        group_size: i32,
        bits: i32,
    },
    /// Native GGUF packed row-major K-quant blocks. The backing array is raw
    /// `uint8` bytes, and kernels decode the GGUF layout directly.
    GgufPacked {
        w: MlxArray,
        format: GgufPackedFormat,
        rows: i32,
        cols: i32,
    },
}

#[cfg(feature = "metal")]
impl WeightTensor {
    /// Returns the element dtype: for Dense the array dtype; for Quantized the scales dtype
    /// (scales hold the dequantization type, e.g. bfloat16).
    pub fn dtype(&self) -> Dtype {
        match self {
            WeightTensor::Dense(a) => a.dtype(),
            WeightTensor::Quantized { scales, .. } => scales.dtype(),
            WeightTensor::GgufPacked { .. } => Dtype::Bfloat16,
        }
    }

    /// Logical output width of the projection.
    ///
    /// Dense tensors are stored transposed as `[in, out]`; quantized tensors
    /// keep MLX packed layout `[out, packed_in]`.
    pub fn output_dim(&self) -> Result<i32> {
        let shape = match self {
            WeightTensor::Dense(w_t) => w_t.shape(),
            WeightTensor::Quantized { w, .. } => w.shape(),
            WeightTensor::GgufPacked { rows, .. } => return Ok(*rows),
        };

        match self {
            WeightTensor::Dense(_) => shape
                .get(1)
                .copied()
                .context("dense projection missing output dimension"),
            WeightTensor::Quantized { .. } => shape
                .first()
                .copied()
                .context("quantized projection missing output dimension"),
            WeightTensor::GgufPacked { .. } => unreachable!(),
        }
    }
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum GgufPackedFormat {
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

#[cfg(feature = "metal")]
impl GgufPackedFormat {
    pub fn block_size(self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// A 3-D stack of affine-quantized expert weights (Qwen3.5/3.6 MoE).
///
/// Mirrors the MLX community Qwen3.6 checkpoint layout after `sanitize()`:
///   - `weight`: packed uint32, shape `[num_experts, out_dim, in_dim / packing]`
///   - `scales` / `biases`: shape `[num_experts, out_dim, in_dim / group_size]`
///
/// Feeds `mlx::core::gather_qmm` inside the Qwen3.5 MoE block (see
/// `crates/mlx-sys/src/mlx_qwen35_moe_block.cpp`). Unlike [`WeightTensor`],
/// this variant assumes 3-D shapes and intentionally does not participate in
/// the 2-D `merge_quantized_projection_rows` fusion.
// GPU required: MlxArrays are backed by Metal buffers.
#[cfg(feature = "metal")]
pub struct StackedQuantized {
    pub weight: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
    pub group_size: i32,
    pub bits: i32,
}

/// Load a 2-D affine-quantized projection with explicit bits/group_size.
///
/// Unlike [`load_proj_from_tensors`], which takes the model-level
/// [`super::config::QuantConfig`], this helper lets the caller override the
/// per-tensor bit width — Qwen3.6 stores its router (`mlp.gate`) and the
/// shared-expert sigmoid gate (`mlp.shared_expert_gate`) as 8-bit while the
/// rest of the MLP is 4-bit. Fails loudly if any of the triple is missing.
#[cfg(feature = "metal")]
pub(super) fn load_quantized_with_bits(
    tensors: &TensorMap,
    base: &str,
    group_size: i32,
    bits: i32,
) -> Result<WeightTensor> {
    let w = tensors
        .get(&format!("{base}.weight"))
        .cloned()
        .with_context(|| format!("missing quantized weight '{base}.weight'"))?;
    let scales = tensors
        .get(&format!("{base}.scales"))
        .cloned()
        .with_context(|| format!("missing quantized scales '{base}.scales'"))?;
    let biases = tensors
        .get(&format!("{base}.biases"))
        .cloned()
        .with_context(|| format!("missing quantized biases '{base}.biases'"))?;
    Ok(WeightTensor::Quantized {
        w,
        scales,
        biases,
        group_size,
        bits,
    })
}

/// Load a 3-D affine-quantized expert stack from a safetensors tensor map.
///
/// Unlike [`load_proj_from_tensors`], this helper always returns
/// [`StackedQuantized`] and fails if any of the three tensors are missing —
/// there is no dense fallback for stacked experts in the Qwen3.5/3.6 MoE
/// checkpoint format. `group_size` and `bits` must be passed explicitly
/// because the per-layer router/expert quantization bits differ from the
/// global default (router is 8-bit, experts are 4-bit in A3B-4bit).
///
/// `base` is the key prefix that owns `.weight`, `.scales`, `.biases` —
/// e.g. `"language_model.model.layers.0.mlp.switch_mlp.gate_proj"`.
#[cfg(feature = "metal")]
pub(super) fn load_stacked_quantized(
    tensors: &TensorMap,
    base: &str,
    group_size: i32,
    bits: i32,
) -> Result<StackedQuantized> {
    let weight = tensors
        .get(&format!("{base}.weight"))
        .cloned()
        .with_context(|| format!("missing stacked quantized weight '{base}.weight'"))?;
    let scales = tensors
        .get(&format!("{base}.scales"))
        .cloned()
        .with_context(|| format!("missing stacked quantized scales '{base}.scales'"))?;
    let biases = tensors
        .get(&format!("{base}.biases"))
        .cloned()
        .with_context(|| format!("missing stacked quantized biases '{base}.biases'"))?;
    anyhow::ensure!(
        weight.shape().len() == 3,
        "stacked quantized weight '{base}.weight' must be 3-D, got shape {:?}",
        weight.shape()
    );
    Ok(StackedQuantized {
        weight,
        scales,
        biases,
        group_size,
        bits,
    })
}

/// Attention input projections for one transformer layer.
#[cfg(feature = "metal")]
pub enum AttentionInputProjection {
    /// Dense/non-merged path.
    Split {
        q_proj: WeightTensor,
        k_proj: WeightTensor,
        v_proj: WeightTensor,
    },
    /// Quantized fallback path with a single `qkv` matmul.
    MergedQuantized {
        qkv_proj: WeightTensor,
        q_dim: i32,
        k_dim: i32,
        v_dim: i32,
    },
}

#[cfg(feature = "metal")]
impl AttentionInputProjection {
    pub(super) fn kv_dtype(&self) -> Dtype {
        match self {
            Self::Split { k_proj, .. } => k_proj.dtype(),
            Self::MergedQuantized { qkv_proj, .. } => qkv_proj.dtype(),
        }
    }

    pub(super) fn project(&self, x: &MlxArray) -> (MlxArray, MlxArray, MlxArray) {
        match self {
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => (
                super::linear(x, q_proj),
                super::linear(x, k_proj),
                super::linear(x, v_proj),
            ),
            Self::MergedQuantized {
                qkv_proj,
                q_dim,
                k_dim,
                v_dim,
            } => {
                let qkv = super::linear(x, qkv_proj);
                let q_raw = slice_last_dim(&qkv, 0, *q_dim);
                let k_raw = slice_last_dim(&qkv, *q_dim, *k_dim);
                let v_raw = slice_last_dim(&qkv, *q_dim + *k_dim, *v_dim);
                (q_raw, k_raw, v_raw)
            }
        }
    }
}

/// MLP input projections for one transformer layer.
#[cfg(feature = "metal")]
pub enum MlpInputProjection {
    /// Dense/non-merged path.
    Split {
        gate_proj: WeightTensor,
        up_proj: WeightTensor,
    },
    /// Quantized fallback path with a single `gate+up` matmul.
    MergedQuantized {
        gate_up_proj: WeightTensor,
        gate_dim: i32,
        up_dim: i32,
    },
}

#[cfg(feature = "metal")]
impl MlpInputProjection {
    pub(super) fn project(&self, x: &MlxArray) -> (MlxArray, MlxArray) {
        match self {
            Self::Split { gate_proj, up_proj } => {
                (super::linear(x, gate_proj), super::linear(x, up_proj))
            }
            Self::MergedQuantized {
                gate_up_proj,
                gate_dim,
                up_dim,
            } => {
                let gate_up = super::linear(x, gate_up_proj);
                let gate_raw = slice_last_dim(&gate_up, 0, *gate_dim);
                let up_raw = slice_last_dim(&gate_up, *gate_dim, *up_dim);
                (gate_raw, up_raw)
            }
        }
    }
}

#[cfg(feature = "metal")]
fn slice_last_dim(array: &MlxArray, offset: i32, len: i32) -> MlxArray {
    use super::mlx::slice;

    let ndim = array.shape().len();
    debug_assert!(ndim >= 1);
    let mut start = vec![0; ndim];
    let mut end = array.shape().to_vec();
    let strides = vec![1; ndim];
    let last = ndim - 1;
    start[last] = offset;
    end[last] = offset + len;
    slice(array, &start, &end, &strides)
}

/// Weight tensors loaded from safetensors shards into Metal unified memory.
// GPU required: all fields are MlxArrays backed by Metal buffers.
#[cfg(feature = "metal")]
pub(super) struct StandardMetalWeights {
    /// Token embedding table — shape [vocab_size, hidden_size] (dequantized to float at load).
    pub(super) embed_tokens: MlxArray,
    /// Per-layer attention + MLP weights.
    pub(super) layers: Vec<StandardMetalLayerWeights>,
    /// Final layer-norm scale — shape [hidden_size].
    pub(super) norm: MlxArray,
    /// Output projection (lm_head) — dense or quantized.
    pub(super) lm_head: WeightTensor,
    /// C++ compiled model (reuses Qwen3.5 model struct with all full-attn layers).
    pub(super) cpp_model: Option<qwen35::CppQwen35Model>,
}

#[cfg(feature = "metal")]
pub(super) enum MetalWeights {
    Qwen3(StandardMetalWeights),
    Qwen35(qwen35::Qwen35MetalWeights),
}

/// Weights for a single transformer layer.
// GPU required: all fields are MlxArrays or WeightTensors.
#[cfg(feature = "metal")]
pub(super) struct StandardMetalLayerWeights {
    pub(super) attention_inputs: AttentionInputProjection,
    pub(super) o_proj: WeightTensor,
    pub(super) mlp_inputs: MlpInputProjection,
    pub(super) down_proj: WeightTensor,
    pub(super) q_norm: MlxArray,
    pub(super) k_norm: MlxArray,
    pub(super) input_layernorm: MlxArray,
    pub(super) post_attention_layernorm: MlxArray,
    /// Individual q/k/v projections for C++ model (separate from merged attention_inputs).
    pub(super) q_proj_individual: Option<WeightTensor>,
    pub(super) k_proj_individual: Option<WeightTensor>,
    pub(super) v_proj_individual: Option<WeightTensor>,
}

#[cfg(feature = "metal")]
pub(super) fn merge_quantized_projection_rows(
    weights: &[&WeightTensor],
) -> Result<Option<WeightTensor>> {
    if weights.is_empty() {
        return Ok(None);
    }

    let mut ws = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len());
    let mut biases = Vec::with_capacity(weights.len());
    let mut expected_w_dtype = None;
    let mut expected_scales_dtype = None;
    let mut expected_biases_dtype = None;
    let mut expected_group_size = None;
    let mut expected_bits = None;
    let mut expected_packed_in = None;
    let mut expected_group_cols = None;

    for weight in weights {
        let WeightTensor::Quantized {
            w,
            scales: scale,
            biases: bias,
            group_size,
            bits,
        } = weight
        else {
            return Ok(None);
        };

        let packed_in = w
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing packed input dimension")?;
        let group_cols = scale
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing scale group dimension")?;

        if let Some(expected) = expected_group_size {
            if *group_size != expected {
                return Ok(None);
            }
        } else {
            expected_group_size = Some(*group_size);
        }

        if let Some(expected) = expected_bits {
            if *bits != expected {
                return Ok(None);
            }
        } else {
            expected_bits = Some(*bits);
        }

        if let Some(expected) = expected_packed_in {
            if packed_in != expected {
                return Ok(None);
            }
        } else {
            expected_packed_in = Some(packed_in);
        }

        if let Some(expected) = expected_group_cols {
            if group_cols != expected {
                return Ok(None);
            }
        } else {
            expected_group_cols = Some(group_cols);
        }

        if let Some(expected) = expected_w_dtype {
            if w.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_w_dtype = Some(w.dtype());
        }

        if let Some(expected) = expected_scales_dtype {
            if scale.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_scales_dtype = Some(scale.dtype());
        }

        if let Some(expected) = expected_biases_dtype {
            if bias.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_biases_dtype = Some(bias.dtype());
        }

        ws.push(w.clone());
        scales.push(scale.clone());
        biases.push(bias.clone());
    }

    let merged_w = concatenate_axis(&ws, 0);
    let merged_scales = concatenate_axis(&scales, 0);
    let merged_biases = concatenate_axis(&biases, 0);
    eval(&[&merged_w, &merged_scales, &merged_biases]);

    Ok(Some(WeightTensor::Quantized {
        w: merged_w,
        scales: merged_scales,
        biases: merged_biases,
        group_size: expected_group_size.unwrap_or_default(),
        bits: expected_bits.unwrap_or_default(),
    }))
}

/// Load safetensors shards into Metal unified memory.
// GPU required: MlxArray is backed by Metal buffers.
#[cfg(feature = "metal")]
pub(super) fn load_qwen3_metal_weights(
    model_dir: &Path,
    config: &MetalModelConfig,
) -> Result<StandardMetalWeights> {
    let tensors = load_tensor_map(model_dir)?;
    let get = |name: &str| tensor_get(&tensors, name);
    let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, config.quantization);

    // embed_tokens: dequantize at load time if quantized (needed for embedding lookup via take_axis).
    let embed_tokens =
        load_embed_tokens_from_tensors(&tensors, "model.embed_tokens", config.quantization)?;

    let norm = get("model.norm.weight")?;

    // lm_head may be weight-tied to embed_tokens; handle both dense and quantized.
    // When tied, use quantized_matmul on the original packed weights (as_linear pattern)
    // instead of dense matmul on dequantized bf16 — saves ~7.5ms/step for large vocabs.
    let lm_head =
        if tensors.contains_key("lm_head.weight") || tensors.contains_key("lm_head.scales") {
            load_proj("lm_head")?
        } else if config.quantization.is_some() {
            // Tied + quantized: load original packed embed weights for quantized_matmul
            load_proj("model.embed_tokens")?
        } else {
            tie_lm_head_from_embed_tokens(&embed_tokens)
        };

    let n = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let q_proj = load_proj(&p("self_attn.q_proj"))?;
        let k_proj = load_proj(&p("self_attn.k_proj"))?;
        let v_proj = load_proj(&p("self_attn.v_proj"))?;
        let q_dim = q_proj.output_dim()?;
        let k_dim = k_proj.output_dim()?;
        let v_dim = v_proj.output_dim()?;
        let attention_inputs = if let Some(qkv_proj) =
            merge_quantized_projection_rows(&[&q_proj, &k_proj, &v_proj])?
        {
            AttentionInputProjection::MergedQuantized {
                qkv_proj,
                q_dim,
                k_dim,
                v_dim,
            }
        } else {
            AttentionInputProjection::Split {
                q_proj,
                k_proj,
                v_proj,
            }
        };

        let gate_proj = load_proj(&p("mlp.gate_proj"))?;
        let up_proj = load_proj(&p("mlp.up_proj"))?;
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

        // Reload individual q/k/v for C++ model (separate from merged)
        let q_ind = load_proj(&p("self_attn.q_proj")).ok();
        let k_ind = load_proj(&p("self_attn.k_proj")).ok();
        let v_ind = load_proj(&p("self_attn.v_proj")).ok();

        layers.push(StandardMetalLayerWeights {
            attention_inputs,
            o_proj: load_proj(&p("self_attn.o_proj"))?,
            mlp_inputs,
            down_proj: load_proj(&p("mlp.down_proj"))?,
            q_norm: get(&p("self_attn.q_norm.weight"))?,
            k_norm: get(&p("self_attn.k_norm.weight"))?,
            input_layernorm: get(&p("input_layernorm.weight"))?,
            post_attention_layernorm: get(&p("post_attention_layernorm.weight"))?,
            q_proj_individual: q_ind,
            k_proj_individual: k_ind,
            v_proj_individual: v_ind,
        });
    }

    // Load quantized embed for as_linear lm_head
    let embed_quantized = if config.quantization.is_some() {
        load_proj("model.embed_tokens").ok()
    } else {
        None
    };

    // Build C++ model (all full-attn layers — reuses Qwen3.5 model struct)
    let cpp_model = build_qwen3_cpp_model(
        &embed_tokens,
        &norm,
        &lm_head,
        embed_quantized.as_ref(),
        &layers,
        config,
    );

    Ok(StandardMetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
        cpp_model,
    })
}

/// Build C++ compiled model for Qwen3 (pure transformer, all full-attn layers).
/// Reuses the Qwen3.5 C++ model struct with zero GDR layers.
#[cfg(feature = "metal")]
// reason: q/k/v/o and packed weight tuples mirror attention/FFI terminology.
#[allow(clippy::many_single_char_names)]
fn build_qwen3_cpp_model(
    embed_tokens: &MlxArray,
    norm: &MlxArray,
    lm_head: &WeightTensor,
    embed_quantized: Option<&WeightTensor>,
    layers: &[StandardMetalLayerWeights],
    config: &MetalModelConfig,
) -> Option<qwen35::CppQwen35Model> {
    use qwen35::CppQwen35Model;

    let model = unsafe { mlx_sys::qwen35_compiled_new() };
    if model.is_null() {
        return None;
    }

    unsafe {
        mlx_sys::qwen35_compiled_set_config(
            model,
            config.rope_theta as f32,
            config.rms_norm_eps as f32,
            config.num_attention_heads as i32,
            config.num_key_value_heads as i32,
            config.head_dim as i32,
            config.head_dim as i32, // rotary_dim = head_dim for Qwen3
            config.hidden_size as i32,
        );
        // Qwen3 has no QK gate (q_dim = nh*hd).
        mlx_sys::qwen35_compiled_set_qk_gate(model, 0);
    }

    // Embed + norm + lm_head
    let (lm_w, lm_s, lm_b, lm_gs, lm_bits) = match lm_head {
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
            0,
        ),
        WeightTensor::GgufPacked { .. } => {
            unsafe { mlx_sys::qwen35_compiled_free(model) };
            return None;
        }
    };
    unsafe {
        mlx_sys::qwen35_compiled_set_embed(
            model,
            embed_tokens.as_raw(),
            norm.as_raw(),
            lm_w,
            lm_s,
            lm_b,
            lm_gs,
            lm_bits,
        );
        if let Some(WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        }) = embed_quantized
        {
            mlx_sys::qwen35_compiled_set_embed_as_linear(
                model,
                w.as_raw(),
                scales.as_raw(),
                biases.as_raw(),
                *group_size,
                *bits,
            );
        }
    }

    // All layers are full attention
    for layer in layers {
        let extract = |wt: &WeightTensor| -> Option<(
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
                WeightTensor::Dense(_) | WeightTensor::GgufPacked { .. } => None,
            }
        };

        // Extract attention projections from AttentionInputProjection
        let (q, k, v) = match &layer.attention_inputs {
            AttentionInputProjection::Split {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let (Some(q), Some(k), Some(v)) =
                    (extract(q_proj), extract(k_proj), extract(v_proj))
                else {
                    unsafe { mlx_sys::qwen35_compiled_free(model) };
                    return None;
                };
                (q, k, v)
            }
            AttentionInputProjection::MergedQuantized { .. } => {
                // Use individual weights stored alongside merged
                let (Some(q), Some(k), Some(v)) = (
                    layer.q_proj_individual.as_ref().and_then(extract),
                    layer.k_proj_individual.as_ref().and_then(extract),
                    layer.v_proj_individual.as_ref().and_then(extract),
                ) else {
                    unsafe { mlx_sys::qwen35_compiled_free(model) };
                    return None;
                };
                (q, k, v)
            }
        };
        let Some(o) = extract(&layer.o_proj) else {
            unsafe { mlx_sys::qwen35_compiled_free(model) };
            return None;
        };

        // MLP
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
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            };
        let Some((dw_w, dw_s, dw_b, _, _)) = extract(&layer.down_proj) else {
            unsafe { mlx_sys::qwen35_compiled_free(model) };
            return None;
        };

        unsafe {
            mlx_sys::qwen35_compiled_push_full_attn(
                model,
                layer.input_layernorm.as_raw(),
                layer.post_attention_layernorm.as_raw(),
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
                layer.q_norm.as_raw(),
                layer.k_norm.as_raw(),
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

    let rc = unsafe { mlx_sys::qwen35_compiled_finalize(model) };
    if rc != 0 {
        unsafe { mlx_sys::qwen35_compiled_free(model) };
        return None;
    }

    log::info!(
        "  C++ Qwen3 model ready ({} full-attn layers)",
        layers.len()
    );
    Some(CppQwen35Model::from_raw(model))
}
