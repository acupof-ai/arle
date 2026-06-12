//! Qwen3.5/Qwen3.6 C++ compiled-model port for the clean Metal executor.

use std::path::Path;

use anyhow::{Context, Result};

use crate::config::{
    MetalGdrConfig, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType,
};
use crate::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
use crate::mlx::{self, Dtype, MlxArray, add, as_dtype, reshape, transpose_axes};
use crate::weights::{
    MlpInputProjection, StackedQuantized, WeightTensor, concat_weight_rows,
    load_quantized_with_bits, load_stacked_quantized, merge_quantized_projection_rows,
};

pub(crate) struct MetalQwen35FullAttentionWeights {
    pub(crate) q_proj: WeightTensor,
    pub(crate) k_proj: WeightTensor,
    pub(crate) v_proj: WeightTensor,
    pub(crate) o_proj: WeightTensor,
    pub(crate) q_norm: MlxArray,
    pub(crate) k_norm: MlxArray,
}

pub(crate) struct MetalLinearAttnWeights {
    pub(crate) in_proj_qkvz: Option<WeightTensor>,
    pub(crate) in_proj_ba: Option<WeightTensor>,
    pub(crate) in_proj_qkv: WeightTensor,
    pub(crate) in_proj_z: WeightTensor,
    pub(crate) in_proj_b: WeightTensor,
    pub(crate) in_proj_a: WeightTensor,
    pub(crate) qkvz_split: (i32, i32),
    pub(crate) ba_num_heads: i32,
    pub(crate) conv1d_weight: MlxArray,
    pub(crate) dt_bias: MlxArray,
    pub(crate) a_log: MlxArray,
    pub(crate) norm_weight: MlxArray,
    pub(crate) out_proj: WeightTensor,
}

pub(crate) enum MetalQwen35Attention {
    Full(MetalQwen35FullAttentionWeights),
    Linear(MetalLinearAttnWeights),
}

pub(crate) struct MetalQwen35DenseMlpWeights {
    pub(crate) inputs: MlpInputProjection,
    pub(crate) down_proj: WeightTensor,
    pub(crate) gate_proj: WeightTensor,
    pub(crate) up_proj: WeightTensor,
}

pub(crate) struct MetalQwen35MoeWeights {
    pub(crate) router: WeightTensor,
    pub(crate) switch_gate: StackedQuantized,
    pub(crate) switch_up: StackedQuantized,
    pub(crate) switch_down: StackedQuantized,
    pub(crate) shared_gate: WeightTensor,
    pub(crate) shared_up: WeightTensor,
    pub(crate) shared_down: WeightTensor,
    pub(crate) shared_expert_gate: WeightTensor,
    pub(crate) num_experts: i32,
    pub(crate) top_k: i32,
    pub(crate) norm_topk_prob: bool,
    pub(crate) router_bits: i32,
    pub(crate) router_group_size: i32,
    pub(crate) expert_bits: i32,
    pub(crate) expert_group_size: i32,
}

pub(crate) enum MlpKind {
    Dense(MetalQwen35DenseMlpWeights),
    Moe(MetalQwen35MoeWeights),
}

pub(crate) struct MetalQwen35BlockWeights {
    pub(crate) input_layernorm: MlxArray,
    pub(crate) attention: MetalQwen35Attention,
    pub(crate) post_attention_layernorm: MlxArray,
    pub(crate) mlp: MlpKind,
}

pub(crate) enum Qwen35Embedding {
    Dense(MlxArray),
}

pub(crate) struct Qwen35MetalWeights {
    pub(crate) embedding: Qwen35Embedding,
    pub(crate) layers: Vec<MetalQwen35BlockWeights>,
    pub(crate) norm: MlxArray,
    pub(crate) lm_head: WeightTensor,
    pub(crate) embed_quantized: Option<WeightTensor>,
    pub(crate) cpp_model: Option<CppQwen35Model>,
}

impl Qwen35MetalWeights {
    pub(crate) fn cpp_model(&self) -> Result<&CppQwen35Model> {
        self.cpp_model
            .as_ref()
            .context("Qwen3.5 C++ compiled model unavailable")
    }
}

pub(crate) fn load_qwen35_metal_weights(
    model_dir: &Path,
    config: &MetalModelConfig,
) -> Result<Qwen35MetalWeights> {
    let arch = &config.arch;
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
            "  Qwen3.5 safetensors use HF offset RMSNorm weights -- normalizing to direct form"
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
        "  {} Qwen3.5/Qwen3.6 layers ({} full attention, {} GDR, {} MoE)",
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
                build_qwen35_full_attention(
                    load_proj(&format!("{attn_prefix}.q_proj"))?,
                    load_proj(&format!("{attn_prefix}.k_proj"))?,
                    load_proj(&format!("{attn_prefix}.v_proj"))?,
                    load_proj(&format!("{attn_prefix}.o_proj"))?,
                    load_norm(&format!("{attn_prefix}.q_norm.weight"))?,
                    load_norm(&format!("{attn_prefix}.k_norm.weight"))?,
                )
            }
            MetalQwen35LayerType::LinearAttention => {
                let attn_prefix = format!("{layer_prefix}.linear_attn");
                build_qwen35_linear_attention(
                    load_proj(&format!("{attn_prefix}.in_proj_qkv"))?,
                    load_proj(&format!("{attn_prefix}.in_proj_z"))?,
                    load_proj(&format!("{attn_prefix}.in_proj_b"))?,
                    load_proj(&format!("{attn_prefix}.in_proj_a"))?,
                    load_conv1d_weight(
                        &get(&format!("{attn_prefix}.conv1d.weight"))?,
                        &arch.linear,
                    )?,
                    get(&format!("{attn_prefix}.dt_bias"))?,
                    as_dtype(&get(&format!("{attn_prefix}.A_log"))?, Dtype::Float32),
                    get(&format!("{attn_prefix}.norm.weight"))?,
                    load_proj(&format!("{attn_prefix}.out_proj"))?,
                )?
            }
        };

        let mlp = if let Some(moe_cfg) = arch.moe.as_ref().filter(|moe| moe.is_moe_layer(i)) {
            MlpKind::Moe(load_qwen35_moe_layer_weights(
                &tensors,
                &layer_prefix,
                moe_cfg,
            )?)
        } else {
            build_qwen35_dense_mlp(
                load_proj(&format!("{layer_prefix}.mlp.gate_proj"))?,
                load_proj(&format!("{layer_prefix}.mlp.up_proj"))?,
                load_proj(&format!("{layer_prefix}.mlp.down_proj"))?,
            )?
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
        embedding: Qwen35Embedding::Dense(embed_tokens),
        layers,
        norm,
        lm_head,
        embed_quantized,
        cpp_model: None,
    };
    weights.cpp_model = CppQwen35Model::build(&weights, config, arch);
    anyhow::ensure!(
        weights.cpp_model.is_some(),
        "R3a requires the Qwen3.5 C++ compiled model; Rust fallback is intentionally not ported"
    );
    Ok(weights)
}

fn build_qwen35_full_attention(
    q_proj: WeightTensor,
    k_proj: WeightTensor,
    v_proj: WeightTensor,
    o_proj: WeightTensor,
    q_norm: MlxArray,
    k_norm: MlxArray,
) -> MetalQwen35Attention {
    MetalQwen35Attention::Full(MetalQwen35FullAttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
    })
}

fn build_qwen35_linear_attention(
    qkv_proj: WeightTensor,
    z_proj: WeightTensor,
    beta_proj: WeightTensor,
    alpha_proj: WeightTensor,
    conv1d_weight: MlxArray,
    dt_bias: MlxArray,
    a_log: MlxArray,
    norm_weight: MlxArray,
    out_proj: WeightTensor,
) -> Result<MetalQwen35Attention> {
    let qkv_dim = qkv_proj.output_dim()?;
    let z_dim = z_proj.output_dim()?;
    let beta_dim = beta_proj.output_dim()?;
    let in_proj_qkvz = match merge_quantized_projection_rows(&[&qkv_proj, &z_proj])? {
        Some(merged) => Some(merged),
        None => concat_weight_rows(&qkv_proj, &z_proj).ok(),
    };
    let in_proj_ba = match merge_quantized_projection_rows(&[&beta_proj, &alpha_proj])? {
        Some(merged) => Some(merged),
        None => concat_weight_rows(&beta_proj, &alpha_proj).ok(),
    };
    Ok(MetalQwen35Attention::Linear(MetalLinearAttnWeights {
        in_proj_qkvz,
        in_proj_ba,
        in_proj_qkv: qkv_proj,
        in_proj_z: z_proj,
        in_proj_b: beta_proj,
        in_proj_a: alpha_proj,
        qkvz_split: (qkv_dim, z_dim),
        ba_num_heads: beta_dim,
        conv1d_weight,
        dt_bias,
        a_log,
        norm_weight,
        out_proj,
    }))
}

fn build_qwen35_dense_mlp(
    gate_proj: WeightTensor,
    up_proj: WeightTensor,
    down_proj: WeightTensor,
) -> Result<MlpKind> {
    let gate_dim = gate_proj.output_dim()?;
    let gate_up_proj = match merge_quantized_projection_rows(&[&gate_proj, &up_proj])? {
        Some(gate_up_proj) => gate_up_proj,
        None => concat_weight_rows(&gate_proj, &up_proj)?,
    };
    let inputs = MlpInputProjection::MergedQuantized {
        gate_up_proj,
        gate_dim,
    };
    Ok(MlpKind::Dense(MetalQwen35DenseMlpWeights {
        inputs,
        down_proj,
        gate_proj,
        up_proj,
    }))
}

fn load_qwen35_moe_layer_weights(
    tensors: &TensorMap,
    layer_prefix: &str,
    moe_cfg: &crate::config::MetalQwen35MoeConfig,
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
        )?,
        switch_up: load_stacked_quantized(tensors, &format!("{mlp_prefix}.switch_mlp.up_proj"))?,
        switch_down: load_stacked_quantized(
            tensors,
            &format!("{mlp_prefix}.switch_mlp.down_proj"),
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
    tensors: &TensorMap,
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

fn load_conv1d_weight(weight: &MlxArray, linear_cfg: &MetalGdrConfig) -> Result<MlxArray> {
    let channels = linear_cfg.qkv_dim() as i32;
    let kernel = linear_cfg.conv_kernel as i32;
    match weight.shape() {
        [ch, ks, 1] if *ch == channels && *ks == kernel => Ok(weight.clone()),
        [ch, 1, ks] if *ch == channels && *ks == kernel => Ok(transpose_axes(weight, &[0, 2, 1])),
        [ch, ks] if *ch == channels && *ks == kernel => Ok(reshape(weight, &[channels, kernel, 1])),
        shape => anyhow::bail!(
            "unsupported conv1d weight shape {:?}, expected [{channels}, {kernel}, 1]",
            shape
        ),
    }
}

fn qwen35_norm_needs_offset_correction(weight: &MlxArray) -> bool {
    let weight_f32 = as_dtype(weight, Dtype::Float32);
    mlx::eval(&[&weight_f32]);
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

fn use_qwen35_cpp_separate_proj() -> bool {
    std::env::var("AGENT_INFER_QWEN35_CPP_SEPARATE").map_or(true, |value| value != "0")
}

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

fn register_qwen35_moe_layer(model: *mut std::ffi::c_void, moe: &MetalQwen35MoeWeights) -> bool {
    let Some(router) = extract_qw(&moe.router) else {
        log::warn!("C++ Qwen3.5 MoE registration requires quantized router weights");
        return false;
    };
    let Some(shared_gate) = extract_qw(&moe.shared_gate) else {
        log::warn!("C++ Qwen3.5 MoE registration requires quantized shared gate weights");
        return false;
    };
    let Some(shared_up) = extract_qw(&moe.shared_up) else {
        log::warn!("C++ Qwen3.5 MoE registration requires quantized shared up weights");
        return false;
    };
    let Some(shared_down) = extract_qw(&moe.shared_down) else {
        log::warn!("C++ Qwen3.5 MoE registration requires quantized shared down weights");
        return false;
    };
    let Some(shared_expert_gate) = extract_qw(&moe.shared_expert_gate) else {
        log::warn!("C++ Qwen3.5 MoE registration requires quantized shared expert gate weights");
        return false;
    };

    unsafe {
        mlx_sys::qwen35_compiled_set_last_moe_mlp(
            model,
            router.0,
            router.1,
            router.2,
            moe.router_group_size,
            moe.router_bits,
            moe.switch_gate.weight.as_raw(),
            moe.switch_gate.scales.as_raw(),
            moe.switch_gate.biases.as_raw(),
            moe.switch_up.weight.as_raw(),
            moe.switch_up.scales.as_raw(),
            moe.switch_up.biases.as_raw(),
            moe.switch_down.weight.as_raw(),
            moe.switch_down.scales.as_raw(),
            moe.switch_down.biases.as_raw(),
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

    if let Err(err) = mlx::check_mlx_error() {
        log::warn!("C++ Qwen3.5 MoE registration failed: {err}");
        return false;
    }
    true
}

/// Owned C++ Qwen35 compiled model handle.
pub(crate) struct CppQwen35Model {
    raw: *mut std::ffi::c_void,
}

impl Drop for CppQwen35Model {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::qwen35_compiled_free(self.raw);
        }
    }
}

impl CppQwen35Model {
    pub(crate) fn build(
        weights: &Qwen35MetalWeights,
        config: &MetalModelConfig,
        arch: &MetalQwen35ArchConfig,
    ) -> Option<Self> {
        let model = unsafe { mlx_sys::qwen35_compiled_new() };
        if model.is_null() {
            return None;
        }

        let add_weight = |weight: &WeightTensor| -> Option<i32> {
            let id = unsafe {
                match weight {
                    WeightTensor::Dense(w) => {
                        mlx_sys::qwen35_compiled_add_dense_weight(model, w.as_raw())
                    }
                    WeightTensor::Quantized {
                        w,
                        scales,
                        biases,
                        group_size,
                        bits,
                    } => mlx_sys::qwen35_compiled_add_affine_weight(
                        model,
                        w.as_raw(),
                        scales.as_raw(),
                        biases.as_raw(),
                        *group_size,
                        *bits,
                    ),
                }
            };
            if id < 0 {
                let err = mlx::check_mlx_error()
                    .err()
                    .map_or_else(|| "unknown MLX error".to_string(), |err| err.to_string());
                log::warn!("C++ Qwen3.5 weight registration failed: {err}");
                None
            } else {
                Some(id)
            }
        };

        macro_rules! add_or_free {
            ($weight:expr) => {
                match add_weight($weight) {
                    Some(id) => id,
                    None => {
                        unsafe { mlx_sys::qwen35_compiled_free(model) };
                        return None;
                    }
                }
            };
        }

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
            mlx_sys::qwen35_compiled_set_qk_gate(model, 1);
        }

        let lm_head_id = add_or_free!(&weights.lm_head);
        match &weights.embedding {
            Qwen35Embedding::Dense(embed_tokens) => unsafe {
                mlx_sys::qwen35_compiled_set_embed_v2(
                    model,
                    embed_tokens.as_raw(),
                    weights.norm.as_raw(),
                    lm_head_id,
                );
            },
        }

        if matches!(weights.lm_head, WeightTensor::Dense(_)) {
            if let Some(embed_quantized) = &weights.embed_quantized {
                let embed_id = add_or_free!(embed_quantized);
                unsafe {
                    mlx_sys::qwen35_compiled_set_embed_as_linear_v2(model, embed_id);
                }
            }
        }

        for layer in &weights.layers {
            let input_ln = layer.input_layernorm.as_raw();
            let post_ln = layer.post_attention_layernorm.as_raw();
            let dense = match &layer.mlp {
                MlpKind::Dense(dense) => Some(dense),
                MlpKind::Moe(_) => None,
            };
            let (gate_up_id, gate_dim, down_id) = if let Some(dense) = dense {
                let MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                } = &dense.inputs;
                (
                    add_or_free!(gate_up_proj),
                    *gate_dim,
                    add_or_free!(&dense.down_proj),
                )
            } else {
                (-1, 0, -1)
            };

            match &layer.attention {
                MetalQwen35Attention::Full(attn) => {
                    let q_id = add_or_free!(&attn.q_proj);
                    let k_id = add_or_free!(&attn.k_proj);
                    let v_id = add_or_free!(&attn.v_proj);
                    let o_id = add_or_free!(&attn.o_proj);
                    unsafe {
                        mlx_sys::qwen35_compiled_push_full_attn_v2(
                            model,
                            input_ln,
                            post_ln,
                            q_id,
                            k_id,
                            v_id,
                            o_id,
                            attn.q_norm.as_raw(),
                            attn.k_norm.as_raw(),
                            gate_up_id,
                            gate_dim,
                            down_id,
                        );
                    }
                }
                MetalQwen35Attention::Linear(attn) => {
                    let qkvz_id = match &attn.in_proj_qkvz {
                        Some(weight) => add_or_free!(weight),
                        None => -1,
                    };
                    let ba_id = match &attn.in_proj_ba {
                        Some(weight) => add_or_free!(weight),
                        None => -1,
                    };
                    let out_id = add_or_free!(&attn.out_proj);
                    unsafe {
                        mlx_sys::qwen35_compiled_push_gdr_v2(
                            model,
                            input_ln,
                            post_ln,
                            qkvz_id,
                            attn.qkvz_split.0,
                            attn.qkvz_split.1,
                            ba_id,
                            attn.ba_num_heads,
                            attn.conv1d_weight.as_raw(),
                            arch.linear.conv_kernel as i32,
                            attn.a_log.as_raw(),
                            attn.dt_bias.as_raw(),
                            attn.norm_weight.as_raw(),
                            arch.linear.rms_norm_eps,
                            out_id,
                            arch.linear.num_key_heads as i32,
                            arch.linear.key_dim as i32,
                            arch.linear.num_value_heads as i32,
                            arch.linear.value_dim as i32,
                            gate_up_id,
                            gate_dim,
                            down_id,
                        );
                    }

                    let need_separate_proj =
                        use_qwen35_cpp_separate_proj() || qkvz_id < 0 || ba_id < 0;
                    if need_separate_proj {
                        let qkv_id = add_or_free!(&attn.in_proj_qkv);
                        let z_id = add_or_free!(&attn.in_proj_z);
                        let b_id = add_or_free!(&attn.in_proj_b);
                        let a_id = add_or_free!(&attn.in_proj_a);
                        let (gate_id, up_id) = if let Some(dense) = dense {
                            (add_or_free!(&dense.gate_proj), add_or_free!(&dense.up_proj))
                        } else {
                            (-1, -1)
                        };
                        unsafe {
                            mlx_sys::qwen35_compiled_set_separate_proj_v2(
                                model, qkv_id, z_id, b_id, a_id, gate_id, up_id,
                            );
                        }
                    } else if let Some(dense) = dense {
                        let gate_id = add_or_free!(&dense.gate_proj);
                        let up_id = add_or_free!(&dense.up_proj);
                        unsafe {
                            mlx_sys::qwen35_compiled_set_separate_mlp_v2(model, gate_id, up_id);
                        }
                    }
                }
            }

            if let MlpKind::Moe(moe) = &layer.mlp
                && !register_qwen35_moe_layer(model, moe)
            {
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            }
        }

        let rc = unsafe { mlx_sys::qwen35_compiled_finalize(model) };
        if rc != 0 {
            log::warn!("C++ Qwen3.5 model finalize failed");
            unsafe { mlx_sys::qwen35_compiled_free(model) };
            return None;
        }
        log::info!(
            "  C++ Qwen3.5 forward model ready ({} layers)",
            weights.layers.len()
        );
        Some(Self { raw: model })
    }

    pub(crate) fn begin_session(
        &self,
        kv_caches: &[MlxArray],
        gdr_states: &[MlxArray],
    ) -> Result<()> {
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let mut gdr_ptrs: Vec<*mut mlx_sys::mlx_array> =
            gdr_states.iter().map(MlxArray::as_raw).collect();
        let rc = unsafe {
            mlx_sys::qwen35_session_begin(
                self.raw,
                kv_ptrs.as_mut_ptr(),
                kv_ptrs.len() as i32,
                gdr_ptrs.as_mut_ptr(),
                gdr_ptrs.len() as i32,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        Ok(())
    }

    pub(crate) fn end_session(
        &self,
        n_kv: usize,
        n_gdr: usize,
    ) -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_kv];
        let mut out_gdr: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); n_gdr];
        let rc = unsafe {
            mlx_sys::qwen35_session_end(
                self.raw,
                out_kv.as_mut_ptr(),
                n_kv as i32,
                out_gdr.as_mut_ptr(),
                n_gdr as i32,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        let kv = out_kv
            .into_iter()
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        let gdr = out_gdr
            .into_iter()
            .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
            .collect();
        Ok((kv, gdr))
    }

    pub(crate) fn prefill_session(
        &self,
        tokens: &MlxArray,
        prompt_len: i32,
        cache_pos: i32,
    ) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc = unsafe {
            mlx_sys::qwen35_compiled_prefill_session(
                self.raw,
                tokens.as_raw(),
                prompt_len,
                cache_pos,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(crate) fn step_session(&self, token: &MlxArray, cache_pos: i32) -> Result<MlxArray> {
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_session(
                self.raw,
                token.as_raw(),
                cache_pos,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(crate) fn step_session_paged_bf16(
        &self,
        token: &MlxArray,
        cache_pos: i32,
        k_full_per_layer: &[MlxArray],
        v_full_per_layer: &[MlxArray],
    ) -> Result<MlxArray> {
        anyhow::ensure!(
            k_full_per_layer.len() == v_full_per_layer.len(),
            "paged session step requires matching K/V layer counts"
        );
        let mut k_ptrs: Vec<*mut mlx_sys::mlx_array> =
            k_full_per_layer.iter().map(MlxArray::as_raw).collect();
        let mut v_ptrs: Vec<*mut mlx_sys::mlx_array> =
            v_full_per_layer.iter().map(MlxArray::as_raw).collect();
        let mut empty_int8_k: Vec<*mut mlx_sys::mlx_array> = Vec::new();
        let mut empty_int8_v: Vec<*mut mlx_sys::mlx_array> = Vec::new();
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_session_paged(
                self.raw,
                token.as_raw(),
                cache_pos,
                k_ptrs.as_mut_ptr(),
                v_ptrs.as_mut_ptr(),
                k_ptrs.len() as i32,
                empty_int8_k.as_mut_ptr(),
                empty_int8_v.as_mut_ptr(),
                0,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }

    pub(crate) fn step_session_paged_int8(
        &self,
        token: &MlxArray,
        cache_pos: i32,
        k_int8_full_per_layer: &[MlxArray],
        v_int8_full_per_layer: &[MlxArray],
    ) -> Result<MlxArray> {
        anyhow::ensure!(
            k_int8_full_per_layer.len() == v_int8_full_per_layer.len(),
            "paged INT8 session step requires matching K/V triple counts"
        );
        anyhow::ensure!(
            k_int8_full_per_layer.len().is_multiple_of(3),
            "paged INT8 session step requires q/scale/bias triples"
        );
        let mut empty_bf16_k: Vec<*mut mlx_sys::mlx_array> = Vec::new();
        let mut empty_bf16_v: Vec<*mut mlx_sys::mlx_array> = Vec::new();
        let mut k_ptrs: Vec<*mut mlx_sys::mlx_array> =
            k_int8_full_per_layer.iter().map(MlxArray::as_raw).collect();
        let mut v_ptrs: Vec<*mut mlx_sys::mlx_array> =
            v_int8_full_per_layer.iter().map(MlxArray::as_raw).collect();
        let mut out_logits: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc = unsafe {
            mlx_sys::qwen35_compiled_step_session_paged(
                self.raw,
                token.as_raw(),
                cache_pos,
                empty_bf16_k.as_mut_ptr(),
                empty_bf16_v.as_mut_ptr(),
                0,
                k_ptrs.as_mut_ptr(),
                v_ptrs.as_mut_ptr(),
                (k_ptrs.len() / 3) as i32,
                &raw mut out_logits,
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        Ok(unsafe { MlxArray::from_raw(out_logits) })
    }
}
