//! DiffusionGemma Metal loader and block-diffusion model wrapper.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use infer_plan::{
    DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerationConfig, DiffusionModelError,
};

use crate::config::{MetalDiffusionGemmaConfig, QuantConfig};
use crate::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
};
use crate::mlx::{self, MlxArray};
use crate::weights::WeightTensor;

pub struct LoadedMetalDiffusionGemma {
    pub model: MetalDiffusionGemmaModel,
    pub generation: DiffusionGenerationConfig,
}

pub struct MetalDiffusionGemmaModel {
    cpp: CppDiffusionGemmaModel,
}

impl MetalDiffusionGemmaModel {
    pub fn load(model_dir: &Path) -> Result<LoadedMetalDiffusionGemma> {
        let _guard = mlx_sys::mlx_guard();
        if let Some(limit) = crate::wired_limit::auto_wired_limit_bytes(model_dir) {
            let previous = mlx::set_wired_limit_bytes(limit as u64);
            log::info!(
                "DiffusionGemma Metal wired limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
        let parsed = crate::config::load_diffusion_gemma_config(model_dir)?;
        let tensors = load_tensor_map(model_dir)?;
        let quant = QuantRegistry::from_model_dir(model_dir)?;
        let cpp = CppDiffusionGemmaModel::build(&parsed, &tensors, &quant)?;
        Ok(LoadedMetalDiffusionGemma {
            model: Self { cpp },
            generation: parsed.generation,
        })
    }
}

impl DiffusionBlockModel for MetalDiffusionGemmaModel {
    fn begin_request(
        &mut self,
        config: &DiffusionGenerationConfig,
    ) -> Result<(), DiffusionModelError> {
        self.cpp
            .begin_request(config.seed)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), DiffusionModelError> {
        let tokens =
            i32_tokens(prompt_tokens).map_err(|err| DiffusionModelError::new(err.to_string()))?;
        self.cpp
            .prefill(&tokens)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }

    fn predict_canvas(
        &mut self,
        canvas: &[u32],
        valid_len: usize,
        step: usize,
        temperature: f32,
    ) -> Result<DiffusionCanvasPrediction, DiffusionModelError> {
        let canvas_i32 =
            i32_tokens(canvas).map_err(|err| DiffusionModelError::new(err.to_string()))?;
        self.cpp
            .predict_canvas(&canvas_i32, valid_len, step, temperature)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }

    fn commit(&mut self, tokens: &[u32]) -> Result<(), DiffusionModelError> {
        let tokens = i32_tokens(tokens).map_err(|err| DiffusionModelError::new(err.to_string()))?;
        self.cpp
            .commit(&tokens)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }
}

fn i32_tokens(tokens: &[u32]) -> Result<Vec<i32>> {
    tokens
        .iter()
        .map(|&token| {
            i32::try_from(token).with_context(|| format!("token id {token} does not fit in i32"))
        })
        .collect()
}

#[derive(Debug, Clone)]
struct QuantRegistry {
    default: Option<QuantConfig>,
    overrides: HashMap<String, QuantConfig>,
}

impl QuantRegistry {
    fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json"))
            .with_context(|| format!("read {}", model_dir.join("config.json").display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw).context("parse config.json")?;
        Ok(Self::from_config_value(&value))
    }

    fn from_config_value(value: &serde_json::Value) -> Self {
        let Some(obj) = value
            .get("quantization")
            .or_else(|| value.get("quantization_config"))
            .and_then(serde_json::Value::as_object)
        else {
            return Self {
                default: None,
                overrides: HashMap::new(),
            };
        };

        let read_qc = |value: &serde_json::Value| -> Option<QuantConfig> {
            let object = value.as_object()?;
            Some(QuantConfig {
                group_size: object
                    .get("group_size")
                    .and_then(serde_json::Value::as_i64)
                    .map_or(64, |n| n as i32),
                bits: object
                    .get("bits")
                    .and_then(serde_json::Value::as_i64)
                    .map_or(4, |n| n as i32),
            })
        };

        let default = Some(QuantConfig {
            group_size: obj
                .get("group_size")
                .and_then(serde_json::Value::as_i64)
                .map_or(64, |n| n as i32),
            bits: obj
                .get("bits")
                .and_then(serde_json::Value::as_i64)
                .map_or(4, |n| n as i32),
        });
        let overrides = obj
            .iter()
            .filter_map(|(key, value)| read_qc(value).map(|qc| (key.clone(), qc)))
            .collect();
        Self { default, overrides }
    }

    fn for_base(&self, base: &str) -> Option<QuantConfig> {
        self.overrides.get(base).copied().or(self.default)
    }
}

struct CppDiffusionGemmaModel {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for CppDiffusionGemmaModel {}

impl Drop for CppDiffusionGemmaModel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::diffusion_gemma_free(self.raw);
        }
    }
}

impl CppDiffusionGemmaModel {
    fn build(
        parsed: &MetalDiffusionGemmaConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
    ) -> Result<Self> {
        let raw = unsafe { mlx_sys::diffusion_gemma_new() };
        anyhow::ensure!(!raw.is_null(), "diffusion_gemma_new returned null");
        let mut builder = CppDiffusionGemmaBuilder { raw };
        if let Err(err) = builder.populate(parsed, tensors, quant) {
            unsafe {
                mlx_sys::diffusion_gemma_free(raw);
            }
            return Err(err);
        }
        Ok(Self { raw })
    }

    fn begin_request(&mut self, seed: u64) -> Result<()> {
        self.check_rc(unsafe { mlx_sys::diffusion_gemma_begin_request(self.raw, seed) })
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<()> {
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_prefill(self.raw, tokens.as_ptr(), tokens.len() as i32)
        })
    }

    fn commit(&mut self, tokens: &[i32]) -> Result<()> {
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_commit(self.raw, tokens.as_ptr(), tokens.len() as i32)
        })
    }

    fn predict_canvas(
        &mut self,
        canvas: &[i32],
        valid_len: usize,
        step: usize,
        temperature: f32,
    ) -> Result<DiffusionCanvasPrediction> {
        anyhow::ensure!(
            valid_len <= canvas.len(),
            "valid_len {} exceeds canvas length {}",
            valid_len,
            canvas.len()
        );
        let mut sampled_tokens = vec![0u32; canvas.len()];
        let mut argmax_tokens = vec![0u32; canvas.len()];
        let mut entropies = vec![0.0; canvas.len()];
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_predict_canvas(
                self.raw,
                canvas.as_ptr(),
                canvas.len() as i32,
                valid_len as i32,
                step as i32,
                temperature,
                sampled_tokens.as_mut_ptr(),
                argmax_tokens.as_mut_ptr(),
                entropies.as_mut_ptr(),
            )
        })?;
        Ok(DiffusionCanvasPrediction {
            sampled_tokens,
            argmax_tokens,
            entropies,
        })
    }

    fn check_rc(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX FFI failed")))
        }
    }
}

struct CppDiffusionGemmaBuilder {
    raw: *mut std::ffi::c_void,
}

impl CppDiffusionGemmaBuilder {
    fn populate(
        &mut self,
        parsed: &MetalDiffusionGemmaConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
    ) -> Result<()> {
        let text = &parsed.config.text_config;
        unsafe {
            mlx_sys::diffusion_gemma_set_config(
                self.raw,
                text.hidden_size as i32,
                text.vocab_size as i32,
                text.rms_norm_eps,
                text.final_logit_softcapping.unwrap_or(0.0),
            );
        }
        mlx::check_mlx_error()?;

        let prefix = "model.decoder";
        let embed = load_embed_tokens_from_tensors(
            tensors,
            &format!("{prefix}.embed_tokens"),
            quant.for_base(&format!("{prefix}.embed_tokens")),
        )?;
        let embed_id = self.add_dense_array(&embed)?;
        let final_norm = tensor_get(tensors, &format!("{prefix}.norm.weight"))?;
        let final_norm_id = self.add_dense_array(&final_norm)?;
        unsafe {
            mlx_sys::diffusion_gemma_set_embed(self.raw, embed_id, final_norm_id);
        }
        mlx::check_mlx_error()?;

        for layer_idx in 0..text.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            self.push_layer(parsed, tensors, quant, &layer_prefix, layer_idx)?;
        }

        self.set_self_conditioning(tensors, quant, prefix)?;
        self.check_rc(unsafe { mlx_sys::diffusion_gemma_finalize(self.raw) })?;
        Ok(())
    }

    fn push_layer(
        &mut self,
        parsed: &MetalDiffusionGemmaConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        layer_prefix: &str,
        layer_idx: usize,
    ) -> Result<()> {
        let text = &parsed.config.text_config;
        let layer_type = text.layer_types[layer_idx];
        let is_full = text.is_global_layer(layer_idx);
        let head_dim = if is_full {
            text.global_attention_head_dim()
        } else {
            text.head_dim
        };
        let num_kv_heads = if is_full {
            text.global_kv_heads()
        } else {
            text.num_key_value_heads
        };
        let (rope_theta, rotary_dim) = rope_for_layer(text, layer_type, head_dim);
        let attn_prefix = format!("{layer_prefix}.self_attn");

        let input_ln_id =
            self.add_dense_name(tensors, &format!("{layer_prefix}.input_layernorm.weight"))?;
        let q_id = self.add_proj(tensors, quant, &format!("{attn_prefix}.q_proj"))?;
        let k_id = self.add_proj(tensors, quant, &format!("{attn_prefix}.k_proj"))?;
        let v_base = format!("{attn_prefix}.v_proj");
        let v_id = if tensors.contains_key(&format!("{v_base}.weight")) {
            self.add_proj(tensors, quant, &v_base)?
        } else {
            anyhow::ensure!(
                is_full,
                "missing v_proj for non-full DiffusionGemma layer {layer_idx}"
            );
            -1
        };
        let o_id = self.add_proj(tensors, quant, &format!("{attn_prefix}.o_proj"))?;
        let q_norm_id = self.add_dense_name(tensors, &format!("{attn_prefix}.q_norm.weight"))?;
        let k_norm_id = self.add_dense_name(tensors, &format!("{attn_prefix}.k_norm.weight"))?;
        let post_attn_ln_id = self.add_dense_name(
            tensors,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
        )?;
        let pre_ff_ln_id = self.add_dense_name(
            tensors,
            &format!("{layer_prefix}.pre_feedforward_layernorm.weight"),
        )?;
        let gate_id = self.add_proj(tensors, quant, &format!("{layer_prefix}.mlp.gate_proj"))?;
        let up_id = self.add_proj(tensors, quant, &format!("{layer_prefix}.mlp.up_proj"))?;
        let down_id = self.add_proj(tensors, quant, &format!("{layer_prefix}.mlp.down_proj"))?;
        let post_ff_ln_id = self.add_dense_name(
            tensors,
            &format!("{layer_prefix}.post_feedforward_layernorm.weight"),
        )?;
        let layer_scalar_id =
            self.add_dense_name(tensors, &format!("{layer_prefix}.layer_scalar"))?;

        let uses_moe = text.uses_moe_block();
        let (
            pre_ff2_ln_id,
            post_ff1_ln_id,
            post_ff2_ln_id,
            router_id,
            router_scale_id,
            per_expert_scale_id,
            expert_gate_up_id,
            expert_down_id,
            num_experts,
            top_k,
        ) = if uses_moe {
            (
                self.add_dense_name(
                    tensors,
                    &format!("{layer_prefix}.pre_feedforward_layernorm_2.weight"),
                )?,
                self.add_dense_name(
                    tensors,
                    &format!("{layer_prefix}.post_feedforward_layernorm_1.weight"),
                )?,
                self.add_dense_name(
                    tensors,
                    &format!("{layer_prefix}.post_feedforward_layernorm_2.weight"),
                )?,
                self.add_proj(tensors, quant, &format!("{layer_prefix}.router.proj"))?,
                self.add_dense_name(tensors, &format!("{layer_prefix}.router.scale"))?,
                self.add_dense_name(tensors, &format!("{layer_prefix}.router.per_expert_scale"))?,
                self.add_proj(
                    tensors,
                    quant,
                    &format!("{layer_prefix}.experts.gate_up_proj"),
                )?,
                self.add_proj(tensors, quant, &format!("{layer_prefix}.experts.down_proj"))?,
                text.num_experts.unwrap_or(0) as i32,
                text.moe_top_k().unwrap_or(0) as i32,
            )
        } else {
            (-1, -1, -1, -1, -1, -1, -1, -1, 0, 0)
        };

        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_push_layer(
                self.raw,
                is_full,
                text.num_attention_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                rope_theta,
                text.sliding_window as i32,
                input_ln_id,
                q_id,
                k_id,
                v_id,
                o_id,
                q_norm_id,
                k_norm_id,
                post_attn_ln_id,
                pre_ff_ln_id,
                gate_id,
                up_id,
                down_id,
                post_ff_ln_id,
                pre_ff2_ln_id,
                post_ff1_ln_id,
                post_ff2_ln_id,
                router_id,
                router_scale_id,
                per_expert_scale_id,
                expert_gate_up_id,
                expert_down_id,
                layer_scalar_id,
                num_experts,
                top_k,
            )
        })
    }

    fn set_self_conditioning(
        &mut self,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        prefix: &str,
    ) -> Result<()> {
        let sc_prefix = format!("{prefix}.self_conditioning");
        let pre_norm_id = self.add_dense_name(tensors, &format!("{sc_prefix}.pre_norm.weight"))?;
        let gate_id = self.add_proj(tensors, quant, &format!("{sc_prefix}.gate_proj"))?;
        let up_id = self.add_proj(tensors, quant, &format!("{sc_prefix}.up_proj"))?;
        let down_id = self.add_proj(tensors, quant, &format!("{sc_prefix}.down_proj"))?;
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_set_self_conditioning(
                self.raw,
                pre_norm_id,
                gate_id,
                up_id,
                down_id,
            )
        })
    }

    fn add_dense_name(&mut self, tensors: &TensorMap, name: &str) -> Result<i32> {
        let array = tensor_get(tensors, name)?;
        self.add_dense_array(&array)
    }

    fn add_dense_array(&mut self, array: &MlxArray) -> Result<i32> {
        let id = unsafe { mlx_sys::diffusion_gemma_add_dense_weight(self.raw, array.as_raw()) };
        self.check_weight_id(id)
    }

    fn add_proj(&mut self, tensors: &TensorMap, quant: &QuantRegistry, base: &str) -> Result<i32> {
        let weight = load_proj_from_tensors(tensors, base, quant.for_base(base))?;
        self.add_weight(&weight)
    }

    fn add_weight(&mut self, weight: &WeightTensor) -> Result<i32> {
        let id = unsafe {
            match weight {
                WeightTensor::Dense(w) => {
                    mlx_sys::diffusion_gemma_add_dense_weight(self.raw, w.as_raw())
                }
                WeightTensor::Quantized {
                    w,
                    scales,
                    biases,
                    group_size,
                    bits,
                } => mlx_sys::diffusion_gemma_add_affine_weight(
                    self.raw,
                    w.as_raw(),
                    scales.as_raw(),
                    biases.as_raw(),
                    *group_size,
                    *bits,
                ),
            }
        };
        self.check_weight_id(id)
    }

    fn check_weight_id(&self, id: i32) -> Result<i32> {
        if id >= 0 {
            Ok(id)
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX weight registration failed")))
        }
    }

    fn check_rc(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX DiffusionGemma registration failed")))
        }
    }
}

fn rope_for_layer(
    text: &gemma_spec::Gemma4TextConfig,
    layer_type: gemma_spec::Gemma4LayerType,
    head_dim: usize,
) -> (f32, usize) {
    let params = text
        .rope_parameters
        .as_ref()
        .and_then(|rope| rope.for_layer_type(layer_type));
    let theta = params
        .and_then(|params| params.rope_theta)
        .unwrap_or(match layer_type {
            gemma_spec::Gemma4LayerType::SlidingAttention => 10_000.0,
            gemma_spec::Gemma4LayerType::FullAttention => 1_000_000.0,
        });
    let partial = params
        .and_then(|params| params.partial_rotary_factor)
        .unwrap_or(1.0);
    let rotary_dim = ((head_dim as f32) * partial).round() as usize;
    (theta, rotary_dim.max(1).min(head_dim))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn quant_registry_uses_per_weight_overrides() {
        let registry = QuantRegistry::from_config_value(&json!({
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "model.decoder.embed_tokens": {"group_size": 64, "bits": 8},
                "model.decoder.layers.0.self_attn.q_proj": {"group_size": 32, "bits": 8}
            }
        }));
        assert_eq!(
            registry
                .for_base("model.decoder.layers.0.experts.down_proj")
                .unwrap()
                .bits,
            4
        );
        let q = registry
            .for_base("model.decoder.layers.0.self_attn.q_proj")
            .unwrap();
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 32);
    }

    #[test]
    fn rope_uses_full_attention_partial_rotary_factor() {
        let cfg = gemma_spec::Gemma4TextConfig::from_json_str(
            r#"{
                "vocab_size": 10,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 128,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "sliding_window": 32,
                "layer_types": ["full_attention"],
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25
                    }
                }
            }"#,
        )
        .unwrap();
        let (theta, rotary_dim) =
            rope_for_layer(&cfg, gemma_spec::Gemma4LayerType::FullAttention, 512);
        assert_eq!(theta, 1_000_000.0);
        assert_eq!(rotary_dim, 128);
    }
}
