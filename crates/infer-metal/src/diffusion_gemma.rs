//! DiffusionGemma Metal loader and block-diffusion model wrapper.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use infer_plan::{
    DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerateOutput,
    DiffusionGenerateStats, DiffusionGenerationConfig, DiffusionModelError, FinishReason,
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
        Self::load_with_resource_plan(model_dir, None)
    }

    pub fn load_with_resource_plan(
        model_dir: &Path,
        resource_plan: Option<crate::resource::MetalResourcePlan>,
    ) -> Result<LoadedMetalDiffusionGemma> {
        let _guard = mlx_sys::mlx_guard();
        crate::resource::apply_startup_mlx_limits(
            model_dir,
            resource_plan.as_ref(),
            Some("DiffusionGemma"),
            true,
        );
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
    fn generate(
        &mut self,
        prompt_tokens: &[u32],
        config: &DiffusionGenerationConfig,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        self.generate_with_cancel(prompt_tokens, config, None)
    }

    fn generate_with_cancel(
        &mut self,
        prompt_tokens: &[u32],
        config: &DiffusionGenerationConfig,
        cancel: Option<&AtomicBool>,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        let tokens =
            i32_tokens(prompt_tokens).map_err(|err| DiffusionModelError::new(err.to_string()))?;
        self.cpp
            .generate(&tokens, config, cancel)
            .map(Some)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }

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

unsafe extern "C" fn diffusion_cancelled(ctx: *const std::ffi::c_void) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    // SAFETY: ctx is the &AtomicBool cancel flag registered with the bridge; it outlives the in-flight call.
    let flag = unsafe { &*(ctx as *const AtomicBool) };
    i32::from(flag.load(Ordering::Acquire))
}

#[derive(Debug, Clone)]
pub(crate) struct QuantRegistry {
    pub(crate) default: Option<QuantConfig>,
    pub(crate) overrides: HashMap<String, QuantConfig>,
}

impl QuantRegistry {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json"))
            .with_context(|| format!("read {}", model_dir.join("config.json").display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw).context("parse config.json")?;
        Ok(Self::from_config_value(&value))
    }

    pub(crate) fn from_config_value(value: &serde_json::Value) -> Self {
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

        let read_qc = |object: &serde_json::Map<String, serde_json::Value>| -> QuantConfig {
            QuantConfig {
                group_size: object
                    .get("group_size")
                    .and_then(serde_json::Value::as_i64)
                    .map_or(64, |n| n as i32),
                bits: object
                    .get("bits")
                    .and_then(serde_json::Value::as_i64)
                    .map_or(4, |n| n as i32),
                // Diffusion Gemma checkpoints ship affine 4-bit.
                mode: crate::config::QuantMode::Affine,
                per_weight: std::sync::Arc::new(HashMap::new()),
            }
        };

        let default = Some(read_qc(obj));
        let overrides = obj
            .iter()
            .filter_map(|(key, value)| value.as_object().map(|o| (key.clone(), read_qc(o))))
            .collect();
        Self { default, overrides }
    }

    pub(crate) fn for_base(&self, base: &str) -> Option<QuantConfig> {
        self.overrides
            .get(base)
            .cloned()
            .or_else(|| self.default.clone())
    }
}

struct CppDiffusionGemmaModel {
    raw: *mut std::ffi::c_void,
}

// SAFETY: the wrapper solely owns its C++ model handle and all bridge/MLX access is serialized, so it may cross threads.
unsafe impl Send for CppDiffusionGemmaModel {}

impl Drop for CppDiffusionGemmaModel {
    fn drop(&mut self) {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let raw = unsafe { mlx_sys::diffusion_gemma_new() };
        anyhow::ensure!(!raw.is_null(), "diffusion_gemma_new returned null");
        let mut builder = CppDiffusionGemmaBuilder { raw };
        if let Err(err) = builder.populate(parsed, tensors, quant) {
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe {
                mlx_sys::diffusion_gemma_free(raw);
            }
            return Err(err);
        }
        Ok(Self { raw })
    }

    fn begin_request(&mut self, seed: u64) -> Result<()> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe { mlx_sys::diffusion_gemma_begin_request(self.raw, seed) })
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<()> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_prefill(self.raw, tokens.as_ptr(), tokens.len() as i32)
        })
    }

    fn commit(&mut self, tokens: &[i32]) -> Result<()> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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

    fn generate(
        &mut self,
        prompt: &[i32],
        config: &DiffusionGenerationConfig,
        cancel: Option<&AtomicBool>,
    ) -> Result<DiffusionGenerateOutput> {
        let max_new_tokens =
            i32::try_from(config.max_new_tokens).context("max_new_tokens does not fit in i32")?;
        let canvas_len =
            i32::try_from(config.canvas_length).context("canvas_length does not fit in i32")?;
        let max_steps = i32::try_from(config.max_denoising_steps)
            .context("max_denoising_steps does not fit in i32")?;
        let stability_threshold = i32::try_from(config.stability_threshold)
            .context("stability_threshold does not fit in i32")?;
        let mut tokens = vec![0u32; config.max_new_tokens];
        let mut out_len = 0i32;
        let mut out_finish = 0i32;
        let mut out_blocks = 0i32;
        let mut out_steps = 0i32;
        let mut out_forced = 0i32;
        let mut out_adaptive = 0i32;
        let (cancel_fn, cancel_ctx) = if let Some(flag) = cancel {
            (
                Some(diffusion_cancelled as unsafe extern "C" fn(*const std::ffi::c_void) -> i32),
                flag as *const AtomicBool as *const std::ffi::c_void,
            )
        } else {
            (None, std::ptr::null())
        };
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_generate(
                self.raw,
                prompt.as_ptr(),
                prompt.len() as i32,
                max_new_tokens,
                canvas_len,
                max_steps,
                config.entropy_bound,
                config.confidence_threshold,
                config.t_min,
                config.t_max,
                stability_threshold,
                config.seed,
                config.stop_token_ids.as_ptr(),
                config.stop_token_ids.len() as i32,
                cancel_fn,
                cancel_ctx,
                tokens.as_mut_ptr(),
                &mut out_len,
                &mut out_finish,
                &mut out_blocks,
                &mut out_steps,
                &mut out_forced,
                &mut out_adaptive,
            )
        })?;
        let len = usize::try_from(out_len).context("negative DiffusionGemma output length")?;
        anyhow::ensure!(
            len <= tokens.len(),
            "DiffusionGemma output length {} exceeds buffer {}",
            len,
            tokens.len()
        );
        tokens.truncate(len);
        let finish = if out_finish == 1 {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        Ok(DiffusionGenerateOutput {
            generated_tokens: tokens,
            finish,
            stats: DiffusionGenerateStats {
                blocks: usize::try_from(out_blocks).unwrap_or_default(),
                denoise_steps: usize::try_from(out_steps).unwrap_or_default(),
                forced_commits: usize::try_from(out_forced).unwrap_or_default(),
                adaptive_commits: usize::try_from(out_adaptive).unwrap_or_default(),
            },
            trace: Vec::new(),
        })
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        let lm_head = load_proj_from_tensors(
            tensors,
            &format!("{prefix}.embed_tokens"),
            quant.for_base(&format!("{prefix}.embed_tokens")),
        )?;
        let lm_head_id = self.add_weight(&lm_head)?;
        let final_norm = tensor_get(tensors, &format!("{prefix}.norm.weight"))?;
        let final_norm_id = self.add_dense_array(&final_norm)?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::diffusion_gemma_set_embed(self.raw, embed_id, lm_head_id, final_norm_id);
        }
        mlx::check_mlx_error()?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_set_requires_self_conditioning(self.raw, true)
        })?;
        self.set_per_layer_embeddings_if_present(text, tensors, quant, prefix)?;

        for layer_idx in 0..text.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            self.push_layer(parsed, tensors, quant, &layer_prefix, layer_idx)?;
        }

        self.set_self_conditioning(tensors, quant, prefix)?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        let head_dim = text.attention_head_dim_for_layer(layer_idx);
        let num_kv_heads = text.kv_heads_for_layer(layer_idx);
        let (rope_theta, rotary_dim) = rope_for_layer(text, layer_type, head_dim);
        let attn_prefix = format!("{layer_prefix}.self_attn");
        let kv_shared_layer_index = text
            .kv_shared_source_layer(layer_idx)
            .map_or(-1, |idx| idx as i32);

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

        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_push_layer(
                self.raw,
                is_full,
                kv_shared_layer_index,
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
        })?;
        self.set_layer_ple_if_present(text, tensors, quant, layer_prefix, layer_idx)
    }

    fn set_per_layer_embeddings_if_present(
        &mut self,
        text: &gemma_spec::Gemma4TextConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        prefix: &str,
    ) -> Result<()> {
        if !text.has_per_layer_embeddings() {
            return Ok(());
        }
        let embed_prefix = format!("{prefix}.embed_tokens_per_layer");
        let embed =
            load_embed_tokens_from_tensors(tensors, &embed_prefix, quant.for_base(&embed_prefix))?;
        let embed_id = self.add_dense_array(&embed)?;
        let projection_prefix = format!("{prefix}.per_layer_model_projection");
        let projection_id = self.add_proj(tensors, quant, &projection_prefix)?;
        let norm_id = self.add_dense_name(
            tensors,
            &format!("{prefix}.per_layer_projection_norm.weight"),
        )?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_set_per_layer_embeddings(
                self.raw,
                embed_id,
                projection_id,
                norm_id,
                text.num_hidden_layers as i32,
                text.hidden_size_per_layer_input.unwrap_or(0) as i32,
                text.vocab_size_per_layer_input.unwrap_or(text.vocab_size) as i32,
            )
        })
    }

    fn set_layer_ple_if_present(
        &mut self,
        text: &gemma_spec::Gemma4TextConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        layer_prefix: &str,
        layer_idx: usize,
    ) -> Result<()> {
        if !text.has_per_layer_embeddings() {
            return Ok(());
        }
        let gate_id = self.add_proj(
            tensors,
            quant,
            &format!("{layer_prefix}.per_layer_input_gate"),
        )?;
        let projection_id = self.add_proj(
            tensors,
            quant,
            &format!("{layer_prefix}.per_layer_projection"),
        )?;
        let norm_id = self.add_dense_name(
            tensors,
            &format!("{layer_prefix}.post_per_layer_input_norm.weight"),
        )?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_set_layer_ple(
                self.raw,
                layer_idx as i32,
                gate_id,
                projection_id,
                norm_id,
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let id = unsafe { mlx_sys::diffusion_gemma_add_dense_weight(self.raw, array.as_raw()) };
        self.check_weight_id(id)
    }

    fn add_proj(&mut self, tensors: &TensorMap, quant: &QuantRegistry, base: &str) -> Result<i32> {
        let weight = load_proj_from_tensors(tensors, base, quant.for_base(base))?;
        self.add_weight(&weight)
    }

    fn add_weight(&mut self, weight: &WeightTensor) -> Result<i32> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
                    ..
                } => mlx_sys::diffusion_gemma_add_affine_weight(
                    self.raw,
                    w.as_raw(),
                    scales.as_raw(),
                    biases.as_ref().map_or(std::ptr::null_mut(), |b| b.as_raw()),
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

pub(crate) fn rope_for_layer(
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
