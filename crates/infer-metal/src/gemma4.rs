//! Gemma4 Metal loader and autoregressive model wrapper.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use infer_plan::{
    DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerateOutput,
    DiffusionGenerateStats, DiffusionGenerationConfig, DiffusionModelError, FinishReason,
    MultimodalImage,
};

use crate::config::{MetalGemma4Config, MetalGemma4VisionConfig};
use crate::diffusion_gemma::{QuantRegistry, rope_for_layer};
use crate::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
use crate::mlx::{self, MlxArray};
use crate::weights::WeightTensor;

pub struct LoadedMetalGemma4 {
    pub model: MetalGemma4Model,
    pub generation: DiffusionGenerationConfig,
    pub image_token_id: Option<u32>,
    pub vision_soft_tokens_per_image: Option<usize>,
}

pub struct MetalGemma4Model {
    cpp: CppGemma4Model,
}

impl MetalGemma4Model {
    pub fn load(model_dir: &Path) -> Result<LoadedMetalGemma4> {
        Self::load_with_resource_plan(model_dir, None)
    }

    pub fn load_with_resource_plan(
        model_dir: &Path,
        resource_plan: Option<crate::resource::MetalResourcePlan>,
    ) -> Result<LoadedMetalGemma4> {
        let _guard = mlx_sys::mlx_guard();
        crate::resource::apply_startup_mlx_limits(
            model_dir,
            resource_plan.as_ref(),
            Some("Gemma4"),
            true,
        );
        let parsed = crate::config::load_gemma4_config(model_dir)?;
        let tensors = load_tensor_map(model_dir)?;
        let quant = QuantRegistry::from_model_dir(model_dir)?;
        let cpp = CppGemma4Model::build(&parsed, &tensors, &quant)?;
        Ok(LoadedMetalGemma4 {
            model: Self { cpp },
            generation: parsed.generation,
            image_token_id: parsed.image_token_id,
            vision_soft_tokens_per_image: parsed.vision_soft_tokens_per_image,
        })
    }
}

impl DiffusionBlockModel for MetalGemma4Model {
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

    fn generate_multimodal_with_cancel(
        &mut self,
        prompt_tokens: &[u32],
        images: &[MultimodalImage],
        config: &DiffusionGenerationConfig,
        cancel: Option<&AtomicBool>,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        let tokens =
            i32_tokens(prompt_tokens).map_err(|err| DiffusionModelError::new(err.to_string()))?;
        self.cpp
            .generate_multimodal(&tokens, images, config, cancel)
            .map(Some)
            .map_err(|err| DiffusionModelError::new(err.to_string()))
    }

    fn prefill(&mut self, _prompt_tokens: &[u32]) -> Result<(), DiffusionModelError> {
        Err(DiffusionModelError::new(
            "Gemma4 uses the backend-owned causal generate fast path",
        ))
    }

    fn predict_canvas(
        &mut self,
        _canvas: &[u32],
        _valid_len: usize,
        _step: usize,
        _temperature: f32,
    ) -> Result<DiffusionCanvasPrediction, DiffusionModelError> {
        Err(DiffusionModelError::new(
            "Gemma4 does not implement the block-diffusion host loop",
        ))
    }

    fn commit(&mut self, _tokens: &[u32]) -> Result<(), DiffusionModelError> {
        Err(DiffusionModelError::new(
            "Gemma4 uses the backend-owned causal generate fast path",
        ))
    }

    fn multimodal_kind(&self) -> Option<infer_plan::MultimodalKind> {
        Some(infer_plan::MultimodalKind::Gemma4)
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

unsafe extern "C" fn gemma4_cancelled(ctx: *const std::ffi::c_void) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    // SAFETY: ctx is the &AtomicBool cancel flag registered with the bridge; it outlives the in-flight call.
    let flag = unsafe { &*(ctx as *const AtomicBool) };
    i32::from(flag.load(Ordering::Acquire))
}

struct CppGemma4Model {
    raw: *mut std::ffi::c_void,
}

// SAFETY: the wrapper solely owns its C++ model handle and all bridge/MLX access is serialized, so it may cross threads.
unsafe impl Send for CppGemma4Model {}

impl Drop for CppGemma4Model {
    fn drop(&mut self) {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::diffusion_gemma_free(self.raw);
        }
    }
}

impl CppGemma4Model {
    fn build(
        parsed: &MetalGemma4Config,
        tensors: &TensorMap,
        quant: &QuantRegistry,
    ) -> Result<Self> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let raw = unsafe { mlx_sys::diffusion_gemma_new() };
        anyhow::ensure!(!raw.is_null(), "diffusion_gemma_new returned null");
        let mut builder = CppGemma4Builder { raw };
        if let Err(err) = builder.populate(parsed, tensors, quant) {
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe {
                mlx_sys::diffusion_gemma_free(raw);
            }
            return Err(err);
        }
        Ok(Self { raw })
    }

    fn check_rc(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX Gemma4 FFI failed")))
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
        let mut tokens = vec![0u32; config.max_new_tokens];
        let mut out_len = 0i32;
        let mut out_finish = 0i32;
        let (cancel_fn, cancel_ctx) = if let Some(flag) = cancel {
            (
                Some(gemma4_cancelled as unsafe extern "C" fn(*const std::ffi::c_void) -> i32),
                flag as *const AtomicBool as *const std::ffi::c_void,
            )
        } else {
            (None, std::ptr::null())
        };
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_generate_causal(
                self.raw,
                prompt.as_ptr(),
                prompt.len() as i32,
                max_new_tokens,
                config.seed,
                config.stop_token_ids.as_ptr(),
                config.stop_token_ids.len() as i32,
                cancel_fn,
                cancel_ctx,
                tokens.as_mut_ptr(),
                &mut out_len,
                &mut out_finish,
            )
        })?;
        let len = usize::try_from(out_len).context("negative Gemma4 output length")?;
        anyhow::ensure!(
            len <= tokens.len(),
            "Gemma4 output length {} exceeds buffer {}",
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
                blocks: len,
                denoise_steps: len,
                ..DiffusionGenerateStats::default()
            },
            trace: Vec::new(),
        })
    }

    fn generate_multimodal(
        &mut self,
        prompt: &[i32],
        images: &[MultimodalImage],
        config: &DiffusionGenerationConfig,
        cancel: Option<&AtomicBool>,
    ) -> Result<DiffusionGenerateOutput> {
        anyhow::ensure!(
            images.len() == 1,
            "Gemma4 Metal VLM currently supports exactly one image per request"
        );
        let image = &images[0];
        anyhow::ensure!(
            image.channels == 3,
            "Gemma4 Metal VLM image input must be RGB"
        );
        anyhow::ensure!(
            image.pixels.len() == image.channels * image.height * image.width,
            "Gemma4 Metal VLM image buffer shape mismatch"
        );
        let max_new_tokens =
            i32::try_from(config.max_new_tokens).context("max_new_tokens does not fit in i32")?;
        let height = i32::try_from(image.height).context("image height does not fit in i32")?;
        let width = i32::try_from(image.width).context("image width does not fit in i32")?;
        let soft_tokens = i32::try_from(image.soft_token_count)
            .context("image soft token count does not fit in i32")?;
        let mut tokens = vec![0u32; config.max_new_tokens];
        let mut out_len = 0i32;
        let mut out_finish = 0i32;
        let (cancel_fn, cancel_ctx) = if let Some(flag) = cancel {
            (
                Some(gemma4_cancelled as unsafe extern "C" fn(*const std::ffi::c_void) -> i32),
                flag as *const AtomicBool as *const std::ffi::c_void,
            )
        } else {
            (None, std::ptr::null())
        };
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_generate_causal_image(
                self.raw,
                prompt.as_ptr(),
                prompt.len() as i32,
                image.pixels.as_ptr(),
                height,
                width,
                soft_tokens,
                max_new_tokens,
                config.seed,
                config.stop_token_ids.as_ptr(),
                config.stop_token_ids.len() as i32,
                cancel_fn,
                cancel_ctx,
                tokens.as_mut_ptr(),
                &mut out_len,
                &mut out_finish,
            )
        })?;
        let len = usize::try_from(out_len).context("negative Gemma4 output length")?;
        anyhow::ensure!(
            len <= tokens.len(),
            "Gemma4 output length {} exceeds buffer {}",
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
                blocks: len,
                denoise_steps: len,
                ..DiffusionGenerateStats::default()
            },
            trace: Vec::new(),
        })
    }
}

struct CppGemma4Builder {
    raw: *mut std::ffi::c_void,
}

impl CppGemma4Builder {
    fn populate(
        &mut self,
        parsed: &MetalGemma4Config,
        tensors: &TensorMap,
        quant: &QuantRegistry,
    ) -> Result<()> {
        let text = &parsed.text;
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

        let prefix = "language_model.model";
        let embed = load_embed_tokens_from_tensors(
            tensors,
            &format!("{prefix}.embed_tokens"),
            quant.for_base(&format!("{prefix}.embed_tokens")),
        )?;
        let embed_id = self.add_dense_array(&embed)?;
        let lm_head = tie_lm_head_from_embed_tokens(&embed);
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
            mlx_sys::diffusion_gemma_set_requires_self_conditioning(self.raw, false)
        })?;
        self.set_per_layer_embeddings_if_present(text, tensors, quant, prefix)?;

        for layer_idx in 0..text.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            self.push_layer(text, tensors, quant, &layer_prefix, layer_idx)?;
        }
        if let Some(vision) = &parsed.vision {
            let image_token_id = parsed
                .image_token_id
                .context("Gemma4 vision_config present but image_token_id is missing")?;
            self.set_vision(vision, image_token_id, tensors, quant)?;
        }

        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe { mlx_sys::diffusion_gemma_finalize(self.raw) })?;
        Ok(())
    }

    fn set_vision(
        &mut self,
        vision: &MetalGemma4VisionConfig,
        image_token_id: u32,
        tensors: &TensorMap,
        quant: &QuantRegistry,
    ) -> Result<()> {
        let patch_proj_id =
            self.add_proj(tensors, quant, "vision_tower.patch_embedder.input_proj")?;
        let position_id = self.add_dense_name(
            tensors,
            "vision_tower.patch_embedder.position_embedding_table",
        )?;
        let projection_id = self.add_proj(tensors, quant, "embed_vision.embedding_projection")?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_set_vision_config(
                self.raw,
                image_token_id as i32,
                vision.hidden_size as i32,
                vision.intermediate_size as i32,
                vision.num_hidden_layers as i32,
                vision.num_attention_heads as i32,
                vision.num_key_value_heads as i32,
                vision.head_dim as i32,
                vision.patch_size as i32,
                vision.pooling_kernel_size as i32,
                vision.default_output_length as i32,
                vision.position_embedding_size as i32,
                vision.rope_theta,
                vision.rms_norm_eps,
                vision.use_clipped_linears,
                patch_proj_id,
                position_id,
                projection_id,
            )
        })?;
        for layer_idx in 0..vision.num_hidden_layers {
            self.push_vision_layer(tensors, quant, layer_idx)?;
        }
        Ok(())
    }

    fn push_vision_layer(
        &mut self,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        layer_idx: usize,
    ) -> Result<()> {
        let prefix = format!("vision_tower.encoder.layers.{layer_idx}");
        let attn = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let input_ln_id =
            self.add_dense_name(tensors, &format!("{prefix}.input_layernorm.weight"))?;
        let q_id = self.add_proj(tensors, quant, &format!("{attn}.q_proj.linear"))?;
        let q_clip = self.add_clip(tensors, &format!("{attn}.q_proj"))?;
        let k_id = self.add_proj(tensors, quant, &format!("{attn}.k_proj.linear"))?;
        let k_clip = self.add_clip(tensors, &format!("{attn}.k_proj"))?;
        let v_id = self.add_proj(tensors, quant, &format!("{attn}.v_proj.linear"))?;
        let v_clip = self.add_clip(tensors, &format!("{attn}.v_proj"))?;
        let o_id = self.add_proj(tensors, quant, &format!("{attn}.o_proj.linear"))?;
        let o_clip = self.add_clip(tensors, &format!("{attn}.o_proj"))?;
        let q_norm_id = self.add_dense_name(tensors, &format!("{attn}.q_norm.weight"))?;
        let k_norm_id = self.add_dense_name(tensors, &format!("{attn}.k_norm.weight"))?;
        let post_attn_ln_id = self.add_dense_name(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let pre_ff_ln_id = self.add_dense_name(
            tensors,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
        )?;
        let gate_id = self.add_proj(tensors, quant, &format!("{mlp}.gate_proj.linear"))?;
        let gate_clip = self.add_clip(tensors, &format!("{mlp}.gate_proj"))?;
        let up_id = self.add_proj(tensors, quant, &format!("{mlp}.up_proj.linear"))?;
        let up_clip = self.add_clip(tensors, &format!("{mlp}.up_proj"))?;
        let down_id = self.add_proj(tensors, quant, &format!("{mlp}.down_proj.linear"))?;
        let down_clip = self.add_clip(tensors, &format!("{mlp}.down_proj"))?;
        let post_ff_ln_id = self.add_dense_name(
            tensors,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
        )?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::diffusion_gemma_push_vision_layer(
                self.raw,
                input_ln_id,
                q_id,
                q_clip[0],
                q_clip[1],
                q_clip[2],
                q_clip[3],
                k_id,
                k_clip[0],
                k_clip[1],
                k_clip[2],
                k_clip[3],
                v_id,
                v_clip[0],
                v_clip[1],
                v_clip[2],
                v_clip[3],
                o_id,
                o_clip[0],
                o_clip[1],
                o_clip[2],
                o_clip[3],
                q_norm_id,
                k_norm_id,
                post_attn_ln_id,
                pre_ff_ln_id,
                gate_id,
                gate_clip[0],
                gate_clip[1],
                gate_clip[2],
                gate_clip[3],
                up_id,
                up_clip[0],
                up_clip[1],
                up_clip[2],
                up_clip[3],
                down_id,
                down_clip[0],
                down_clip[1],
                down_clip[2],
                down_clip[3],
                post_ff_ln_id,
            )
        })
    }

    fn add_clip(&mut self, tensors: &TensorMap, base: &str) -> Result<[i32; 4]> {
        Ok([
            self.add_dense_name(tensors, &format!("{base}.input_min"))?,
            self.add_dense_name(tensors, &format!("{base}.input_max"))?,
            self.add_dense_name(tensors, &format!("{base}.output_min"))?,
            self.add_dense_name(tensors, &format!("{base}.output_max"))?,
        ])
    }

    fn push_layer(
        &mut self,
        text: &gemma_spec::Gemma4TextConfig,
        tensors: &TensorMap,
        quant: &QuantRegistry,
        layer_prefix: &str,
        layer_idx: usize,
    ) -> Result<()> {
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
        let v_id = self.add_proj(tensors, quant, &format!("{attn_prefix}.v_proj"))?;
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
                    MlxArray::as_raw_opt(biases.as_ref()),
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
                .unwrap_or_else(|| anyhow::anyhow!("MLX Gemma4 weight registration failed")))
        }
    }

    fn check_rc(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX Gemma4 registration failed")))
        }
    }
}
