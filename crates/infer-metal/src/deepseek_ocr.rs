//! Mirrors `gemma4.rs`: a thin Rust shell over the C++ MLX bridge
//! (`mlx_deepseek_ocr_model.cpp`). The model rides the shared autoregressive
//! `Engine` through `DiffusionBlockModel` + `BufferedDiffusionExecutor`
//! (`canvas_length=1`, `max_denoising_steps=1`). Decoder/projector weights are
//! MXFP8 (uint8 scales, no biases); SAM/CLIP weights are dense BF16.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use infer_plan::{
    DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerateOutput,
    DiffusionGenerateStats, DiffusionGenerationConfig, DiffusionModelError, FinishReason,
    MultimodalImage,
};

use crate::config::MetalDeepseekOcrConfig;
use crate::loader::{TensorMap, load_tensor_map, tensor_get};
use crate::mlx;

pub struct LoadedMetalDeepseekOcr {
    pub model: MetalDeepseekOcrModel,
    pub generation: DiffusionGenerationConfig,
    pub image_token_id: u32,
}

pub struct MetalDeepseekOcrModel {
    cpp: CppDeepseekOcrModel,
}

impl MetalDeepseekOcrModel {
    pub fn load(model_dir: &Path) -> Result<LoadedMetalDeepseekOcr> {
        Self::load_with_resource_plan(model_dir, None)
    }

    pub fn load_with_resource_plan(
        model_dir: &Path,
        resource_plan: Option<crate::resource::MetalResourcePlan>,
    ) -> Result<LoadedMetalDeepseekOcr> {
        let _guard = mlx_sys::mlx_guard();
        crate::resource::apply_startup_mlx_limits(
            model_dir,
            resource_plan.as_ref(),
            Some("DeepSeek-OCR"),
            false,
        );
        let parsed = crate::config::load_deepseek_ocr_config(model_dir)?;
        let tensors = load_tensor_map(model_dir)?;
        let cpp = CppDeepseekOcrModel::build(&parsed, &tensors)?;
        Ok(LoadedMetalDeepseekOcr {
            model: Self { cpp },
            generation: parsed.generation,
            image_token_id: parsed.image_token_id,
        })
    }
}

impl DiffusionBlockModel for MetalDeepseekOcrModel {
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
            "DeepSeek-OCR uses the backend-owned causal generate fast path",
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
            "DeepSeek-OCR does not implement the block-diffusion host loop",
        ))
    }

    fn commit(&mut self, _tokens: &[u32]) -> Result<(), DiffusionModelError> {
        Err(DiffusionModelError::new(
            "DeepSeek-OCR uses the backend-owned causal generate fast path",
        ))
    }

    fn multimodal_kind(&self) -> Option<infer_plan::MultimodalKind> {
        Some(infer_plan::MultimodalKind::DeepseekOcr)
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

unsafe extern "C" fn deepseek_ocr_cancelled(ctx: *const std::ffi::c_void) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    // SAFETY: ctx is the &AtomicBool cancel flag registered with the bridge; it outlives the in-flight call.
    let flag = unsafe { &*(ctx as *const AtomicBool) };
    i32::from(flag.load(Ordering::Acquire))
}

struct CppDeepseekOcrModel {
    raw: *mut std::ffi::c_void,
}

// SAFETY: the wrapper solely owns its C++ model handle and all bridge/MLX access is serialized, so it may cross threads.
unsafe impl Send for CppDeepseekOcrModel {}

impl Drop for CppDeepseekOcrModel {
    fn drop(&mut self) {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::deepseek_ocr_free(self.raw);
        }
    }
}

impl CppDeepseekOcrModel {
    fn build(parsed: &MetalDeepseekOcrConfig, tensors: &TensorMap) -> Result<Self> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let raw = unsafe { mlx_sys::deepseek_ocr_new() };
        anyhow::ensure!(!raw.is_null(), "deepseek_ocr_new returned null");
        let mut builder = CppDeepseekOcrBuilder { raw };
        if let Err(err) = builder.populate(parsed, tensors) {
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe {
                mlx_sys::deepseek_ocr_free(raw);
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
                .unwrap_or_else(|| anyhow::anyhow!("MLX DeepSeek-OCR FFI failed")))
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
        let (cancel_fn, cancel_ctx) = cancel_pair(cancel);
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_generate_causal(
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
        finalize_output(tokens, out_len, out_finish)
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
            "DeepSeek-OCR Metal VLM currently supports exactly one image per request"
        );
        let image = &images[0];
        anyhow::ensure!(
            image.channels == 3,
            "DeepSeek-OCR Metal VLM image input must be RGB"
        );
        anyhow::ensure!(
            image.pixels.len() == image.channels * image.height * image.width,
            "DeepSeek-OCR Metal VLM image buffer shape mismatch"
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
        let (cancel_fn, cancel_ctx) = cancel_pair(cancel);
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_generate_causal_image(
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
        finalize_output(tokens, out_len, out_finish)
    }
}

fn cancel_pair(
    cancel: Option<&AtomicBool>,
) -> (
    Option<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    *const std::ffi::c_void,
) {
    if let Some(flag) = cancel {
        (
            Some(deepseek_ocr_cancelled as unsafe extern "C" fn(*const std::ffi::c_void) -> i32),
            flag as *const AtomicBool as *const std::ffi::c_void,
        )
    } else {
        (None, std::ptr::null())
    }
}

fn finalize_output(
    mut tokens: Vec<u32>,
    out_len: i32,
    out_finish: i32,
) -> Result<DiffusionGenerateOutput> {
    let len = usize::try_from(out_len).context("negative DeepSeek-OCR output length")?;
    anyhow::ensure!(
        len <= tokens.len(),
        "DeepSeek-OCR output length {} exceeds buffer {}",
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

struct CppDeepseekOcrBuilder {
    raw: *mut std::ffi::c_void,
}

impl CppDeepseekOcrBuilder {
    fn populate(&mut self, parsed: &MetalDeepseekOcrConfig, tensors: &TensorMap) -> Result<()> {
        let text = parsed.text();
        let quant = parsed
            .spec
            .quantization
            .as_ref()
            .context("DeepSeek-OCR config missing quantization block")?;
        let group_size = quant.group_size;
        let bits = quant.bits;

        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::deepseek_ocr_set_config(
                self.raw,
                text.hidden_size as i32,
                text.vocab_size as i32,
                text.num_attention_heads as i32,
                text.num_key_value_heads as i32,
                text.head_dim() as i32,
                text.v_head_dim as i32,
                text.rms_norm_eps,
                text.rope_theta,
            );
        }
        mlx::check_mlx_error()?;

        // lm_head is its own weight here, not tied to the embedding.
        let prefix = "language_model.model";
        let embed_id = self.add_mxfp8(tensors, &format!("{prefix}.embed_tokens"))?;
        let lm_head_id = self.add_mxfp8(tensors, "language_model.lm_head")?;
        let final_norm_id = self.add_dense_name(tensors, &format!("{prefix}.norm.weight"))?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::deepseek_ocr_set_embed(
                self.raw,
                embed_id,
                embed_id + 1,
                lm_head_id,
                lm_head_id + 1,
                final_norm_id,
                group_size,
                bits,
            );
        }
        mlx::check_mlx_error()?;

        for layer_idx in 0..text.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            self.push_layer(parsed, tensors, &layer_prefix, layer_idx)?;
        }

        self.set_vision(parsed, tensors)?;

        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe { mlx_sys::deepseek_ocr_finalize(self.raw) })?;
        Ok(())
    }

    fn push_layer(
        &mut self,
        parsed: &MetalDeepseekOcrConfig,
        tensors: &TensorMap,
        layer_prefix: &str,
        layer_idx: usize,
    ) -> Result<()> {
        let text = parsed.text();
        let attn = format!("{layer_prefix}.self_attn");
        let mlp = format!("{layer_prefix}.mlp");
        let input_ln_id =
            self.add_dense_name(tensors, &format!("{layer_prefix}.input_layernorm.weight"))?;
        let post_attn_ln_id = self.add_dense_name(
            tensors,
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
        )?;
        let q_id = self.add_mxfp8(tensors, &format!("{attn}.q_proj"))?;
        let k_id = self.add_mxfp8(tensors, &format!("{attn}.k_proj"))?;
        let v_id = self.add_mxfp8(tensors, &format!("{attn}.v_proj"))?;
        let o_id = self.add_mxfp8(tensors, &format!("{attn}.o_proj"))?;

        let is_moe = text.is_moe_layer(layer_idx);
        let (
            dense_gate_id,
            dense_up_id,
            dense_down_id,
            router_id,
            switch_gate_id,
            switch_up_id,
            switch_down_id,
            shared_gate_id,
            shared_up_id,
            shared_down_id,
            num_experts,
            top_k,
        ) = if is_moe {
            (
                -1,
                -1,
                -1,
                self.add_dense_name(tensors, &format!("{mlp}.gate.weight"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.switch_mlp.gate_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.switch_mlp.up_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.switch_mlp.down_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.shared_experts.gate_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.shared_experts.up_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.shared_experts.down_proj"))?,
                text.n_routed_experts as i32,
                text.num_experts_per_tok as i32,
            )
        } else {
            (
                self.add_mxfp8(tensors, &format!("{mlp}.gate_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.up_proj"))?,
                self.add_mxfp8(tensors, &format!("{mlp}.down_proj"))?,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                0,
            )
        };

        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_push_layer(
                self.raw,
                input_ln_id,
                post_attn_ln_id,
                q_id,
                k_id,
                v_id,
                o_id,
                dense_gate_id,
                dense_up_id,
                dense_down_id,
                router_id,
                switch_gate_id,
                switch_up_id,
                switch_down_id,
                shared_gate_id,
                shared_up_id,
                shared_down_id,
                num_experts,
                top_k,
                text.routed_scaling_factor,
            )
        })
    }

    fn set_vision(&mut self, parsed: &MetalDeepseekOcrConfig, tensors: &TensorMap) -> Result<()> {
        let v = &parsed.spec.vision;
        let sam = &parsed.spec.sam;
        let proj = &parsed.spec.projector;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_set_vision_config(
                self.raw,
                parsed.image_token_id as i32,
                v.hidden_size as i32,
                v.intermediate_size as i32,
                v.num_hidden_layers as i32,
                v.num_attention_heads as i32,
                v.patch_size as i32,
                v.layer_norm_eps,
                sam.width as i32,
                sam.layers as i32,
                sam.heads as i32,
                sam.patch_size as i32,
                sam.window_size as i32,
                sam.image_size as i32,
                proj.input_dim as i32,
                proj.n_embed as i32,
            )
        })?;

        let patch_embed_w = self.add_dense_name(tensors, "sam_model.patch_embed.proj.weight")?;
        let patch_embed_b = self.add_dense_name(tensors, "sam_model.patch_embed.proj.bias")?;
        let pos_embed = self.add_dense_name(tensors, "sam_model.pos_embed")?;
        let neck0 = self.add_dense_name(tensors, "sam_model.neck.0.weight")?;
        let neck1_w = self.add_dense_name(tensors, "sam_model.neck.1.weight")?;
        let neck1_b = self.add_dense_name(tensors, "sam_model.neck.1.bias")?;
        let neck2 = self.add_dense_name(tensors, "sam_model.neck.2.weight")?;
        let neck3_w = self.add_dense_name(tensors, "sam_model.neck.3.weight")?;
        let neck3_b = self.add_dense_name(tensors, "sam_model.neck.3.bias")?;
        let net2 = self.add_dense_name(tensors, "sam_model.net_2.weight")?;
        let net3 = self.add_dense_name(tensors, "sam_model.net_3.weight")?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_set_sam_stem(
                self.raw,
                patch_embed_w,
                patch_embed_b,
                pos_embed,
                neck0,
                neck1_w,
                neck1_b,
                neck2,
                neck3_w,
                neck3_b,
                net2,
                net3,
            )
        })?;
        for i in 0..sam.layers {
            let window_size = if sam.global_attn_indexes.contains(&i) {
                0
            } else {
                sam.window_size as i32
            };
            self.push_sam_block(tensors, i, window_size)?;
        }

        let class_embed =
            self.add_dense_name(tensors, "vision_model.embeddings.class_embedding")?;
        let pos =
            self.add_dense_name(tensors, "vision_model.embeddings.position_embedding.weight")?;
        let pre_ln_w = self.add_dense_name(tensors, "vision_model.pre_layrnorm.weight")?;
        let pre_ln_b = self.add_dense_name(tensors, "vision_model.pre_layrnorm.bias")?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_set_clip_stem(self.raw, class_embed, pos, pre_ln_w, pre_ln_b)
        })?;
        for i in 0..v.num_hidden_layers {
            self.push_clip_layer(tensors, i)?;
        }

        let projector_w = self.add_mxfp8(tensors, "projector.layers")?;
        let projector_bias = self.add_dense_name(tensors, "projector.layers.bias")?;
        let image_newline = self.add_dense_name(tensors, "image_newline")?;
        let view_separator = self.add_dense_name(tensors, "view_separator")?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_set_projector(
                self.raw,
                projector_w,
                projector_bias,
                image_newline,
                view_separator,
            )
        })
    }

    fn push_sam_block(&mut self, tensors: &TensorMap, idx: usize, window_size: i32) -> Result<()> {
        let prefix = format!("sam_model.blocks.{idx}");
        let norm1_w = self.add_dense_name(tensors, &format!("{prefix}.norm1.weight"))?;
        let norm1_b = self.add_dense_name(tensors, &format!("{prefix}.norm1.bias"))?;
        let qkv_w = self.add_dense_name(tensors, &format!("{prefix}.attn.qkv.weight"))?;
        let qkv_b = self.add_dense_name(tensors, &format!("{prefix}.attn.qkv.bias"))?;
        let proj_w = self.add_dense_name(tensors, &format!("{prefix}.attn.proj.weight"))?;
        let proj_b = self.add_dense_name(tensors, &format!("{prefix}.attn.proj.bias"))?;
        let rel_pos_h = self.add_dense_name(tensors, &format!("{prefix}.attn.rel_pos_h"))?;
        let rel_pos_w = self.add_dense_name(tensors, &format!("{prefix}.attn.rel_pos_w"))?;
        let norm2_w = self.add_dense_name(tensors, &format!("{prefix}.norm2.weight"))?;
        let norm2_b = self.add_dense_name(tensors, &format!("{prefix}.norm2.bias"))?;
        let lin1_w = self.add_dense_name(tensors, &format!("{prefix}.mlp.lin1.weight"))?;
        let lin1_b = self.add_dense_name(tensors, &format!("{prefix}.mlp.lin1.bias"))?;
        let lin2_w = self.add_dense_name(tensors, &format!("{prefix}.mlp.lin2.weight"))?;
        let lin2_b = self.add_dense_name(tensors, &format!("{prefix}.mlp.lin2.bias"))?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_push_sam_block(
                self.raw,
                window_size,
                norm1_w,
                norm1_b,
                qkv_w,
                qkv_b,
                proj_w,
                proj_b,
                rel_pos_h,
                rel_pos_w,
                norm2_w,
                norm2_b,
                lin1_w,
                lin1_b,
                lin2_w,
                lin2_b,
            )
        })
    }

    fn push_clip_layer(&mut self, tensors: &TensorMap, idx: usize) -> Result<()> {
        let prefix = format!("vision_model.transformer.layers.{idx}");
        let ln1_w = self.add_dense_name(tensors, &format!("{prefix}.layer_norm1.weight"))?;
        let ln1_b = self.add_dense_name(tensors, &format!("{prefix}.layer_norm1.bias"))?;
        let qkv_w = self.add_dense_name(tensors, &format!("{prefix}.self_attn.qkv_proj.weight"))?;
        let qkv_b = self.add_dense_name(tensors, &format!("{prefix}.self_attn.qkv_proj.bias"))?;
        let out_w = self.add_dense_name(tensors, &format!("{prefix}.self_attn.out_proj.weight"))?;
        let out_b = self.add_dense_name(tensors, &format!("{prefix}.self_attn.out_proj.bias"))?;
        let ln2_w = self.add_dense_name(tensors, &format!("{prefix}.layer_norm2.weight"))?;
        let ln2_b = self.add_dense_name(tensors, &format!("{prefix}.layer_norm2.bias"))?;
        let fc1_w = self.add_dense_name(tensors, &format!("{prefix}.mlp.fc1.weight"))?;
        let fc1_b = self.add_dense_name(tensors, &format!("{prefix}.mlp.fc1.bias"))?;
        let fc2_w = self.add_dense_name(tensors, &format!("{prefix}.mlp.fc2.weight"))?;
        let fc2_b = self.add_dense_name(tensors, &format!("{prefix}.mlp.fc2.bias"))?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        self.check_rc(unsafe {
            mlx_sys::deepseek_ocr_push_clip_layer(
                self.raw, ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, fc1_w, fc1_b,
                fc2_w, fc2_b,
            )
        })
    }

    fn add_dense_name(&mut self, tensors: &TensorMap, name: &str) -> Result<i32> {
        let array = tensor_get(tensors, name)?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let id = unsafe { mlx_sys::deepseek_ocr_add_dense_weight(self.raw, array.as_raw()) };
        self.check_weight_id(id)
    }

    /// Register an MXFP8 weight (`{base}.weight` + `{base}.scales`); the C++ side
    /// stores the weight at the returned id and the scales at id+1.
    fn add_mxfp8(&mut self, tensors: &TensorMap, base: &str) -> Result<i32> {
        let w = tensor_get(tensors, &format!("{base}.weight"))?;
        let scales = tensor_get(tensors, &format!("{base}.scales"))?;
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let id = unsafe {
            mlx_sys::deepseek_ocr_add_mxfp8_weight(self.raw, w.as_raw(), scales.as_raw(), 32, 8)
        };
        self.check_weight_id(id)
    }

    fn check_weight_id(&self, id: i32) -> Result<i32> {
        if id >= 0 {
            Ok(id)
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX DeepSeek-OCR weight registration failed")))
        }
    }

    fn check_rc(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(mlx::check_mlx_error()
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("MLX DeepSeek-OCR registration failed")))
        }
    }
}

impl MetalDeepseekOcrConfig {
    pub(crate) fn text(&self) -> &deepseek_ocr_spec::DeepseekOcrTextConfig {
        &self.spec.text
    }
}
