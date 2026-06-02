use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};

use super::config::{MetalModelArch, MetalModelConfig, QuantConfig};
use super::dflash::{
    Qwen35VerifyStateGuard, drain_current_qwen35_gdr_tapes, materialize_token_array,
    qwen35_rollback_to_accepted_varlen,
};
use super::loader::{load_proj_from_tensors, load_tensor_map, tensor_get};
use super::mlx::{
    Dtype, MlxArray, add, as_dtype, async_eval, concatenate_axis, multiply, reshape, rope,
    scaled_dot_product_attention, sigmoid, slice, take_axis, transpose_axes,
};
use super::ops::linear;
use super::qwen35::{
    CppQwen35Model, MetalQwen35Attention, MetalQwen35BlockWeights, MetalQwen35FullAttentionWeights,
    MlpKind, capture_qwen35_final_hidden_from_cpp_outputs, load_qwen35_moe_layer_weights,
    moe_mlp_forward, qwen35_norm_needs_offset_correction, qwen35_normalize_direct_norm_weight,
    rms_norm_last_dim,
};
use super::sampling::gpu_sample_token;
use super::weights::WeightTensor;
use crate::{gguf::GgufFile, hf_hub, sampler::SamplingParams};

const SAFETENSORS_HEADER_LIMIT_BYTES: u64 = 64 * 1024 * 1024;
const MTP_EXAMPLE_LIMIT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMtpMode {
    Auto,
    Explicit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMtpOptions {
    pub mode: MetalMtpMode,
    pub draft_model: Option<String>,
    pub draft_tokens: Option<usize>,
}

impl MetalMtpOptions {
    pub const fn auto() -> Self {
        Self {
            mode: MetalMtpMode::Auto,
            draft_model: None,
            draft_tokens: None,
        }
    }

    pub const fn explicit() -> Self {
        Self {
            mode: MetalMtpMode::Explicit,
            draft_model: None,
            draft_tokens: None,
        }
    }

    #[must_use]
    pub fn with_draft_model(mut self, draft_model: String) -> Self {
        self.draft_model = Some(draft_model);
        self
    }

    #[must_use]
    pub fn with_draft_tokens(mut self, draft_tokens: usize) -> Self {
        self.draft_tokens = Some(draft_tokens);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalMtpTensorSource {
    SafetensorsIndex(PathBuf),
    SafetensorsHeader,
    Gguf,
    NotFound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMtpProbe {
    pub tensor_count: usize,
    pub examples: Vec<String>,
    pub source: MetalMtpTensorSource,
}

impl MetalMtpProbe {
    pub fn has_tensors(&self) -> bool {
        self.tensor_count > 0
    }

    pub fn examples_label(&self) -> String {
        if self.examples.is_empty() {
            "none".to_string()
        } else {
            self.examples.join(", ")
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalMtpDraftProbe {
    pub requested_model: String,
    pub resolved_path: PathBuf,
    pub model_type: String,
    pub block_size: Option<u64>,
    pub tensor_count: usize,
    pub examples: Vec<String>,
    pub source: MetalMtpTensorSource,
}

pub(crate) struct MetalMtpRuntime {
    requested_model: String,
    resolved_path: PathBuf,
    block_size: usize,
    quantization: QuantConfig,
    pre_fc_norm_embedding: MlxArray,
    pre_fc_norm_hidden: MlxArray,
    fc: WeightTensor,
    block: MetalQwen35BlockWeights,
    norm: MlxArray,
}

pub(super) struct MetalMtpBlockResult {
    pub(super) accepted_tokens: Vec<u32>,
    pub(super) updated_seed_hidden: MlxArray,
    pub(super) accepted_inputs: usize,
}

impl MetalMtpRuntime {
    pub(super) fn draft_model_id(&self) -> &str {
        &self.requested_model
    }

    pub(super) fn resolved_path(&self) -> &Path {
        &self.resolved_path
    }

    pub(super) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(super) fn quantization(&self) -> QuantConfig {
        self.quantization
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_draft_block(
        &self,
        current_token: u32,
        seed_hidden: &MlxArray,
        target_embed: &MlxArray,
        target_lm_head: &WeightTensor,
        target_config: &MetalModelConfig,
        target_kv_flat: &[MlxArray],
        target_cache_len: i32,
        params: &SamplingParams,
    ) -> Result<MlxArray> {
        ensure!(
            self.block_size >= 2,
            "Metal MTP block_size must be >= 2, got {}",
            self.block_size
        );
        let mut token_arr = MlxArray::from_slice_i32(&[current_token as i32], &[1]);
        let mut hidden = seed_hidden.clone();
        let mut block_tokens = Vec::with_capacity(self.block_size);
        block_tokens.push(token_arr.clone());
        for _ in 1..self.block_size {
            let (sampled, next_hidden) = self.draft_step(
                &token_arr,
                &hidden,
                target_embed,
                target_lm_head,
                target_config,
                target_kv_flat,
                target_cache_len,
                params,
            )?;
            token_arr = reshape(&as_dtype(&sampled, Dtype::Int32), &[1]);
            block_tokens.push(token_arr.clone());
            hidden = next_hidden;
        }
        Ok(concatenate_axis(&block_tokens, 0))
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_step(
        &self,
        token_arr: &MlxArray,
        seed_hidden: &MlxArray,
        target_embed: &MlxArray,
        target_lm_head: &WeightTensor,
        target_config: &MetalModelConfig,
        target_kv_flat: &[MlxArray],
        target_cache_len: i32,
        params: &SamplingParams,
    ) -> Result<(MlxArray, MlxArray)> {
        let input_embed = take_axis(target_embed, token_arr, 0);
        let input_embed = reshape(&input_embed, &[1, target_config.hidden_size as i32]);
        let seed_hidden = reshape(seed_hidden, &[1, target_config.hidden_size as i32]);
        let input_embed = rms_norm_last_dim(
            &input_embed,
            &self.pre_fc_norm_embedding,
            target_config.rms_norm_eps as f32,
            false,
        );
        let seed_hidden = rms_norm_last_dim(
            &seed_hidden,
            &self.pre_fc_norm_hidden,
            target_config.rms_norm_eps as f32,
            false,
        );
        let x = concatenate_axis(&[input_embed, seed_hidden], 1);
        let mut x = linear(&x, &self.fc);

        let residual = x.clone();
        let xn = rms_norm_last_dim(
            &x,
            &self.block.input_layernorm,
            target_config.rms_norm_eps as f32,
            false,
        );
        let attn_out =
            self.frozen_full_attention_step(&xn, target_config, target_kv_flat, target_cache_len)?;
        x = add(&residual, &attn_out);

        let residual = x.clone();
        let xn = rms_norm_last_dim(
            &x,
            &self.block.post_attention_layernorm,
            target_config.rms_norm_eps as f32,
            false,
        );
        let mlp_out = match &self.block.mlp {
            MlpKind::Moe(moe) => moe_mlp_forward(&xn, moe),
            MlpKind::Dense(_) => {
                bail!("Metal MTP currently supports Qwen3.6 MoE draft layers only")
            }
        };
        let x = add(&residual, &mlp_out);
        let final_hidden =
            rms_norm_last_dim(&x, &self.norm, target_config.rms_norm_eps as f32, false);
        let logits = linear(&final_hidden, target_lm_head);
        let sampled = gpu_sample_token(&logits, params);
        Ok((sampled, final_hidden))
    }

    fn frozen_full_attention_step(
        &self,
        x: &MlxArray,
        target_config: &MetalModelConfig,
        target_kv_flat: &[MlxArray],
        target_cache_len: i32,
    ) -> Result<MlxArray> {
        let MetalModelArch::Qwen35(arch) = &target_config.arch else {
            bail!("Metal MTP frozen attention requires Qwen3.5/Qwen3.6 target config");
        };
        let MetalQwen35Attention::Full(attn) = &self.block.attention else {
            bail!("Metal MTP draft layer must be full attention");
        };
        ensure!(
            target_kv_flat.len() >= 2,
            "Metal MTP frozen attention requires target full-attn KV pair 0"
        );
        ensure!(
            target_cache_len >= 0,
            "Metal MTP frozen attention got negative target_cache_len={target_cache_len}"
        );
        let k_cache = &target_kv_flat[0];
        let v_cache = &target_kv_flat[1];
        let k_shape = k_cache.shape();
        ensure!(
            k_shape.len() == 4 && target_cache_len <= k_shape[2],
            "Metal MTP target K cache shape {:?} cannot serve cache_len {}",
            k_shape,
            target_cache_len
        );

        let n_heads = target_config.num_attention_heads as i32;
        let n_kv_heads = target_config.num_key_value_heads as i32;
        let head_dim = target_config.head_dim as i32;
        let q_dim = n_heads * head_dim;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
        let rope_pos = target_cache_len.saturating_sub(1);

        let q_full = linear(x, &attn.q_proj);
        let q_full = reshape(&q_full, &[1, 1, n_heads, head_dim * 2]);
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
        let mut q = rms_norm_last_dim(
            &q_heads,
            &attn.q_norm,
            target_config.rms_norm_eps as f32,
            false,
        );
        q = transpose_axes(&q, &[0, 2, 1, 3]);
        q = rope(
            &q,
            arch.rotary_dim as i32,
            false,
            target_config.rope_theta as f32,
            1.0,
            rope_pos,
        );

        let k_raw = linear(x, &attn.k_proj);
        let mut k = reshape(&k_raw, &[1, 1, n_kv_heads, head_dim]);
        k = rms_norm_last_dim(&k, &attn.k_norm, target_config.rms_norm_eps as f32, false);
        k = transpose_axes(&k, &[0, 2, 1, 3]);
        k = rope(
            &k,
            arch.rotary_dim as i32,
            false,
            target_config.rope_theta as f32,
            1.0,
            rope_pos,
        );

        let v_raw = linear(x, &attn.v_proj);
        let v = transpose_axes(
            &reshape(&v_raw, &[1, 1, n_kv_heads, head_dim]),
            &[0, 2, 1, 3],
        );

        let (k_full, v_full) = if target_cache_len > 0 {
            let k_prefix = slice(
                k_cache,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, target_cache_len, head_dim],
                &[1, 1, 1, 1],
            );
            let v_prefix = slice(
                v_cache,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, target_cache_len, head_dim],
                &[1, 1, 1, 1],
            );
            (
                concatenate_axis(&[k_prefix, k], 2),
                concatenate_axis(&[v_prefix, v], 2),
            )
        } else {
            (k, v)
        };
        let attn_out = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None);
        let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = reshape(&attn_out, &[1, q_dim]);
        let gate = reshape(&gate_heads, &[1, q_dim]);
        let gate = sigmoid(&as_dtype(&gate, Dtype::Float32));
        let gated = as_dtype(
            &multiply(&as_dtype(&attn_out, Dtype::Float32), &gate),
            Dtype::Bfloat16,
        );
        Ok(linear(&gated, &attn.o_proj))
    }
}

impl MetalMtpDraftProbe {
    pub fn examples_label(&self) -> String {
        if self.examples.is_empty() {
            "none".to_string()
        } else {
            self.examples.join(", ")
        }
    }

    pub fn block_size_label(&self) -> String {
        self.block_size
            .map_or_else(|| "unknown".to_string(), |value| value.to_string())
    }
}

pub(super) fn probe_mtp_tensors(
    model_root: &Path,
    gguf: Option<&GgufFile>,
) -> Result<MetalMtpProbe> {
    if let Some(gguf) = gguf {
        return Ok(probe_gguf(gguf));
    }

    if let Some(probe) = probe_safetensors_index(model_root)? {
        return Ok(probe);
    }

    let mut all_matches = Vec::new();
    for shard in collect_safetensors_files(model_root)? {
        all_matches.extend(probe_safetensors_header(&shard)?);
    }
    Ok(probe_from_matches(
        all_matches,
        MetalMtpTensorSource::SafetensorsHeader,
    ))
}

pub(super) fn resolve_mtp_draft_model(draft_model: &str) -> Result<MetalMtpDraftProbe> {
    ensure!(
        !draft_model.trim().is_empty(),
        "Metal MTP draft model must not be empty"
    );
    let resolved_path = hf_hub::resolve_model_path(draft_model)
        .with_context(|| format!("failed to resolve Metal MTP draft model '{draft_model}'"))?;
    let model_root = model_root_from_path(&resolved_path);
    probe_mtp_draft_model_root(draft_model, &model_root, resolved_path)
}

pub(super) fn load_mtp_runtime(
    draft_model: &str,
    draft_tokens_override: Option<usize>,
    target_config: &MetalModelConfig,
) -> Result<(MetalMtpRuntime, MetalMtpDraftProbe)> {
    let probe = resolve_mtp_draft_model(draft_model)?;
    let model_root = model_root_from_path(&probe.resolved_path);
    let text_config = read_mtp_text_config(&model_root)?;
    ensure!(
        text_config.hidden_size == target_config.hidden_size,
        "Metal MTP hidden_size {} does not match target hidden_size {}",
        text_config.hidden_size,
        target_config.hidden_size
    );
    ensure!(
        text_config.vocab_size == target_config.vocab_size,
        "Metal MTP vocab_size {} does not match target vocab_size {}",
        text_config.vocab_size,
        target_config.vocab_size
    );
    ensure!(
        text_config.num_attention_heads == target_config.num_attention_heads
            && text_config.num_key_value_heads == target_config.num_key_value_heads
            && text_config.head_dim == target_config.head_dim,
        "Metal MTP attention shape heads={}/{} hd={} does not match target heads={}/{} hd={}",
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
        target_config.num_attention_heads,
        target_config.num_key_value_heads,
        target_config.head_dim
    );
    let MetalModelArch::Qwen35(target_arch) = &target_config.arch else {
        bail!("Metal MTP currently requires a Qwen3.5/Qwen3.6 target model");
    };
    let target_moe_cfg = target_arch
        .moe
        .as_ref()
        .context("Metal MTP currently supports the Qwen3.6 MoE target only")?;
    ensure!(
        text_config.block_size >= 2,
        "Metal MTP draft config block_size must be >= 2, got {}",
        text_config.block_size
    );
    let block_size = if let Some(draft_tokens) = draft_tokens_override {
        ensure!(draft_tokens > 0, "Metal MTP draft tokens must be >= 1");
        ensure!(
            draft_tokens <= 16,
            "Metal MTP draft tokens must be <= 16 for local experimentation, got {}",
            draft_tokens
        );
        let effective_block_size = draft_tokens
            .checked_add(1)
            .context("Metal MTP draft token override overflowed block_size")?;
        if effective_block_size > text_config.block_size {
            log::warn!(
                "Metal MTP draft-token override exceeds model-declared block_size: draft_tokens={} effective_block_size={} declared_block_size={}; running explicit experiment",
                draft_tokens,
                effective_block_size,
                text_config.block_size
            );
        }
        effective_block_size
    } else {
        text_config.block_size
    };

    let tensors = load_tensor_map(&model_root).with_context(|| {
        format!(
            "loading Metal MTP draft tensors from {}",
            model_root.display()
        )
    })?;
    let quantization = text_config.quantization;
    let mut mtp_moe_cfg = target_moe_cfg.clone();
    mtp_moe_cfg.router_bits = quantization.bits;
    mtp_moe_cfg.router_group_size = quantization.group_size;
    mtp_moe_cfg.expert_bits = quantization.bits;
    mtp_moe_cfg.expert_group_size = quantization.group_size;
    let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, Some(quantization));
    let get = |name: &str| tensor_get(&tensors, name);
    let norms_need_offset_correction = {
        let sample = get("layers.0.input_layernorm.weight")?;
        qwen35_norm_needs_offset_correction(&sample)
    };
    if norms_need_offset_correction {
        log::info!(
            "  Metal MTP draft uses HF offset RMSNorm weights — normalizing to direct form at load"
        );
    }
    let load_norm = |name: &str| -> Result<MlxArray> {
        let weight = get(name)?;
        Ok(qwen35_normalize_direct_norm_weight(
            &weight,
            norms_need_offset_correction,
        ))
    };

    let attn_prefix = "layers.0.self_attn";
    let attention = MetalQwen35Attention::Full(MetalQwen35FullAttentionWeights {
        q_proj: load_proj(&format!("{attn_prefix}.q_proj"))?,
        k_proj: load_proj(&format!("{attn_prefix}.k_proj"))?,
        v_proj: load_proj(&format!("{attn_prefix}.v_proj"))?,
        o_proj: load_proj(&format!("{attn_prefix}.o_proj"))?,
        q_norm: load_norm(&format!("{attn_prefix}.q_norm.weight"))?,
        k_norm: load_norm(&format!("{attn_prefix}.k_norm.weight"))?,
    });
    let block = MetalQwen35BlockWeights {
        input_layernorm: load_norm("layers.0.input_layernorm.weight")?,
        attention,
        post_attention_layernorm: load_norm("layers.0.post_attention_layernorm.weight")?,
        mlp: MlpKind::Moe(load_qwen35_moe_layer_weights(
            &tensors,
            "layers.0",
            &mtp_moe_cfg,
        )?),
    };

    let runtime = MetalMtpRuntime {
        requested_model: draft_model.to_string(),
        resolved_path: probe.resolved_path.clone(),
        block_size,
        quantization,
        pre_fc_norm_embedding: load_norm("pre_fc_norm_embedding.weight")?,
        pre_fc_norm_hidden: load_norm("pre_fc_norm_hidden.weight")?,
        fc: load_proj("fc")?,
        block,
        norm: load_norm("norm.weight")?,
    };
    Ok((runtime, probe))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qwen35_mtp_speculative_block(
    runtime: &MetalMtpRuntime,
    current_token: u32,
    seed_hidden: &MlxArray,
    target_embed: &MlxArray,
    target_lm_head: &WeightTensor,
    target_config: &MetalModelConfig,
    cpp_model: &CppQwen35Model,
    params: &SamplingParams,
    target_kv_flat: &mut [MlxArray],
    target_gdr_flat: &mut [MlxArray],
    target_cache_len: &mut i32,
) -> Result<MetalMtpBlockResult> {
    let profile = std::env::var("QWEN35_MTP_PROFILE").is_ok();
    let t_start = std::time::Instant::now();
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("Metal MTP block_size does not fit i32")?;
    let block_tokens = runtime.prepare_draft_block(
        current_token,
        seed_hidden,
        target_embed,
        target_lm_head,
        target_config,
        target_kv_flat,
        *target_cache_len,
        params,
    )?;
    let t_draft = t_start.elapsed();

    let expected_tape_count = target_gdr_flat.len() / 2;
    let gdr_snapshot = target_gdr_flat.to_vec();
    unsafe {
        mlx_sys::qwen35_set_tape_mode(cpp_model.as_raw(), true);
        mlx_sys::qwen35_set_capture_final_hidden(cpp_model.as_raw(), true);
    }
    let _verify_state_guard = Qwen35VerifyStateGuard {
        raw: cpp_model.as_raw(),
    };

    let verify_summary = cpp_model.verify_block_summary(
        &block_tokens,
        block_size_i32,
        *target_cache_len,
        target_kv_flat,
        target_gdr_flat,
        params,
        None,
    )?;
    let tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
    let final_hidden = capture_qwen35_final_hidden_from_cpp_outputs(cpp_model.as_raw())?
        .context("Metal MTP verifier did not capture final hidden")?;
    let matched = verify_summary.matched_prefix_len;
    ensure!(
        matched < runtime.block_size,
        "Metal MTP verify summary returned matched_prefix_len={} for block_size={}",
        matched,
        runtime.block_size
    );
    let accepted_inputs = matched + 1;
    if accepted_inputs < runtime.block_size {
        qwen35_rollback_to_accepted_varlen(
            target_gdr_flat,
            &gdr_snapshot,
            &tapes,
            &[accepted_inputs as i32],
        )?;
    }
    *target_cache_len += accepted_inputs as i32;

    let mut accepted_tokens = if matched == 0 {
        Vec::new()
    } else {
        let accepted_prefix = slice(&block_tokens, &[1], &[1 + matched as i32], &[1]);
        materialize_token_array(&accepted_prefix)
    };
    accepted_tokens.push(verify_summary.next_token);

    let hidden_shape = final_hidden.shape();
    ensure!(
        hidden_shape.len() == 2 && hidden_shape[0] >= accepted_inputs as i32,
        "Metal MTP final hidden shape {:?} cannot provide accepted row {}",
        hidden_shape,
        accepted_inputs
    );
    let hidden_width = hidden_shape[1];
    let row = accepted_inputs as i32 - 1;
    let updated_seed_hidden = slice(&final_hidden, &[row, 0], &[row + 1, hidden_width], &[1, 1]);

    let mut to_eval: Vec<&MlxArray> = vec![&updated_seed_hidden];
    to_eval.extend(target_gdr_flat.iter());
    to_eval.extend(target_kv_flat.iter());
    async_eval(&to_eval);

    if profile {
        let t_total = t_start.elapsed();
        log::info!(
            "qwen35_mtp: accepted={}/{} matched_prefix={} draft={:.1}ms verify_total={:.1}ms",
            accepted_inputs,
            runtime.block_size,
            matched,
            t_draft.as_secs_f32() * 1000.0,
            t_total.saturating_sub(t_draft).as_secs_f32() * 1000.0,
        );
    }

    Ok(MetalMtpBlockResult {
        accepted_tokens,
        updated_seed_hidden,
        accepted_inputs,
    })
}

fn probe_gguf(gguf: &GgufFile) -> MetalMtpProbe {
    let mut matches = Vec::new();
    for (key, value) in &gguf.metadata {
        if is_mtp_metadata_key(key)
            && value
                .as_u32()
                .or_else(|| value.as_str().and_then(|raw| raw.parse::<u32>().ok()))
                .is_some_and(|n| n > 0)
        {
            matches.push(format!("{key}={}", value_label(key, value)));
        }
    }
    matches.extend(
        gguf.tensors
            .keys()
            .filter(|name| is_mtp_tensor_name(name))
            .cloned(),
    );
    probe_from_matches(matches, MetalMtpTensorSource::Gguf)
}

fn probe_mtp_draft_model_root(
    requested_model: &str,
    model_root: &Path,
    resolved_path: PathBuf,
) -> Result<MetalMtpDraftProbe> {
    let config_path = model_root.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading MTP draft config {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing MTP draft config {}", config_path.display()))?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    ensure!(
        model_type == "qwen3_5_mtp",
        "Metal MTP draft model '{}' has model_type='{}', expected 'qwen3_5_mtp'",
        requested_model,
        if model_type.is_empty() {
            "<missing>"
        } else {
            &model_type
        }
    );
    let block_size = config.get("block_size").and_then(serde_json::Value::as_u64);
    let (tensor_names, source) = collect_safetensors_tensor_names(model_root)?;
    ensure!(
        !tensor_names.is_empty(),
        "Metal MTP draft model '{}' has no safetensors tensors",
        requested_model
    );
    ensure!(
        tensor_names.iter().any(|name| name == "fc.weight"),
        "Metal MTP draft model '{}' is missing fc.weight",
        requested_model
    );
    ensure!(
        tensor_names
            .iter()
            .any(|name| name == "pre_fc_norm_embedding.weight"),
        "Metal MTP draft model '{}' is missing pre_fc_norm_embedding.weight",
        requested_model
    );
    ensure!(
        tensor_names
            .iter()
            .any(|name| name == "pre_fc_norm_hidden.weight"),
        "Metal MTP draft model '{}' is missing pre_fc_norm_hidden.weight",
        requested_model
    );
    ensure!(
        tensor_names
            .iter()
            .any(|name| name.starts_with("layers.0.")),
        "Metal MTP draft model '{}' is missing layers.0.* weights",
        requested_model
    );

    let mut examples = tensor_names
        .iter()
        .filter(|name| is_split_mtp_draft_tensor_name(name))
        .cloned()
        .collect::<Vec<_>>();
    examples.sort();
    examples.dedup();
    examples.truncate(MTP_EXAMPLE_LIMIT);
    Ok(MetalMtpDraftProbe {
        requested_model: requested_model.to_string(),
        resolved_path,
        model_type,
        block_size,
        tensor_count: tensor_names.len(),
        examples,
        source,
    })
}

#[derive(Clone, Copy, Debug)]
struct MtpTextConfig {
    block_size: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    quantization: QuantConfig,
}

fn read_mtp_text_config(model_root: &Path) -> Result<MtpTextConfig> {
    let config_path = model_root.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading MTP draft config {}", config_path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parsing MTP draft config {}", config_path.display()))?;
    let text = root
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .context("Metal MTP draft config missing text_config")?;
    let block_size = root
        .get("block_size")
        .and_then(serde_json::Value::as_u64)
        .context("Metal MTP draft config missing block_size")? as usize;
    let quant_root = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
        .context("Metal MTP draft config missing quantization")?;
    let quantization = QuantConfig {
        group_size: quant_root
            .get("group_size")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(64) as i32,
        bits: quant_root
            .get("bits")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(4) as i32,
    };
    let get_usize = |key: &str| -> Result<usize> {
        text.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .with_context(|| format!("Metal MTP draft text_config missing {key}"))
    };
    ensure!(
        text.get("mtp_num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1)
            == 1,
        "Metal MTP currently supports exactly one MTP hidden layer"
    );
    ensure!(
        !text
            .get("mtp_use_dedicated_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        "Metal MTP currently requires target-tied embeddings"
    );
    Ok(MtpTextConfig {
        block_size,
        hidden_size: get_usize("hidden_size")?,
        num_attention_heads: get_usize("num_attention_heads")?,
        num_key_value_heads: get_usize("num_key_value_heads")?,
        head_dim: get_usize("head_dim")?,
        vocab_size: get_usize("vocab_size")?,
        quantization,
    })
}

fn probe_safetensors_index(model_root: &Path) -> Result<Option<MetalMtpProbe>> {
    let index_path = model_root.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;
    let Some(weight_map) = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(Some(MetalMtpProbe {
            tensor_count: 0,
            examples: Vec::new(),
            source: MetalMtpTensorSource::SafetensorsIndex(index_path),
        }));
    };
    let matches = weight_map
        .keys()
        .filter(|name| is_mtp_tensor_name(name))
        .cloned()
        .collect::<Vec<_>>();
    Ok(Some(probe_from_matches(
        matches,
        MetalMtpTensorSource::SafetensorsIndex(index_path),
    )))
}

fn collect_safetensors_tensor_names(
    model_root: &Path,
) -> Result<(Vec<String>, MetalMtpTensorSource)> {
    let index_path = model_root.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let names = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .map(|weight_map| weight_map.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        return Ok((names, MetalMtpTensorSource::SafetensorsIndex(index_path)));
    }

    let mut names = Vec::new();
    for shard in collect_safetensors_files(model_root)? {
        names.extend(probe_all_safetensors_header_names(&shard)?);
    }
    Ok((
        names,
        if model_root.is_dir() {
            MetalMtpTensorSource::SafetensorsHeader
        } else {
            MetalMtpTensorSource::NotFound
        },
    ))
}

fn probe_safetensors_header(path: &Path) -> Result<Vec<String>> {
    Ok(probe_all_safetensors_header_names(path)?
        .into_iter()
        .filter(|name| is_mtp_tensor_name(name))
        .collect())
}

fn probe_all_safetensors_header_names(path: &Path) -> Result<Vec<String>> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .with_context(|| format!("reading safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    anyhow::ensure!(
        header_len <= SAFETENSORS_HEADER_LIMIT_BYTES,
        "safetensors header in {} is too large: {} bytes",
        path.display(),
        header_len
    );
    file.seek(SeekFrom::Start(8))
        .with_context(|| format!("seeking safetensors header in {}", path.display()))?;
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading safetensors header from {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parsing safetensors header from {}", path.display()))?;
    let Some(root) = value.as_object() else {
        return Ok(Vec::new());
    };
    Ok(root
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect())
}

fn collect_safetensors_files(model_root: &Path) -> Result<Vec<PathBuf>> {
    if !model_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = std::fs::read_dir(model_root)
        .with_context(|| format!("reading {}", model_root.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn probe_from_matches(mut matches: Vec<String>, source: MetalMtpTensorSource) -> MetalMtpProbe {
    matches.sort();
    matches.dedup();
    let tensor_count = matches.len();
    matches.truncate(MTP_EXAMPLE_LIMIT);
    MetalMtpProbe {
        tensor_count,
        examples: matches,
        source: if tensor_count == 0 {
            match source {
                MetalMtpTensorSource::SafetensorsIndex(path) => {
                    MetalMtpTensorSource::SafetensorsIndex(path)
                }
                _ => MetalMtpTensorSource::NotFound,
            }
        } else {
            source
        },
    }
}

fn is_mtp_tensor_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("mtp.")
        || lower.contains(".mtp.")
        || lower.contains("nextn")
        || lower.contains("next_n")
        || lower.contains(".eh_proj.")
        || lower.contains(".enorm.")
        || lower.contains(".hnorm.")
}

fn is_split_mtp_draft_tensor_name(name: &str) -> bool {
    name == "fc.weight"
        || name == "pre_fc_norm_embedding.weight"
        || name == "pre_fc_norm_hidden.weight"
        || name.starts_with("layers.0.")
        || name == "norm.weight"
}

fn is_mtp_metadata_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.ends_with(".nextn_predict_layers")
        || lower.contains("nextn_predict_layers")
        || lower.contains("next_n_predict_layers")
        || lower.contains("mtp_num_hidden_layers")
}

fn value_label(key: &str, value: &crate::gguf::GgufValue) -> String {
    value
        .as_u32()
        .map(|v| v.to_string())
        .or_else(|| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{key}:present"))
}

fn model_root_from_path(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mtp_from_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"a.safetensors","mtp.fc.weight":"b.safetensors","mtp.layers.0.self_attn.q_proj.weight":"b.safetensors","model.layers.0.nextn.weight":"b.safetensors"}}"#,
        )
        .expect("write index");

        let probe = probe_mtp_tensors(dir.path(), None).expect("probe");
        assert_eq!(probe.tensor_count, 3);
        assert_eq!(
            probe.examples,
            vec![
                "model.layers.0.nextn.weight".to_string(),
                "mtp.fc.weight".to_string(),
                "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            ]
        );
        assert!(matches!(
            probe.source,
            MetalMtpTensorSource::SafetensorsIndex(_)
        ));
    }

    #[test]
    fn explicit_no_index_no_shards_is_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let probe = probe_mtp_tensors(dir.path(), None).expect("probe");
        assert_eq!(probe.tensor_count, 0);
        assert_eq!(probe.source, MetalMtpTensorSource::NotFound);
    }

    #[test]
    fn detects_mtp_from_gguf_metadata_and_tensor_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("mtp.gguf");
        write_minimal_gguf(
            &gguf_path,
            &[("qwen35.nextn_predict_layers", 2)],
            &[
                "blk.48.nextn.eh_proj.weight",
                "blk.48.nextn.enorm.weight",
                "blk.48.attn_q.weight",
            ],
        );
        let gguf = GgufFile::open(gguf_path.to_str().expect("path")).expect("gguf");

        let probe = probe_mtp_tensors(dir.path(), Some(&gguf)).expect("probe");
        assert_eq!(probe.tensor_count, 3);
        assert_eq!(
            probe.examples,
            vec![
                "blk.48.nextn.eh_proj.weight".to_string(),
                "blk.48.nextn.enorm.weight".to_string(),
                "qwen35.nextn_predict_layers=2".to_string(),
            ]
        );
        assert_eq!(probe.source, MetalMtpTensorSource::Gguf);
    }

    #[test]
    fn gguf_without_mtp_signals_is_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let gguf_path = dir.path().join("plain.gguf");
        write_minimal_gguf(&gguf_path, &[("qwen35.nextn_predict_layers", 0)], &[]);
        let gguf = GgufFile::open(gguf_path.to_str().expect("path")).expect("gguf");

        let probe = probe_mtp_tensors(dir.path(), Some(&gguf)).expect("probe");
        assert_eq!(probe.tensor_count, 0);
        assert_eq!(probe.source, MetalMtpTensorSource::NotFound);
    }

    #[test]
    fn detects_split_mtp_draft_model_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_mtp_draft_config(dir.path(), "qwen3_5_mtp");
        write_mtp_draft_index(
            dir.path(),
            &[
                "fc.weight",
                "fc.scales",
                "fc.biases",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.self_attn.q_proj.weight",
                "norm.weight",
            ],
        );

        let probe =
            probe_mtp_draft_model_root("draft", dir.path(), dir.path().to_path_buf()).unwrap();
        assert_eq!(probe.model_type, "qwen3_5_mtp");
        assert_eq!(probe.block_size, Some(3));
        assert_eq!(probe.tensor_count, 8);
        assert!(probe.examples.contains(&"fc.weight".to_string()));
        assert!(
            probe
                .examples
                .contains(&"pre_fc_norm_embedding.weight".to_string())
        );
    }

    #[test]
    fn rejects_non_mtp_draft_model_type() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_mtp_draft_config(dir.path(), "qwen3_5_moe");
        write_mtp_draft_index(dir.path(), &["fc.weight"]);

        let err = probe_mtp_draft_model_root("draft", dir.path(), dir.path().to_path_buf())
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 'qwen3_5_mtp'"), "{err}");
    }

    fn write_minimal_gguf(path: &Path, metadata: &[(&str, u32)], tensors: &[&str]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for tensor in tensors {
            push_string(&mut bytes, tensor);
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write gguf");
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn write_mtp_draft_config(path: &Path, model_type: &str) {
        std::fs::write(
            path.join("config.json"),
            format!(
                r#"{{
                    "block_size": 3,
                    "model_type": "{model_type}",
                    "text_config": {{
                        "mtp_num_hidden_layers": 1
                    }}
                }}"#
            ),
        )
        .expect("write config");
    }

    fn write_mtp_draft_index(path: &Path, names: &[&str]) {
        let mut weight_map = serde_json::Map::new();
        for name in names {
            weight_map.insert(
                (*name).to_string(),
                serde_json::Value::String("model.safetensors".to_string()),
            );
        }
        let value = serde_json::json!({
            "metadata": {"total_size": 1},
            "weight_map": weight_map,
        });
        std::fs::write(
            path.join("model.safetensors.index.json"),
            serde_json::to_vec(&value).expect("json"),
        )
        .expect("write index");
    }
}
