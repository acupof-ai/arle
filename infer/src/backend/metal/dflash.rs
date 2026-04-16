use std::{collections::HashSet, path::Path, time::Instant};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

use super::{
    config::{MetalModelArch, MetalModelConfig, QuantConfig},
    forward::rust_transformer_layer,
    generate::{KV_CACHE_CHUNK, MetalGenerateOutput},
    loader::{load_proj_from_tensors, load_tensor_map, tensor_get},
    mlx::{MlxArray, concatenate_axis, eval, rms_norm, slice, take_axis, zeros},
    ops::{extend_kv_cache, linear},
    sampling::{gpu_sample_token, validate_metal_sampling_params},
    weights::{MlpInputProjection, StandardMetalWeights, WeightTensor},
};
use crate::{hf_hub, sampler::SamplingParams};

#[derive(Clone, Debug)]
pub struct MetalDflashOptions {
    pub draft_model: String,
    pub speculative_tokens: Option<usize>,
}

impl MetalDflashOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.draft_model.trim().is_empty(),
            "Metal DFlash draft model must not be empty"
        );
        if let Some(tokens) = self.speculative_tokens {
            ensure!(
                tokens > 0,
                "Metal DFlash speculative token override must be >= 1 when set"
            );
        }
        Ok(())
    }
}

pub(super) struct MetalDflashRuntime {
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    draft_model_id: String,
    draft_config: DFlashDraftConfig,
    draft_weights: DFlashDraftWeights,
}

impl MetalDflashRuntime {
    pub(super) fn load(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> Result<Self> {
        options.validate()?;
        ensure!(
            matches!(target_config.arch, MetalModelArch::Qwen3),
            "Metal DFlash currently supports Qwen3 only; Qwen3.5 still needs recurrent rollback integration"
        );

        let draft_model_dir = hf_hub::resolve_weighted_model_path(&options.draft_model)
            .with_context(|| {
                format!(
                    "failed to resolve DFlash draft model '{}'",
                    options.draft_model
                )
            })?;
        let draft_config = DFlashDraftConfig::load(&draft_model_dir)?;
        ensure!(
            !draft_config.target_layer_ids.is_empty(),
            "DFlash draft config must declare at least one target layer id"
        );
        ensure!(
            draft_config.hidden_size == target_config.hidden_size,
            "DFlash draft hidden_size {} != target hidden_size {}",
            draft_config.hidden_size,
            target_config.hidden_size
        );

        let draft_weights = DFlashDraftWeights::load(&draft_model_dir, &draft_config)
            .with_context(|| {
                format!(
                    "failed to load DFlash draft weights from {}",
                    draft_model_dir.display()
                )
            })?;
        let default_block_size = draft_config.block_size.max(1);
        let requested_block_size = options
            .speculative_tokens
            .unwrap_or(default_block_size)
            .max(1);
        if let Some(requested) = options.speculative_tokens {
            if requested < default_block_size {
                log::warn!(
                    "Metal DFlash speculative block override {} is below the draft default {}; this can reduce acceptance and throughput",
                    requested,
                    default_block_size
                );
            } else if requested > default_block_size {
                log::warn!(
                    "Metal DFlash speculative block override {} exceeds the draft default {}; clamping to {}",
                    requested,
                    default_block_size,
                    default_block_size
                );
            }
        }
        let block_size = requested_block_size.min(default_block_size);

        log::info!(
            "Metal DFlash enabled: draft='{}', block_size={}, target_layers={:?}",
            options.draft_model,
            block_size,
            draft_config.target_layer_ids
        );

        Ok(Self {
            block_size,
            mask_token_id: draft_config.mask_token_id,
            target_layer_ids: draft_config.target_layer_ids.clone(),
            draft_model_id: options.draft_model.clone(),
            draft_config,
            draft_weights,
        })
    }

    pub(super) fn draft_model_id(&self) -> &str {
        &self.draft_model_id
    }

    pub(super) fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub(super) fn draft_num_hidden_layers(&self) -> usize {
        self.draft_config.num_hidden_layers
    }

    pub(super) fn draft_n_kv_heads(&self) -> i32 {
        self.draft_config.num_key_value_heads as i32
    }

    pub(super) fn draft_head_dim(&self) -> i32 {
        self.draft_config.head_dim as i32
    }
}

#[derive(Clone, Debug)]
pub(super) struct DFlashDraftConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    quantization: Option<QuantConfig>,
}

#[derive(Debug, Deserialize)]
struct RawDraftQuantConfig {
    group_size: Option<i32>,
    bits: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct RawDflashConfig {
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct RawDraftConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    block_size: Option<usize>,
    dflash_config: RawDflashConfig,
    quantization: Option<RawDraftQuantConfig>,
    quantization_config: Option<RawDraftQuantConfig>,
}

impl DFlashDraftConfig {
    fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let raw: RawDraftConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("cannot read {}", config_path.display()))?,
        )
        .with_context(|| format!("cannot parse {}", config_path.display()))?;

        let quant_source = raw.quantization.or(raw.quantization_config);
        let quantization = quant_source.map(|q| QuantConfig {
            group_size: q.group_size.unwrap_or(64),
            bits: q.bits.unwrap_or(4),
        });

        Ok(Self {
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps as f32,
            rope_theta: raw.rope_theta as f32,
            block_size: raw.block_size.unwrap_or(16),
            mask_token_id: raw.dflash_config.mask_token_id,
            target_layer_ids: raw.dflash_config.target_layer_ids,
            quantization,
        })
    }
}

struct DFlashDraftLayerWeights {
    q_proj: WeightTensor,
    k_proj: WeightTensor,
    v_proj: WeightTensor,
    o_proj: WeightTensor,
    input_layernorm: MlxArray,
    post_attention_layernorm: MlxArray,
    q_norm: MlxArray,
    k_norm: MlxArray,
    mlp_inputs: MlpInputProjection,
    down_proj: WeightTensor,
}

struct DFlashDraftWeights {
    layers: Vec<DFlashDraftLayerWeights>,
    fc: WeightTensor,
    hidden_norm: MlxArray,
    norm: MlxArray,
}

impl DFlashDraftWeights {
    fn load(model_dir: &Path, config: &DFlashDraftConfig) -> Result<Self> {
        let tensors = load_tensor_map(model_dir)?;
        let get = |name: &str| tensor_get(&tensors, name);
        let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, config.quantization);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = |suffix: &str| format!("layers.{i}.{suffix}");

            let gate_proj = load_proj(&p("mlp.gate_proj"))?;
            let up_proj = load_proj(&p("mlp.up_proj"))?;
            let gate_dim = gate_proj.output_dim()?;
            let up_dim = up_proj.output_dim()?;
            let mlp_inputs = if let Some(gate_up_proj) =
                super::weights::merge_quantized_projection_rows(&[&gate_proj, &up_proj])?
            {
                MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                    up_dim,
                }
            } else {
                MlpInputProjection::Split { gate_proj, up_proj }
            };

            layers.push(DFlashDraftLayerWeights {
                q_proj: load_proj(&p("self_attn.q_proj"))?,
                k_proj: load_proj(&p("self_attn.k_proj"))?,
                v_proj: load_proj(&p("self_attn.v_proj"))?,
                o_proj: load_proj(&p("self_attn.o_proj"))?,
                input_layernorm: get(&p("input_layernorm.weight"))?,
                post_attention_layernorm: get(&p("post_attention_layernorm.weight"))?,
                q_norm: get(&p("self_attn.q_norm.weight"))?,
                k_norm: get(&p("self_attn.k_norm.weight"))?,
                mlp_inputs,
                down_proj: load_proj(&p("mlp.down_proj"))?,
            });
        }

        Ok(Self {
            layers,
            fc: load_proj("fc")?,
            hidden_norm: get("hidden_norm.weight")?,
            norm: get("norm.weight")?,
        })
    }
}

pub(super) struct ContiguousKvState {
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    len: i32,
    capacity: i32,
    n_kv_heads: i32,
    head_dim: i32,
}

impl ContiguousKvState {
    pub(super) fn new(
        num_layers: usize,
        n_kv_heads: i32,
        head_dim: i32,
        initial_tokens: usize,
    ) -> Self {
        let initial_cap = ((i32::try_from(initial_tokens).unwrap_or_default() + KV_CACHE_CHUNK
            - 1)
            / KV_CACHE_CHUNK
            + 1)
            * KV_CACHE_CHUNK;
        let cache_shape = [1i32, n_kv_heads, initial_cap.max(KV_CACHE_CHUNK), head_dim];
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(zeros(&cache_shape, super::mlx::Dtype::Bfloat16));
            v_caches.push(zeros(&cache_shape, super::mlx::Dtype::Bfloat16));
        }
        Self {
            k_caches,
            v_caches,
            len: 0,
            capacity: cache_shape[2],
            n_kv_heads,
            head_dim,
        }
    }

    pub(super) fn from_dtype(
        num_layers: usize,
        n_kv_heads: i32,
        head_dim: i32,
        initial_tokens: usize,
        dtype: super::mlx::Dtype,
    ) -> Self {
        let initial_cap = ((i32::try_from(initial_tokens).unwrap_or_default() + KV_CACHE_CHUNK
            - 1)
            / KV_CACHE_CHUNK
            + 1)
            * KV_CACHE_CHUNK;
        let cache_shape = [1i32, n_kv_heads, initial_cap.max(KV_CACHE_CHUNK), head_dim];
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(zeros(&cache_shape, dtype));
            v_caches.push(zeros(&cache_shape, dtype));
        }
        Self {
            k_caches,
            v_caches,
            len: 0,
            capacity: cache_shape[2],
            n_kv_heads,
            head_dim,
        }
    }

    fn ensure_capacity(&mut self, required_len: i32) {
        if required_len <= self.capacity {
            return;
        }
        let new_capacity = ((required_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
        for cache in &mut self.k_caches {
            extend_kv_cache(cache, self.n_kv_heads, self.head_dim, new_capacity);
        }
        for cache in &mut self.v_caches {
            extend_kv_cache(cache, self.n_kv_heads, self.head_dim, new_capacity);
        }
        self.capacity = new_capacity;
    }

    fn trim(&mut self, num_tokens: usize) {
        self.len = self
            .len
            .saturating_sub(i32::try_from(num_tokens).unwrap_or_default());
    }
}

pub(super) fn metal_generate_dflash_qwen3(
    runtime: &MetalDflashRuntime,
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<MetalGenerateOutput> {
    ensure!(
        !input_ids.is_empty(),
        "Metal DFlash requires at least one prompt token"
    );
    validate_metal_sampling_params(params)?;

    if max_new_tokens == 0 {
        return Ok(MetalGenerateOutput {
            tokens: Vec::new(),
            finish_reason: "length",
            ttft_ms: 0.0,
            total_time_ms: 0.0,
        });
    }

    let dtype = weights.layers[0].attention_inputs.kv_dtype();
    let mut target_state = ContiguousKvState::from_dtype(
        config.num_hidden_layers,
        config.num_key_value_heads as i32,
        config.head_dim as i32,
        input_ids.len() + max_new_tokens,
        dtype,
    );
    let mut draft_state = ContiguousKvState::new(
        runtime.draft_config.num_hidden_layers,
        runtime.draft_config.num_key_value_heads as i32,
        runtime.draft_config.head_dim as i32,
        input_ids.len() + max_new_tokens,
    );

    let (prompt_norm_hidden, mut target_hidden) = qwen3_forward_with_hidden_states(
        input_ids,
        weights,
        config,
        &runtime.target_layer_ids,
        &mut target_state,
    )?;
    let prompt_logits = linear(&prompt_norm_hidden, &weights.lm_head);
    let first_token = sample_last_token(&prompt_logits, params)?;
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut generated = vec![first_token];
    on_token(first_token)?;
    if is_stop_token(config, params, first_token) || generated.len() >= max_new_tokens {
        let total_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return Ok(MetalGenerateOutput {
            tokens: generated,
            finish_reason: if is_stop_token(config, params, first_token) {
                "stop"
            } else {
                "length"
            },
            ttft_ms,
            total_time_ms,
        });
    }

    let mut current_token = first_token;
    let mut acceptance_lengths = Vec::new();
    let finish_reason = loop {
        let mut block_tokens = vec![runtime.mask_token_id; runtime.block_size];
        block_tokens[0] = current_token;

        let noise_embedding = embed_tokens(&weights.embed_tokens, &block_tokens);
        let draft_hidden =
            dflash_draft_forward(runtime, &noise_embedding, &target_hidden, &mut draft_state)?;
        let block_hidden = slice(
            &draft_hidden,
            &[1, 0],
            &[
                i32::try_from(runtime.block_size).unwrap_or_default(),
                i32::try_from(config.hidden_size).unwrap_or_default(),
            ],
            &[1, 1],
        );
        let draft_logits = linear(&block_hidden, &weights.lm_head);
        let drafted_suffix = sample_rows(&draft_logits, params)?;
        draft_state.trim(runtime.block_size);
        for (dst, src) in block_tokens.iter_mut().skip(1).zip(drafted_suffix.iter()) {
            *dst = *src;
        }

        let (verifier_norm_hidden, verifier_hidden) = qwen3_forward_with_hidden_states(
            &block_tokens,
            weights,
            config,
            &runtime.target_layer_ids,
            &mut target_state,
        )?;
        let verifier_logits = linear(&verifier_norm_hidden, &weights.lm_head);
        let posterior = sample_rows(&verifier_logits, params)?;
        let matched = block_tokens
            .iter()
            .skip(1)
            .zip(posterior.iter())
            .take(runtime.block_size.saturating_sub(1))
            .take_while(|(draft, target)| draft == target)
            .count();
        let accepted_inputs = matched + 1;
        let posterior_token = *posterior
            .get(matched)
            .ok_or_else(|| anyhow!("DFlash verifier produced too few tokens"))?;
        if accepted_inputs < runtime.block_size {
            target_state.trim(runtime.block_size - accepted_inputs);
        }

        acceptance_lengths.push(accepted_inputs);
        target_hidden = slice(
            &verifier_hidden,
            &[0, 0],
            &[
                i32::try_from(accepted_inputs).unwrap_or_default(),
                i32::try_from(runtime.target_layer_ids.len() * config.hidden_size)
                    .unwrap_or_default(),
            ],
            &[1, 1],
        );

        let mut accepted_finish_reason = None;
        for token in block_tokens
            .iter()
            .skip(1)
            .take(accepted_inputs.saturating_sub(1))
        {
            generated.push(*token);
            on_token(*token)?;
            if is_stop_token(config, params, *token) {
                log::info!(
                    "Metal DFlash: accepted {:?} (avg {:.2}) before stop",
                    acceptance_lengths,
                    average_acceptance(&acceptance_lengths)
                );
                accepted_finish_reason = Some("stop");
                break;
            }
            if generated.len() >= max_new_tokens {
                log::info!(
                    "Metal DFlash: accepted {:?} (avg {:.2}) before length stop",
                    acceptance_lengths,
                    average_acceptance(&acceptance_lengths)
                );
                accepted_finish_reason = Some("length");
                break;
            }
        }
        if let Some(reason) = accepted_finish_reason {
            break reason;
        }

        generated.push(posterior_token);
        on_token(posterior_token)?;
        current_token = posterior_token;
        if is_stop_token(config, params, posterior_token) {
            log::info!(
                "Metal DFlash: accepted {:?} (avg {:.2})",
                acceptance_lengths,
                average_acceptance(&acceptance_lengths)
            );
            break "stop";
        }
        if generated.len() >= max_new_tokens {
            log::info!(
                "Metal DFlash: accepted {:?} (avg {:.2})",
                acceptance_lengths,
                average_acceptance(&acceptance_lengths)
            );
            break "length";
        }
    };

    let total_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms,
    })
}

/// Result of one DFlash speculative block (draft → verify → accept/reject).
pub(super) struct DFlashBlockResult {
    pub accepted_tokens: Vec<u32>,
    pub updated_target_hidden: MlxArray,
    pub accepted_inputs: usize,
}

/// Run one DFlash speculative block: draft N tokens, verify against target
/// model, accept the longest matching prefix, trim rejected KV.
pub(super) fn dflash_speculative_block(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    target_state: &mut ContiguousKvState,
    draft_state: &mut ContiguousKvState,
) -> Result<DFlashBlockResult> {
    let mut block_tokens = vec![runtime.mask_token_id; runtime.block_size];
    block_tokens[0] = current_token;
    let noise_embedding = embed_tokens(&weights.embed_tokens, &block_tokens);
    let draft_hidden = dflash_draft_forward(runtime, &noise_embedding, target_hidden, draft_state)?;
    let block_hidden = slice(
        &draft_hidden,
        &[1, 0],
        &[
            i32::try_from(runtime.block_size).unwrap_or_default(),
            i32::try_from(config.hidden_size).unwrap_or_default(),
        ],
        &[1, 1],
    );
    let draft_logits = linear(&block_hidden, &weights.lm_head);
    let drafted_suffix = sample_rows(&draft_logits, params)?;
    draft_state.trim(runtime.block_size);
    for (dst, src) in block_tokens.iter_mut().skip(1).zip(drafted_suffix.iter()) {
        *dst = *src;
    }
    let (verifier_norm_hidden, verifier_hidden) = qwen3_forward_with_hidden_states(
        &block_tokens,
        weights,
        config,
        &runtime.target_layer_ids,
        target_state,
    )?;
    let verifier_logits = linear(&verifier_norm_hidden, &weights.lm_head);
    let posterior = sample_rows(&verifier_logits, params)?;
    let matched = block_tokens
        .iter()
        .skip(1)
        .zip(posterior.iter())
        .take(runtime.block_size.saturating_sub(1))
        .take_while(|(draft, target)| draft == target)
        .count();
    let accepted_inputs = matched + 1;
    let posterior_token = *posterior
        .get(matched)
        .ok_or_else(|| anyhow!("DFlash verifier produced too few tokens"))?;
    if accepted_inputs < runtime.block_size {
        target_state.trim(runtime.block_size - accepted_inputs);
    }
    let updated_target_hidden = slice(
        &verifier_hidden,
        &[0, 0],
        &[
            i32::try_from(accepted_inputs).unwrap_or_default(),
            i32::try_from(runtime.target_layer_ids.len() * config.hidden_size).unwrap_or_default(),
        ],
        &[1, 1],
    );
    let mut accepted_tokens = Vec::with_capacity(accepted_inputs);
    for &token in block_tokens
        .iter()
        .skip(1)
        .take(accepted_inputs.saturating_sub(1))
    {
        accepted_tokens.push(token);
    }
    accepted_tokens.push(posterior_token);
    Ok(DFlashBlockResult {
        accepted_tokens,
        updated_target_hidden,
        accepted_inputs,
    })
}

/// Public wrapper for the scheduler path — runs the full Qwen3 forward
/// on `ContiguousKvState` and captures hidden states at target layers.
pub(super) fn qwen3_forward_with_hidden_states_on_state(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    target_layer_ids: &[usize],
    state: &mut ContiguousKvState,
) -> Result<(MlxArray, MlxArray)> {
    qwen3_forward_with_hidden_states(input_ids, weights, config, target_layer_ids, state)
}

fn qwen3_forward_with_hidden_states(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    target_layer_ids: &[usize],
    state: &mut ContiguousKvState,
) -> Result<(MlxArray, MlxArray)> {
    let seq = i32::try_from(input_ids.len()).context("input length exceeds i32")?;
    state.ensure_capacity(state.len + seq);
    let n_heads = i32::try_from(config.num_attention_heads).unwrap_or_default();
    let n_kv_heads = i32::try_from(config.num_key_value_heads).unwrap_or_default();
    let head_dim = i32::try_from(config.head_dim).unwrap_or_default();
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = config.rope_theta as f32;
    let eps = config.rms_norm_eps as f32;
    let selected: HashSet<_> = target_layer_ids.iter().copied().collect();
    let mut selected_hidden = Vec::with_capacity(target_layer_ids.len());

    let mut x = embed_tokens(&weights.embed_tokens, input_ids);
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = rust_transformer_layer(
            x,
            layer,
            layer_idx,
            &mut state.k_caches,
            &mut state.v_caches,
            seq,
            state.len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            None,
            0,
        )?;
        if selected.contains(&layer_idx) {
            selected_hidden.push(x.clone());
        }
    }
    state.len += seq;

    ensure!(
        !selected_hidden.is_empty(),
        "DFlash target_layer_ids selected no hidden states"
    );

    let norm_hidden = rms_norm(&x, &weights.norm, eps);
    Ok((norm_hidden, concatenate_axis(&selected_hidden, 1)))
}

fn dflash_draft_forward(
    runtime: &MetalDflashRuntime,
    noise_embedding: &MlxArray,
    target_hidden: &MlxArray,
    state: &mut ContiguousKvState,
) -> Result<MlxArray> {
    let context_len = *target_hidden
        .shape()
        .first()
        .ok_or_else(|| anyhow!("target_hidden must be rank-2"))?;
    let seq = *noise_embedding
        .shape()
        .first()
        .ok_or_else(|| anyhow!("noise_embedding must be rank-2"))?;
    state.ensure_capacity(state.len + context_len + seq);

    let n_heads = i32::try_from(runtime.draft_config.num_attention_heads).unwrap_or_default();
    let n_kv_heads = i32::try_from(runtime.draft_config.num_key_value_heads).unwrap_or_default();
    let head_dim = i32::try_from(runtime.draft_config.head_dim).unwrap_or_default();
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = runtime.draft_config.rope_theta;
    let eps = runtime.draft_config.rms_norm_eps;

    let target_hidden = linear(target_hidden, &runtime.draft_weights.fc);
    let target_hidden = rms_norm(&target_hidden, &runtime.draft_weights.hidden_norm, eps);
    let mut hidden_states = noise_embedding.clone();

    for (layer_idx, layer) in runtime.draft_weights.layers.iter().enumerate() {
        hidden_states = dflash_draft_layer_forward(
            &hidden_states,
            &target_hidden,
            layer,
            layer_idx,
            state,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
        );
    }

    state.len += context_len + seq;
    Ok(rms_norm(&hidden_states, &runtime.draft_weights.norm, eps))
}

#[allow(clippy::too_many_arguments)]
fn dflash_draft_layer_forward(
    hidden_states: &MlxArray,
    target_hidden: &MlxArray,
    layer: &DFlashDraftLayerWeights,
    layer_idx: usize,
    state: &mut ContiguousKvState,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
) -> MlxArray {
    use super::mlx::{
        add, multiply, reshape, rope, scaled_dot_product_attention, silu, slice_update,
        transpose_axes,
    };

    let seq = hidden_states.shape()[0];
    let context_len = target_hidden.shape()[0];
    let residual = hidden_states.clone();
    let normed_hidden_states = rms_norm(hidden_states, &layer.input_layernorm, eps);

    let q_raw = linear(&normed_hidden_states, &layer.q_proj);
    let kv_states = concatenate_axis(&[target_hidden.clone(), normed_hidden_states], 0);
    let k_raw = linear(&kv_states, &layer.k_proj);
    let v_raw = linear(&kv_states, &layer.v_proj);

    let q = reshape(&q_raw, &[1, seq, n_heads, head_dim]);
    let q = rms_norm(&q, &layer.q_norm, eps);
    let q = transpose_axes(&q, &[0, 2, 1, 3]);
    let q = rope(&q, head_dim, false, rope_base, 1.0, state.len + context_len);

    let total_len = context_len + seq;
    let k = reshape(&k_raw, &[1, total_len, n_kv_heads, head_dim]);
    let k = rms_norm(&k, &layer.k_norm, eps);
    let k = transpose_axes(&k, &[0, 2, 1, 3]);
    let k = rope(&k, head_dim, false, rope_base, 1.0, state.len);

    let v = reshape(&v_raw, &[1, total_len, n_kv_heads, head_dim]);
    let v = transpose_axes(&v, &[0, 2, 1, 3]);

    let end_pos = state.len + total_len;
    state.k_caches[layer_idx] = slice_update(
        &mut state.k_caches[layer_idx],
        &k,
        &[0, 0, state.len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );
    state.v_caches[layer_idx] = slice_update(
        &mut state.v_caches[layer_idx],
        &v,
        &[0, 0, state.len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );

    let k_full = slice(
        &state.k_caches[layer_idx],
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );
    let v_full = slice(
        &state.v_caches[layer_idx],
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );

    let attn = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None);
    let attn = transpose_axes(&attn, &[0, 2, 1, 3]);
    let attn = reshape(&attn, &[seq, n_heads * head_dim]);
    let attn = linear(&attn, &layer.o_proj);
    let hidden_states = add(&residual, &attn);

    let residual = hidden_states.clone();
    let hidden_states = rms_norm(&hidden_states, &layer.post_attention_layernorm, eps);
    let (gate_raw, up) = layer.mlp_inputs.project(&hidden_states);
    let mlp = linear(&multiply(&silu(&gate_raw), &up), &layer.down_proj);
    add(&residual, &mlp)
}

fn embed_tokens(embed_table: &MlxArray, input_ids: &[u32]) -> MlxArray {
    let ids: Vec<i32> = input_ids.iter().map(|&token| token as i32).collect();
    let indices = MlxArray::from_slice_i32(&ids, &[i32::try_from(ids.len()).unwrap_or_default()]);
    take_axis(embed_table, &indices, 0)
}

fn sample_last_token(logits: &MlxArray, params: &SamplingParams) -> Result<u32> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 logits, got shape {shape:?}"
    );
    let last_row = slice(logits, &[shape[0] - 1, 0], &[shape[0], shape[1]], &[1, 1]);
    let token = gpu_sample_token(&last_row, params);
    eval(&[&token]);
    Ok(token.item_i32() as u32)
}

fn sample_rows(logits: &MlxArray, params: &SamplingParams) -> Result<Vec<u32>> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 logits, got shape {shape:?}"
    );
    let mut row_tokens = Vec::with_capacity(shape[0] as usize);
    for row in 0..shape[0] {
        let row_logits = slice(logits, &[row, 0], &[row + 1, shape[1]], &[1, 1]);
        row_tokens.push(gpu_sample_token(&row_logits, params));
    }
    let refs: Vec<_> = row_tokens.iter().collect();
    eval(&refs);
    Ok(row_tokens
        .iter()
        .map(|token| token.item_i32() as u32)
        .collect())
}

fn is_stop_token(config: &MetalModelConfig, params: &SamplingParams, token: u32) -> bool {
    (!params.ignore_eos && config.is_stop_token(token)) || params.stop_token_ids.contains(&token)
}

fn average_acceptance(lengths: &[usize]) -> f64 {
    if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
    }
}
