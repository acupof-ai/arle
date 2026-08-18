//! Backend-neutral block-diffusion generation loop.
//!
//! Diffusion-style text models do not fit the autoregressive
//! prefill-then-one-token-decode [`ForwardPlan`](crate::ForwardPlan) contract:
//! one generation block owns a fixed canvas, repeatedly denoises all positions,
//! and commits the whole converged canvas. This module keeps that outer loop in
//! pure host code while backends provide the block model implementation.

use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

use crate::FinishReason;

pub const DEFAULT_DIFFUSION_CANVAS_LENGTH: usize = 256;
pub const DEFAULT_DIFFUSION_MAX_DENOISING_STEPS: usize = 48;
pub const DEFAULT_DIFFUSION_ENTROPY_BOUND: f32 = 0.1;
pub const DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD: f32 = 0.005;
pub const DEFAULT_DIFFUSION_T_MAX: f32 = 0.8;
pub const DEFAULT_DIFFUSION_T_MIN: f32 = 0.4;
pub const DEFAULT_DIFFUSION_STABILITY_THRESHOLD: usize = 1;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionGenerationConfig {
    pub canvas_length: usize,
    pub max_denoising_steps: usize,
    pub max_new_tokens: usize,
    pub vocab_size: u32,
    pub stop_token_ids: Vec<u32>,
    /// Fills the unused tail of a short final canvas.
    pub pad_token_id: u32,
    pub entropy_bound: f32,
    /// Average entropy threshold for adaptive stopping.
    pub confidence_threshold: f32,
    pub t_min: f32,
    pub t_max: f32,
    /// Required number of stable argmax canvases before adaptive commit.
    pub stability_threshold: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalImage {
    /// RGB pixels in channel-first `[3, height, width]` order, `0..1` float,
    /// resized to the model processor's patch grid.
    pub pixels: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    /// Soft-token embeddings the vision tower emits after pooling.
    pub soft_token_count: usize,
}

/// VLM image-preprocessing / marker convention a backend expects. The serving
/// layer dispatches preprocessing on this so each VLM uses its own resize +
/// marker logic without the server depending on backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultimodalKind {
    /// Gemma4 SigLIP vision tower (`<|image|>` markers, pooled soft tokens).
    Gemma4,
    /// DeepSeek-OCR DeepEncoder (`<image>` markers, 1024x1024 base view).
    DeepseekOcr,
}

impl DiffusionGenerationConfig {
    /// DiffusionGemma defaults from the public model config.
    #[must_use]
    pub fn diffusion_gemma(max_new_tokens: usize, vocab_size: u32) -> Self {
        Self {
            canvas_length: DEFAULT_DIFFUSION_CANVAS_LENGTH,
            max_denoising_steps: DEFAULT_DIFFUSION_MAX_DENOISING_STEPS,
            max_new_tokens,
            vocab_size,
            // 1 = `<eos>`, 106 = `<end_of_turn>`, 50 = the model's extra stop id
            // (same fallback as `infer-metal`'s config loader).
            stop_token_ids: vec![1, 106, 50],
            pad_token_id: 0,
            entropy_bound: DEFAULT_DIFFUSION_ENTROPY_BOUND,
            confidence_threshold: DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD,
            t_min: DEFAULT_DIFFUSION_T_MIN,
            t_max: DEFAULT_DIFFUSION_T_MAX,
            stability_threshold: DEFAULT_DIFFUSION_STABILITY_THRESHOLD,
            seed: 0,
        }
    }

    fn validate(&self) -> Result<(), DiffusionGenerateError> {
        if self.canvas_length == 0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "canvas_length must be greater than zero",
            ));
        }
        if self.max_denoising_steps == 0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "max_denoising_steps must be greater than zero",
            ));
        }
        if self.vocab_size == 0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "vocab_size must be greater than zero",
            ));
        }
        if !(self.t_min.is_finite() && self.t_max.is_finite()) || self.t_min < 0.0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "temperature schedule must be finite and non-negative",
            ));
        }
        if !self.entropy_bound.is_finite() || self.entropy_bound < 0.0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "entropy_bound must be finite and non-negative",
            ));
        }
        if !self.confidence_threshold.is_finite() || self.confidence_threshold < 0.0 {
            return Err(DiffusionGenerateError::InvalidConfig(
                "confidence_threshold must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Compact per-position facts for one denoise pass.
///
/// Backends should keep logits and probabilities on device and only return
/// these to the outer loop.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionCanvasPrediction {
    pub sampled_tokens: Vec<u32>,
    pub argmax_tokens: Vec<u32>,
    pub entropies: Vec<f32>,
}

impl DiffusionCanvasPrediction {
    fn validate(&self, canvas_len: usize) -> Result<(), DiffusionGenerateError> {
        if self.sampled_tokens.len() != canvas_len {
            return Err(DiffusionGenerateError::InvalidPrediction {
                field: "sampled_tokens",
                expected: canvas_len,
                got: self.sampled_tokens.len(),
            });
        }
        if self.argmax_tokens.len() != canvas_len {
            return Err(DiffusionGenerateError::InvalidPrediction {
                field: "argmax_tokens",
                expected: canvas_len,
                got: self.argmax_tokens.len(),
            });
        }
        if self.entropies.len() != canvas_len {
            return Err(DiffusionGenerateError::InvalidPrediction {
                field: "entropies",
                expected: canvas_len,
                got: self.entropies.len(),
            });
        }
        if self.entropies.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(DiffusionGenerateError::InvalidConfig(
                "prediction entropies must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Backend hook consumed by [`generate_diffusion_with_cancel`].
pub trait DiffusionBlockModel {
    /// Optional backend-owned generation fast path.
    ///
    /// Implementors that can keep the whole denoise loop on device may return a
    /// completed generation here. The default falls back to the portable host
    /// loop below.
    fn generate(
        &mut self,
        _prompt_tokens: &[u32],
        _config: &DiffusionGenerationConfig,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        Ok(None)
    }

    fn generate_with_cancel(
        &mut self,
        prompt_tokens: &[u32],
        config: &DiffusionGenerationConfig,
        _cancel: Option<&AtomicBool>,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        self.generate(prompt_tokens, config)
    }

    fn generate_multimodal_with_cancel(
        &mut self,
        _prompt_tokens: &[u32],
        _images: &[MultimodalImage],
        _config: &DiffusionGenerationConfig,
        _cancel: Option<&AtomicBool>,
    ) -> Result<Option<DiffusionGenerateOutput>, DiffusionModelError> {
        Ok(None)
    }

    /// VLM preprocessing/marker convention this model expects, when it is a VLM.
    /// Default `None` = text-only / no multimodal preprocessing.
    fn multimodal_kind(&self) -> Option<MultimodalKind> {
        None
    }

    fn begin_request(
        &mut self,
        _config: &DiffusionGenerationConfig,
    ) -> Result<(), DiffusionModelError> {
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), DiffusionModelError>;

    fn predict_canvas(
        &mut self,
        canvas: &[u32],
        valid_len: usize,
        step: usize,
        temperature: f32,
    ) -> Result<DiffusionCanvasPrediction, DiffusionModelError>;

    /// Commit a finalized canvas back into the model's causal context so the
    /// next canvas can condition on it.
    fn commit(&mut self, tokens: &[u32]) -> Result<(), DiffusionModelError>;
}

/// Stringly backend error wrapper, intentionally dependency-free.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct DiffusionModelError {
    pub message: String,
}

impl DiffusionModelError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum DiffusionGenerateError {
    #[error("invalid diffusion config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid diffusion prediction {field}: expected {expected}, got {got}")]
    InvalidPrediction {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("diffusion generation cancelled")]
    Cancelled,
    #[error("diffusion model error: {0}")]
    Model(#[from] DiffusionModelError),
}

/// Per-step trace useful for tests and future `/v1/stats` counters.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStepTrace {
    pub block_index: usize,
    pub step: usize,
    pub accepted_positions: usize,
    pub mean_entropy: f32,
    pub confident: bool,
    pub stable: bool,
    pub committed: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiffusionGenerateStats {
    pub blocks: usize,
    pub denoise_steps: usize,
    pub forced_commits: usize,
    pub adaptive_commits: usize,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionGenerateOutput {
    pub generated_tokens: Vec<u32>,
    pub finish: FinishReason,
    pub stats: DiffusionGenerateStats,
    pub trace: Vec<DiffusionStepTrace>,
}

pub fn generate_diffusion_with_cancel<M: DiffusionBlockModel>(
    model: &mut M,
    prompt_tokens: &[u32],
    config: &DiffusionGenerationConfig,
    cancel: Option<&AtomicBool>,
) -> Result<DiffusionGenerateOutput, DiffusionGenerateError> {
    config.validate()?;
    if config.max_new_tokens == 0 {
        return Ok(DiffusionGenerateOutput {
            generated_tokens: Vec::new(),
            finish: FinishReason::Length,
            stats: DiffusionGenerateStats::default(),
            trace: Vec::new(),
        });
    }
    if cancelled(cancel) {
        return Err(DiffusionGenerateError::Cancelled);
    }
    if let Some(output) = model.generate_with_cancel(prompt_tokens, config, cancel)? {
        return Ok(output);
    }

    model.begin_request(config)?;
    model.prefill(prompt_tokens)?;

    let mut output = Vec::with_capacity(config.max_new_tokens);
    let mut trace = Vec::new();
    let mut stats = DiffusionGenerateStats::default();
    let mut finish = FinishReason::Length;
    let mut block_index = 0usize;

    while output.len() < config.max_new_tokens {
        if cancelled(cancel) {
            return Err(DiffusionGenerateError::Cancelled);
        }
        let remaining = config.max_new_tokens - output.len();
        let valid_len = remaining.min(config.canvas_length);
        let mut canvas = initial_canvas(config, block_index, valid_len);
        let mut history: Vec<Vec<u32>> = Vec::new();

        for step in 0..config.max_denoising_steps {
            if cancelled(cancel) {
                return Err(DiffusionGenerateError::Cancelled);
            }
            let temperature = diffusion_temperature(config, step);
            let prediction = model.predict_canvas(&canvas, valid_len, step, temperature)?;
            prediction.validate(config.canvas_length)?;

            let accepted_mask = entropy_bound_acceptance_mask(
                &prediction.entropies[..valid_len],
                config.entropy_bound,
            );
            let accepted_positions = accepted_mask.iter().filter(|&&accepted| accepted).count();
            let mean_entropy = mean(&prediction.entropies[..valid_len]);
            let confident = mean_entropy < config.confidence_threshold;

            for i in 0..valid_len {
                canvas[i] = if accepted_mask[i] {
                    prediction.sampled_tokens[i]
                } else {
                    renoise_token(config, block_index, step, i)
                };
            }
            for token in canvas.iter_mut().take(config.canvas_length).skip(valid_len) {
                *token = config.pad_token_id;
            }

            history.push(prediction.argmax_tokens[..valid_len].to_vec());
            if history.len() > config.stability_threshold.max(1) {
                history.remove(0);
            }
            let stable = history_is_stable(&history, config.stability_threshold);
            let forced = step + 1 >= config.max_denoising_steps;
            let committed = forced || (stable && confident);

            trace.push(DiffusionStepTrace {
                block_index,
                step,
                accepted_positions,
                mean_entropy,
                confident,
                stable,
                committed,
            });
            stats.denoise_steps += 1;

            if !committed {
                continue;
            }

            if forced {
                stats.forced_commits += 1;
            } else {
                stats.adaptive_commits += 1;
            }

            let mut commit_tokens = prediction.argmax_tokens[..valid_len].to_vec();
            let stop_at = first_stop_position(&commit_tokens, &config.stop_token_ids);
            if let Some(pos) = stop_at {
                commit_tokens.truncate(pos + 1);
                finish = FinishReason::Stop;
            }
            model.commit(&commit_tokens)?;
            output.extend_from_slice(&commit_tokens);
            break;
        }

        stats.blocks += 1;
        block_index += 1;
        if matches!(finish, FinishReason::Stop) || stats.blocks > config.max_new_tokens {
            break;
        }
    }

    if output.len() > config.max_new_tokens {
        output.truncate(config.max_new_tokens);
    }

    Ok(DiffusionGenerateOutput {
        generated_tokens: output,
        finish,
        stats,
        trace,
    })
}

fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|flag| flag.load(Ordering::Acquire))
}

/// DiffusionGemma entropy-bound acceptance mask.
///
/// Sort positions by entropy, accept positions whose cumulative entropy
/// excluding the current maximum remains within the bound.
#[must_use]
pub fn entropy_bound_acceptance_mask(entropies: &[f32], entropy_bound: f32) -> Vec<bool> {
    let mut sorted: Vec<(usize, f32)> = entropies.iter().copied().enumerate().collect();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut mask = vec![false; entropies.len()];
    let mut cumsum = 0.0f32;
    let mut cummax = 0.0f32;
    for (idx, entropy) in sorted {
        cumsum += entropy;
        cummax = cummax.max(entropy);
        if cumsum - cummax <= entropy_bound {
            mask[idx] = true;
        }
    }
    mask
}

fn diffusion_temperature(config: &DiffusionGenerationConfig, step: usize) -> f32 {
    let remaining = config.max_denoising_steps.saturating_sub(step).max(1) as f32;
    config.t_min + (config.t_max - config.t_min) * (remaining / config.max_denoising_steps as f32)
}

fn initial_canvas(
    config: &DiffusionGenerationConfig,
    block_index: usize,
    valid_len: usize,
) -> Vec<u32> {
    (0..config.canvas_length)
        .map(|i| {
            if i < valid_len {
                renoise_token(config, block_index, usize::MAX, i)
            } else {
                config.pad_token_id
            }
        })
        .collect()
}

fn renoise_token(
    config: &DiffusionGenerationConfig,
    block_index: usize,
    step: usize,
    position: usize,
) -> u32 {
    let bits = splitmix64(
        config.seed
            ^ ((block_index as u64) << 32)
            ^ ((step as u64).wrapping_mul(0x9E37_79B9))
            ^ position as u64,
    );
    (bits % u64::from(config.vocab_size)) as u32
}

fn history_is_stable(history: &[Vec<u32>], threshold: usize) -> bool {
    let threshold = threshold.max(1);
    if history.len() < threshold {
        return false;
    }
    let first = &history[0];
    history.iter().all(|row| row == first)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn first_stop_position(tokens: &[u32], stop_token_ids: &[u32]) -> Option<usize> {
    tokens
        .iter()
        .position(|token| stop_token_ids.iter().any(|stop| stop == token))
}

fn predict_row(
    logits: &[f32],
    temperature: f32,
    seed: u64,
    step: usize,
    row: u64,
) -> (u32, u32, f32) {
    let mut argmax = 0usize;
    let mut argmax_v = f32::NEG_INFINITY;
    for (idx, &logit) in logits.iter().enumerate() {
        if logit > argmax_v {
            argmax = idx;
            argmax_v = logit;
        }
    }

    let temp = temperature.max(0.0);
    let inv_t = if temp > 0.0 { 1.0 / temp } else { 1.0 };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&logit| ((logit - max) * inv_t).exp())
        .collect();
    let sum: f32 = probs.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return (argmax as u32, argmax as u32, 0.0);
    }

    let mut entropy = 0.0f32;
    for p in &mut probs {
        *p /= sum;
        if *p > 0.0 {
            entropy -= *p * p.ln();
        }
    }

    if temp <= 0.0 {
        return (argmax as u32, argmax as u32, entropy);
    }

    let mut sampled = 0usize;
    let mut sampled_v = f32::NEG_INFINITY;
    for (idx, &logit) in logits.iter().enumerate() {
        let gumbel = sample_gumbel(seed, step, row, idx as u64);
        let noisy = logit * inv_t + gumbel;
        if noisy > sampled_v {
            sampled_v = noisy;
            sampled = idx;
        }
    }
    (sampled as u32, argmax as u32, entropy)
}

fn sample_gumbel(seed: u64, step: usize, row: u64, vocab_idx: u64) -> f32 {
    let bits = splitmix64(
        seed ^ ((step as u64) << 40) ^ (row << 20) ^ vocab_idx.wrapping_mul(0xD1B5_4A32_D192_ED03),
    );
    let u = (((bits >> 40) as f32) + 1.0) / ((1u32 << 24) as f32 + 2.0);
    -(-u.ln()).ln()
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
