use anyhow::{Result, ensure};

use super::mlx::MlxArray;
use crate::sampler::SamplingParams;

#[cfg(feature = "metal")]
#[allow(clippy::float_cmp)]
pub(super) fn validate_metal_sampling_params(params: &SamplingParams) -> Result<()> {
    let mut unsupported = Vec::new();

    if params.top_k != -1 && params.top_k != 1 {
        unsupported.push(format!("top_k={}", params.top_k));
    }
    if params.top_p < 1.0 {
        unsupported.push(format!("top_p={}", params.top_p));
    }
    if params.min_p > 0.0 {
        unsupported.push(format!("min_p={}", params.min_p));
    }
    if params.repetition_penalty != 1.0 {
        unsupported.push(format!("repetition_penalty={}", params.repetition_penalty));
    }
    if params.frequency_penalty != 0.0 {
        unsupported.push(format!("frequency_penalty={}", params.frequency_penalty));
    }
    if params.presence_penalty != 0.0 {
        unsupported.push(format!("presence_penalty={}", params.presence_penalty));
    }
    if let Some(seed) = params.seed {
        unsupported.push(format!("seed={seed}"));
    }

    ensure!(
        unsupported.is_empty(),
        "Metal backend currently supports only temperature sampling and greedy decoding (top_k=1); unsupported sampling params: {}",
        unsupported.join(", ")
    );

    Ok(())
}

/// P4 — GPU-side sampling: argmax or categorical, stays on GPU until `.item()`.
#[cfg(feature = "metal")]
pub(super) fn gpu_sample_token(logits: &MlxArray, params: &SamplingParams) -> MlxArray {
    if params.temperature <= 1e-6 || params.top_k == 1 {
        return greedy_sample_token(logits);
    }

    // Temperature scaling then GPU categorical sample.
    let inv_t = MlxArray::scalar_f32(1.0f32 / params.temperature);
    let scaled = super::mlx::multiply(logits, &inv_t);
    categorical_sample_token(&scaled)
}

#[cfg(feature = "metal")]
pub(super) fn gpu_sample_token_batched(logits: &MlxArray, params: &SamplingParams) -> MlxArray {
    if params.temperature <= 1e-6 || params.top_k == 1 {
        return super::mlx::argmax_axis(logits, -1);
    }

    let inv_t = MlxArray::scalar_f32(1.0f32 / params.temperature);
    let scaled = super::mlx::multiply(logits, &inv_t);
    super::mlx::categorical(&scaled)
}

#[cfg(feature = "metal")]
fn greedy_sample_token(logits: &MlxArray) -> MlxArray {
    super::mlx::argmax(logits)
}

#[cfg(feature = "metal")]
fn categorical_sample_token(logits: &MlxArray) -> MlxArray {
    super::mlx::categorical(logits)
}
