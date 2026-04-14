//! Sampling parameters and logit post-processing.
//!
//! This module is pure Rust with no GPU dependencies.
//! Actual logit transformation kernels live in `ops/sampling.rs` (CUDA-only).

/// Parameters controlling token sampling from the model's output distribution.
///
/// All penalty fields are applied in order:
/// 1. `repetition_penalty`
/// 2. `frequency_penalty` + `presence_penalty`
/// 3. `temperature` scaling
/// 4. `top_k` truncation
/// 5. `top_p` (nucleus) filtering
/// 6. Argmax / categorical sample
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Sampling temperature. 0.0 = greedy (argmax). Higher = more random.
    pub temperature: f32,

    /// Top-K: keep only the K highest-probability tokens. -1 = disabled.
    pub top_k: i32,

    /// Top-P (nucleus): keep the smallest set of tokens whose cumulative
    /// probability ≥ top_p. 1.0 = disabled.
    pub top_p: f32,

    /// Min-P: filter tokens whose probability is less than `min_p × p_max`.
    /// 0.0 = disabled. Typical value: 0.05.
    pub min_p: f32,

    /// Repetition penalty applied to all previously generated token ids.
    /// Values > 1.0 discourage repetition; < 1.0 encourage it. 1.0 = no-op.
    pub repetition_penalty: f32,

    /// OpenAI-style frequency penalty. Subtracts `frequency_penalty × count`
    /// from the logit of each token id that has appeared `count` times.
    /// Range: [-2.0, 2.0]. 0.0 = disabled.
    pub frequency_penalty: f32,

    /// OpenAI-style presence penalty. Subtracts `presence_penalty` from the
    /// logit of any token id that has appeared at least once.
    /// Range: [-2.0, 2.0]. 0.0 = disabled.
    pub presence_penalty: f32,

    /// Stop generating when the EOS token is produced.
    /// Set to `true` to generate past EOS (useful for benchmarks).
    pub ignore_eos: bool,

    /// Additional token ids that trigger stop (beyond the model's EOS).
    pub stop_token_ids: Vec<u32>,

    /// RNG seed for deterministic sampling. `None` = use the engine-level seed.
    pub seed: Option<u64>,

    /// Optional override for the number of tokens to generate.
    /// `None` means the backend should use its default.
    pub max_new_tokens: Option<usize>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ignore_eos: false,
            stop_token_ids: vec![],
            seed: None,
            max_new_tokens: None,
        }
    }
}

impl SamplingParams {
    /// Returns `true` when deterministic greedy decoding should be used.
    /// Greedy when temperature ≤ 0 (or top_k == 1) AND no stochastic filters.
    #[allow(dead_code)]
    pub(crate) fn is_greedy(&self) -> bool {
        (self.temperature <= 0.0 || self.top_k == 1) && self.top_p >= 1.0 && self.min_p <= 0.0
    }

    /// Returns `true` when any penalty modifies the logits before sampling.
    #[allow(clippy::float_cmp)]
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// Apply repetition, frequency, and presence penalties to a logit slice
    /// in-place. `token_counts[token_id]` is the number of times that token
    /// has appeared in the generated sequence so far.
    ///
    /// This pure-Rust implementation is used for correctness tests. The
    /// production path calls the CUDA kernel in `ops/sampling.rs`.
    #[allow(clippy::float_cmp)]
    pub fn apply_penalties(&self, logits: &mut [f32], token_counts: &[u32]) {
        if !self.has_penalties() {
            return;
        }

        let rep = self.repetition_penalty;
        let freq = self.frequency_penalty;
        let pres = self.presence_penalty;

        for (token_id, count) in token_counts.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            if token_id >= logits.len() {
                break;
            }

            let logit = logits[token_id];

            // Repetition penalty: divide by penalty if logit > 0, multiply if < 0.
            // This matches Transformers / SGLang semantics.
            let penalized = if rep == 1.0 {
                logit
            } else if logit >= 0.0 {
                logit / rep
            } else {
                logit * rep
            };

            // Frequency + presence penalties (additive, like OpenAI).
            let penalized =
                penalized - freq * (*count as f32) - if *count > 0 { pres } else { 0.0 };

            logits[token_id] = penalized;
        }
    }
}

// ============================================================================
// HTTP request → SamplingParams conversion helpers
// ============================================================================

/// Build `SamplingParams` from the common OpenAI request fields.
/// Called by both `/v1/completions` and `/v1/chat/completions` handlers.
pub fn sampling_params_from_request(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    ignore_eos: Option<bool>,
    seed: Option<u64>,
    stop_token_ids: Option<Vec<u32>>,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(-1),
        top_p: top_p.unwrap_or(1.0),
        min_p: min_p.unwrap_or(0.0),
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        presence_penalty: presence_penalty.unwrap_or(0.0),
        ignore_eos: ignore_eos.unwrap_or(false),
        seed,
        stop_token_ids: stop_token_ids.unwrap_or_default(),
        max_new_tokens: None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_defaults() {
        let params = SamplingParams::default();
        assert!(params.is_greedy());
    }

    #[test]
    fn test_greedy_top_k_1() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 1,
            top_p: 1.0,
            ..Default::default()
        };
        assert!(params.is_greedy());
    }

    #[test]
    fn test_not_greedy() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: -1,
            top_p: 1.0,
            ..Default::default()
        };
        assert!(!params.is_greedy());
    }

    #[test]
    fn test_min_p_disables_greedy() {
        let params = SamplingParams {
            temperature: 0.0,
            min_p: 0.05,
            ..Default::default()
        };
        assert!(!params.is_greedy());
    }

    #[test]
    fn test_no_penalty_when_defaults() {
        assert!(!SamplingParams::default().has_penalties());
    }

    #[test]
    fn test_repetition_penalty_detected() {
        let params = SamplingParams {
            repetition_penalty: 1.1,
            ..Default::default()
        };
        assert!(params.has_penalties());
    }

    #[test]
    fn test_apply_repetition_penalty_positive_logit() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        let counts = vec![1u32, 0, 1];
        let params = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        params.apply_penalties(&mut logits, &counts);
        // token 0: 1.0 / 2.0 = 0.5
        // token 1: unchanged = 2.0
        // token 2: 3.0 / 2.0 = 1.5
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_repetition_penalty_negative_logit() {
        let mut logits = vec![-2.0_f32];
        let counts = vec![1u32];
        let params = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        params.apply_penalties(&mut logits, &counts);
        // token 0: -2.0 * 2.0 = -4.0
        assert!((logits[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_frequency_penalty() {
        let mut logits = vec![5.0_f32, 5.0, 5.0];
        // token 0 appeared 3 times, token 1 once, token 2 never
        let counts = vec![3u32, 1, 0];
        let params = SamplingParams {
            frequency_penalty: 1.0,
            ..Default::default()
        };
        params.apply_penalties(&mut logits, &counts);
        // token 0: 5.0 - 1.0*3 = 2.0
        // token 1: 5.0 - 1.0*1 = 4.0
        // token 2: unchanged = 5.0
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_presence_penalty() {
        let mut logits = vec![5.0_f32, 5.0, 5.0];
        let counts = vec![2u32, 1, 0];
        let params = SamplingParams {
            presence_penalty: 0.5,
            ..Default::default()
        };
        params.apply_penalties(&mut logits, &counts);
        // tokens 0 and 1 penalized by 0.5 each; token 2 untouched
        assert!((logits[0] - 4.5).abs() < 1e-6);
        assert!((logits[1] - 4.5).abs() < 1e-6);
        assert!((logits[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_no_penalty_on_zero_count() {
        let mut logits = vec![3.0_f32];
        let counts = vec![0u32];
        let params = SamplingParams {
            repetition_penalty: 1.5,
            frequency_penalty: 1.0,
            presence_penalty: 1.0,
            ..Default::default()
        };
        params.apply_penalties(&mut logits, &counts);
        // count = 0 → skip
        assert!((logits[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sampling_params_from_request() {
        let p = sampling_params_from_request(
            Some(0.7),
            Some(0.9),
            None,
            Some(0.05),
            Some(1.1),
            Some(0.2),
            Some(0.1),
            None,
            Some(42),
            Some(vec![1, 2, 3]),
        );
        assert!((p.temperature - 0.7).abs() < 1e-6);
        assert!((p.top_p - 0.9).abs() < 1e-6);
        assert_eq!(p.top_k, -1);
        assert!((p.min_p - 0.05).abs() < 1e-6);
        assert!((p.repetition_penalty - 1.1).abs() < 1e-6);
        assert!((p.frequency_penalty - 0.2).abs() < 1e-6);
        assert!((p.presence_penalty - 0.1).abs() < 1e-6);
        assert_eq!(p.seed, Some(42));
        assert_eq!(p.stop_token_ids, vec![1, 2, 3]);
    }
}
