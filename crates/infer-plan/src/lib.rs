//! Backend-independent inference planning data.
//!
//! This crate is the pure data layer shared by engine-core and executor
//! implementations. It intentionally carries no device tensor, stream, graph,
//! or backend runtime types.

/// Forward execution mode requested by the engine core.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub enum ForwardMode {
    /// A prompt or prompt chunk is being inserted into KV.
    Prefill,
    /// One token is being decoded for each active row.
    Decode,
    /// Decode rows and prefill rows share the same executor step.
    Mixed,
    /// No executor work is available for this scheduler tick.
    Idle,
    /// Target-model verification for speculative decoding.
    TargetVerify,
    /// Draft-model extension for speculative decoding.
    DraftExtend,
}

/// One decode row in a [`ForwardPlan`].
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct DecodeRow {
    /// Engine slot that owns the request state.
    pub slot: usize,
    /// Last token produced for this slot, used as the next decode input.
    pub last_token: u32,
    /// Logical KV sequence length already present for this slot.
    pub kv_seq_len: usize,
    /// Sampling parameters for converting this row's logits into a token.
    pub params: SamplingParams,
}

/// One prefill row in a [`ForwardPlan`].
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct PrefillRow {
    /// Engine slot that owns the request state.
    pub slot: usize,
    /// Token ids for the prompt chunk assigned to this row.
    pub tokens: Vec<u32>,
    /// Logical starting position for `tokens` in this slot.
    pub start_pos: usize,
    /// Total logical token count for the request after this prefill chunk.
    pub total_tokens: usize,
    /// Sampling parameters used when the final chunk produces the first token.
    pub params: SamplingParams,
}

/// Minimal speculative-decode plan placeholder.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct SpecPlan {
    /// Draft decode rows associated with this speculative step.
    pub draft_rows: Vec<DecodeRow>,
}

/// Backend-independent forward plan produced by engine-core.
///
/// This is the engine-core to executor bridge. It only names slots, host token
/// ids, logical positions, and scheduler metadata. Device tensors and buffers
/// stay behind the executor/model implementation boundary.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct ForwardPlan {
    /// Execution mode for this step.
    pub mode: ForwardMode,
    /// Decode rows scheduled for this step.
    pub decode_rows: Vec<DecodeRow>,
    /// Prefill rows scheduled for this step.
    pub prefill_rows: Vec<PrefillRow>,
    /// Optional pipeline-parallel microbatch identifier.
    pub microbatch: Option<u32>,
    /// Optional speculative-decode metadata.
    pub spec: Option<SpecPlan>,
}

impl ForwardPlan {
    /// Build an empty plan for a scheduler tick with no executor work.
    #[must_use]
    pub fn idle() -> Self {
        Self {
            mode: ForwardMode::Idle,
            decode_rows: Vec::new(),
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        }
    }

    /// Return whether this plan is explicitly an idle plan.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        matches!(self.mode, ForwardMode::Idle)
    }
}

/// Reason a slot stopped producing tokens.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub enum FinishReason {
    /// A configured stop token or stop condition fired.
    Stop,
    /// The generation length limit was reached.
    Length,
    /// The request was aborted before natural completion.
    Abort,
}

/// One sampled token returned from an executor step.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct SlotToken {
    /// Engine slot that owns this output token.
    pub slot: usize,
    /// Sampled token id.
    pub token: u32,
    /// Optional log probability for the sampled token.
    pub logprob: Option<f32>,
    /// Optional finish reason when this token terminates the slot.
    pub finish: Option<FinishReason>,
}

/// Host-visible output from one executor step.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// Tokens sampled by this step, keyed by engine slot.
    pub tokens: Vec<SlotToken>,
}

/// Parameters controlling token sampling from a logits distribution.
///
/// This is the pure-data subset of the existing runtime sampling contract.
/// Penalty and filtering kernels live below the executor seam.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Sampling temperature. `0.0` requests greedy argmax decoding.
    pub temperature: f32,
    /// Top-K filter. `-1` disables the filter.
    pub top_k: i32,
    /// Top-P nucleus threshold. `1.0` disables the filter.
    pub top_p: f32,
    /// Min-P threshold relative to the maximum probability. `0.0` disables it.
    pub min_p: f32,
    /// Repetition penalty for previously generated token ids.
    pub repetition_penalty: f32,
    /// OpenAI-style frequency penalty.
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty.
    pub presence_penalty: f32,
    /// Whether EOS should be ignored as a stopping condition.
    pub ignore_eos: bool,
    /// Additional token ids that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Optional deterministic sampling seed.
    pub seed: Option<u64>,
    /// Optional override for the number of newly generated tokens.
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
            stop_token_ids: Vec::new(),
            seed: None,
            max_new_tokens: None,
        }
    }
}

impl SamplingParams {
    /// Whether this configuration decodes greedily (argmax, no randomness).
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// Index of the maximum logit (greedy / argmax). Ties resolve to the lowest index.
#[must_use]
pub fn argmax_logit(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// SplitMix64 — a tiny dependency-free mixer turning a seed into a u64 stream.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample one token id from a logits row under `params`.
///
/// Greedy (`temperature <= 0`) returns `argmax_logit` — bit-identical to the
/// backends' device argmax, so it preserves greedy parity. For `temperature > 0`
/// it applies temperature scaling, then optional top-k / top-p (nucleus) / min-p
/// filtering, then a multinomial draw. Randomness is derived deterministically
/// from `(params.seed, position)` so a run is reproducible and a discrete poll
/// needs no per-slot RNG state. Pure and host-side: one logits copy at c=1 is
/// sub-millisecond, so no new GPU sampling kernel is required.
#[must_use]
pub fn sample_token(logits: &[f32], params: &SamplingParams, position: u64) -> u32 {
    if params.is_greedy() || logits.is_empty() {
        return argmax_logit(logits);
    }

    // Temperature-scaled, numerically stable softmax over all candidates.
    let inv_t = 1.0 / params.temperature;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, (((l - max) * inv_t) as f32).exp()))
        .collect();
    let sum: f32 = cand.iter().map(|(_, p)| *p).sum();
    if sum > 0.0 {
        for c in &mut cand {
            c.1 /= sum;
        }
    }

    // Descending by probability for top-k / top-p truncation.
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    if params.top_k > 0 && (params.top_k as usize) < cand.len() {
        cand.truncate(params.top_k as usize);
    }
    if params.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = cand.len();
        for (i, (_, p)) in cand.iter().enumerate() {
            cum += *p;
            if cum >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut.max(1));
    }
    if params.min_p > 0.0 {
        let top = cand.first().map_or(0.0, |(_, p)| *p);
        let thresh = params.min_p * top;
        cand.retain(|(_, p)| *p >= thresh);
    }

    // Multinomial draw over the surviving candidates.
    let total: f32 = cand.iter().map(|(_, p)| *p).sum();
    let bits = splitmix64(
        params
            .seed
            .unwrap_or(0)
            .wrapping_add(position)
            .wrapping_add(1),
    );
    let unit = (bits >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
    let mut acc = 0.0;
    let target = unit * total;
    for (idx, p) in &cand {
        acc += *p;
        if target < acc {
            return *idx;
        }
    }
    cand.last().map_or(0, |(idx, _)| *idx)
}

#[cfg(test)]
mod sampler_tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.1).collect()
    }

    #[test]
    fn greedy_equals_argmax() {
        let logits = vec![0.2, 5.0, 1.0, 4.9];
        let p = SamplingParams::default(); // temperature 0.0
        assert!(p.is_greedy());
        assert_eq!(sample_token(&logits, &p, 0), 1);
        assert_eq!(argmax_logit(&logits), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = ramp(50);
        let p = SamplingParams {
            temperature: 1.5,
            top_k: 1,
            ..SamplingParams::default()
        };
        // Only the max survives, so every position samples it.
        for pos in 0..16 {
            assert_eq!(sample_token(&logits, &p, pos), 49);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic_and_varies_by_position() {
        let logits = ramp(64);
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(42),
            ..SamplingParams::default()
        };
        // Reproducible for a fixed (seed, position).
        assert_eq!(sample_token(&logits, &p, 3), sample_token(&logits, &p, 3));
        // Advancing position changes the stream (not stuck on one token).
        let draws: Vec<u32> = (0..32).map(|pos| sample_token(&logits, &p, pos)).collect();
        assert!(draws.iter().any(|&t| t != draws[0]));
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // One dominant logit: nucleus collapses to it regardless of temperature.
        let mut logits = vec![0.0; 100];
        logits[7] = 50.0;
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            seed: Some(1),
            ..SamplingParams::default()
        };
        assert_eq!(sample_token(&logits, &p, 0), 7);
    }
}
