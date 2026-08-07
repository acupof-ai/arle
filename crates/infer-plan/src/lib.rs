//! Backend-independent inference planning data.
//!
//! This crate is the pure data layer shared by engine-core and executor
//! implementations. It intentionally carries no device tensor, stream, graph,
//! or backend runtime types.

mod diffusion;
mod sample;

pub use diffusion::{
    DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerateError,
    DiffusionGenerateOutput, DiffusionGenerateStats, DiffusionGenerationConfig,
    DiffusionModelError, DiffusionStepTrace, MultimodalImage, MultimodalKind,
    diffusion_prediction_from_logits, entropy_bound_acceptance_mask, generate_diffusion,
    generate_diffusion_with_cancel,
};
pub use sample::{argmax_logit, merge_vocab_shard_argmax, sample_token, sample_token_logprob};

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Per-step grammar constraint (xgrammar next-token bitmask, bit set =
    /// allowed). Host-side and backend-neutral; the engine refreshes it after
    /// every accepted token.
    pub grammar_bitmask: Option<std::sync::Arc<[u32]>>,
    /// Token-id → logit bias added before sampling.
    pub logit_bias: std::collections::HashMap<u32, f32>,
    /// Number of completions to generate. Engine supports one; values > 1 are
    /// handled by the API layer (re-run with offset seeds).
    pub n: usize,
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
            grammar_bitmask: None,
            logit_bias: std::collections::HashMap::new(),
            n: 1,
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
