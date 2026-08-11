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

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub enum ForwardMode {
    Prefill,
    Decode,
    /// Decode rows and prefill rows share the same executor step.
    Mixed,
    Idle,
    /// Target-model verification for speculative decoding.
    TargetVerify,
    /// Draft-model extension for speculative decoding.
    DraftExtend,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct DecodeRow {
    pub slot: usize,
    pub last_token: u32,
    /// Logical KV sequence length already present for this slot.
    pub kv_seq_len: usize,
    pub params: SamplingParams,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct PrefillRow {
    pub slot: usize,
    pub tokens: Vec<u32>,
    /// Logical starting position for `tokens` in this slot.
    pub start_pos: usize,
    /// Total logical token count for the request after this prefill chunk.
    pub total_tokens: usize,
    pub params: SamplingParams,
}

/// Minimal speculative-decode plan placeholder.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct SpecPlan {
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
    pub mode: ForwardMode,
    pub decode_rows: Vec<DecodeRow>,
    pub prefill_rows: Vec<PrefillRow>,
    /// Optional pipeline-parallel microbatch identifier.
    pub microbatch: Option<u32>,
    pub spec: Option<SpecPlan>,
}

impl ForwardPlan {
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

    #[must_use]
    pub fn is_idle(&self) -> bool {
        matches!(self.mode, ForwardMode::Idle)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// A configured stop token or stop condition fired.
    Stop,
    Length,
    /// The request was aborted before natural completion.
    Abort,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct SlotToken {
    pub slot: usize,
    pub token: u32,
    pub logprob: Option<f32>,
    /// Set when this token terminates the slot.
    pub finish: Option<FinishReason>,
}

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
    /// `0.0` requests greedy argmax decoding.
    pub temperature: f32,
    /// `-1` disables the filter.
    pub top_k: i32,
    /// `1.0` disables the filter.
    pub top_p: f32,
    /// Threshold relative to the maximum probability. `0.0` disables it.
    pub min_p: f32,
    pub repetition_penalty: f32,
    /// OpenAI-style frequency penalty.
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty.
    pub presence_penalty: f32,
    pub ignore_eos: bool,
    pub stop_token_ids: Vec<u32>,
    pub seed: Option<u64>,
    pub max_new_tokens: Option<usize>,
    /// xgrammar next-token bitmask: bit set = allowed. Refreshed after every
    /// accepted token.
    pub grammar_bitmask: Option<std::sync::Arc<[u32]>>,
    /// Token-id → logit bias added before sampling.
    pub logit_bias: std::collections::HashMap<u32, f32>,
    /// Engine supports one; values > 1 are handled by the API layer.
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
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}
