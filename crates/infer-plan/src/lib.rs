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
    entropy_bound_acceptance_mask, generate_diffusion_with_cancel,
};
pub use sample::{
    PenaltyHistory, argmax_logit, sample_token, sample_token_logprob,
    sample_token_logprob_penalized, sample_token_penalized, sampled_top_logprobs,
};

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub enum ForwardMode {
    Prefill,
    Decode,
    /// Decode rows and prefill rows share the same executor step.
    Mixed,
    Idle,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct DecodeRow {
    pub slot: usize,
    pub last_token: u32,
    /// Logical KV sequence length already present for this slot.
    pub kv_seq_len: usize,
    pub params: SamplingParams,
    /// `prompt ++ generated` snapshot, set only when [`SamplingParams::has_penalty`].
    pub penalty_history: Option<std::sync::Arc<[u32]>>,
    /// Split point in `penalty_history`: repetition scores the whole slice,
    /// frequency/presence only the generated tail.
    pub penalty_prompt_len: usize,
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
    /// `prompt ++ generated` snapshot, set only when [`SamplingParams::has_penalty`].
    pub penalty_history: Option<std::sync::Arc<[u32]>>,
    /// Split point in `penalty_history`: repetition scores the whole slice,
    /// frequency/presence only the generated tail.
    pub penalty_prompt_len: usize,
}

impl PrefillRow {
    #[must_use]
    pub fn end_pos(&self) -> usize {
        self.start_pos + self.tokens.len()
    }

    /// One predicate for every backend: the open-coded form was written both as
    /// `==` and as `>=`, so a chunk that overshot `total_tokens` counted as final
    /// on some paths and not on others.
    #[must_use]
    pub fn is_final_chunk(&self) -> bool {
        self.end_pos() >= self.total_tokens
    }
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
}

impl ForwardPlan {
    #[must_use]
    pub fn idle() -> Self {
        Self {
            mode: ForwardMode::Idle,
            decode_rows: Vec::new(),
            prefill_rows: Vec::new(),
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
    /// OpenAI logprobs capture, present only when [`SamplingParams::top_logprobs`]
    /// asked for it: entry 0 = the sampled token's logprob under the full
    /// (rewritten, temperature-scaled) softmax; entries 1.. = the top-N
    /// alternatives, probability-descending.
    #[cfg_attr(feature = "serde", serde(default))]
    pub top_logprobs: Vec<(u32, f32)>,
    /// Set when this token terminates the slot.
    pub finish: Option<FinishReason>,
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// Tokens sampled by this step, keyed by engine slot.
    pub tokens: Vec<SlotToken>,
}

/// Pure-data subset of the runtime sampling contract; penalty and filtering
/// kernels live below the executor seam.
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
    /// A map would serialize its keys as JSON strings, and the multiproc relay
    /// then fails to read them back as `u32` — one biased request killed every
    /// TP>1 worker. Pairs, sorted by token id, survive the round trip.
    pub logit_bias: Vec<(u32, f32)>,
    /// Engine supports one; values > 1 are handled by the API layer.
    pub n: usize,
    /// `Some(n)`: capture OpenAI-style logprobs — the sampled token's logprob
    /// under the full softmax plus the top-n alternatives (`n` may be 0:
    /// sampled-token logprob only). Vetoes the device-argmax fast path (the
    /// capture needs host logits) and speculative decode for this request.
    #[cfg_attr(feature = "serde", serde(default))]
    pub top_logprobs: Option<usize>,
    /// When set, the sampler returns this token directly, bypassing all
    /// sampling. Set by the engine to force think-end after a reasoning
    /// budget; cleared after one use.
    #[cfg_attr(feature = "serde", serde(default))]
    pub force_next_token: Option<u32>,
    /// Token that ends a thinking block. When set with `max_thinking_tokens`,
    /// the engine tracks reasoning tokens and forces this token after the budget.
    #[cfg_attr(feature = "serde", serde(default))]
    pub think_end_token_id: Option<u32>,
    /// Token that starts a thinking block (multi-segment thinking re-entry).
    #[cfg_attr(feature = "serde", serde(default))]
    pub think_start_token_id: Option<u32>,
    /// Max reasoning tokens before forced think-end. `None` = no budget.
    #[cfg_attr(feature = "serde", serde(default))]
    pub max_thinking_tokens: Option<usize>,
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
            logit_bias: Vec::new(),
            n: 1,
            top_logprobs: None,
            force_next_token: None,
            think_end_token_id: None,
            think_start_token_id: None,
            max_thinking_tokens: None,
        }
    }
}

impl SamplingParams {
    /// Sampling POLICY only. To decide whether a backend may emit a device
    /// argmax over raw logits, ask [`Self::is_raw_argmax`].
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Any of the three penalties is off its no-op value, so the request needs
    /// a token history. Single source of truth: engine-core populates
    /// `penalty_history` on exactly these requests, and [`Self::is_raw_argmax`]
    /// vetoes exactly these requests.
    #[must_use]
    pub fn has_penalty(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// The token equals `argmax` over the model's raw logits, so a backend may
    /// skip [`crate::sample_token`]. Greedy is necessary but not sufficient:
    /// `grammar_bitmask` and `logit_bias` rewrite the logits first.
    ///
    /// The destructure is load-bearing — a new field breaks this at compile
    /// time, forcing the author to say whether it rewrites logits. `logit_bias`
    /// silently kept the fast path until 2026-08-13.
    #[must_use]
    pub fn is_raw_argmax(&self) -> bool {
        let Self {
            temperature,
            // Inert at temperature 0.
            top_k: _,
            top_p: _,
            min_p: _,
            seed: _,
            // Downstream of token choice.
            ignore_eos: _,
            stop_token_ids: _,
            max_new_tokens: _,
            n: _,
            // Rewrite the logits — must reach the argmax.
            grammar_bitmask,
            logit_bias,
            // Also rewrite the logits; read through `has_penalty` so the veto
            // and engine-core's history decision cannot drift apart.
            repetition_penalty: _,
            frequency_penalty: _,
            presence_penalty: _,
            // The capture reads host logits, so the request must take the
            // host sampling path.
            top_logprobs,
            // Bypasses sampling entirely.
            force_next_token,
            // Lifecycle config, not logit rewriting.
            think_end_token_id: _,
            think_start_token_id: _,
            max_thinking_tokens: _,
        } = self;
        *temperature <= 0.0
            && grammar_bitmask.is_none()
            && logit_bias.is_empty()
            && top_logprobs.is_none()
            && force_next_token.is_none()
            && !self.has_penalty()
    }
}
