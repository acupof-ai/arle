//! Public types mirroring the legacy `infer::server_engine` contract.
//!
//! These are re-exported from the crate root so a consumer that imports
//! `infer_api::{CompletionRequest, CompletionOutput, ...}` sees the SAME shape
//! it imports today from `infer::server_engine::{...}`. Where the rewrite stack
//! already ships a field-identical type (`SamplingParams`), it is re-exported
//! directly instead of duplicated.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

// `SamplingParams` in `infer-plan` is field-for-field identical to the legacy
// `infer::sampler::SamplingParams` (temperature, top_k, top_p, min_p,
// repetition_penalty, frequency_penalty, presence_penalty, ignore_eos,
// stop_token_ids, seed, max_new_tokens). Re-export it so the public surface
// converges on one canonical sampling contract.
pub use infer_plan::SamplingParams;

/// Sticky-routing / prefix-cache affinity key.
///
/// Mirrors `infer::types::SessionId` (a cheap `Arc<str>` newtype). Carried on
/// [`CompletionRequest::session_id`] so the swap from `infer` -> `infer-api`
/// needs no caller change; the rewrite engine does not yet consume it
/// (host-side sticky routing is a follow-up), so it is currently advisory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(std::sync::Arc<str>);

impl SessionId {
    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SessionId {
    fn from(value: &str) -> Self {
        Self(std::sync::Arc::from(value))
    }
}

impl From<String> for SessionId {
    fn from(value: String) -> Self {
        Self(std::sync::Arc::from(value.as_str()))
    }
}

/// A single completion request.
///
/// Field-for-field compatible with `infer::server_engine::CompletionRequest`.
/// `session_id` / `trace_context` / `cancel` are accepted but only partially
/// honored by the rewrite stack today (see [`super::LoadedInferenceEngine`]
/// doc gaps): a caller passing `None` (every current consumer) works unchanged.
#[derive(Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
    /// Return per-token log-probabilities (greedy sampling only).
    pub logprobs: bool,
    /// Optional client-supplied session identifier for sticky routing / prefix
    /// affinity. Advisory until the rewrite engine wires host-side sticky
    /// routing.
    pub session_id: Option<SessionId>,
    /// Parent tracing context to attach to the engine-side request. Carried for
    /// surface-compatibility; the rewrite engine does not yet propagate it.
    pub trace_context: Option<fastrace::collector::SpanContext>,
    /// Cooperative cancel flag. The blocking `complete()` path observes this on
    /// finish; full mid-generation cancellation is a follow-up (the rewrite
    /// `ServeHandle` exposes blocking `collect` only).
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Why generation stopped.
///
/// The rewrite `infer_plan::FinishReason` adds an `Abort` variant; this 2-state
/// enum is the legacy public shape. [`from_plan`](FinishReason::from_plan) maps
/// `Abort` -> `Stop` so the public surface stays binary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    /// Render to the OpenAI wire string.
    #[must_use]
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }

    /// Map a rewrite-stack `infer_plan::FinishReason` into this binary shape.
    /// `Abort` collapses to `Stop` (the legacy enum has no abort variant).
    #[must_use]
    pub fn from_plan(reason: &infer_plan::FinishReason) -> Self {
        match reason {
            infer_plan::FinishReason::Length => Self::Length,
            infer_plan::FinishReason::Stop | infer_plan::FinishReason::Abort => Self::Stop,
        }
    }
}

/// A completed (non-streaming) generation result.
///
/// Field-for-field compatible with `infer::server_engine::CompletionOutput`.
pub struct CompletionOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    /// Per-token log-probabilities (greedy only). Empty if not requested or the
    /// backend does not surface them (the rewrite stack does not yet).
    pub token_logprobs: Vec<f32>,
    /// Tokenized prompt the engine actually saw.
    pub prompt_token_ids: Vec<u32>,
    /// Generated token ids.
    pub response_token_ids: Vec<u32>,
}

/// Prompt / completion / total token accounting.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// One streamed delta.
///
/// Field-for-field compatible with `infer::server_engine::CompletionStreamDelta`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionStreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    /// Log-probability of the generated token (greedy only).
    pub logprob: Option<f32>,
    /// Token ids newly emitted in this delta.
    pub token_ids: Vec<u32>,
    /// Terminal failure, if the request failed before a normal finish delta.
    pub error: Option<CompletionStreamError>,
}

impl CompletionStreamDelta {
    /// Create a text-only delta (no finish, no usage, no error).
    #[must_use]
    pub fn text(s: String) -> Self {
        Self {
            text_delta: s,
            finish_reason: None,
            usage: None,
            logprob: None,
            token_ids: Vec::new(),
            error: None,
        }
    }

    /// Create a terminal error delta.
    #[must_use]
    pub fn error(kind: impl Into<String>, chain: Vec<String>) -> Self {
        Self {
            text_delta: String::new(),
            finish_reason: None,
            usage: None,
            logprob: None,
            token_ids: Vec::new(),
            error: Some(CompletionStreamError::from_chain(kind, chain)),
        }
    }
}

/// A terminal inference/scheduler failure attached to a stream delta.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionStreamError {
    pub kind: String,
    pub message: String,
    pub chain: Vec<String>,
}

impl CompletionStreamError {
    /// Build from an error-cause chain (most-recent first).
    #[must_use]
    pub fn from_chain(kind: impl Into<String>, chain: Vec<String>) -> Self {
        let message = chain
            .first()
            .cloned()
            .unwrap_or_else(|| "inference request failed".to_string());
        Self {
            kind: kind.into(),
            message,
            chain,
        }
    }

    /// Flatten into an `anyhow::Error`.
    #[must_use]
    pub fn into_anyhow(self) -> anyhow::Error {
        let chain = if self.chain.is_empty() {
            self.message
        } else {
            self.chain.join(": ")
        };
        anyhow!("{}: {}", self.kind, chain)
    }
}

/// Mixed decode+prefill path counters (telemetry sub-shape).
#[derive(Clone, Debug, Default, Serialize)]
pub struct PrefillPathStats {
    pub ok_true_count: u64,
    pub ok_false_count: u64,
    pub ok_false_reasons: HashMap<String, u64>,
}

/// Backend-agnostic engine-level telemetry snapshot.
///
/// Field-for-field compatible with `infer::server_engine::EngineTelemetry`
/// minus the `model_arch` field (legacy `ModelArchSummary` is an infer-internal
/// type the rewrite stack does not yet model; see the doc gaps in
/// [`super::LoadedInferenceEngine`]). The rewrite `ServeHandle` does not surface
/// any of these counters today, so the adapter returns the default (empty)
/// shape — callers must treat empty as "unavailable", never as zero.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EngineTelemetry {
    pub ttft_us: Option<f64>,
    pub itl_p50_us: Option<f64>,
    pub itl_p99_us: Option<f64>,
    pub queue_depth: u32,
    pub active_requests: u32,
    pub batch_occupancy: f64,
    pub kv_tier_hit_rates: HashMap<String, f64>,
    pub spec_acceptance_rate: Option<f64>,
    pub prefill_path_stats: PrefillPathStats,
    pub timestamp_ms: u64,
}

/// The public inference contract.
///
/// Identical in signature to `infer::server_engine::InferenceEngine`, so a
/// consumer's `&mut dyn InferenceEngine` / `impl InferenceEngine` bounds compile
/// unchanged after the `infer` -> `infer-api` swap.
pub trait InferenceEngine: Send {
    /// The model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    /// Run a complete generation synchronously and return the full output.
    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput>;

    /// Run a generation, streaming token deltas through `tx` as produced.
    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()>;

    /// Encode `text` to token ids with the backend's loaded tokenizer.
    ///
    /// The default impl errors so the trait stays object-safe; backends with a
    /// tokenizer override it. Callers must treat `Err(_)` as "tokenize
    /// unavailable" and downgrade `tokens` to `None`, never substitute an empty
    /// `Vec`.
    fn tokenize(&self, _text: &str) -> Result<Vec<u32>> {
        Err(anyhow!("backend does not expose tokenize()"))
    }

    /// Backend-agnostic engine-level telemetry snapshot.
    fn telemetry(&self) -> EngineTelemetry {
        EngineTelemetry::default()
    }
}
