use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

use crate::model_arch::ModelArchSummary;
use crate::sampler::SamplingParams;

#[derive(Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
    /// Return per-token log-probabilities (greedy sampling only).
    pub logprobs: bool,
    /// Optional client-supplied session identifier used for sticky routing /
    /// prefix-cache affinity. Forwarded onto `IncomingRequest::session_id`
    /// when this request is routed through a `RequestHandle`. CLI agent
    /// callers may populate this; otherwise leave `None`.
    pub session_id: Option<crate::types::SessionId>,
    /// Parent tracing context to attach to the scheduler-side request.
    /// Forwarded onto `IncomingRequest::trace_context`. `None` for
    /// non-traced callers.
    pub trace_context: Option<fastrace::collector::SpanContext>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub(crate) fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

pub struct CompletionOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    /// Per-token log-probabilities (greedy only). Empty if logprobs not requested.
    pub token_logprobs: Vec<f32>,
    /// Tokenized prompt the engine actually saw. Empty when the backend
    /// has not yet populated this field — callers must treat empty as
    /// "unavailable", not "zero tokens".
    pub prompt_token_ids: Vec<u32>,
    /// Generated token IDs (concatenation of every stream delta's
    /// `token_ids`). Redundant with the streaming channel but cheap and
    /// useful for non-streaming callers / RL trajectory export. Empty
    /// when the backend has not populated per-delta token IDs.
    pub response_token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct CompletionStreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    /// Log-probability of the generated token (greedy only, None otherwise).
    #[allow(dead_code)]
    pub logprob: Option<f32>,
    /// Token IDs newly emitted in this delta (Phase 2 trajectory token
    /// layer). Empty for backends that have not yet populated this — the
    /// agent loop treats an empty cumulative response as "unavailable"
    /// and surfaces `tokens = None` rather than fabricating partial data.
    pub token_ids: Vec<u32>,
}

impl CompletionStreamDelta {
    /// Create a text delta (no finish, no logprob, no token IDs).
    pub fn text(s: String) -> Self {
        Self {
            text_delta: s,
            finish_reason: None,
            usage: None,
            logprob: None,
            token_ids: Vec::new(),
        }
    }
}

/// Backend-agnostic snapshot of engine-level telemetry (`InferenceEngine::telemetry()`).
///
/// M1 unification surface (see `docs/plans/backend-unification.md` §M1):
/// CUDA and Metal both project from their respective scheduler-side metrics
/// into this struct so HTTP / bench / observability code can read one shape
/// regardless of which backend is loaded. Fields a backend cannot supply stay
/// `None` / `0` — callers must treat empty as "unavailable", never as zero.
///
/// `kv_tier_hit_rates` is keyed by tier label (`"T0"`, `"T1"`, `"T2"`, `"T3"`).
/// Backends without a particular tier omit that key rather than reporting `0.0`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EngineTelemetry {
    /// Time-to-first-token (microseconds, p50). `None` when no requests
    /// have completed since boot.
    pub ttft_us: Option<f64>,
    /// Inter-token latency p50 in microseconds (TPOT-style: time per
    /// output token after the first). `None` until at least one request
    /// generated >1 token.
    pub itl_p50_us: Option<f64>,
    /// Inter-token latency p99 in microseconds.
    pub itl_p99_us: Option<f64>,
    /// Requests currently waiting in the scheduler queue.
    pub queue_depth: u32,
    /// Requests currently active in scheduler slots.
    pub active_requests: u32,
    /// Fraction of allocated KV slots currently in use (0.0..=1.0).
    pub batch_occupancy: f64,
    /// Backend-neutral model architecture summary. `None` for legacy/mock
    /// engines that have not wired the M5 architecture contract.
    pub model_arch: Option<ModelArchSummary>,
    /// Per-tier hit rates keyed by `"T0"` / `"T1"` / `"T2"` / `"T3"`.
    pub kv_tier_hit_rates: HashMap<String, f64>,
    /// Aggregate speculative-decode acceptance rate (accepted / verified
    /// draft tokens), 0.0..=1.0. `None` until at least one verified
    /// speculation step has landed; matches the `infer_spec_acceptance_rate`
    /// Prometheus gauge.
    pub spec_acceptance_rate: Option<f64>,
    /// Wall-clock timestamp of the snapshot (millis since UNIX epoch).
    pub timestamp_ms: u64,
}

pub trait InferenceEngine: Send {
    /// Returns the model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    /// Run a complete generation request synchronously and return the full output.
    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput>;

    /// Run a generation request, streaming token deltas through `tx` as they are produced.
    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()>;

    /// Encode `text` to token IDs using whatever tokenizer the backend
    /// already loaded. The agent loop calls this to interleave tool
    /// results into the trajectory's `response_ids` (with mask=0) so an
    /// RL trainer can mask environment tokens out of the policy loss.
    ///
    /// The default impl errors so the trait stays object-safe and Phase
    /// 1 backends keep compiling untouched. Phase 2 backends override
    /// it. Callers must treat an `Err(_)` as "tokenize unavailable" and
    /// downgrade `tokens` to `None` per the trajectory contract — never
    /// substitute an empty Vec.
    fn tokenize(&self, _text: &str) -> Result<Vec<u32>> {
        Err(anyhow!("backend does not expose tokenize()"))
    }

    /// Backend-agnostic engine-level telemetry snapshot. Default returns
    /// the empty/zero shape so legacy backends keep compiling. Backends
    /// that drive a `SchedulerHandle` / `MetalSchedulerHandle` override
    /// this to project from their `ServerMetrics`.
    fn telemetry(&self) -> EngineTelemetry {
        EngineTelemetry::default()
    }
}
