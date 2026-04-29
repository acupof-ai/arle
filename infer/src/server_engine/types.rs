use anyhow::{Result, anyhow};
use tokio::sync::mpsc::UnboundedSender;

use crate::sampler::SamplingParams;

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
}
