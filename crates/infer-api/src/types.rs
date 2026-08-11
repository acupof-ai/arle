//! Public types mirroring the legacy `infer::server_engine` contract, so a
//! consumer importing `infer_api::{...}` sees the same shape it imports from
//! `infer::server_engine::{...}` today. `SamplingParams` is re-exported from
//! `infer-plan` rather than duplicated.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub use infer_plan::SamplingParams;

#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceVec};
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtr;

/// Raw `[seq_len, vocab]` logits produced by the CUDA OPD-teacher forward,
/// carried back to `train` without sampling.
///
/// Mirrors the legacy `infer::server_engine::RawLogits` shape so `train` can swap
/// `infer` -> `infer-api` with no code change. `logits` is a row-major
/// `[seq_len, vocab]` device buffer; `device` is the model's context, needed to
/// sync / consume the buffer (D2H or a D2D import into the train backend).
#[cfg(feature = "cuda")]
pub struct RawLogits {
    pub logits: DeviceVec,
    pub shape: [usize; 2],
    pub device: DeviceContext,
}

#[cfg(feature = "cuda")]
impl RawLogits {
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.shape[1]
    }

    pub fn to_host_f32(&self) -> Result<Vec<f32>> {
        self.logits.to_host(&self.device)
    }

    /// Run `f` with the raw device pointer (`u64`) of the logits buffer, e.g. for
    /// a D2D import into the train backend. The pointer is valid only for the
    /// duration of `f`.
    pub fn with_logits_device_ptr<T>(&self, f: impl FnOnce(u64) -> T) -> T {
        let (ptr, _guard) = self.logits.data.device_ptr(&self.device.stream);
        f(ptr)
    }
}

// SAFETY: `RawLogits` owns a CUDA allocation plus the context needed to consume
// it. It is produced on the engine thread and handed to a single OPD-teacher
// caller; callers must not share the contained mutable device allocation across
// threads.
#[cfg(feature = "cuda")]
unsafe impl Send for RawLogits {}

/// Sticky-routing / prefix-cache affinity key (an `Arc<str>` newtype). Carried
/// on [`CompletionRequest::session_id`]; advisory until the rewrite engine wires
/// host-side sticky routing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(std::sync::Arc<str>);

impl SessionId {
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

/// A single completion request (field-for-field compatible with the legacy
/// `CompletionRequest`). `session_id`/`trace_context`/`cancel` are accepted but
/// only partially honored (see the crate-level doc gaps).
#[derive(Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
    /// Return per-token log-probabilities (greedy sampling only).
    pub logprobs: bool,
    /// Session identifier for sticky routing / prefix affinity (advisory).
    pub session_id: Option<SessionId>,
    /// Parent tracing context (carried for compatibility; not yet propagated).
    pub trace_context: Option<fastrace::collector::SpanContext>,
    /// Cooperative cancel flag (observed on finish; mid-generation cancel is a
    /// follow-up).
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl CompletionRequest {
    /// Build a request with the two always-specified fields; all others default.
    #[must_use]
    pub fn new(prompt: impl Into<String>, max_tokens: usize) -> Self {
        Self {
            prompt: prompt.into(),
            max_tokens,
            sampling: SamplingParams::default(),
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
            cancel: None,
        }
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    #[must_use]
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<SessionId>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    #[must_use]
    pub fn with_logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = logprobs;
        self
    }
}

/// Raw image bytes attached to a backend-native chat message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatPromptImage {
    pub source: String,
    pub mime_type: Option<String>,
    pub data: Vec<u8>,
}

impl ChatPromptImage {
    #[must_use]
    pub fn new(source: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            source: source.into(),
            mime_type: None,
            data,
        }
    }

    #[must_use]
    pub fn with_mime_type(mut self, mime_type: impl Into<Option<String>>) -> Self {
        self.mime_type = mime_type.into();
        self
    }
}

/// A multimodal chat request for backends that expose image soft-token support.
#[derive(Debug)]
pub struct MultimodalChatRequest {
    pub messages: Vec<ChatPromptMessage>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
}

impl MultimodalChatRequest {
    #[must_use]
    pub fn new(messages: Vec<ChatPromptMessage>, max_tokens: usize) -> Self {
        Self {
            messages,
            max_tokens,
            sampling: SamplingParams::default(),
        }
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }
}

/// Why generation stopped (the legacy 2-state public shape;
/// [`from_plan`](FinishReason::from_plan) maps the rewrite's `Abort` -> `Stop`).
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    #[must_use]
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }

    /// Map a rewrite `infer_plan::FinishReason` into this binary shape
    /// (`Abort` -> `Stop`).
    #[must_use]
    pub fn from_plan(reason: &infer_plan::FinishReason) -> Self {
        match reason {
            infer_plan::FinishReason::Length => Self::Length,
            infer_plan::FinishReason::Stop | infer_plan::FinishReason::Abort => Self::Stop,
        }
    }
}

/// A completed (non-streaming) generation result (legacy `CompletionOutput`).
pub struct CompletionOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    /// Per-token log-probabilities (greedy only); always empty (not yet surfaced).
    pub token_logprobs: Vec<f32>,
    pub prompt_token_ids: Vec<u32>,
    pub response_token_ids: Vec<u32>,
}

/// Prompt / completion / total token accounting.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    /// Build usage with `total_tokens` set to `prompt_tokens + completion_tokens`.
    #[must_use]
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Backend-native chat rendering input.
///
/// This is intentionally smaller than the OpenAI wire type: CLI/agent callers
/// only need role + text content, while HTTP continues to own tools and other
/// request-shape details. Backends with checkpoint chat templates override
/// [`InferenceEngine::render_chat_prompt`]; the default remains ChatML for
/// legacy autoregressive paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatPromptMessage {
    pub role: String,
    pub content: String,
    pub images: Vec<ChatPromptImage>,
}

impl ChatPromptMessage {
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_images(mut self, images: Vec<ChatPromptImage>) -> Self {
        self.images = images;
        self
    }

    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    #[must_use]
    pub fn user_with_images(content: impl Into<String>, images: Vec<ChatPromptImage>) -> Self {
        Self::user(content).with_images(images)
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// One streamed delta (legacy `CompletionStreamDelta`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionStreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    /// Log-probability of the generated token (greedy only).
    pub logprob: Option<f32>,
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

/// Backend-agnostic engine-level telemetry snapshot (legacy `EngineTelemetry`
/// minus `model_arch`). The rewrite `ServeHandle` surfaces none of these today,
/// so the adapter returns the default — treat empty as "unavailable", not zero.
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

/// The public inference contract (signature-identical to the legacy
/// `InferenceEngine`, so consumer bounds compile unchanged after the swap).
pub trait InferenceEngine: Send {
    /// The model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput>;

    fn complete_multimodal_chat(
        &mut self,
        _req: MultimodalChatRequest,
    ) -> Result<CompletionOutput> {
        Err(anyhow!(
            "backend does not expose multimodal chat completion"
        ))
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()>;

    /// Encode `text` to token ids with the backend's tokenizer. The default
    /// errors (object-safety); treat `Err(_)` as "unavailable", never empty `Vec`.
    fn tokenize(&self, _text: &str) -> Result<Vec<u32>> {
        Err(anyhow!("backend does not expose tokenize()"))
    }

    /// Render chat messages using the backend's native chat template.
    ///
    /// The default keeps the historical Qwen-style ChatML prompt for engines
    /// that have not exposed a checkpoint template yet. Template-aware
    /// [`ServeInferenceEngine`](crate::ServeInferenceEngine) instances override
    /// this to use the tokenizer's checkpoint template.
    fn render_chat_prompt(&self, messages: &[ChatPromptMessage]) -> Result<String> {
        anyhow::ensure!(
            !messages.is_empty(),
            "messages must contain at least one message"
        );
        let mut out = String::new();
        for message in messages {
            let role = message.role.trim();
            anyhow::ensure!(!role.is_empty(), "message role must not be empty");
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(&message.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    }

    fn telemetry(&self) -> EngineTelemetry {
        EngineTelemetry::default()
    }
}
