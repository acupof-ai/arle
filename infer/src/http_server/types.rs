//! HTTP server types: config, request/response containers, identity.
//!
//! Split out of `http_server.rs` (pure structural refactor — no behavior change).

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::openai_v1::{
    ChatCompletionRequest, CompletionRequest as OpenAiCompletionRequest, ResponsesRequest,
};
use crate::metrics::ServerMetrics;
use crate::request_handle::RequestHandle;
use crate::sampler::{SamplingParams, sampling_params_from_request};
use crate::scheduler::{IncomingRequest, RequestPriority};
use crate::server_engine::{CompletionOutput, CompletionStreamDelta, FinishReason, TokenUsage};
use crate::tokenizer::Tokenizer;
use fastrace::collector::SpanContext;

/// Maximum wall-clock time allowed for a non-streaming request to complete.
/// Streaming responses have natural per-chunk flow control and are not capped here.
pub(super) const RESPONSE_TIMEOUT: Duration = Duration::from_mins(5);
pub(in crate::http_server) const HTTP_REQUEST_BODY_LIMIT_BYTES: usize = 16 * 1024 * 1024;
pub(super) const HTTP_REQUEST_ID_HEADER: &str = "x-request-id";

pub(super) struct AppState {
    pub(super) handle: Arc<dyn RequestHandle>,
    pub(super) tokenizer: Option<Tokenizer>,
    pub(super) identity: ServingIdentity,
    pub(super) metrics: ServerMetrics,
    pub(super) config: HttpServerConfig,
}

/// Boot-time serving identity captured once when the router is built.
///
/// `RequestHandle` remains the submission path; this snapshot owns the
/// served model metadata that HTTP responses need on every request.
#[derive(Clone, Debug)]
pub(super) struct ServingIdentity {
    pub(super) model_id: String,
    pub(super) dflash_status: Option<crate::request_handle::DflashStatus>,
}

#[derive(Clone, Debug, Default)]
pub struct HttpServerConfig {
    pub api_key: Option<Arc<str>>,
    pub train_control_target: Option<TrainControlTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainControlTarget {
    authority: Arc<str>,
    base_path: Arc<str>,
}

pub(super) fn normalize_train_control_base_path(path: &str) -> String {
    if path.is_empty() || path == "/" {
        return "/".to_string();
    }
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

impl TrainControlTarget {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let uri = raw
            .parse::<axum::http::Uri>()
            .map_err(|err| format!("invalid train control URL '{raw}': {err}"))?;
        if uri.scheme_str() != Some("http") {
            return Err(format!("train control URL must use http://, got '{raw}'"));
        }
        if uri.query().is_some() {
            return Err(format!(
                "train control URL must not include a query string: '{raw}'"
            ));
        }
        let authority = uri
            .authority()
            .ok_or_else(|| format!("train control URL is missing host: '{raw}'"))?
            .as_str();
        let base_path = normalize_train_control_base_path(uri.path());
        Ok(Self {
            authority: Arc::<str>::from(authority),
            base_path: Arc::<str>::from(base_path),
        })
    }

    pub(super) fn request_path(&self, route_suffix: &str, query: Option<&str>) -> String {
        let mut path = String::new();
        if self.base_path.as_ref() != "/" {
            path.push_str(self.base_path.as_ref());
        }
        path.push_str(route_suffix);
        if let Some(query) = query.filter(|value| !value.is_empty()) {
            path.push('?');
            path.push_str(query);
        }
        path
    }

    pub(super) fn authority(&self) -> &str {
        self.authority.as_ref()
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct TrainEventsQuery {
    pub(super) after_seq: Option<u64>,
}

#[derive(Debug)]
pub(super) struct ProxiedTrainResponse {
    pub(super) status: axum::http::StatusCode,
    pub(super) body: Vec<u8>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct HealthResponse {
    status: String,
    service: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

impl HealthResponse {
    pub(super) fn live() -> Self {
        Self {
            status: "ok".to_string(),
            service: "agent-infer".to_string(),
            model: None,
        }
    }

    pub(super) fn ready(model_id: &str) -> Self {
        Self {
            status: "ready".to_string(),
            service: "agent-infer".to_string(),
            model: Some(model_id.to_string()),
        }
    }
}

pub(super) struct RequestExecutionOptions {
    pub(super) max_tokens: usize,
    pub(super) stream: bool,
    pub(super) include_usage: bool,
    pub(super) continuous_usage_stats: bool,
    pub(super) sampling: SamplingParams,
    pub(super) stop: Option<Vec<String>>,
    pub(super) session_id: Option<crate::types::SessionId>,
}

impl RequestExecutionOptions {
    pub(super) fn from_completion(req: &OpenAiCompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: req.include_usage_or_default(),
            continuous_usage_stats: req.continuous_usage_stats_or_default(),
            sampling: sampling_params_from_request(
                req.temperature,
                req.top_p,
                req.top_k,
                req.min_p,
                req.repetition_penalty,
                req.frequency_penalty,
                req.presence_penalty,
                req.ignore_eos,
                req.seed,
                req.stop_token_ids.clone(),
            ),
            stop: req.stop.clone(),
            session_id: req.session_id_parsed(),
        }
    }

    pub(super) fn from_chat(req: &ChatCompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: req.include_usage_or_default(),
            continuous_usage_stats: req.continuous_usage_stats_or_default(),
            sampling: sampling_params_from_request(
                req.temperature,
                req.top_p,
                req.top_k,
                req.min_p,
                req.repetition_penalty,
                req.frequency_penalty,
                req.presence_penalty,
                req.ignore_eos,
                req.seed,
                req.stop_token_ids.clone(),
            ),
            stop: req.stop.clone(),
            session_id: req.session_id_parsed(),
        }
    }

    pub(super) fn from_responses(req: &ResponsesRequest) -> Self {
        Self {
            max_tokens: req.max_output_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: false,
            continuous_usage_stats: false,
            sampling: sampling_params_from_request(
                req.temperature,
                req.top_p,
                req.top_k,
                req.min_p,
                req.repetition_penalty,
                req.frequency_penalty,
                req.presence_penalty,
                req.ignore_eos,
                req.seed,
                req.stop_token_ids.clone(),
            ),
            stop: req.stop.clone(),
            session_id: req.session_id_parsed(),
        }
    }

    pub(super) fn into_incoming_request(
        self,
        prompt: String,
        prompt_tokens: Option<Vec<u32>>,
        delta_tx: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
        trace_context: Option<SpanContext>,
    ) -> IncomingRequest {
        IncomingRequest {
            prompt,
            prompt_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            stop: self.stop,
            priority: RequestPriority::default(),
            session_id: self.session_id,
            delta_tx,
            trace_context,
        }
    }
}

pub(super) struct BufferedResponse {
    pub(super) text: String,
    pub(super) finish_reason: FinishReason,
    pub(super) usage: TokenUsage,
    pub(super) token_logprobs: Vec<f32>,
    pub(super) response_token_ids: Vec<u32>,
    /// `true` once a delta with `finish_reason: Some(_)` has been observed.
    ///
    /// Distinguishes a clean scheduler-side completion from an aborted request
    /// (e.g. prefill OOM → `EmitCommand::Abort` → `delta_tx` dropped without a
    /// terminal delta). The non-streaming HTTP handlers use this to return a
    /// 503 instead of an empty 200 when the server gave up on the request.
    pub(super) terminal_seen: bool,
}

impl Default for BufferedResponse {
    fn default() -> Self {
        Self {
            text: String::new(),
            finish_reason: FinishReason::Length,
            usage: TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            token_logprobs: Vec::new(),
            response_token_ids: Vec::new(),
            terminal_seen: false,
        }
    }
}

impl BufferedResponse {
    pub(super) fn apply_delta(&mut self, delta: &CompletionStreamDelta) {
        self.text.push_str(&delta.text_delta);
        if let Some(reason) = delta.finish_reason {
            self.finish_reason = reason;
            self.terminal_seen = true;
        }
        if let Some(usage) = delta.usage {
            self.usage = usage;
        }
        if let Some(lp) = delta.logprob {
            self.token_logprobs.push(lp);
        }
        if !delta.token_ids.is_empty() {
            self.response_token_ids.extend(delta.token_ids.iter());
        }
    }

    pub(super) fn into_output(self) -> CompletionOutput {
        CompletionOutput {
            text: self.text,
            finish_reason: self.finish_reason,
            usage: self.usage,
            token_logprobs: self.token_logprobs,
            // The HTTP non-streaming path doesn't tokenize the prompt
            // here — the OpenAI-compat response shape doesn't expose
            // token IDs to the client anyway. Phase 2 trajectory export
            // runs through the agent loop, which calls `engine.tokenize`
            // directly.
            prompt_token_ids: Vec::new(),
            response_token_ids: self.response_token_ids,
        }
    }
}
