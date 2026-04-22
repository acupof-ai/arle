#[allow(clippy::struct_field_names, clippy::needless_pass_by_value)]
mod openai_v1;
pub mod sessions;

use std::convert::Infallible;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::Request as AxumRequest;
use axum::extract::rejection::{BytesRejection, JsonRejection};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, Method, header},
    middleware,
    routing::{get, post},
};
use chat::openai_messages_to_prompt as chat_messages_to_prompt;
use fastrace::Span;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use fastrace::local::LocalSpan;
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use serde::Deserialize;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Maximum wall-clock time allowed for a non-streaming request to complete.
/// Streaming responses have natural per-chunk flow control and are not capped here.
const RESPONSE_TIMEOUT: Duration = Duration::from_mins(5);

use crate::error::ApiError;
use crate::metrics::ServerMetrics;
use crate::request_handle::RequestHandle;
use crate::sampler::{SamplingParams, sampling_params_from_request};
#[cfg(test)]
use crate::scheduler::SchedulerHandle;
use crate::scheduler::{IncomingRequest, RequestPriority};
use crate::server_engine::CompletionStreamDelta;
use crate::server_engine::{CompletionOutput, FinishReason, TokenUsage};
use crate::session_persistence::SessionPersistence;
use crate::tokenizer::Tokenizer;
use crate::trace_reporter::trace_runtime;
use openai_v1::{
    ChatCompletionRequest, ChatCompletionResponse, ChatStreamChunk, ChatStreamUsageChunk,
    CompletionRequest as OpenAiCompletionRequest, CompletionResponse, DflashStatusPayload,
    ModelsListResponse, ResponsesInput, ResponsesRequest, ResponsesResponse,
    ResponsesStreamCreatedEvent, ResponsesStreamDeltaEvent, StreamChunk, StreamUsageChunk,
};

struct AppState {
    handle: Arc<dyn RequestHandle>,
    tokenizer: Option<Tokenizer>,
    identity: ServingIdentity,
    metrics: ServerMetrics,
    config: HttpServerConfig,
}

/// Boot-time serving identity captured once when the router is built.
///
/// `RequestHandle` remains the submission path; this snapshot owns the
/// served model metadata that HTTP responses need on every request.
#[derive(Clone, Debug)]
struct ServingIdentity {
    model_id: String,
    dflash_status: Option<crate::request_handle::DflashStatus>,
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

fn normalize_train_control_base_path(path: &str) -> String {
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

    fn request_path(&self, route_suffix: &str, query: Option<&str>) -> String {
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

    fn authority(&self) -> &str {
        self.authority.as_ref()
    }
}

#[derive(Debug, Deserialize)]
struct TrainEventsQuery {
    after_seq: Option<u64>,
}

#[derive(Debug)]
struct ProxiedTrainResponse {
    status: axum::http::StatusCode,
    body: Vec<u8>,
}

struct RequestExecutionOptions {
    max_tokens: usize,
    stream: bool,
    include_usage: bool,
    sampling: SamplingParams,
    stop: Option<Vec<String>>,
    session_id: Option<crate::types::SessionId>,
}

impl RequestExecutionOptions {
    fn from_completion(req: &OpenAiCompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: req.include_usage_or_default(),
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

    fn from_chat(req: &ChatCompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: req.include_usage_or_default(),
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

    fn from_responses(req: &ResponsesRequest) -> Self {
        Self {
            max_tokens: req.max_output_tokens_or_default(),
            stream: req.stream_or_default(),
            include_usage: false,
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

    fn into_incoming_request(
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

struct BufferedResponse {
    text: String,
    finish_reason: FinishReason,
    usage: TokenUsage,
    token_logprobs: Vec<f32>,
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
        }
    }
}

impl BufferedResponse {
    fn apply_delta(&mut self, delta: &CompletionStreamDelta) {
        self.text.push_str(&delta.text_delta);
        if let Some(reason) = delta.finish_reason {
            self.finish_reason = reason;
        }
        if let Some(usage) = delta.usage {
            self.usage = usage;
        }
        if let Some(lp) = delta.logprob {
            self.token_logprobs.push(lp);
        }
    }

    fn into_output(self) -> CompletionOutput {
        CompletionOutput {
            text: self.text,
            finish_reason: self.finish_reason,
            usage: self.usage,
            token_logprobs: self.token_logprobs,
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn request_parent_context(headers: &HeaderMap) -> SpanContext {
    headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(SpanContext::decode_w3c_traceparent)
        .unwrap_or_else(SpanContext::random)
}

fn http_request_span(
    route: &'static str,
    stream: bool,
    max_tokens: usize,
    session_id: Option<&crate::types::SessionId>,
    headers: &HeaderMap,
) -> Span {
    let decision = trace_runtime().decide_request(uuid::Uuid::new_v4().as_bytes());
    let parent = request_parent_context(headers).sampled(decision.sampled);
    Span::root("http", parent).with_properties(|| {
        [
            ("route", route.to_string()),
            ("stream", stream.to_string()),
            ("max_tokens", max_tokens.to_string()),
            ("trace_level", decision.effective_level().to_string()),
            (
                "session_id",
                session_id
                    .map(std::string::ToString::to_string)
                    .unwrap_or_default(),
            ),
        ]
    })
}

// ============================================================================
// SSE helpers — shared between /v1/completions and /v1/chat/completions
// ============================================================================

/// Returns the terminal `[DONE]` SSE event that ends every streaming response.
fn sse_done_stream() -> impl futures_util::Stream<Item = Result<Event, Infallible>> {
    stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) })
}

async fn collect_buffered_response(
    mut delta_rx: UnboundedReceiver<CompletionStreamDelta>,
    request_kind: &str,
) -> Result<BufferedResponse, ApiError> {
    let collect = async {
        let mut buffered = BufferedResponse::default();
        while let Some(delta) = delta_rx.recv().await {
            buffered.apply_delta(&delta);
        }
        buffered
    };

    tokio::time::timeout(RESPONSE_TIMEOUT, collect)
        .await
        .map_err(|_| {
            error!(
                "Non-streaming {request_kind} timed out after {}s",
                RESPONSE_TIMEOUT.as_secs()
            );
            ApiError::timeout(RESPONSE_TIMEOUT.as_secs())
        })
}

fn parse_json_request<T>(payload: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    payload.map(|Json(value)| value).map_err(|err| match err {
        JsonRejection::MissingJsonContentType(_) => {
            ApiError::bad_request("Expected `Content-Type: application/json`", "invalid_json")
        }
        JsonRejection::JsonSyntaxError(inner) => ApiError::bad_request(
            format!("Malformed JSON request body: {inner}"),
            "invalid_json",
        ),
        JsonRejection::JsonDataError(inner) => ApiError::bad_request(
            format!("Invalid JSON request body: {inner}"),
            "invalid_json",
        ),
        JsonRejection::BytesRejection(inner) => bytes_rejection_to_api_error(&inner),
        other => ApiError::bad_request(
            format!("Failed to decode JSON request body: {other}"),
            "invalid_json",
        ),
    })
}

fn bytes_rejection_to_api_error(err: &BytesRejection) -> ApiError {
    let status = err.status();
    let body_text = err.body_text();
    if status == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::payload_too_large(body_text, "payload_too_large")
    } else {
        ApiError::bad_request(body_text, "invalid_body")
    }
}

fn route_not_found_error(path: &str) -> ApiError {
    ApiError::not_found(format!("Route `{path}` was not found"), "route_not_found")
}

fn method_not_allowed_error(method: &Method, path: &str) -> ApiError {
    ApiError::method_not_allowed(
        format!("Method `{method}` is not allowed for `{path}`"),
        "method_not_allowed",
    )
}

async fn route_not_found_handler(request: AxumRequest) -> ApiError {
    route_not_found_error(request.uri().path())
}

async fn method_not_allowed_handler(request: AxumRequest) -> ApiError {
    method_not_allowed_error(request.method(), request.uri().path())
}

fn submit_request(
    state: &AppState,
    options: RequestExecutionOptions,
    prompt: String,
) -> Result<UnboundedReceiver<CompletionStreamDelta>, ApiError> {
    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let prompt_tokens = {
        let _tokenize_span = LocalSpan::enter_with_local_parent("tokenize");
        match state.tokenizer.as_ref() {
            Some(tokenizer) => Some(tokenizer.encode(&prompt).map_err(|err| {
                error!("Prompt tokenization failed before scheduler submission: {err}");
                ApiError::service_unavailable("Failed to tokenize request prompt")
            })?),
            None => None,
        }
    };
    let enqueue_context = {
        let _enqueue_span = LocalSpan::enter_with_local_parent("enqueue");
        SpanContext::current_local_parent()
    };
    let incoming = options.into_incoming_request(prompt, prompt_tokens, delta_tx, enqueue_context);

    if let Err(e) = state.handle.submit(incoming) {
        warn!("Scheduler at capacity: {e}");
        return Err(ApiError::service_unavailable(
            "Server is at capacity, please retry later",
        ));
    }

    Ok(delta_rx)
}

fn authorize_headers(headers: &HeaderMap, expected_api_key: Option<&str>) -> Result<(), ApiError> {
    let Some(expected_api_key) = expected_api_key else {
        return Ok(());
    };

    let auth_header = headers
        .get(header::AUTHORIZATION)
        .ok_or_else(|| ApiError::unauthorized("Missing Authorization: Bearer <token> header"))?;
    let auth_value = auth_header
        .to_str()
        .map_err(|_| ApiError::unauthorized("Authorization header must be valid ASCII"))?;
    let (scheme, supplied_api_key) = auth_value
        .split_once(' ')
        .ok_or_else(|| ApiError::unauthorized("Authorization header must use Bearer auth"))?;
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return Err(ApiError::unauthorized(
            "Authorization header must use Bearer auth",
        ));
    }
    if supplied_api_key != expected_api_key {
        return Err(ApiError::unauthorized("Invalid API key"));
    }

    Ok(())
}

fn authorize_v1_request(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
    authorize_headers(headers, state.config.api_key.as_deref())
}

async fn authorize_session_request(
    State(expected_api_key): State<Option<Arc<str>>>,
    headers: HeaderMap,
    request: AxumRequest,
    next: middleware::Next,
) -> Result<Response, ApiError> {
    authorize_headers(&headers, expected_api_key.as_deref())?;
    Ok(next.run(request).await)
}

fn build_responses_prompt(req: &ResponsesRequest) -> Result<String, ApiError> {
    let mut messages = Vec::new();
    if let Some(instructions) = req.instructions.as_deref() {
        if !instructions.trim().is_empty() {
            messages.push(chat::OpenAiChatMessage {
                role: "system".into(),
                content: Some(instructions.into()),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            });
        }
    }

    match &req.input {
        ResponsesInput::Text(text) => {
            if text.trim().is_empty() {
                return Err(ApiError::bad_request(
                    "Input must not be empty",
                    "empty_input",
                ));
            }
            messages.push(chat::OpenAiChatMessage {
                role: "user".into(),
                content: Some(text.clone().into()),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Message(message) => messages.push(message.clone()),
        ResponsesInput::Messages(items) => {
            if items.is_empty() {
                return Err(ApiError::bad_request(
                    "Input messages must not be empty",
                    "empty_input",
                ));
            }
            messages.extend(items.iter().cloned());
        }
    }

    Ok(chat_messages_to_prompt(&messages, &req.tools))
}

/// Build the SSE event(s) for a single [`CompletionStreamDelta`].
///
/// Always returns one event for the main chunk. If `include_usage` is true and
/// this is the terminal delta (has a finish_reason), appends a second event with
/// usage statistics.
///
/// `make_chunk` converts the delta into the serializable chunk type.
/// `make_usage` converts [`TokenUsage`] into the serializable usage-chunk type.
fn delta_sse_events<C, U>(
    delta: crate::server_engine::CompletionStreamDelta,
    include_usage: bool,
    make_chunk: impl FnOnce(crate::server_engine::CompletionStreamDelta) -> C,
    make_usage: impl FnOnce(crate::server_engine::TokenUsage) -> U,
) -> Vec<Result<Event, Infallible>>
where
    C: serde::Serialize,
    U: serde::Serialize,
{
    let usage = delta.usage;
    let is_terminal = delta.finish_reason.is_some();
    let chunk = make_chunk(delta);
    let mut events = vec![Ok(
        Event::default().data(serde_json::to_string(&chunk).expect("chunk serialization"))
    )];

    if include_usage && is_terminal {
        if let Some(u) = usage {
            let usage_chunk = make_usage(u);
            events.push(Ok(Event::default().data(
                serde_json::to_string(&usage_chunk).expect("usage chunk serialization"),
            )));
        }
    }
    events
}

fn sse_json_event<T: serde::Serialize>(event_name: &'static str, payload: &T) -> Event {
    Event::default()
        .event(event_name)
        .data(serde_json::to_string(payload).expect("SSE payload serialization"))
}

enum ResponsesSseState {
    Start {
        response_id: String,
        created_at: u64,
        model_id: String,
        delta_rx: UnboundedReceiver<CompletionStreamDelta>,
        buffered: BufferedResponse,
    },
    Streaming {
        response_id: String,
        created_at: u64,
        model_id: String,
        delta_rx: UnboundedReceiver<CompletionStreamDelta>,
        buffered: BufferedResponse,
        final_pending: bool,
    },
    Done,
}

fn responses_sse_stream(
    delta_rx: UnboundedReceiver<CompletionStreamDelta>,
    response_id: String,
    created_at: u64,
    model_id: String,
) -> impl futures_util::Stream<Item = Result<Event, Infallible>> {
    stream::unfold(
        ResponsesSseState::Start {
            response_id,
            created_at,
            model_id,
            delta_rx,
            buffered: BufferedResponse::default(),
        },
        |state| async move {
            match state {
                ResponsesSseState::Start {
                    response_id,
                    created_at,
                    model_id,
                    delta_rx,
                    buffered,
                } => {
                    let event = sse_json_event(
                        "response.created",
                        &ResponsesStreamCreatedEvent::new(
                            response_id.clone(),
                            created_at,
                            model_id.clone(),
                        ),
                    );
                    Some((
                        Ok(event),
                        ResponsesSseState::Streaming {
                            response_id,
                            created_at,
                            model_id,
                            delta_rx,
                            buffered,
                            final_pending: false,
                        },
                    ))
                }
                ResponsesSseState::Streaming {
                    response_id,
                    created_at,
                    model_id,
                    mut delta_rx,
                    mut buffered,
                    final_pending,
                } => {
                    if final_pending {
                        let response = ResponsesResponse::from_output_with_id(
                            response_id.clone(),
                            model_id.clone(),
                            created_at,
                            buffered.into_output(),
                        );
                        let event = sse_json_event("response.completed", &response);
                        return Some((Ok(event), ResponsesSseState::Done));
                    }

                    while let Some(delta) = delta_rx.recv().await {
                        let has_text = !delta.text_delta.is_empty();
                        let is_terminal = delta.finish_reason.is_some();
                        let text_delta = delta.text_delta.clone();
                        buffered.apply_delta(&delta);

                        if has_text {
                            let event = sse_json_event(
                                "response.output_text.delta",
                                &ResponsesStreamDeltaEvent::new(
                                    response_id.clone(),
                                    created_at,
                                    model_id.clone(),
                                    text_delta,
                                ),
                            );
                            return Some((
                                Ok(event),
                                ResponsesSseState::Streaming {
                                    response_id,
                                    created_at,
                                    model_id,
                                    delta_rx,
                                    buffered,
                                    final_pending: is_terminal,
                                },
                            ));
                        }

                        if is_terminal {
                            let response = ResponsesResponse::from_output_with_id(
                                response_id.clone(),
                                model_id.clone(),
                                created_at,
                                buffered.into_output(),
                            );
                            let event = sse_json_event("response.completed", &response);
                            return Some((Ok(event), ResponsesSseState::Done));
                        }
                    }

                    let response = ResponsesResponse::from_output_with_id(
                        response_id.clone(),
                        model_id.clone(),
                        created_at,
                        buffered.into_output(),
                    );
                    let event = sse_json_event("response.completed", &response);
                    Some((Ok(event), ResponsesSseState::Done))
                }
                ResponsesSseState::Done => None,
            }
        },
    )
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Result<Json<OpenAiCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let req = parse_json_request(payload)?;
    let options = RequestExecutionOptions::from_completion(&req);
    let http_span = http_request_span(
        "/v1/completions",
        options.stream,
        options.max_tokens,
        options.session_id.as_ref(),
        &headers,
    );

    async move {
        authorize_v1_request(&headers, state.as_ref())?;
        req.validate()?;
        let max_tokens = options.max_tokens;
        let stream = options.stream;
        let include_usage = options.include_usage;
        let model_id = state.identity.model_id.clone();

        if req.prompt.trim().is_empty() {
            warn!("Rejecting empty prompt request");
            return Err(ApiError::bad_request(
                "Prompt must not be empty",
                "empty_prompt",
            ));
        }

        info!(
            "Received request: prompt_len={}, max_tokens={}, stream={}",
            req.prompt.len(),
            max_tokens,
            stream,
        );

        let delta_rx = submit_request(state.as_ref(), options, req.prompt)?;

        if stream {
            let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
            let created = now_secs();

            let sse_stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
                stream::iter(delta_sse_events(
                    delta,
                    include_usage,
                    |d| StreamChunk::from_delta(&request_id, created, &model_id, d),
                    |u| StreamUsageChunk::from_usage(&request_id, created, &model_id, u),
                ))
            });

            Ok(Sse::new(sse_stream.chain(sse_done_stream())).into_response())
        } else {
            let stream_parent = SpanContext::current_local_parent().unwrap_or_default();
            let stream_span = Span::root("stream_flush", stream_parent)
                .with_properties(|| [("route", "/v1/completions".to_string())]);
            let finish_parent = SpanContext::from_span(&stream_span).unwrap_or(stream_parent);
            let buffered = async move { collect_buffered_response(delta_rx, "request").await }
                .in_span(stream_span)
                .await?;

            info!(
                "Request completed: prompt_tokens={}, completion_tokens={}",
                buffered.usage.prompt_tokens, buffered.usage.completion_tokens
            );

            async move {
                let response =
                    CompletionResponse::from_output(model_id, now_secs(), buffered.into_output());
                Ok(Json(response).into_response())
            }
            .in_span(
                Span::root("finish", finish_parent)
                    .with_properties(|| [("route", "/v1/completions".to_string())]),
            )
            .await
        }
    }
    .in_span(http_span)
    .await
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let req = parse_json_request(payload)?;
    let options = RequestExecutionOptions::from_chat(&req);
    let http_span = http_request_span(
        "/v1/chat/completions",
        options.stream,
        options.max_tokens,
        options.session_id.as_ref(),
        &headers,
    );

    async move {
        authorize_v1_request(&headers, state.as_ref())?;
        req.validate()?;
        if req.messages.is_empty() {
            warn!("Rejecting empty messages request");
            return Err(ApiError::bad_request(
                "Messages array must not be empty",
                "empty_messages",
            ));
        }

        let max_tokens = options.max_tokens;
        let do_stream = options.stream;
        let include_usage = options.include_usage;
        let model_id = state.identity.model_id.clone();

        // Convert messages → ChatML prompt.
        let prompt = chat_messages_to_prompt(&req.messages, &req.tools);

        info!(
            "chat/completions: messages={}, prompt_len={}, max_tokens={}, stream={}",
            req.messages.len(),
            prompt.len(),
            max_tokens,
            do_stream,
        );

        let delta_rx = submit_request(state.as_ref(), options, prompt)?;

        if do_stream {
            let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let created = now_secs();

            let role_event =
                {
                    let chunk = ChatStreamChunk::role_chunk(&request_id, created, &model_id);
                    Ok::<_, Infallible>(Event::default().data(
                        serde_json::to_string(&chunk).expect("ChatStreamChunk serialization"),
                    ))
                };

            let req_id = request_id;
            let mid = model_id.clone();
            let content_stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
                stream::iter(delta_sse_events(
                    delta,
                    include_usage,
                    |d| ChatStreamChunk::content_chunk(&req_id, created, &mid, d),
                    |u| ChatStreamUsageChunk::from_usage(&req_id, created, &mid, u),
                ))
            });

            let full_stream = stream::once(async move { role_event })
                .chain(content_stream)
                .chain(sse_done_stream());

            Ok(Sse::new(full_stream).into_response())
        } else {
            let stream_parent = SpanContext::current_local_parent().unwrap_or_default();
            let stream_span = Span::root("stream_flush", stream_parent)
                .with_properties(|| [("route", "/v1/chat/completions".to_string())]);
            let finish_parent = SpanContext::from_span(&stream_span).unwrap_or(stream_parent);
            let buffered = async move { collect_buffered_response(delta_rx, "chat request").await }
                .in_span(stream_span)
                .await?;

            info!(
                "chat/completions done: prompt_tokens={}, completion_tokens={}",
                buffered.usage.prompt_tokens, buffered.usage.completion_tokens
            );

            async move {
                let output = buffered.into_output();
                let response = ChatCompletionResponse::from_output(model_id, now_secs(), &output);
                Ok(Json(response).into_response())
            }
            .in_span(
                Span::root("finish", finish_parent)
                    .with_properties(|| [("route", "/v1/chat/completions".to_string())]),
            )
            .await
        }
    }
    .in_span(http_span)
    .await
}

async fn models_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    let dflash = state
        .identity
        .dflash_status
        .as_ref()
        .map(|status| DflashStatusPayload {
            enabled: true,
            draft: status.draft_model.clone(),
            speculative_tokens: status.speculative_tokens,
            acceptance_rate: state
                .metrics
                .dflash_acceptance_rate_opt()
                .filter(|rate| rate.is_finite()),
        });
    let response = ModelsListResponse::single(state.identity.model_id.as_str(), now_secs(), dflash);
    Ok(Json(response).into_response())
}

async fn responses_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let req = parse_json_request(payload)?;
    let options = RequestExecutionOptions::from_responses(&req);
    let http_span = http_request_span(
        "/v1/responses",
        options.stream,
        options.max_tokens,
        options.session_id.as_ref(),
        &headers,
    );

    async move {
        authorize_v1_request(&headers, state.as_ref())?;
        req.validate()?;
        let prompt = build_responses_prompt(&req)?;
        let max_tokens = options.max_tokens;
        let stream = options.stream;
        let model_id = state.identity.model_id.clone();

        info!(
            "responses: prompt_len={}, max_output_tokens={}",
            prompt.len(),
            max_tokens,
        );

        let delta_rx = submit_request(state.as_ref(), options, prompt)?;
        if stream {
            let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
            let created_at = now_secs();
            let stream = responses_sse_stream(delta_rx, response_id, created_at, model_id);
            Ok(Sse::new(stream.chain(sse_done_stream())).into_response())
        } else {
            let stream_parent = SpanContext::current_local_parent().unwrap_or_default();
            let stream_span = Span::root("stream_flush", stream_parent)
                .with_properties(|| [("route", "/v1/responses".to_string())]);
            let finish_parent = SpanContext::from_span(&stream_span).unwrap_or(stream_parent);
            let buffered =
                async move { collect_buffered_response(delta_rx, "responses request").await }
                    .in_span(stream_span)
                    .await?;
            async move {
                let response =
                    ResponsesResponse::from_output(model_id, now_secs(), buffered.into_output());
                Ok(Json(response).into_response())
            }
            .in_span(
                Span::root("finish", finish_parent)
                    .with_properties(|| [("route", "/v1/responses".to_string())]),
            )
            .await
        }
    }
    .in_span(http_span)
    .await
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_prometheus();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

async fn stats_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(err) = authorize_v1_request(&headers, state.as_ref()) {
        return err.into_response();
    }
    let body = state.metrics.render_summary();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

async fn train_status_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    proxy_train_control(state.as_ref(), "GET", "/v1/train/status", None).await
}

async fn train_events_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<TrainEventsQuery>,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    let query = query
        .after_seq
        .map(|after_seq| format!("after_seq={after_seq}"));
    proxy_train_control(state.as_ref(), "GET", "/v1/train/events", query.as_deref()).await
}

async fn train_stop_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    proxy_train_control(state.as_ref(), "POST", "/v1/train/stop", None).await
}

async fn train_save_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    proxy_train_control(state.as_ref(), "POST", "/v1/train/save", None).await
}

async fn proxy_train_control(
    state: &AppState,
    method: &'static str,
    route_suffix: &'static str,
    query: Option<&str>,
) -> Result<Response, ApiError> {
    let Some(target) = state.config.train_control_target.clone() else {
        return Err(ApiError::not_found(
            "Train control plane is not configured on this infer server",
            "train_control_unconfigured",
        ));
    };
    let path = target.request_path(route_suffix, query);
    let proxied =
        tokio::task::spawn_blocking(move || blocking_train_control_request(&target, method, &path))
            .await
            .map_err(|err| {
                error!("train control proxy task failed: {err}");
                ApiError::service_unavailable("Train control plane task failed")
            })??;
    Ok((
        proxied.status,
        [(
            axum::http::header::CONTENT_TYPE,
            "application/json; charset=utf-8",
        )],
        proxied.body,
    )
        .into_response())
}

fn blocking_train_control_request(
    target: &TrainControlTarget,
    method: &str,
    path: &str,
) -> Result<ProxiedTrainResponse, ApiError> {
    let mut stream = TcpStream::connect(target.authority()).map_err(|err| {
        warn!("train control proxy connect failed: {err}");
        ApiError::service_unavailable("Train control plane is unavailable")
    })?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n",
        host = target.authority(),
    );
    stream.write_all(request.as_bytes()).map_err(|err| {
        warn!("train control proxy write failed: {err}");
        ApiError::service_unavailable("Train control plane write failed")
    })?;
    stream.flush().map_err(|err| {
        warn!("train control proxy flush failed: {err}");
        ApiError::service_unavailable("Train control plane flush failed")
    })?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|err| {
        warn!("train control proxy read failed: {err}");
        ApiError::service_unavailable("Train control plane read failed")
    })?;
    parse_train_control_response(&raw)
}

fn parse_train_control_response(raw: &[u8]) -> Result<ProxiedTrainResponse, ApiError> {
    let Some(header_end) = raw.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err(ApiError::service_unavailable(
            "Train control plane returned an invalid HTTP response",
        ));
    };
    let header_bytes = &raw[..header_end];
    let body = raw[header_end + 4..].to_vec();
    let header_text = std::str::from_utf8(header_bytes).map_err(|_| {
        ApiError::service_unavailable("Train control plane returned non-UTF8 headers")
    })?;
    let status_code = header_text
        .lines()
        .next()
        .and_then(|status_line| status_line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .and_then(|code| axum::http::StatusCode::from_u16(code).ok())
        .ok_or_else(|| {
            ApiError::service_unavailable("Train control plane returned an invalid status line")
        })?;
    Ok(ProxiedTrainResponse {
        status: status_code,
        body,
    })
}

/// Build the Axum router with default (empty) metrics.
pub fn build_app<H>(handle: H) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(
        handle,
        ServerMetrics::new(""),
        HttpServerConfig::default(),
        None,
    )
}

/// Build the Axum router with a pre-configured `ServerMetrics` instance.
pub fn build_app_with_metrics<H>(handle: H, metrics: ServerMetrics) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(handle, metrics, HttpServerConfig::default(), None)
}

/// Build the Axum router with explicit metrics and server configuration.
pub fn build_app_with_config<H>(
    handle: H,
    metrics: ServerMetrics,
    config: HttpServerConfig,
) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(handle, metrics, config, None)
}

pub fn build_app_with_session_engine<H, E>(handle: H, engine: Arc<tokio::sync::RwLock<E>>) -> Router
where
    H: RequestHandle + 'static,
    E: SessionPersistence + Send + Sync + 'static,
{
    build_app_inner(
        handle,
        ServerMetrics::new(""),
        HttpServerConfig::default(),
        Some(sessions::session_router(engine)),
    )
}

pub fn build_app_with_config_and_session_engine<H, E>(
    handle: H,
    engine: Arc<tokio::sync::RwLock<E>>,
    metrics: ServerMetrics,
    config: HttpServerConfig,
) -> Router
where
    H: RequestHandle + 'static,
    E: SessionPersistence + Send + Sync + 'static,
{
    build_app_inner(
        handle,
        metrics,
        config,
        Some(sessions::session_router(engine)),
    )
}

fn build_app_inner<H>(
    handle: H,
    metrics: ServerMetrics,
    config: HttpServerConfig,
    session_routes: Option<Router>,
) -> Router
where
    H: RequestHandle + 'static,
{
    let session_api_key = config.api_key.clone();
    let tokenizer = handle.tokenizer_clone();
    let identity = ServingIdentity {
        model_id: handle.model_id().to_string(),
        dflash_status: handle.dflash_status(),
    };
    let state = Arc::new(AppState {
        handle: Arc::new(handle),
        tokenizer,
        identity,
        metrics,
        config,
    });

    // The session subtree (if present) is already fully-routed by
    // `sessions::session_router(engine)`, so we apply the auth middleware
    // via `.layer(...)` and mount the whole subtree as a service.
    let mut router: Router<Arc<AppState>> = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/train/status", get(train_status_handler))
        .route("/v1/train/events", get(train_events_handler))
        .route("/v1/train/stop", post(train_stop_handler))
        .route("/v1/train/save", post(train_save_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/stats", get(stats_handler));

    if let Some(session_routes) = session_routes {
        let guarded = session_routes.layer(middleware::from_fn_with_state(
            session_api_key,
            authorize_session_request,
        ));
        router = router.nest_service("/v1/sessions", guarded);
    }

    router
        .method_not_allowed_fallback(method_not_allowed_handler)
        .fallback(route_not_found_handler)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use tower::util::ServiceExt;

    fn mock_scheduler(model_id: &str) -> SchedulerHandle {
        mock_scheduler_with_deltas(
            model_id,
            vec![
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                },
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                },
            ],
            true,
        )
    }

    fn mock_scheduler_with_deltas(
        model_id: &str,
        deltas: Vec<CompletionStreamDelta>,
        prefix_prompt_on_first_delta: bool,
    ) -> SchedulerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IncomingRequest>();
        let model_id = model_id.to_string();

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                for (index, delta) in deltas.iter().enumerate() {
                    let text_delta = if prefix_prompt_on_first_delta && index == 0 {
                        format!("ok:{}{}", req.prompt, delta.text_delta)
                    } else {
                        delta.text_delta.clone()
                    };
                    let _ = req.delta_tx.send(CompletionStreamDelta {
                        text_delta,
                        finish_reason: delta.finish_reason,
                        usage: delta.usage,
                        logprob: delta.logprob,
                    });
                }
            }
        });

        SchedulerHandle::from_parts(tx, &model_id)
    }

    fn spawn_train_control_stub_once(
        expected_method: &'static str,
        expected_target: &'static str,
        status: u16,
        body: &'static str,
    ) -> (TrainControlTarget, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind train control stub");
        let addr = listener.local_addr().expect("stub addr");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept train control stub");
            let mut reader = BufReader::new(stream.try_clone().expect("clone stub stream"));
            let mut request_line = String::new();
            reader
                .read_line(&mut request_line)
                .expect("read request line");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("");
            let target = parts.next().unwrap_or("");
            assert_eq!(method, expected_method);
            assert_eq!(target, expected_target);

            loop {
                let mut line = String::new();
                let bytes = reader.read_line(&mut line).expect("read header");
                if bytes == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
            }

            write!(
                stream,
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .expect("write stub response");
        });
        let target = TrainControlTarget::parse(&format!("http://{addr}"))
            .expect("parse train control target");
        (target, handle)
    }

    #[test]
    fn train_control_target_parses_and_normalizes_base_path() {
        let target =
            TrainControlTarget::parse("http://127.0.0.1:9123/base/child/").expect("parse target");
        assert_eq!(target.authority(), "127.0.0.1:9123");
        assert_eq!(
            target.request_path("/v1/train/status", None),
            "/base/child/v1/train/status"
        );
        assert_eq!(
            target.request_path("/v1/train/events", Some("after_seq=7")),
            "/base/child/v1/train/events?after_seq=7"
        );
    }

    #[tokio::test]
    async fn completion_response_uses_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["choices"][0]["text"], "ok:hello");
    }

    #[tokio::test]
    async fn streaming_response_uses_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""model":"Qwen3-8B""#),
            "payload={payload}"
        );
        assert!(
            !payload.contains(r#""model":"qwen3-4b""#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"));
    }

    #[tokio::test]
    async fn streaming_response_includes_usage_when_requested() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload
                .contains(r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_rejects_empty_prompt() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"   ","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn completion_rejects_malformed_json_with_openai_error_body() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":"oops"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "invalid_json");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Invalid JSON request body")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_requires_json_content_type() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":1}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_json");
        assert!(
            payload["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Content-Type")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn completion_rejects_payload_too_large_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let oversized_body = format!(
            r#"{{"prompt":"{}","max_tokens":1}}"#,
            "x".repeat(3 * 1024 * 1024)
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(oversized_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "payload_too_large");
    }

    #[tokio::test]
    async fn unknown_route_returns_structured_not_found_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/unknown")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "route_not_found");
        assert_eq!(
            payload["error"]["message"],
            "Route `/v1/unknown` was not found"
        );
    }

    #[tokio::test]
    async fn wrong_method_returns_structured_method_not_allowed_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/completions")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(
            response.headers()["content-type"],
            "application/json; charset=utf-8"
        );

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["code"], "method_not_allowed");
        assert_eq!(
            payload["error"]["message"],
            "Method `GET` is not allowed for `/v1/completions`"
        );
    }

    #[tokio::test]
    async fn completion_rejects_zero_max_tokens_with_structured_error() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"prompt":"hello","max_tokens":0}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("max_tokens")
        );
    }

    // -----------------------------------------------------------------------
    // /v1/chat/completions tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn chat_completion_basic() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Model id comes from the loaded model, not the request.
        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        // Content should contain something (mock returns "ok:<prompt>").
        assert!(
            payload["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "expected non-empty content, got: {payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_rejects_empty_messages() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","messages":[],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_completion_rejects_stream_options_without_stream() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stream_options")
        );
    }

    #[tokio::test]
    async fn chat_completion_streaming() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("chat.completion.chunk"),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"), "payload={payload}");
        // First chunk must carry role.
        assert!(
            payload.contains(r#""role":"assistant""#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains("infer_requests_total"),
            "payload={payload}"
        );
        assert!(
            payload.contains("infer_requests_active"),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn stats_endpoint_returns_text() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();
        assert!(!payload.is_empty(), "stats body should not be empty");
    }

    #[tokio::test]
    async fn train_status_returns_not_found_when_bridge_is_unconfigured() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn train_status_proxies_json_from_control_plane() {
        let (target, handle) = spawn_train_control_stub_once(
            "GET",
            "/v1/train/status",
            200,
            r#"{"iter":3,"finished":false}"#,
        );
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                train_control_target: Some(target),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"iter":3,"finished":false})
        );
        handle.join().expect("join train control stub");
    }

    #[tokio::test]
    async fn train_events_proxy_forwards_after_seq_query() {
        let (target, handle) = spawn_train_control_stub_once(
            "GET",
            "/v1/train/events?after_seq=7",
            200,
            r#"{"events":[],"latest_seq":7}"#,
        );
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                train_control_target: Some(target),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/events?after_seq=7")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"events":[],"latest_seq":7})
        );
        handle.join().expect("join train control stub");
    }

    #[tokio::test]
    async fn completions_reject_missing_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn completions_accept_valid_bearer_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret-token")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn stats_reject_missing_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/stats")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn train_status_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                train_control_target: Some(
                    TrainControlTarget::parse("http://127.0.0.1:9123").expect("parse target"),
                ),
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/train/status")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn chat_completion_streaming_with_usage() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""prompt_tokens":1"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn chat_completion_returns_structured_tool_calls() {
        let app = build_app(mock_scheduler_with_deltas(
            "Qwen3-4B",
            vec![
                CompletionStreamDelta {
                    text_delta:
                        "\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>"
                            .to_string(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                },
                CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                },
            ],
            false,
        ));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
        assert!(payload["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "shell"
        );
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!({"command":"pwd"}).to_string()
        );
    }

    #[tokio::test]
    async fn models_endpoint_returns_loaded_model_id() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["object"], "list");
        assert_eq!(payload["data"][0]["id"], "Qwen3-8B");
    }

    #[tokio::test]
    async fn serving_identity_is_snapshotted_once_and_reused_by_http_handlers() {
        use crate::request_handle::{DflashStatus, RequestHandle, SubmitError};
        use crate::scheduler::IncomingRequest;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct SnapshotHandle {
            submit_tx: tokio::sync::mpsc::UnboundedSender<IncomingRequest>,
            model_id: String,
            dflash_status: Option<DflashStatus>,
            model_id_calls: Arc<AtomicUsize>,
            dflash_calls: Arc<AtomicUsize>,
        }

        impl RequestHandle for SnapshotHandle {
            fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
                self.submit_tx.send(req).map_err(|_| SubmitError)
            }

            fn model_id(&self) -> &str {
                let calls = self.model_id_calls.fetch_add(1, Ordering::SeqCst) + 1;
                assert_eq!(calls, 1, "model_id() should only be called at build time");
                &self.model_id
            }

            fn dflash_status(&self) -> Option<DflashStatus> {
                let calls = self.dflash_calls.fetch_add(1, Ordering::SeqCst) + 1;
                assert_eq!(
                    calls, 1,
                    "dflash_status() should only be called at build time"
                );
                self.dflash_status.clone()
            }
        }

        let model_id_calls = Arc::new(AtomicUsize::new(0));
        let dflash_calls = Arc::new(AtomicUsize::new(0));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<IncomingRequest>();
        let model_id = "BootModel".to_string();
        let dflash_status = Some(DflashStatus {
            draft_model: "draft/boot-model".to_string(),
            speculative_tokens: 4,
        });
        let handle = SnapshotHandle {
            submit_tx: tx,
            model_id: model_id.clone(),
            dflash_status: dflash_status.clone(),
            model_id_calls: model_id_calls.clone(),
            dflash_calls: dflash_calls.clone(),
        };

        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: format!("ok:{}", req.prompt),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    }),
                    logprob: None,
                });
            }
        });

        let app = build_app(handle);
        assert_eq!(model_id_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dflash_calls.load(Ordering::SeqCst), 1);

        for (method, uri, body) in [
            (
                "POST",
                "/v1/completions",
                r#"{"prompt":"hello","max_tokens":1}"#,
            ),
            (
                "POST",
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#,
            ),
            (
                "POST",
                "/v1/responses",
                r#"{"input":"hello","max_output_tokens":1}"#,
            ),
        ] {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();

            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(payload["model"], model_id, "uri={uri}");
        }

        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["data"][0]["id"], model_id);
        let dflash = &payload["data"][0]["dflash"];
        assert_eq!(dflash["enabled"], true);
        assert_eq!(dflash["draft"], "draft/boot-model");
        assert_eq!(dflash["speculative_tokens"], 4);
        assert!(dflash["acceptance_rate"].is_null());

        assert_eq!(model_id_calls.load(Ordering::SeqCst), 1);
        assert_eq!(dflash_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn models_endpoint_omits_dflash_when_handle_reports_none() {
        let app = build_app(mock_scheduler("Qwen3-8B"));
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Baseline (non-DFlash) runtimes must keep the original JSON shape —
        // the `dflash` key is skipped entirely, not emitted as `null`.
        assert!(
            !payload["data"][0]
                .as_object()
                .unwrap()
                .contains_key("dflash"),
            "dflash key must be omitted when RequestHandle reports None, got {payload}"
        );
    }

    #[tokio::test]
    async fn models_endpoint_surfaces_dflash_status_when_reported() {
        use crate::request_handle::{DflashStatus, RequestHandle, SubmitError};
        use crate::scheduler::IncomingRequest;

        struct DflashHandle;
        impl RequestHandle for DflashHandle {
            fn submit(&self, _req: IncomingRequest) -> Result<(), SubmitError> {
                Ok(())
            }
            fn model_id(&self) -> &str {
                "Qwen3.5-4B-MLX-4bit"
            }
            fn dflash_status(&self) -> Option<DflashStatus> {
                Some(DflashStatus {
                    draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
                    speculative_tokens: 5,
                })
            }
        }

        let app = build_app(DflashHandle);
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let dflash = &payload["data"][0]["dflash"];
        assert_eq!(dflash["enabled"], true);
        assert_eq!(dflash["draft"], "z-lab/Qwen3.5-4B-DFlash");
        assert_eq!(dflash["speculative_tokens"], 5);
        // No speculative blocks have run in a test build → `acceptance_rate`
        // must serialise as JSON `null`, not 0.0, so dashboards can tell
        // "no data yet" apart from "everything rejected".
        assert!(
            dflash["acceptance_rate"].is_null(),
            "acceptance_rate must be null before any blocks run, got {dflash}"
        );
    }

    #[tokio::test]
    async fn models_endpoint_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn responses_endpoint_returns_openai_style_response_object() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","input":"hello","max_output_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["object"], "response");
        assert_eq!(payload["model"], "Qwen3-4B");
        assert_eq!(payload["usage"]["input_tokens"], 1);
        assert!(
            payload["output_text"]
                .as_str()
                .is_some_and(|text| text.contains("hello")),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_rejects_invalid_sampling_knob() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"hello","top_p":1.5}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "invalid_parameter");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("top_p")
        );
    }

    #[tokio::test]
    async fn responses_endpoint_streams_deltas_and_final_event_before_done() {
        let app = build_app(mock_scheduler("Qwen3-4B"));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"input":"hello","max_output_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        let created_pos = payload
            .find("event: response.created")
            .unwrap_or_else(|| panic!("missing created event: {payload}"));
        let delta_pos = payload
            .find("event: response.output_text.delta")
            .unwrap_or_else(|| panic!("missing delta event: {payload}"));
        let completed_pos = payload
            .find("event: response.completed")
            .unwrap_or_else(|| panic!("missing completed event: {payload}"));
        let done_pos = payload
            .find("[DONE]")
            .unwrap_or_else(|| panic!("missing terminal done event: {payload}"));

        assert!(
            created_pos < delta_pos && delta_pos < completed_pos && completed_pos < done_pos,
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""delta":"ok:<|im_start|>user"#),
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""status":"completed""#),
            "payload={payload}"
        );
        assert!(
            payload.contains(r#""usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }

    #[tokio::test]
    async fn responses_endpoint_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
                ..Default::default()
            },
        );
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"hello","max_output_tokens":1}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
