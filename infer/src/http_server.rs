mod openai_v1;
pub mod sessions;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, header},
    routing::{get, post},
};
use futures_util::{StreamExt, stream};
use infer_chat::openai_messages_to_prompt as chat_messages_to_prompt;
use log::{error, info, warn};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Maximum wall-clock time allowed for a non-streaming request to complete.
/// Streaming responses have natural per-chunk flow control and are not capped here.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);

use crate::error::ApiError;
use crate::metrics::ServerMetrics;
use crate::request_handle::RequestHandle;
use crate::sampler::{SamplingParams, sampling_params_from_request};
#[cfg(test)]
use crate::scheduler::SchedulerHandle;
use crate::scheduler::{IncomingRequest, RequestPriority};
use crate::server_engine::CompletionStreamDelta;
use crate::server_engine::{CompletionOutput, FinishReason, TokenUsage};
use openai_v1::{
    ChatCompletionRequest, ChatCompletionResponse, ChatStreamChunk, ChatStreamUsageChunk,
    CompletionRequest as OpenAiCompletionRequest, CompletionResponse, ModelsListResponse,
    ResponsesInput, ResponsesRequest, ResponsesResponse, ResponsesStreamCreatedEvent,
    ResponsesStreamDeltaEvent, StreamChunk, StreamUsageChunk,
};

struct AppState {
    handle: Arc<dyn RequestHandle>,
    metrics: ServerMetrics,
    config: HttpServerConfig,
}

#[derive(Clone, Debug, Default)]
pub struct HttpServerConfig {
    pub api_key: Option<Arc<str>>,
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
        delta_tx: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    ) -> IncomingRequest {
        IncomingRequest {
            prompt,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            stop: self.stop,
            priority: RequestPriority::default(),
            session_id: self.session_id,
            delta_tx,
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

fn submit_request(
    handle: &dyn RequestHandle,
    options: RequestExecutionOptions,
    prompt: String,
) -> Result<UnboundedReceiver<CompletionStreamDelta>, ApiError> {
    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let incoming = options.into_incoming_request(prompt, delta_tx);

    if let Err(e) = handle.submit(incoming) {
        error!("Scheduler unavailable or full: {e}");
        return Err(ApiError::service_unavailable(
            "Server is at capacity, please retry later",
        ));
    }

    Ok(delta_rx)
}

fn authorize_v1_request(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
    let Some(expected_api_key) = state.config.api_key.as_deref() else {
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

fn build_responses_prompt(req: &ResponsesRequest) -> Result<String, ApiError> {
    let mut messages = Vec::new();
    if let Some(instructions) = req.instructions.as_deref() {
        if !instructions.trim().is_empty() {
            messages.push(infer_chat::OpenAiChatMessage {
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
            messages.push(infer_chat::OpenAiChatMessage {
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

fn sse_json_event<T: serde::Serialize>(
    event_name: &'static str,
    payload: &T,
) -> Result<Event, Infallible> {
    Ok(Event::default()
        .event(event_name)
        .data(serde_json::to_string(payload).expect("SSE payload serialization")))
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
                        event,
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
                        return Some((event, ResponsesSseState::Done));
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
                                event,
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
                            return Some((event, ResponsesSseState::Done));
                        }
                    }

                    let response = ResponsesResponse::from_output_with_id(
                        response_id.clone(),
                        model_id.clone(),
                        created_at,
                        buffered.into_output(),
                    );
                    let event = sse_json_event("response.completed", &response);
                    Some((event, ResponsesSseState::Done))
                }
                ResponsesSseState::Done => None,
            }
        },
    )
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<OpenAiCompletionRequest>,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    let options = RequestExecutionOptions::from_completion(&req);
    let max_tokens = options.max_tokens;
    let stream = options.stream;
    let include_usage = options.include_usage;
    let model_id = state.handle.model_id().to_string();

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

    let delta_rx = submit_request(state.handle.as_ref(), options, req.prompt)?;

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
        let buffered = collect_buffered_response(delta_rx, "request").await?;

        info!(
            "Request completed: prompt_tokens={}, completion_tokens={}",
            buffered.usage.prompt_tokens, buffered.usage.completion_tokens
        );

        let response =
            CompletionResponse::from_output(model_id, now_secs(), buffered.into_output());
        Ok(Json(response).into_response())
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    if req.messages.is_empty() {
        warn!("Rejecting empty messages request");
        return Err(ApiError::bad_request(
            "Messages array must not be empty",
            "empty_messages",
        ));
    }

    let options = RequestExecutionOptions::from_chat(&req);
    let max_tokens = options.max_tokens;
    let do_stream = options.stream;
    let include_usage = options.include_usage;
    let model_id = state.handle.model_id().to_string();

    // Convert messages → ChatML prompt.
    let prompt = chat_messages_to_prompt(&req.messages, &req.tools);

    info!(
        "chat/completions: messages={}, prompt_len={}, max_tokens={}, stream={}",
        req.messages.len(),
        prompt.len(),
        max_tokens,
        do_stream,
    );

    let delta_rx = submit_request(state.handle.as_ref(), options, prompt)?;

    if do_stream {
        let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = now_secs();

        // First chunk carries the role field.
        let role_event = {
            let chunk = ChatStreamChunk::role_chunk(&request_id, created, &model_id);
            Ok::<_, Infallible>(
                Event::default()
                    .data(serde_json::to_string(&chunk).expect("ChatStreamChunk serialization")),
            )
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
        let buffered = collect_buffered_response(delta_rx, "chat request").await?;

        info!(
            "chat/completions done: prompt_tokens={}, completion_tokens={}",
            buffered.usage.prompt_tokens, buffered.usage.completion_tokens
        );

        let output = buffered.into_output();
        let response = ChatCompletionResponse::from_output(model_id, now_secs(), &output);
        Ok(Json(response).into_response())
    }
}

async fn models_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    let response = ModelsListResponse::single(state.handle.model_id(), now_secs());
    Ok(Json(response).into_response())
}

async fn responses_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    let prompt = build_responses_prompt(&req)?;
    let options = RequestExecutionOptions::from_responses(&req);
    let max_tokens = options.max_tokens;
    let stream = options.stream;
    let model_id = state.handle.model_id().to_string();

    info!(
        "responses: prompt_len={}, max_output_tokens={}",
        prompt.len(),
        max_tokens,
    );

    let delta_rx = submit_request(state.handle.as_ref(), options, prompt)?;
    if stream {
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let created_at = now_secs();
        let stream = responses_sse_stream(delta_rx, response_id, created_at, model_id);
        Ok(Sse::new(stream.chain(sse_done_stream())).into_response())
    } else {
        let buffered = collect_buffered_response(delta_rx, "responses request").await?;
        let response = ResponsesResponse::from_output(model_id, now_secs(), buffered.into_output());
        Ok(Json(response).into_response())
    }
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

/// Build the Axum router with default (empty) metrics.
pub fn build_app<H>(handle: H) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_with_config(handle, ServerMetrics::new(""), HttpServerConfig::default())
}

/// Build the Axum router with a pre-configured `ServerMetrics` instance.
pub fn build_app_with_metrics<H>(handle: H, metrics: ServerMetrics) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_with_config(handle, metrics, HttpServerConfig::default())
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
    let state = Arc::new(AppState {
        handle: Arc::new(handle),
        metrics,
        config,
    });

    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/models", get(models_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/stats", get(stats_handler))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
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
    async fn completions_reject_missing_api_key_when_auth_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
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
    async fn models_endpoint_requires_auth_when_enabled() {
        let app = build_app_with_config(
            mock_scheduler("Qwen3-4B"),
            ServerMetrics::new(""),
            HttpServerConfig {
                api_key: Some(Arc::<str>::from("secret-token")),
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
            .expect(&format!("missing created event: {payload}"));
        let delta_pos = payload
            .find("event: response.output_text.delta")
            .expect(&format!("missing delta event: {payload}"));
        let completed_pos = payload
            .find("event: response.completed")
            .expect(&format!("missing completed event: {payload}"));
        let done_pos = payload
            .find("[DONE]")
            .expect(&format!("missing terminal done event: {payload}"));

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
