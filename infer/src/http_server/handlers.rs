//! Request handlers, SSE streaming helpers, JSON/route helpers, and the
//! train control proxy.
//!
//! Split out of `http_server.rs` (pure structural refactor — no behavior change).

use std::convert::Infallible;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::Request as AxumRequest;
use axum::extract::rejection::{BytesRejection, JsonRejection};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, Method, header};
use axum::middleware;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use chat::{
    DeepSeekV4ChatTemplateOptions, openai_messages_to_deepseek_v4_prompt,
    openai_messages_to_prompt as chat_messages_to_prompt,
};
use fastrace::Span;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use fastrace::local::LocalSpan;
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use serde::Deserialize;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::openai_v1::{
    ChatCompletionRequest, ChatCompletionResponse, ChatStreamChunk, ChatStreamUsageChunk,
    CompletionRequest as OpenAiCompletionRequest, CompletionResponse, DflashStatusPayload,
    ModelsListResponse, ResponsesInput, ResponsesRequest, ResponsesResponse,
    ResponsesStreamCreatedEvent, ResponsesStreamDeltaEvent, StreamChunk, StreamUsageChunk,
};
use super::preprocess::PreprocessPermitGuard;
use super::types::{
    AppState, BufferedResponse, HTTP_REQUEST_ID_HEADER, HealthResponse, ProxiedTrainResponse,
    RESPONSE_TIMEOUT, RequestExecutionOptions, TrainControlTarget, TrainEventsQuery,
};
use crate::error::ApiError;
use crate::metrics::ServerMetrics;
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::trace_reporter::trace_runtime;

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

#[derive(Debug, Deserialize)]
pub(super) struct StatsQuery {
    #[serde(default)]
    format: Option<String>,
}

fn wants_json_stats(headers: &HeaderMap, query: &StatsQuery) -> bool {
    if query
        .format
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("json"))
    {
        return true;
    }

    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|accept| {
            accept
                .split(',')
                .any(|part| part.trim().starts_with("application/json"))
        })
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

fn sse_keep_alive() -> KeepAlive {
    KeepAlive::new()
        .interval(std::time::Duration::from_secs(15))
        .text("arle-keepalive")
}

fn non_streaming_timeout_error(request_kind: &str) -> ApiError {
    error!(
        "Non-streaming {request_kind} timed out after {}s",
        RESPONSE_TIMEOUT.as_secs()
    );
    ApiError::timeout(RESPONSE_TIMEOUT.as_secs())
}

struct RequestTraceState {
    request_id: String,
    route: &'static str,
    stream: bool,
    max_tokens: usize,
    prompt_bytes: usize,
    started_at: Instant,
    first_token_at: Option<Instant>,
    chunks_seen: u64,
    text_bytes: usize,
    token_ids_seen: usize,
    last_usage: Option<TokenUsage>,
    metrics: ServerMetrics,
    finished: bool,
}

impl RequestTraceState {
    fn new(
        route: &'static str,
        request_id: String,
        stream: bool,
        max_tokens: usize,
        prompt_bytes: usize,
        metrics: ServerMetrics,
    ) -> Self {
        Self {
            request_id,
            route,
            stream,
            max_tokens,
            prompt_bytes,
            started_at: Instant::now(),
            first_token_at: None,
            chunks_seen: 0,
            text_bytes: 0,
            token_ids_seen: 0,
            last_usage: None,
            metrics,
            finished: false,
        }
    }

    fn observe_delta(&mut self, delta: &CompletionStreamDelta) {
        self.chunks_seen = self.chunks_seen.saturating_add(1);
        self.text_bytes = self.text_bytes.saturating_add(delta.text_delta.len());
        self.token_ids_seen = self.token_ids_seen.saturating_add(delta.token_ids.len());
        if self.first_token_at.is_none()
            && (!delta.text_delta.is_empty() || !delta.token_ids.is_empty())
        {
            self.first_token_at = Some(Instant::now());
        }
        if let Some(usage) = delta.usage {
            self.last_usage = Some(usage);
        }
    }

    fn finish(
        &mut self,
        terminal_seen: bool,
        finish_reason: Option<FinishReason>,
        error: Option<&'static str>,
    ) {
        if self.finished {
            return;
        }
        self.finished = true;

        let elapsed = self.started_at.elapsed();
        let ttft_ms = self
            .first_token_at
            .map(|first| first.duration_since(self.started_at).as_secs_f64() * 1_000.0);
        let total_ms = elapsed.as_secs_f64() * 1_000.0;
        let usage = self.last_usage.unwrap_or(TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });
        let scheduler_phase = self.metrics.scheduler_step_phase_us();
        let scheduler_loop = self.metrics.scheduler_loop_phase_us();
        let scheduler_pipeline = self.metrics.scheduler_pipeline_us();
        let preprocess_stage = self.metrics.preprocess_stage_us();

        info!(
            target: "infer::request_trace",
            "request_trace {}",
            serde_json::json!({
                "request_id": self.request_id,
                "route": self.route,
                "stream": self.stream,
                "terminal_seen": terminal_seen,
                "finish_reason": finish_reason.map(FinishReason::as_openai_str),
                "error": error,
                "max_tokens": self.max_tokens,
                "prompt_bytes": self.prompt_bytes,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "chunks_seen": self.chunks_seen,
                "text_bytes": self.text_bytes,
                "token_ids_seen": self.token_ids_seen,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "throughput": {
                    "prompt_tokens_per_s_at_ttft": ttft_ms
                        .filter(|value| *value > 0.0)
                        .map(|value| usage.prompt_tokens as f64 * 1_000.0 / value),
                    "completion_tokens_per_s_e2e": (total_ms > 0.0)
                        .then(|| usage.completion_tokens as f64 * 1_000.0 / total_ms),
                    "total_tokens_per_s_e2e": (total_ms > 0.0)
                        .then(|| usage.total_tokens as f64 * 1_000.0 / total_ms),
                },
                "kv": {
                    "gpu_util": self.metrics.kv_gpu_utilization(),
                    "fetch_queue_depth": self.metrics.kv_fetch_queue_depth(),
                    "fetch_waiters": self.metrics.kv_fetch_waiters(),
                    "store_queue_depth": self.metrics.kv_store_queue_depth(),
                    "fetch_backpressure": self.metrics.kv_fetch_backpressure(),
                    "store_backpressure": self.metrics.kv_store_backpressure(),
                },
                "prefix": {
                    "hit_rate": self.metrics.prefix_hit_rate(),
                    "skip_rate": self.metrics.prefix_skip_rate(),
                    "request_hit_rate": self.metrics.prefix_request_hit_rate(),
                    "request_skip_rate": self.metrics.prefix_request_skip_rate(),
                    "matched_tokens": self.metrics.matched_prefix_tokens(),
                    "resume_prefill_tokens": self.metrics.resume_prefill_tokens(),
                    "lookup_latency_us": self.metrics.prefix_lookup_latency_us(),
                    "lookup_reusable_tokens": self.metrics.prefix_lookup_reusable_tokens(),
                    "ready_on_gpu": self.metrics.prefix_lookup_ready_on_gpu(),
                    "direct_gpu_attach": self.metrics.prefix_lookup_direct_gpu_attach(),
                    "staged": self.metrics.prefix_lookup_staged(),
                    "prefetch": self.metrics.prefix_lookup_prefetch(),
                    "recompute": self.metrics.prefix_lookup_recompute(),
                },
                "scheduler": {
                    "active": self.metrics.requests_active(),
                    "waiting": self.metrics.requests_waiting(),
                    "running_batch": self.metrics.scheduler_running_batch(),
                    "prefill_queue": self.metrics.scheduler_prefill_queue(),
                    "scheduled_rows": self.metrics.scheduler_scheduled_rows(),
                    "scheduled_decode_rows": self.metrics.scheduler_scheduled_decode_rows(),
                    "scheduled_prefill_rows": self.metrics.scheduler_scheduled_prefill_rows(),
                    "decode_tokens": self.metrics.scheduler_decode_tokens(),
                    "prefill_tokens": self.metrics.scheduler_prefill_tokens(),
                    "batch_width": self.metrics.scheduler_batch_width(),
                    "step_last_s": self.metrics.scheduler_step_last_seconds(),
                    "phase_us": scheduler_phase.map(|(
                        admission,
                        prefill,
                        decode,
                        emit,
                        total,
                    )| serde_json::json!({
                        "admission": admission,
                        "prefill": prefill,
                        "decode": decode,
                        "emit": emit,
                        "total": total,
                    })),
                    "loop_us": scheduler_loop.map(|(cleanup, total)| serde_json::json!({
                        "cleanup": cleanup,
                        "total": total,
                    })),
                    "pipeline_us": {
                        "snapshot": scheduler_pipeline.0,
                        "cpu_plan": scheduler_pipeline.1,
                        "gpu_completion_wait": scheduler_pipeline.2,
                        "gpu_command_queue_depth": scheduler_pipeline.3,
                    },
                },
                "preprocess": {
                    "queue_depth": preprocess_stage.0,
                    "wait_us": preprocess_stage.1,
                    "tokenize_us": preprocess_stage.2,
                },
            })
        );
    }
}

fn trace_streaming_deltas(
    mut source: UnboundedReceiver<CompletionStreamDelta>,
    mut trace: RequestTraceState,
) -> UnboundedReceiver<CompletionStreamDelta> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(delta) = source.recv().await {
            trace.observe_delta(&delta);
            let finish_reason = delta.finish_reason;
            let terminal = finish_reason.is_some();
            if terminal {
                trace.finish(true, finish_reason, None);
            }
            if tx.send(delta).is_err() {
                trace.finish(false, finish_reason, Some("client_channel_closed"));
                while let Some(delta) = source.recv().await {
                    trace.observe_delta(&delta);
                    if let Some(finish_reason) = delta.finish_reason {
                        trace.finish(true, Some(finish_reason), None);
                        break;
                    }
                }
                return;
            }
            if terminal {
                return;
            }
        }
        trace.finish(false, None, Some("scheduler_channel_closed"));
    });
    rx
}

async fn collect_buffered_response_inner(
    mut delta_rx: UnboundedReceiver<CompletionStreamDelta>,
    request_kind: &str,
    mut trace: RequestTraceState,
) -> Result<BufferedResponse, ApiError> {
    let mut buffered = BufferedResponse::default();
    while let Some(delta) = delta_rx.recv().await {
        trace.observe_delta(&delta);
        buffered.apply_delta(&delta);
    }

    // Channel closed without a terminal delta — the scheduler aborted this
    // request (e.g. prefill OOM, slot teardown). Returning the buffered
    // (empty) body as a 200 silently swallows the error and confuses
    // clients (see K7 in docs/projects/2026-04-29-perf-bug-roundup.md);
    // surface a 503 instead so callers retry.
    if !buffered.terminal_seen {
        warn!(
            "{request_kind} channel closed without finish_reason ({} completion tokens, {} bytes text); returning 503",
            buffered.usage.completion_tokens,
            buffered.text.len(),
        );
        trace.finish(false, None, Some("scheduler_channel_closed"));
        return Err(ApiError::service_unavailable(
            "Inference request aborted before completion (server overloaded or out of memory). Please retry.",
        ));
    }

    trace.finish(true, Some(buffered.finish_reason), None);
    Ok(buffered)
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
        JsonRejection::JsonDataError(inner) => json_data_rejection_to_api_error(&inner),
        JsonRejection::BytesRejection(inner) => bytes_rejection_to_api_error(&inner),
        other => ApiError::bad_request(
            format!("Failed to decode JSON request body: {other}"),
            "invalid_json",
        ),
    })
}

fn json_data_rejection_to_api_error(err: &axum::extract::rejection::JsonDataError) -> ApiError {
    let detail = err.to_string();
    if let Some(field) = unsupported_json_field(&detail) {
        return ApiError::bad_request(
            format!("Invalid `{field}`: is not supported on this server yet"),
            "invalid_parameter",
        )
        .with_param(field);
    }
    ApiError::bad_request(
        format!("Invalid JSON request body: {detail}"),
        "invalid_json",
    )
}

fn unsupported_json_field(message: &str) -> Option<&str> {
    let (_, tail) = message.split_once("unknown field `")?;
    let (field, _) = tail.split_once('`')?;
    Some(field).filter(|field| !field.is_empty())
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

fn allow_header_value_for_path(path: &str) -> Option<axum::http::HeaderValue> {
    let allow = match path {
        "/v1/completions"
        | "/v1/chat/completions"
        | "/v1/responses"
        | "/v1/train/stop"
        | "/v1/train/save" => "POST",
        "/v1/models" | "/metrics" | "/v1/stats" | "/v1/train/status" | "/v1/train/events"
        | "/healthz" | "/readyz" => "GET, HEAD",
        _ => return None,
    };
    Some(axum::http::HeaderValue::from_static(allow))
}

fn method_not_allowed_error(method: &Method, path: &str) -> ApiError {
    let error = ApiError::method_not_allowed(
        format!("Method `{method}` is not allowed for `{path}`"),
        "method_not_allowed",
    );
    if let Some(allow) = allow_header_value_for_path(path) {
        error.with_header(header::ALLOW, allow)
    } else {
        error
    }
}

pub(super) async fn route_not_found_handler(request: AxumRequest) -> ApiError {
    route_not_found_error(request.uri().path())
}

pub(super) async fn method_not_allowed_handler(request: AxumRequest) -> ApiError {
    method_not_allowed_error(request.method(), request.uri().path())
}

fn request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get(HTTP_REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map_or_else(|| uuid::Uuid::new_v4().to_string(), ToOwned::to_owned)
}

fn preprocess_active_depth(state: &AppState) -> u64 {
    state
        .preprocess_capacity
        .saturating_sub(state.preprocess_permits.available_permits()) as u64
}

pub(super) async fn attach_request_id(
    mut request: AxumRequest,
    next: middleware::Next,
) -> Response {
    let request_id = request_id_from_headers(request.headers());
    request.extensions_mut().insert(request_id.clone());

    let mut response = next.run(request).await;
    if let Ok(value) = header::HeaderValue::from_str(&request_id) {
        response.headers_mut().insert(HTTP_REQUEST_ID_HEADER, value);
    }
    response
}

async fn preprocess_prompt_tokens(
    state: &AppState,
    prompt: String,
) -> Result<(String, Option<Vec<u32>>, Option<i32>), ApiError> {
    let Some(preprocess_pool) = state.preprocess_pool.clone() else {
        return Ok((prompt, None, None));
    };

    let wait_started_at = std::time::Instant::now();
    let permit = state
        .preprocess_permits
        .clone()
        .acquire_owned()
        .await
        .map_err(|err| {
            error!("Prompt preprocessing queue closed before scheduler submission: {err}");
            ApiError::service_unavailable("Failed to preprocess request prompt")
        })?;
    let wait_us = wait_started_at.elapsed().as_micros() as u64;
    state
        .metrics
        .set_preprocess_stage(preprocess_active_depth(state), wait_us, 0);
    let tokenize_started_at = std::time::Instant::now();
    let permit_guard = PreprocessPermitGuard::new(
        state.preprocess_permits.clone(),
        state.preprocess_capacity,
        state.metrics.clone(),
        permit,
        wait_us,
        tokenize_started_at,
    );
    let output = preprocess_pool
        .encode(prompt, permit_guard)
        .await
        .map_err(|err| {
            error!("Prompt tokenization failed before scheduler submission: {err}");
            ApiError::service_unavailable("Failed to tokenize request prompt")
        })?;
    Ok((output.prompt, Some(output.prompt_tokens), output.numa_node))
}

async fn submit_request(
    state: &AppState,
    options: RequestExecutionOptions,
    prompt: String,
) -> Result<UnboundedReceiver<CompletionStreamDelta>, ApiError> {
    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let preprocess_parent = SpanContext::current_local_parent().unwrap_or_default();
    let preprocess_span = Span::root("preprocess", preprocess_parent);
    let (prompt, prompt_tokens, ingress_numa_node) = preprocess_prompt_tokens(state, prompt)
        .in_span(preprocess_span)
        .await?;
    let enqueue_context = {
        let _enqueue_span = LocalSpan::enter_with_local_parent("enqueue");
        SpanContext::current_local_parent()
    };
    let incoming = options.into_incoming_request(
        prompt,
        prompt_tokens,
        ingress_numa_node,
        delta_tx,
        enqueue_context,
    );

    if let Err(e) = state.handle.submit(incoming) {
        warn!("Scheduler at capacity: {e}");
        return Err(ApiError::service_unavailable(
            "Server is at capacity, please retry later",
        ));
    }

    Ok(delta_rx)
}

async fn submit_and_collect_buffered_response(
    state: &AppState,
    options: RequestExecutionOptions,
    prompt: String,
    request_kind: &'static str,
    route: &'static str,
) -> Result<(BufferedResponse, SpanContext), ApiError> {
    let trace = RequestTraceState::new(
        route,
        format!("http-{}", uuid::Uuid::new_v4()),
        false,
        options.max_tokens,
        prompt.len(),
        state.metrics.clone(),
    );
    let stream_parent = SpanContext::current_local_parent().unwrap_or_default();
    let stream_span = Span::root("stream_flush", stream_parent)
        .with_properties(|| [("route", route.to_string())]);
    let finish_parent = SpanContext::from_span(&stream_span).unwrap_or(stream_parent);
    let collect = async move {
        let mut trace = trace;
        let delta_rx = match submit_request(state, options, prompt).await {
            Ok(delta_rx) => delta_rx,
            Err(err) => {
                trace.finish(false, None, Some("submit_failed"));
                return Err(err);
            }
        };
        async move { collect_buffered_response_inner(delta_rx, request_kind, trace).await }
            .in_span(stream_span)
            .await
    };
    let buffered = tokio::time::timeout(RESPONSE_TIMEOUT, collect)
        .await
        .map_err(|_| non_streaming_timeout_error(request_kind))??;
    Ok((buffered, finish_parent))
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

fn is_deepseek_v4_model(model_id: &str) -> bool {
    let normalized = model_id.to_ascii_lowercase().replace(['_', '-'], "");
    normalized.contains("deepseekv4") || normalized.contains("dsv4")
}

fn build_chat_prompt(model_id: &str, req: &ChatCompletionRequest) -> String {
    if is_deepseek_v4_model(model_id) {
        let kwargs = req.chat_template_kwargs.as_ref();
        let options = DeepSeekV4ChatTemplateOptions {
            thinking: kwargs.and_then(|value| value.thinking).unwrap_or(false),
            reasoning_effort: kwargs.and_then(|value| value.reasoning_effort.clone()),
        };
        openai_messages_to_deepseek_v4_prompt(&req.messages, &req.tools, &options)
    } else {
        chat_messages_to_prompt(&req.messages, &req.tools)
    }
}

fn build_responses_prompt(req: &ResponsesRequest) -> String {
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
            messages.push(chat::OpenAiChatMessage {
                role: "user".into(),
                content: Some(text.clone().into()),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Message(message) => messages.push(message.clone()),
        ResponsesInput::Messages(items) => messages.extend(items.iter().cloned()),
    }

    chat_messages_to_prompt(&messages, &req.tools)
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
    continuous_usage_stats: bool,
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

    let emit_usage = include_usage && (is_terminal || continuous_usage_stats);
    if emit_usage {
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

pub(super) async fn completions(
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
        let model_id = state.identity.model_id.clone();
        req.validate_for_model(&model_id)?;
        let max_tokens = options.max_tokens;
        let stream = options.stream;
        let include_usage = options.include_usage;
        let continuous_usage_stats = options.continuous_usage_stats;
        let return_token_ids = req.return_token_ids_or_default();

        info!(
            "Received request: prompt_bytes={}, max_tokens={}, stream={}",
            req.prompt.len(),
            max_tokens,
            stream,
        );

        if stream {
            let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
            let mut trace = RequestTraceState::new(
                "/v1/completions",
                request_id.clone(),
                true,
                max_tokens,
                req.prompt.len(),
                state.metrics.clone(),
            );
            let delta_rx = match submit_request(state.as_ref(), options, req.prompt).await {
                Ok(delta_rx) => delta_rx,
                Err(err) => {
                    trace.finish(false, None, Some("submit_failed"));
                    return Err(err);
                }
            };
            let delta_rx = trace_streaming_deltas(delta_rx, trace);
            let created = now_secs();

            let sse_stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
                stream::iter(delta_sse_events(
                    delta,
                    include_usage,
                    continuous_usage_stats,
                    |d| StreamChunk::from_delta(&request_id, created, &model_id, d),
                    |u| StreamUsageChunk::from_usage(&request_id, created, &model_id, u),
                ))
            });

            Ok(Sse::new(sse_stream.chain(sse_done_stream()))
                .keep_alive(sse_keep_alive())
                .into_response())
        } else {
            let (buffered, finish_parent) = submit_and_collect_buffered_response(
                state.as_ref(),
                options,
                req.prompt,
                "request",
                "/v1/completions",
            )
            .await?;

            info!(
                "Request completed: prompt_tokens={}, completion_tokens={}",
                buffered.usage.prompt_tokens, buffered.usage.completion_tokens
            );

            async move {
                let response = CompletionResponse::from_output(
                    model_id,
                    now_secs(),
                    buffered.into_output(),
                    return_token_ids,
                );
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

pub(super) async fn chat_completions(
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
        let model_id = state.identity.model_id.clone();
        req.validate_for_model(&model_id)?;

        let max_tokens = options.max_tokens;
        let do_stream = options.stream;
        let include_usage = options.include_usage;
        let continuous_usage_stats = options.continuous_usage_stats;

        let prompt = build_chat_prompt(&model_id, &req);

        info!(
            "chat/completions: messages={}, prompt_bytes={}, max_tokens={}, stream={}",
            req.messages.len(),
            prompt.len(),
            max_tokens,
            do_stream,
        );

        if do_stream {
            let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let mut trace = RequestTraceState::new(
                "/v1/chat/completions",
                request_id.clone(),
                true,
                max_tokens,
                prompt.len(),
                state.metrics.clone(),
            );
            let delta_rx = match submit_request(state.as_ref(), options, prompt).await {
                Ok(delta_rx) => delta_rx,
                Err(err) => {
                    trace.finish(false, None, Some("submit_failed"));
                    return Err(err);
                }
            };
            let delta_rx = trace_streaming_deltas(delta_rx, trace);
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
                    continuous_usage_stats,
                    |d| ChatStreamChunk::content_chunk(&req_id, created, &mid, d),
                    |u| ChatStreamUsageChunk::from_usage(&req_id, created, &mid, u),
                ))
            });

            let full_stream = stream::once(async move { role_event })
                .chain(content_stream)
                .chain(sse_done_stream());

            Ok(Sse::new(full_stream)
                .keep_alive(sse_keep_alive())
                .into_response())
        } else {
            let (buffered, finish_parent) = submit_and_collect_buffered_response(
                state.as_ref(),
                options,
                prompt,
                "chat request",
                "/v1/chat/completions",
            )
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

pub(super) async fn models_handler(
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
    let response = ModelsListResponse::from_pool_specs(
        state.identity.model_id.as_str(),
        now_secs(),
        dflash,
        &state.config.pool_models,
    );
    Ok(Json(response).into_response())
}

pub(super) async fn responses_handler(
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
        let model_id = state.identity.model_id.clone();
        req.validate_for_model(&model_id)?;
        let prompt = build_responses_prompt(&req);
        let max_tokens = options.max_tokens;
        let stream = options.stream;

        info!(
            "responses: prompt_bytes={}, max_output_tokens={}",
            prompt.len(),
            max_tokens,
        );

        if stream {
            let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
            let mut trace = RequestTraceState::new(
                "/v1/responses",
                response_id.clone(),
                true,
                max_tokens,
                prompt.len(),
                state.metrics.clone(),
            );
            let delta_rx = match submit_request(state.as_ref(), options, prompt).await {
                Ok(delta_rx) => delta_rx,
                Err(err) => {
                    trace.finish(false, None, Some("submit_failed"));
                    return Err(err);
                }
            };
            let delta_rx = trace_streaming_deltas(delta_rx, trace);
            let created_at = now_secs();
            let stream = responses_sse_stream(delta_rx, response_id, created_at, model_id);
            Ok(Sse::new(stream.chain(sse_done_stream()))
                .keep_alive(sse_keep_alive())
                .into_response())
        } else {
            let (buffered, finish_parent) = submit_and_collect_buffered_response(
                state.as_ref(),
                options,
                prompt,
                "responses request",
                "/v1/responses",
            )
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

pub(super) async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_prometheus();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

pub(super) async fn healthz_handler() -> Json<HealthResponse> {
    Json(HealthResponse::live())
}

pub(super) async fn readyz_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse::ready(&state.identity.model_id))
}

pub(super) async fn stats_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<StatsQuery>,
) -> Response {
    if let Err(err) = authorize_v1_request(&headers, state.as_ref()) {
        return err.into_response();
    }
    if wants_json_stats(&headers, &query) {
        return Json(state.metrics.render_stats_json()).into_response();
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

pub(super) async fn train_status_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    proxy_train_control(state.as_ref(), "GET", "/v1/train/status", None).await
}

pub(super) async fn train_events_handler(
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

pub(super) async fn train_stop_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    authorize_v1_request(&headers, state.as_ref())?;
    proxy_train_control(state.as_ref(), "POST", "/v1/train/stop", None).await
}

pub(super) async fn train_save_handler(
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
