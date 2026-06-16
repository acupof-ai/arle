//! axum router and route handlers for the OpenAI v1 facade.
//!
//! Wires the [`ServeHandle`] engine front door to `/v1/completions`
//! and non-streaming `/v1/chat/completions` endpoints. Request/response wire shapes live in
//! [`crate::schema`]; the tokenizer adapter in [`crate::tokenizer`]. This file
//! owns only request ingress: state, routing, and the per-request
//! encode -> submit -> collect -> decode flow.

use std::convert::Infallible;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use infer_core::CompletedRequest;
use infer_plan::{DiffusionGenerateOutput, FinishReason, MultimodalImage, SamplingParams};
use infer_seam::{BackendExecutor, KvPool};
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use crate::multimodal::{expand_gemma4_image_markers, preprocess_gemma4_image};
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, ChatContent, ChatContentPart,
    CompletionRequest, CompletionResponse, ModelsResponse, StatsResponse,
};
use crate::tokenizer::OpenAiTokenizer;
use crate::{RequestTicket, ServeHandle, StreamItem};

struct HttpState<E: BackendExecutor, K: KvPool> {
    model: String,
    tokenizer: Mutex<OpenAiTokenizer>,
    serve: ServeHandle<E, K>,
}

/// Build an axum router exposing non-streaming OpenAI v1 completions.
pub fn openai_router<E, K>(
    serve: ServeHandle<E, K>,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
) -> Router
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let state = Arc::new(HttpState {
        model: model.into(),
        tokenizer: Mutex::new(tokenizer),
        serve,
    });
    Router::new()
        .route("/v1/completions", post(completions::<E, K>))
        .route("/v1/chat/completions", post(chat_completions::<E, K>))
        .route("/v1/models", get(list_models::<E, K>))
        .route("/v1/stats", get(stats::<E, K>))
        .route("/metrics", get(metrics::<E, K>))
        // Long-context prompts (e.g. a ~900K-token needle) serialize to several MB,
        // far over axum's 2 MiB default. Allow up to 256 MiB (a 1M-token prompt is
        // only a handful of MB; the cap stays well under any real DoS concern here).
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

/// `GET /v1/models` — list the single served model (OpenAI-compatible; clients
/// like openai-python call this to discover the model id).
async fn list_models<E, K>(State(state): State<Arc<HttpState<E, K>>>) -> Json<ModelsResponse>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    Json(ModelsResponse::single(state.model.clone()))
}

/// `GET /v1/stats` — compact runtime counters for bench and smoke probes.
async fn stats<E, K>(
    State(state): State<Arc<HttpState<E, K>>>,
) -> Result<Json<StatsResponse>, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let counters = state.serve.counters();
    Ok(Json(StatsResponse::from_counters(counters)))
}

/// `GET /metrics` — Prometheus text exposition of the same counters `/v1/stats`
/// serves as JSON (scrape surface for bench/monitoring tooling).
async fn metrics<E, K>(
    State(state): State<Arc<HttpState<E, K>>>,
) -> Result<([(header::HeaderName, &'static str); 1], String), ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let counters = state.serve.counters();
    Ok((
        [(
            header::CONTENT_TYPE,
            crate::metrics::PROMETHEUS_CONTENT_TYPE,
        )],
        crate::metrics::render_prometheus(&counters, &state.model),
    ))
}

async fn completions<E, K>(
    State(state): State<Arc<HttpState<E, K>>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    request.validate()?;
    let sampling = request.sampling_params();
    let max_tokens = sampling.max_new_tokens.unwrap_or(16);
    let prompt_tokens = encode(&state, &request.prompt)?;
    if request.stream.unwrap_or(false) {
        return stream_completion(state, prompt_tokens, max_tokens, sampling);
    }
    let completed = generate(&state, prompt_tokens, max_tokens, sampling)?;
    let text = decode(&state, &completed.generated_tokens)?;
    Ok(Json(CompletionResponse::from_completed(
        state.model.clone(),
        completed,
        text,
    ))
    .into_response())
}

async fn chat_completions<E, K>(
    State(state): State<Arc<HttpState<E, K>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    request.validate()?;
    let sampling = request.sampling_params();
    let max_tokens = sampling.max_new_tokens.unwrap_or(16);
    let images = extract_gemma4_images(&request.messages)?;
    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer.render_chat(&request.messages)?
    };
    if !images.is_empty() {
        let prompt = expand_gemma4_image_markers(&prompt, &images)?;
        let prompt_tokens = encode(&state, &prompt)?;
        let prompt_token_count = prompt_tokens.len();
        let output = generate_multimodal(&state, prompt_tokens, images, max_tokens, sampling)?;
        let content = decode(&state, &output.generated_tokens)?;
        return Ok(Json(ChatCompletionResponse::from_parts(
            state.model.clone(),
            content,
            prompt_token_count,
            output.generated_tokens.len(),
            Some(&output.finish),
        )));
    }
    let prompt_tokens = encode(&state, &prompt)?;
    let completed = generate(&state, prompt_tokens, max_tokens, sampling)?;
    let content = decode(&state, &completed.generated_tokens)?;
    Ok(Json(ChatCompletionResponse::from_completed(
        state.model.clone(),
        completed,
        content,
    )))
}

fn encode<E: BackendExecutor, K: KvPool>(
    state: &Arc<HttpState<E, K>>,
    text: &str,
) -> Result<Vec<u32>, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.encode(text)?)
}

fn decode<E: BackendExecutor, K: KvPool>(
    state: &Arc<HttpState<E, K>>,
    tokens: &[u32],
) -> Result<String, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.decode(tokens)?)
}

fn generate<E, K>(
    state: &Arc<HttpState<E, K>>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<CompletedRequest, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let ticket = state.serve.submit(prompt_tokens, max_tokens, sampling)?;
    ticket.collect().map_err(ApiError::from)
}

fn stream_completion<E, K>(
    state: Arc<HttpState<E, K>>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<Response, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let prompt_token_count = prompt_tokens.len();
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?
        .clone();
    let (ticket, stream_rx) = state
        .serve
        .submit_streaming(prompt_tokens, max_tokens, sampling)?;
    let model = state.model.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(16);

    thread::spawn(move || {
        drive_completion_sse(model, tokenizer, prompt_token_count, ticket, stream_rx, tx);
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn drive_completion_sse(
    model: String,
    tokenizer: OpenAiTokenizer,
    prompt_token_count: usize,
    ticket: RequestTicket,
    stream_rx: Receiver<StreamItem>,
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_time_secs();
    let mut acc_ids = Vec::new();
    let mut emitted = String::new();

    while let Ok(item) = stream_rx.recv() {
        match item {
            StreamItem::Token { token, .. } => {
                acc_ids.push(token);
                let full = tokenizer.decode(&acc_ids).unwrap_or_default();
                if full.len() > emitted.len() && full.starts_with(&emitted) {
                    let delta = full[emitted.len()..].to_string();
                    emitted.push_str(&delta);
                    if !send_sse_json(
                        &tx,
                        completion_stream_chunk(&id, created, &model, delta, None, None),
                    ) {
                        // Client (SSE receiver) disconnected. Stop sending, but
                        // do NOT drop the ticket here: the engine request is
                        // still running on the backend and continues to occupy a
                        // live slot. Returning now would drop `ticket` and
                        // decrement `live_requests` while GPU work is in flight,
                        // letting the frontend over-admit past the backend cap.
                        // Break to `ticket.collect()` below, which keeps the slot
                        // counted until the engine actually finishes (the stream
                        // channel is unbounded, so the engine never blocks on our
                        // stopped draining).
                        break;
                    }
                }
            }
            StreamItem::Done(completed) => {
                finish_completion_sse(
                    &model,
                    &tokenizer,
                    &id,
                    created,
                    prompt_token_count,
                    completed,
                    emitted,
                    &tx,
                );
                return;
            }
        }
    }

    if let Ok(completed) = ticket.collect() {
        finish_completion_sse(
            &model,
            &tokenizer,
            &id,
            created,
            prompt_token_count,
            completed,
            emitted,
            &tx,
        );
    }
}

fn finish_completion_sse(
    model: &str,
    tokenizer: &OpenAiTokenizer,
    id: &str,
    created: u64,
    prompt_token_count: usize,
    completed: CompletedRequest,
    mut emitted: String,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let full = tokenizer
        .decode(&completed.generated_tokens)
        .unwrap_or_default();
    if full.len() > emitted.len() && full.starts_with(&emitted) {
        let delta = full[emitted.len()..].to_string();
        emitted.push_str(&delta);
        if !send_sse_json(
            tx,
            completion_stream_chunk(id, created, model, delta, None, None),
        ) {
            return;
        }
    }

    let usage = json!({
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completed.generated_tokens.len(),
        "total_tokens": prompt_token_count + completed.generated_tokens.len(),
    });
    let _ = send_sse_json(
        tx,
        completion_stream_chunk(
            id,
            created,
            model,
            String::new(),
            Some(finish_reason(completed.finish.as_ref())),
            Some(usage),
        ),
    );
    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
}

fn send_sse_json(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    value: serde_json::Value,
) -> bool {
    tx.blocking_send(Ok(Event::default().data(value.to_string())))
        .is_ok()
}

fn completion_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    finish: Option<&str>,
    usage: Option<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": finish,
        }],
        "usage": usage,
    })
}

fn finish_reason(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) | None => "length",
        Some(FinishReason::Abort) => "abort",
    }
}

fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn generate_multimodal<E, K>(
    state: &Arc<HttpState<E, K>>,
    prompt_tokens: Vec<u32>,
    images: Vec<MultimodalImage>,
    max_tokens: usize,
    sampling: SamplingParams,
) -> Result<DiffusionGenerateOutput, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    let output = state.serve.run_on_executor(move |executor| {
        executor.generate_multimodal(&prompt_tokens, &images, max_tokens, &sampling)
    })??;
    output.ok_or_else(|| {
        ApiError::bad_request("image content requires a backend with Gemma4 VLM support")
    })
}

fn extract_gemma4_images(
    messages: &[crate::schema::ChatMessage],
) -> Result<Vec<MultimodalImage>, ApiError> {
    let mut images = Vec::new();
    for message in messages {
        let Some(ChatContent::Parts(parts)) = &message.content else {
            continue;
        };
        for part in parts {
            match part.normalized_kind() {
                "image" => {
                    let url = image_data_url(part)?;
                    let bytes = decode_image_data_url(url)?;
                    images.push(preprocess_gemma4_image(&bytes).map_err(|err| {
                        ApiError::bad_request(format!("invalid image data: {err}"))
                    })?);
                }
                "audio" | "video" => {
                    return Err(ApiError::bad_request(
                        "audio/video content parts are not supported by Gemma4 VLM",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(images)
}

fn image_data_url(part: &ChatContentPart) -> Result<&str, ApiError> {
    fn value_url(value: &serde_json::Value) -> Option<&str> {
        value
            .as_str()
            .or_else(|| value.get("url").and_then(serde_json::Value::as_str))
    }
    part.image_url
        .as_ref()
        .and_then(value_url)
        .or_else(|| part.input_image.as_ref().and_then(value_url))
        .or_else(|| part.extra.get("image_url").and_then(value_url))
        .or_else(|| part.extra.get("url").and_then(serde_json::Value::as_str))
        .ok_or_else(|| ApiError::bad_request("image content part must include image_url.url"))
}

fn decode_image_data_url(url: &str) -> Result<Vec<u8>, ApiError> {
    let Some((header, payload)) = url.split_once(',') else {
        return Err(ApiError::bad_request(
            "image_url must be a data:image/...;base64 URL",
        ));
    };
    let header = header.to_ascii_lowercase();
    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
        return Err(ApiError::bad_request(
            "image_url must be a data:image/...;base64 URL",
        ));
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.as_bytes())
        .map_err(|err| ApiError::bad_request(format!("invalid base64 image data: {err}")))
}
