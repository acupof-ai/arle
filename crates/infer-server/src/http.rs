//! axum router and route handlers for the OpenAI v1 facade.
//!
//! Wires the [`ServeHandle`] engine front door to non-streaming `/v1/completions`
//! and `/v1/chat/completions` endpoints. Request/response wire shapes live in
//! [`crate::schema`]; the tokenizer adapter in [`crate::tokenizer`]. This file
//! owns only request ingress: state, routing, and the per-request
//! encode -> submit -> collect -> decode flow.

use std::sync::{Arc, Mutex};

use axum::extract::{DefaultBodyLimit, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use infer_core::CompletedRequest;
use infer_plan::SamplingParams;
use infer_seam::{BackendExecutor, KvPool};

use crate::ServeHandle;
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    ModelsResponse,
};
use crate::tokenizer::OpenAiTokenizer;

struct HttpState<E: BackendExecutor, K: KvPool> {
    model: String,
    tokenizer: Mutex<OpenAiTokenizer>,
    serve: Mutex<ServeHandle<E, K>>,
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
        serve: Mutex::new(serve),
    });
    Router::new()
        .route("/v1/completions", post(completions::<E, K>))
        .route("/v1/chat/completions", post(chat_completions::<E, K>))
        .route("/v1/models", get(list_models::<E, K>))
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

async fn completions<E, K>(
    State(state): State<Arc<HttpState<E, K>>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    request.validate()?;
    let sampling = request.sampling_params();
    let max_tokens = sampling.max_new_tokens.unwrap_or(16);
    let prompt_tokens = encode(&state, &request.prompt)?;
    let completed = generate(&state, prompt_tokens, max_tokens, sampling)?;
    let text = decode(&state, &completed.generated_tokens)?;
    Ok(Json(CompletionResponse::from_completed(
        state.model.clone(),
        completed,
        text,
    )))
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
    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer.render_chat(&request.messages)?
    };
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
    let ticket = {
        let serve = state
            .serve
            .lock()
            .map_err(|_| ApiError::internal("serve lock poisoned"))?;
        serve.submit(prompt_tokens, max_tokens, sampling)?
    };
    ticket.collect().map_err(ApiError::from)
}
