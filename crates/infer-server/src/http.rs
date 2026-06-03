//! axum router and route handlers for the OpenAI v1 facade.
//!
//! Wires the [`ServeHandle`] engine front door to non-streaming `/v1/completions`
//! and `/v1/chat/completions` endpoints. Request/response wire shapes live in
//! [`crate::schema`]; the tokenizer adapter in [`crate::tokenizer`]. This file
//! owns only request ingress: state, routing, and the per-request
//! encode -> submit -> collect -> decode flow.

#[cfg(feature = "metal")]
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
#[cfg(feature = "metal")]
use anyhow::{Context, anyhow};
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use infer_core::CompletedRequest;
#[cfg(feature = "metal")]
use infer_core::{Engine, SchedulerConfig};
use infer_seam::{BackendExecutor, KvPool};

use crate::ServeHandle;
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
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
        .with_state(state)
}

/// Build the R5 Metal serving facade from a model path or HuggingFace id.
#[cfg(feature = "metal")]
pub fn metal_openai_router_from_model_path(model_path: impl AsRef<Path>) -> Result<Router> {
    let model_source = model_path.as_ref().to_string_lossy().to_string();
    let tokenizer_dir = resolve_tokenizer_dir(&model_source)?;
    let tokenizer = OpenAiTokenizer::from_model_dir(&tokenizer_dir)?;
    let engine_model_source = model_source.clone();
    let serve = ServeHandle::spawn_with_engine_builder(move || {
        let executor = infer_metal::MetalExecutor::from_model_path(&engine_model_source)?;
        let mut config = SchedulerConfig::for_slots(4);
        config.max_prompt_tokens = 32_768;
        config.max_total_tokens = 65_536;
        config.chunked_prefill_size = 64;
        Ok(Engine::with_config(
            executor,
            infer_metal::MetalKvPool::new(4, 8192, 16),
            config,
        ))
    })?;
    Ok(openai_router(serve, tokenizer, model_source))
}

#[cfg(feature = "metal")]
fn resolve_tokenizer_dir(model_id_or_path: &str) -> Result<PathBuf> {
    if let Some(local) = local_tokenizer_dir(model_id_or_path) {
        return Ok(local);
    }

    if looks_like_local_path(model_id_or_path) {
        anyhow::bail!("model path does not contain tokenizer.json: {model_id_or_path}");
    }

    let mut builder = hf_hub::api::sync::ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.trim().is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .context("failed to initialise HuggingFace API for tokenizer")?;
    let repo = api.repo(hf_hub::Repo::new(
        model_id_or_path.to_string(),
        hf_hub::RepoType::Model,
    ));
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let _ = repo.get(name);
    }
    let tokenizer_json = repo
        .get("tokenizer.json")
        .with_context(|| format!("resolve tokenizer.json for {model_id_or_path}"))?;
    tokenizer_json
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("tokenizer.json has no parent path"))
}

#[cfg(feature = "metal")]
fn local_tokenizer_dir(model_id_or_path: &str) -> Option<PathBuf> {
    let local = Path::new(model_id_or_path);
    if local.join("tokenizer.json").exists() {
        return Some(local.to_path_buf());
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        let name = model_id_or_path
            .rsplit('/')
            .next()
            .unwrap_or(model_id_or_path);
        candidates.push(cwd.join("models").join(name));
        candidates.push(cwd.join("infer").join("models").join(name));
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join("models").join(model_id_or_path));
        if let Some((org, repo)) = model_id_or_path.split_once('/') {
            let cache = home
                .join(".cache")
                .join("huggingface")
                .join("hub")
                .join(format!("models--{org}--{repo}"));
            candidates.extend(snapshot_dirs(&cache));
        }
    }

    candidates
        .into_iter()
        .find(|candidate| candidate.join("tokenizer.json").exists())
}

#[cfg(feature = "metal")]
fn snapshot_dirs(cache_repo_dir: &Path) -> Vec<PathBuf> {
    let snapshot_root = cache_repo_dir.join("snapshots");
    let Ok(entries) = std::fs::read_dir(snapshot_root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .collect();
    dirs.sort();
    dirs.reverse();
    dirs
}

#[cfg(feature = "metal")]
fn looks_like_local_path(input: &str) -> bool {
    let trimmed = input.trim();
    trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with('~')
        || trimmed.matches('/').count() > 1
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
    let completed = generate(&state, prompt_tokens, max_tokens)?;
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
    let completed = generate(&state, prompt_tokens, max_tokens)?;
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
        serve.submit(prompt_tokens, max_tokens)?
    };
    ticket.collect().map_err(ApiError::from)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "metal")]
    use super::*;
    #[cfg(feature = "metal")]
    use axum::body::{Body, to_bytes};
    #[cfg(feature = "metal")]
    use axum::http::{Request, StatusCode};
    #[cfg(feature = "metal")]
    use serde_json::{Value, json};
    #[cfg(feature = "metal")]
    use tower::ServiceExt;

    #[cfg(feature = "metal")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "loads MLX weights and runs real Metal generation"]
    async fn metal_openai_chat_completions_returns_real_text() -> Result<()> {
        let model = std::env::var("INFER_SERVER_E2E_MODEL")
            .unwrap_or_else(|_| "mlx-community/Qwen3.5-0.8B-MLX-4bit".to_string());
        let app = metal_openai_router_from_model_path(&model)?;
        let body = json!({
            "model": model,
            "messages": [
                {"role": "user", "content": "Write one short sentence about local inference."}
            ],
            "max_tokens": 16,
            "temperature": 0.0
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body)?))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 20).await?;
        let value: Value = serde_json::from_slice(&bytes)?;
        let text = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let completion_tokens = value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        println!("generated_text={text:?}");
        println!("completion_tokens={completion_tokens}");
        assert!(!text.is_empty(), "generated text must not be empty");
        assert!(completion_tokens > 0, "completion token count must be > 0");
        Ok(())
    }
}
