use std::path::Path;
#[cfg(feature = "metal")]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use infer_core::CompletedRequest;
#[cfg(feature = "metal")]
use infer_core::{Engine, SchedulerConfig};
use infer_plan::{FinishReason, SamplingParams};
use infer_seam::{BackendExecutor, KvPool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::ServeHandle;

/// Tokenizer and chat-template adapter for the OpenAI v1 facade.
///
/// The tokenizer is loaded directly from `tokenizer.json`. `tokenizer_config`
/// is read to detect a model-provided chat template; R5 keeps rendering minimal
/// and Qwen-compatible rather than carrying the legacy prompt stack.
#[derive(Clone)]
pub struct OpenAiTokenizer {
    inner: Tokenizer,
    chat_template: Option<String>,
}

impl OpenAiTokenizer {
    /// Load `tokenizer.json` and optional `tokenizer_config.json` from a model dir.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {} failed: {err}", tokenizer_path.display()))?;
        let chat_template = read_chat_template(model_dir)?;
        Ok(Self {
            inner,
            chat_template,
        })
    }

    /// Encode text into token ids without adding special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids into text, skipping special tokens.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))
    }

    /// Render OpenAI chat messages into the Qwen ChatML prompt form.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        ensure!(
            !messages.is_empty(),
            "messages must contain at least one message"
        );
        if let Some(template) = &self.chat_template
            && !template.contains("<|im_start|>")
        {
            log::warn!(
                "tokenizer_config.json has an unknown chat_template shape; using Qwen ChatML fallback"
            );
        }

        let mut out = String::new();
        for message in messages {
            let role = message.role.trim();
            ensure!(!role.is_empty(), "message role must not be empty");
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            if let Some(content) = message.content.as_deref() {
                out.push_str(content);
            }
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    }
}

fn read_chat_template(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join("tokenizer_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read tokenizer config {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse tokenizer config {}", path.display()))?;
    Ok(value
        .get("chat_template")
        .and_then(|template| template.as_str().map(str::to_owned)))
}

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

/// Minimal `/v1/completions` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub ignore_eos: Option<bool>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

impl CompletionRequest {
    fn validate(&self) -> Result<(), ApiError> {
        if self.prompt.trim().is_empty() {
            return Err(ApiError::bad_request("prompt must not be empty"));
        }
        validate_common(self.stream, self.max_tokens)
    }

    /// Convert compatible sampling fields into the shared pure-data contract.
    #[must_use]
    pub fn sampling_params(&self) -> SamplingParams {
        sampling_params(
            self.max_tokens,
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
            self.ignore_eos,
            self.stop_token_ids.clone(),
            self.seed,
        )
    }
}

/// Minimal `/v1/chat/completions` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub ignore_eos: Option<bool>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

impl ChatCompletionRequest {
    fn validate(&self) -> Result<(), ApiError> {
        if self.messages.is_empty() {
            return Err(ApiError::bad_request(
                "messages must contain at least one message",
            ));
        }
        validate_common(self.stream, self.max_tokens)
    }

    /// Convert compatible sampling fields into the shared pure-data contract.
    #[must_use]
    pub fn sampling_params(&self) -> SamplingParams {
        sampling_params(
            self.max_tokens,
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
            self.ignore_eos,
            self.stop_token_ids.clone(),
            self.seed,
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

fn validate_common(stream: Option<bool>, max_tokens: Option<usize>) -> Result<(), ApiError> {
    if stream.unwrap_or(false) {
        return Err(ApiError::bad_request(
            "stream=true is deferred in R5 tranche 2",
        ));
    }
    if matches!(max_tokens, Some(0)) {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sampling_params(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    ignore_eos: Option<bool>,
    stop_token_ids: Option<Vec<u32>>,
    seed: Option<u64>,
) -> SamplingParams {
    let default = SamplingParams::default();
    SamplingParams {
        temperature: temperature.unwrap_or(default.temperature),
        top_k: top_k.unwrap_or(default.top_k),
        top_p: top_p.unwrap_or(default.top_p),
        min_p: min_p.unwrap_or(default.min_p),
        repetition_penalty: repetition_penalty.unwrap_or(default.repetition_penalty),
        frequency_penalty: frequency_penalty.unwrap_or(default.frequency_penalty),
        presence_penalty: presence_penalty.unwrap_or(default.presence_penalty),
        ignore_eos: ignore_eos.unwrap_or(default.ignore_eos),
        stop_token_ids: stop_token_ids.unwrap_or_default(),
        seed,
        max_new_tokens: max_tokens,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    fn from_completed(model: String, completed: CompletedRequest, text: String) -> Self {
        let usage = Usage::new(
            completed.prompt_tokens.len(),
            completed.generated_tokens.len(),
        );
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "text_completion",
            created: unix_time_secs(),
            model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: finish_reason(completed.finish.as_ref()).to_string(),
            }],
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    fn from_completed(model: String, completed: CompletedRequest, content: String) -> Self {
        let usage = Usage::new(
            completed.prompt_tokens.len(),
            completed.generated_tokens.len(),
        );
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion",
            created: unix_time_secs(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content,
                },
                finish_reason: finish_reason(completed.finish.as_ref()).to_string(),
            }],
            usage,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
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

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: value.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": "invalid_request_error",
                    "param": null,
                    "code": null
                }
            })),
        )
            .into_response()
    }
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
    use serde_json::Value;
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
