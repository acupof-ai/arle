//! axum router and route handlers for the OpenAI v1 facade.
//!
//! Wires the [`ServeHandle`] engine front door to non-streaming `/v1/completions`
//! and `/v1/chat/completions` endpoints. Request/response wire shapes live in
//! [`crate::schema`]; the tokenizer adapter in [`crate::tokenizer`]. This file
//! owns only request ingress: state, routing, and the per-request
//! encode -> submit -> collect -> decode flow.

use std::sync::{Arc, Mutex};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use infer_core::CompletedRequest;
use infer_plan::{DiffusionGenerateOutput, MultimodalImage, SamplingParams};
use infer_seam::{BackendExecutor, KvPool};

use crate::ServeHandle;
use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, ChatContent, ChatContentPart,
    CompletionRequest, CompletionResponse, ModelsResponse, StatsResponse,
};
use crate::tokenizer::OpenAiTokenizer;

const GEMMA4_IMAGE_MARKER: &str = "<|image|>";
const GEMMA4_BOI_MARKER: &str = "<|image>";
const GEMMA4_EOI_MARKER: &str = "<image|>";
const GEMMA4_PATCH_SIZE: usize = 16;
const GEMMA4_POOLING_KERNEL: usize = 3;
const GEMMA4_MAX_SOFT_TOKENS: usize = 280;

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
    let counters = state
        .serve
        .lock()
        .map_err(|_| ApiError::internal("serve lock poisoned"))?
        .counters();
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
    let counters = state
        .serve
        .lock()
        .map_err(|_| ApiError::internal("serve lock poisoned"))?
        .counters();
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
    let ticket = {
        let serve = state
            .serve
            .lock()
            .map_err(|_| ApiError::internal("serve lock poisoned"))?;
        serve.submit(prompt_tokens, max_tokens, sampling)?
    };
    ticket.collect().map_err(ApiError::from)
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
    let output = {
        let serve = state
            .serve
            .lock()
            .map_err(|_| ApiError::internal("serve lock poisoned"))?;
        serve.run_on_executor(move |executor| {
            executor.generate_multimodal(&prompt_tokens, &images, max_tokens, &sampling)
        })?
    }?;
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
                    images.push(preprocess_gemma4_image(&bytes)?);
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

fn preprocess_gemma4_image(bytes: &[u8]) -> Result<MultimodalImage, ApiError> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| ApiError::bad_request(format!("invalid image data: {err}")))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(ApiError::bad_request("image dimensions must be non-zero"));
    }
    let (target_width, target_height) = gemma4_resize_shape(width as usize, height as usize)?;
    let resized = image::imageops::resize(
        &image,
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let mut pixels = vec![0.0f32; 3 * target_height * target_width];
    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let base = y * target_width + x;
            pixels[base] = f32::from(pixel[0]) / 255.0;
            pixels[target_height * target_width + base] = f32::from(pixel[1]) / 255.0;
            pixels[2 * target_height * target_width + base] = f32::from(pixel[2]) / 255.0;
        }
    }
    let patches = (target_height / GEMMA4_PATCH_SIZE) * (target_width / GEMMA4_PATCH_SIZE);
    let soft_token_count = patches / (GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL);
    Ok(MultimodalImage {
        pixels,
        channels: 3,
        height: target_height,
        width: target_width,
        soft_token_count,
    })
}

fn gemma4_resize_shape(width: usize, height: usize) -> Result<(usize, usize), ApiError> {
    let side_multiple = GEMMA4_POOLING_KERNEL * GEMMA4_PATCH_SIZE;
    let max_patches = GEMMA4_MAX_SOFT_TOKENS * GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL;
    let target_pixels = (max_patches * GEMMA4_PATCH_SIZE * GEMMA4_PATCH_SIZE) as f64;
    let factor = (target_pixels / (width * height) as f64).sqrt();
    let quantize = |side: usize| -> usize {
        (((factor * side as f64) / side_multiple as f64).floor() as usize) * side_multiple
    };
    let mut target_width = quantize(width);
    let mut target_height = quantize(height);
    if target_width == 0 && target_height == 0 {
        return Err(ApiError::bad_request(
            "image is too small for Gemma4 patch preprocessing",
        ));
    }
    let max_side_length =
        (max_patches / (GEMMA4_POOLING_KERNEL * GEMMA4_POOLING_KERNEL)) * side_multiple;
    if target_width == 0 {
        target_width = side_multiple;
        target_height = (((height as f64 / width as f64) * side_multiple as f64).floor() as usize)
            .min(max_side_length)
            .max(side_multiple);
    }
    if target_height == 0 {
        target_height = side_multiple;
        target_width = (((width as f64 / height as f64) * side_multiple as f64).floor() as usize)
            .min(max_side_length)
            .max(side_multiple);
    }
    Ok((target_width, target_height))
}

fn expand_gemma4_image_markers(
    prompt: &str,
    images: &[MultimodalImage],
) -> Result<String, ApiError> {
    let mut output = String::with_capacity(prompt.len() + images.len() * 4096);
    let mut rest = prompt;
    for image in images {
        let Some(pos) = rest.find(GEMMA4_IMAGE_MARKER) else {
            return Err(ApiError::bad_request(
                "chat template did not emit enough Gemma4 image markers",
            ));
        };
        output.push_str(&rest[..pos]);
        output.push_str(GEMMA4_BOI_MARKER);
        for _ in 0..image.soft_token_count {
            output.push_str(GEMMA4_IMAGE_MARKER);
        }
        output.push_str(GEMMA4_EOI_MARKER);
        rest = &rest[pos + GEMMA4_IMAGE_MARKER.len()..];
    }
    output.push_str(rest);
    if output.matches(GEMMA4_IMAGE_MARKER).count()
        != images
            .iter()
            .map(|image| image.soft_token_count)
            .sum::<usize>()
    {
        return Err(ApiError::bad_request(
            "chat template emitted more Gemma4 image markers than provided images",
        ));
    }
    Ok(output)
}
