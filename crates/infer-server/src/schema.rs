//! OpenAI v1 wire types and the API error shape (COLD — fixed external contract).
//!
//! Request/response bodies for `/v1/completions` and `/v1/chat/completions`, the
//! sampling-field mapping into the shared [`SamplingParams`] contract, and
//! [`ApiError`] / its [`IntoResponse`] rendering. The HTTP handlers in
//! [`crate::http`] own request ingress; this file owns only the wire shapes and
//! their validation/conversion.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use infer_core::CompletedRequest;
use infer_plan::{FinishReason, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::execution::CounterSnapshot;

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
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
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
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
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
    pub(crate) fn from_completed(model: String, completed: CompletedRequest, text: String) -> Self {
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

/// `GET /v1/models` response — the OpenAI model-list shape. The server serves a
/// single loaded model, so `data` always has one card.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

impl ModelsResponse {
    pub(crate) fn single(model: String) -> Self {
        Self {
            object: "list",
            data: vec![ModelCard {
                id: model,
                object: "model",
                created: unix_time_secs(),
                owned_by: "arle",
            }],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StatsResponse {
    pub scheduler: SchedulerStats,
    pub throughput: ThroughputStatsResponse,
    pub prefix_cache: PrefixCacheStatsResponse,
    pub kv_tier: KvTierStatsResponse,
    pub ssd_recall: SsdRecallStats,
}

impl StatsResponse {
    pub(crate) fn from_counters(counters: CounterSnapshot) -> Self {
        Self {
            scheduler: SchedulerStats {
                active_requests: counters.active_requests,
                queue_depth: counters.queue_depth,
                kv_free_pages: counters.kv_free_pages,
            },
            throughput: ThroughputStatsResponse {
                steps: counters.throughput.steps,
                prefill_tokens: counters.throughput.prefill_tokens,
                generated_tokens: counters.throughput.generated_tokens,
                requests_completed: counters.throughput.requests_completed,
            },
            kv_tier: KvTierStatsResponse {
                available: counters.kv_tier.demoted_pages > 0
                    || counters.kv_tier.resident_blocks > 0
                    || counters.kv_tier.demoted_slots > 0,
                demoted_pages: counters.kv_tier.demoted_pages,
                promoted_pages: counters.kv_tier.promoted_pages,
                promote_failures: counters.kv_tier.promote_failures,
                resident_blocks: counters.kv_tier.resident_blocks,
                demoted_slots: counters.kv_tier.demoted_slots,
                promoted_slots: counters.kv_tier.promoted_slots,
                slot_promote_failures: counters.kv_tier.slot_promote_failures,
            },
            prefix_cache: PrefixCacheStatsResponse {
                lookups: counters.prefix_cache.lookups,
                hits: counters.prefix_cache.hits,
                hit_rate: ratio(counters.prefix_cache.hits, counters.prefix_cache.lookups),
                hit_tokens: counters.prefix_cache.hit_tokens,
                hit_pages: counters.prefix_cache.hit_pages,
                published_pages: counters.prefix_cache.published_pages,
                cached_pages: counters.prefix_cache.cached_pages,
            },
            ssd_recall: SsdRecallStats {
                available: false,
                lookups: 0,
                hits: 0,
                recall_rate: None,
                not_available_reason: "per-level ssd recall counters are not split out yet; \
                                       T2 disk spill activity is included in the kv_tier block",
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SchedulerStats {
    pub active_requests: usize,
    pub queue_depth: usize,
    pub kv_free_pages: usize,
}

/// Engine throughput counters (monotonic since engine start), for QPS/TPS
/// computation by polling clients.
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStatsResponse {
    pub steps: u64,
    pub prefill_tokens: u64,
    pub generated_tokens: u64,
    pub requests_completed: u64,
}

/// KV host-tier (T1 DRAM) counters. All zero until a backend with a tier
/// store is configured (`available` keys off observed tier activity).
#[derive(Debug, Clone, Serialize)]
pub struct KvTierStatsResponse {
    pub available: bool,
    pub demoted_pages: u64,
    pub promoted_pages: u64,
    pub promote_failures: u64,
    pub resident_blocks: usize,
    pub demoted_slots: u64,
    pub promoted_slots: u64,
    pub slot_promote_failures: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrefixCacheStatsResponse {
    pub lookups: u64,
    pub hits: u64,
    pub hit_rate: Option<f64>,
    pub hit_tokens: u64,
    pub hit_pages: u64,
    pub published_pages: u64,
    pub cached_pages: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SsdRecallStats {
    pub available: bool,
    pub lookups: u64,
    pub hits: u64,
    pub recall_rate: Option<f64>,
    pub not_available_reason: &'static str,
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
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
    pub(crate) fn from_completed(
        model: String,
        completed: CompletedRequest,
        content: String,
    ) -> Self {
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
pub(crate) struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    pub(crate) fn too_many_requests(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        let message = value.to_string();
        if message.starts_with("server is busy:") {
            return Self::too_many_requests(message);
        }
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
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
    use super::*;

    #[test]
    fn models_response_is_openai_single_model_list() {
        let resp = ModelsResponse::single("Qwen3-8B".to_string());
        let v = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"].as_array().expect("data array").len(), 1);
        assert_eq!(v["data"][0]["id"], "Qwen3-8B");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "arle");
        assert!(v["data"][0]["created"].is_u64());
    }

    #[test]
    fn stats_response_reports_prefix_rate_and_ssd_unavailable() {
        let resp = StatsResponse::from_counters(CounterSnapshot {
            active_requests: 0,
            queue_depth: 0,
            kv_free_pages: 7,
            prefix_cache: infer_core::PrefixCacheStats {
                lookups: 4,
                hits: 3,
                hit_tokens: 96,
                hit_pages: 6,
                published_pages: 8,
                cached_pages: 8,
            },
            throughput: infer_core::ThroughputStats {
                steps: 12,
                prefill_tokens: 300,
                generated_tokens: 48,
                requests_completed: 3,
            },
            kv_tier: infer_core::KvTierStats::default(),
        });

        let v = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(v["scheduler"]["kv_free_pages"], 7);
        assert_eq!(v["prefix_cache"]["hit_rate"], 0.75);
        assert_eq!(v["prefix_cache"]["hit_tokens"], 96);
        assert_eq!(v["throughput"]["generated_tokens"], 48);
        assert_eq!(v["throughput"]["requests_completed"], 3);
        assert_eq!(v["kv_tier"]["available"], false);
        assert_eq!(v["kv_tier"]["demoted_pages"], 0);
        assert_eq!(v["ssd_recall"]["available"], false);
        assert!(v["ssd_recall"]["recall_rate"].is_null());
    }
}
