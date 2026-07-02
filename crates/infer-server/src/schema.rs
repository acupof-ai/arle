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
use infer_plan::{FinishReason, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::multiproc_relay::WireStats;

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
    pub return_token_ids: Option<bool>,
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
    /// Pass-through extra arguments for the checkpoint's Jinja chat template
    /// (vLLM / SGLang field name). The HF chat-template render receives these
    /// as top-level context variables — most notably `enable_thinking: bool`,
    /// which toggles the Qwen `{% if enable_thinking %}` reasoning branch.
    /// Absent (the default) renders byte-identically to before this field.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
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

    /// Whether `chat_template_kwargs.enable_thinking` is set truthy. Used to
    /// decide if the thinking budget applies AND if the serve lifts reasoning
    /// into `reasoning_content`. The request value wins when present; absent or
    /// non-boolean falls back to `default_on` (the server-side default — clients
    /// that can't set the kwarg, e.g. terminal-bench's terminus/litellm, still
    /// get the split when the operator turns the budget on).
    #[must_use]
    pub(crate) fn enable_thinking(&self, default_on: bool) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(default_on)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<ChatContent>,
}

impl ChatMessage {
    #[must_use]
    pub fn content_text(&self) -> String {
        self.content
            .as_ref()
            .map_or_else(String::new, ChatContent::to_text)
    }

    #[must_use]
    pub fn has_media_content(&self) -> bool {
        self.content
            .as_ref()
            .is_some_and(ChatContent::has_media_content)
    }

    #[must_use]
    pub(crate) fn template_content(&self) -> serde_json::Value {
        self.content.as_ref().map_or(
            serde_json::Value::String(String::new()),
            ChatContent::to_template_value,
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl ChatContent {
    #[must_use]
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| {
                    (part.kind == "text")
                        .then_some(part.text.as_deref())
                        .flatten()
                })
                .collect(),
        }
    }

    #[must_use]
    pub fn has_media_content(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(ChatContentPart::is_media),
        }
    }

    #[must_use]
    pub(crate) fn to_template_value(&self) -> serde_json::Value {
        match self {
            Self::Text(text) => serde_json::Value::String(text.clone()),
            Self::Parts(parts) => serde_json::Value::Array(
                parts
                    .iter()
                    .map(ChatContentPart::to_template_value)
                    .collect(),
            ),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<serde_json::Value>,
    #[serde(default)]
    pub input_image: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatContentPart {
    pub(crate) fn normalized_kind(&self) -> &str {
        match self.kind.as_str() {
            "image_url" | "input_image" => "image",
            "input_audio" => "audio",
            other => other,
        }
    }

    fn is_media(&self) -> bool {
        matches!(self.normalized_kind(), "image" | "audio" | "video")
    }

    fn to_template_value(&self) -> serde_json::Value {
        let mut object = self.extra.clone();
        object.insert(
            "type".to_string(),
            serde_json::Value::String(self.normalized_kind().to_string()),
        );
        if let Some(text) = &self.text {
            object.insert("text".to_string(), serde_json::Value::String(text.clone()));
        }
        if let Some(image_url) = &self.image_url {
            object.insert("image_url".to_string(), image_url.clone());
        }
        if let Some(input_image) = &self.input_image {
            object.insert("input_image".to_string(), input_image.clone());
        }
        serde_json::Value::Object(object)
    }
}

fn validate_common(_stream: Option<bool>, max_tokens: Option<usize>) -> Result<(), ApiError> {
    if max_tokens == Some(0) {
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
    /// Build a completion response from already-decoded parts (the multiproc
    /// coordinator path, which has no in-process `CompletedRequest`).
    pub(crate) fn from_parts(
        model: String,
        text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish: Option<&FinishReason>,
        token_ids: Option<Vec<u32>>,
    ) -> Self {
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "text_completion",
            created: unix_time_secs(),
            model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: finish_reason(finish).to_string(),
                token_ids,
            }],
            usage: Usage::new(prompt_tokens, completion_tokens),
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
    pub kv_system: KvSystemMetricsResponse,
    pub ssd_recall: SsdRecallStats,
}

impl StatsResponse {
    pub(crate) fn from_wire(w: WireStats) -> Self {
        Self::from_counters(w.into_counter_snapshot())
    }

    pub(crate) fn from_counters(counters: crate::execution::CounterSnapshot) -> Self {
        let prefix = counters.prefix_cache;
        let tier = counters.kv_tier;
        let system = counters.kv_system;
        // Stored pages count as tier activity too: a prefix blob parked in the
        // store (host or disk) proves the tier is live before the first
        // demote/promote flips a counter.
        let tier_available = tier.demoted_pages > 0
            || tier.promoted_pages > 0
            || tier.promote_failures > 0
            || tier.resident_blocks > 0
            || tier.demoted_slots > 0
            || tier.promoted_slots > 0
            || tier.slot_promote_failures > 0
            || system.host_demoted_pages > 0
            || system.disk_pages > 0;
        let ssd_available = system.disk_pages > 0 || system.reuse_hit_disk > 0;
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
            prefix_cache: PrefixCacheStatsResponse {
                lookups: prefix.lookups,
                hits: prefix.hits,
                hit_rate: ratio(prefix.hits.min(prefix.lookups), prefix.lookups),
                hit_tokens: prefix.hit_tokens,
                hit_pages: prefix.hit_pages,
                published_pages: prefix.published_pages,
                cached_pages: prefix.cached_pages,
            },
            kv_tier: KvTierStatsResponse {
                available: tier_available,
                demoted_pages: tier.demoted_pages,
                promoted_pages: tier.promoted_pages,
                promote_failures: tier.promote_failures,
                resident_blocks: tier.resident_blocks,
                demoted_slots: tier.demoted_slots,
                promoted_slots: tier.promoted_slots,
                slot_promote_failures: tier.slot_promote_failures,
            },
            kv_system: KvSystemMetricsResponse {
                resident_pages: system.resident_pages,
                resident_evictable_pages: system.resident_evictable_pages,
                host_demoted_pages: system.host_demoted_pages,
                host_demoted_pending_inflight: system.host_demoted_pending_inflight,
                disk_pages: system.disk_pages,
                reuse_hit_resident: system.reuse_hit_resident,
                reuse_hit_host_demoted: system.reuse_hit_host_demoted,
                reuse_hit_disk: system.reuse_hit_disk,
                reuse_miss: system.reuse_miss,
                demote_mset_count: system.demote_mset_count,
                demote_mset_copy_bytes: system.demote_mset_copy_bytes,
                demote_mset_copy_ms: system.demote_mset_copy_ms,
                promote_mget_count: system.promote_mget_count,
                promote_mget_copy_bytes: system.promote_mget_copy_bytes,
                promote_mget_copy_ms: system.promote_mget_copy_ms,
                fetch_wait_ms: system.fetch_wait_ms,
                fallback_recompute: system.fallback_recompute,
                prefix_match_full_blocks: system.prefix_match_full_blocks,
                prefix_match_clamped_blocks: system.prefix_match_clamped_blocks,
            },
            ssd_recall: SsdRecallStats {
                available: ssd_available,
                not_available_reason: if ssd_available {
                    ""
                } else {
                    "no SSD recall activity observed"
                },
                ..SsdRecallStats::default()
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

/// KV host-demoted counters. All zero until a backend with a tier
/// store is configured (`available` keys off observed tier activity).
#[derive(Debug, Clone, Default, Serialize)]
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

#[derive(Debug, Clone, Default, Serialize)]
pub struct KvSystemMetricsResponse {
    pub resident_pages: usize,
    pub resident_evictable_pages: usize,
    pub host_demoted_pages: usize,
    pub host_demoted_pending_inflight: usize,
    pub disk_pages: usize,
    pub reuse_hit_resident: u64,
    pub reuse_hit_host_demoted: u64,
    pub reuse_hit_disk: u64,
    pub reuse_miss: u64,
    pub demote_mset_count: u64,
    pub demote_mset_copy_bytes: u64,
    pub demote_mset_copy_ms: u64,
    pub promote_mget_count: u64,
    pub promote_mget_copy_bytes: u64,
    pub promote_mget_copy_ms: u64,
    pub fetch_wait_ms: u64,
    pub fallback_recompute: u64,
    pub prefix_match_full_blocks: u64,
    pub prefix_match_clamped_blocks: u64,
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

#[derive(Debug, Clone, Default, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
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
    pub(crate) fn from_parts(
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish: Option<&FinishReason>,
        enable_thinking: bool,
    ) -> Self {
        let (reasoning_content, content) = split_reasoning(&content, enable_thinking);
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
                    reasoning_content,
                },
                finish_reason: finish_reason(finish).to_string(),
            }],
            usage: Usage::new(prompt_tokens, completion_tokens),
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
    /// Thinking-model reasoning lifted out of `content` (everything before
    /// `</think>`). Absent for non-thinking outputs so the wire shape stays
    /// byte-identical to before for those.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

const THINK_END: &str = "</think>";

/// Split a thinking-model's generated text into `(reasoning_content, content)`.
///
/// The chat template pre-fills `<think>\n` into the *prompt*, so the model emits
/// `reasoning</think>answer` (closer + answer). We split on the closer to give
/// API consumers a clean `content` (the actual answer/commands).
/// - Thinking OFF: byte-identical passthrough `(None, text)` — no scan, so an
///   answer that legitimately contains a literal `</think>` is never truncated
///   (matches SGLang/vLLM: parse reasoning only when the lane is active).
/// - Thinking ON, has `</think>`: reasoning = text before it (leading `<think>`
///   stripped, trimmed); content = text after it (leading whitespace trimmed).
/// - Thinking ON, no `</think>`: truncated thinking → all reasoning, empty
///   content (the answer never arrived; a max_tokens concern, not ours).
///
/// Empty reasoning collapses to `None` so the field is omitted.
///
// ponytail: chat SSE splits incrementally via
// `sse_util::StreamingReasoningSplitter` — keep the two policies in lockstep.
fn split_reasoning(text: &str, enable_thinking: bool) -> (Option<String>, String) {
    if !enable_thinking {
        return (None, text.to_string());
    }
    let (reasoning, content) = match text.find(THINK_END) {
        Some(idx) => (
            text[..idx]
                .trim_start()
                .trim_start_matches("<think>")
                .trim(),
            text[idx + THINK_END.len()..].trim_start(),
        ),
        // Truncated thinking: the closer never arrived, so neither did the answer.
        None => (text.trim(), ""),
    };
    (
        (!reasoning.is_empty()).then(|| reasoning.to_string()),
        content.to_string(),
    )
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
    fn chat_content_parts_parse_text_and_image() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        }))
        .unwrap();
        assert_eq!(request.messages[0].content_text(), "what is this?");
        assert!(request.messages[0].has_media_content());
        let template = request.messages[0].template_content();
        assert_eq!(template[1]["type"], "image");
        assert!(request.validate().is_ok());
    }

    #[test]
    fn text_completions_allow_streaming_for_guidellm() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "prompt": "hello",
            "stream": true,
            "max_tokens": 1
        }))
        .unwrap();
        assert!(request.validate().is_ok());
    }

    #[test]
    fn completion_token_ids_are_opt_in() {
        let hidden = CompletionResponse::from_parts(
            "m".to_string(),
            String::new(),
            3,
            2,
            Some(&FinishReason::Length),
            None,
        );
        let hidden_json = serde_json::to_value(&hidden).expect("serialize hidden");
        assert!(hidden_json["choices"][0].get("token_ids").is_none());

        let visible = CompletionResponse::from_parts(
            "m".to_string(),
            String::new(),
            3,
            2,
            Some(&FinishReason::Length),
            Some(vec![1, 2]),
        );
        let visible_json = serde_json::to_value(&visible).expect("serialize visible");
        assert_eq!(visible_json["choices"][0]["token_ids"], json!([1, 2]));
    }

    #[test]
    fn stats_response_relays_tier_counters() {
        let mut counters = crate::execution::CounterSnapshot::default();
        counters.prefix_cache.lookups = 2;
        counters.prefix_cache.hits = 5;
        counters.kv_tier.demoted_slots = 2;
        counters.kv_system.reuse_hit_host_demoted = 3;
        counters.kv_system.disk_pages = 4;
        let stats = StatsResponse::from_counters(counters);
        assert_eq!(stats.prefix_cache.hit_rate, Some(1.0));
        assert_eq!(stats.prefix_cache.hits, 5);
        assert!(stats.kv_tier.available);
        assert_eq!(stats.kv_tier.demoted_slots, 2);
        assert_eq!(stats.kv_system.reuse_hit_host_demoted, 3);
        assert!(stats.ssd_recall.available);
    }

    #[test]
    fn enable_thinking_request_value_wins_over_server_default() {
        // request omits the kwarg → falls back to the server default
        let absent: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(!absent.enable_thinking(false), "absent + default off → off");
        assert!(
            absent.enable_thinking(true),
            "absent + default on → on (terminus/litellm path: split fires)"
        );
        // request sets it explicitly → that wins regardless of the default
        let on: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": true}
        }))
        .unwrap();
        assert!(
            on.enable_thinking(false),
            "request true overrides default off"
        );
        let off: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .unwrap();
        assert!(
            !off.enable_thinking(true),
            "request false overrides default on"
        );
    }

    #[test]
    fn split_reasoning_lifts_thinking_out_of_content() {
        // (a) closer + answer
        assert_eq!(
            split_reasoning("reason</think>answer", true),
            (Some("reason".to_string()), "answer".to_string())
        );
        // (b) leading <think> opener stripped
        assert_eq!(
            split_reasoning("<think>r</think>a", true),
            (Some("r".to_string()), "a".to_string())
        );
        // (c) no closer + thinking on → all reasoning, empty content
        assert_eq!(
            split_reasoning("only thinking no close", true),
            (Some("only thinking no close".to_string()), String::new())
        );
        // (d) no closer + thinking off → byte-identical passthrough
        assert_eq!(
            split_reasoning("plain answer", false),
            (None, "plain answer".to_string())
        );
        // (e) whitespace around the closer is trimmed off the content
        assert_eq!(
            split_reasoning("r\n</think>\n\nanswer", true),
            (Some("r".to_string()), "answer".to_string())
        );
        // (f) BLOCKER fix: thinking OFF never scans — a literal `</think>` in a
        // non-thinking answer passes through byte-identically (not truncated).
        assert_eq!(
            split_reasoning("The closing tag is </think> in HTML.", false),
            (None, "The closing tag is </think> in HTML.".to_string())
        );
        // (g) multiple closers → split on the FIRST; stray closer stays in content
        assert_eq!(
            split_reasoning("r1</think>mid</think>answer", true),
            (Some("r1".to_string()), "mid</think>answer".to_string())
        );
        // (h) empty reasoning → None (no `"reasoning_content":""` on the wire)
        assert_eq!(
            split_reasoning("</think>answer", true),
            (None, "answer".to_string())
        );
        assert_eq!(
            split_reasoning("<think></think>answer", true),
            (None, "answer".to_string())
        );
        // (i) empty output
        assert_eq!(split_reasoning("", true), (None, String::new()));
        // (j) leading whitespace before the opener still strips it
        assert_eq!(
            split_reasoning(" <think>r</think>a", true),
            (Some("r".to_string()), "a".to_string())
        );
    }

    #[test]
    fn assistant_message_omits_reasoning_when_absent() {
        let resp = ChatCompletionResponse::from_parts(
            "Qwen3-8B".to_string(),
            "plain answer".to_string(),
            3,
            2,
            Some(&FinishReason::Stop),
            false,
        );
        let v = serde_json::to_value(&resp).expect("serialize");
        let message = &v["choices"][0]["message"];
        assert_eq!(message["content"], "plain answer");
        assert!(
            message.get("reasoning_content").is_none(),
            "reasoning_content must be omitted for non-thinking output"
        );
    }

    #[test]
    fn chat_completions_allow_streaming() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 1
        }))
        .unwrap();
        assert!(request.validate().is_ok());
    }
}
