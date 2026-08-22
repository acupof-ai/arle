//! OpenAI v1 wire types and the API error shape (COLD — fixed external contract).
//!
//! Request/response bodies for `/v1/completions` and `/v1/chat/completions`, the
//! sampling-field mapping into the shared [`SamplingParams`] contract, and
//! [`ApiError`] / its [`IntoResponse`] rendering. The HTTP handlers in
//! [`crate::http`] own request ingress; this file owns only the wire shapes and
//! their validation/conversion.

use std::path::Path;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use infer_plan::{FinishReason, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::multiproc_relay::WireStats;

/// Serve-wide sampling defaults for request fields left unset. Both halves of
/// the default matter: temperature drives greedy-vs-sample, and the nucleus
/// (`top_k`/`top_p`/`min_p`) truncates the tail — omitting the nucleus while
/// forcing temperature>0 draws over the full vocab tail (token salad).
/// Initialized to the shipped greedy/no-filter default, byte-identical to
/// [`SamplingParams::default`] until serve init overrides it from the model's
/// `generation_config.json`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingDefaults {
    /// Non-zero keeps rollout logprobs non-empty (greedy emits `logprob: None`)
    /// — the cc rollout F.6 invariant; the model nucleus then truncates the tail.
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}

impl SamplingDefaults {
    /// Read `temperature`/`top_k`/`top_p` from the model's
    /// `generation_config.json`; a missing file, unreadable file, or missing key
    /// keeps the greedy/no-filter default for that field (tolerant parse, mirrors
    /// `Qwen35Config::load_stop_token_ids`). `min_p` has no wire key — stays default.
    #[must_use]
    pub fn from_generation_config(model_dir: impl AsRef<Path>) -> Self {
        let mut defaults = Self::default();
        let path = model_dir.as_ref().join("generation_config.json");
        let Ok(content) = std::fs::read_to_string(&path) else {
            return defaults;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) else {
            return defaults;
        };
        if let Some(t) = value.get("temperature").and_then(serde_json::Value::as_f64) {
            defaults.temperature = t as f32;
        }
        if let Some(k) = value.get("top_k").and_then(serde_json::Value::as_i64) {
            defaults.top_k = k as i32;
        }
        if let Some(p) = value.get("top_p").and_then(serde_json::Value::as_f64) {
            defaults.top_p = p as f32;
        }
        defaults
    }
}

// Set-once-at-serve-init, read-per-request. Const initializer = the shipped
// greedy/no-filter default (matches `SamplingParams::default()`).
static SAMPLING_DEFAULTS: RwLock<SamplingDefaults> = RwLock::new(SamplingDefaults {
    temperature: 0.0,
    top_k: -1,
    top_p: 1.0,
    min_p: 0.0,
});

pub fn set_sampling_defaults(defaults: SamplingDefaults) {
    *SAMPLING_DEFAULTS.write().unwrap() = defaults;
}

fn sampling_defaults() -> SamplingDefaults {
    *SAMPLING_DEFAULTS.read().unwrap()
}

/// OpenAI `stream_options`: `{"include_usage": true}` asks the streaming
/// response for a trailing usage-only chunk (empty `choices`, populated
/// `usage`) right before `[DONE]` — mirrors vLLM/SGLang. Absent/false keeps
/// every streamed chunk's `usage: null`, byte-identical to before this field.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// `/v1/completions` `prompt`: a text string (tokenized as usual) OR a raw
/// token-id array (fed to the engine verbatim, skipping tokenization). The
/// token-id form preserves EXACT ids across a multi-turn feed-back — re-encoding
/// a decoded completion shifts boundary tokens and truncates prefix reuse.
/// Untagged: a JSON string → [`Self::Text`], a JSON number array → [`Self::Tokens`].
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Tokens(Vec<u32>),
}

impl PromptInput {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.trim().is_empty(),
            Self::Tokens(ids) => ids.is_empty(),
        }
    }
}

/// OpenAI `stop`: a single string OR an array of strings (vLLM/SGLang/OpenAI
/// all accept both). Normalizes either into `Vec<String>`; absence stays `None`.
fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        One(String),
        Many(Vec<String>),
    }
    Ok(
        Option::<StringOrVec>::deserialize(deserializer)?.map(|stop| match stop {
            StringOrVec::One(one) => vec![one],
            StringOrVec::Many(many) => many,
        }),
    )
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: PromptInput,
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
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    pub return_token_ids: Option<bool>,
    /// OpenAI structured output. `json_schema` constrains generation to the
    /// schema; `json_object` to any valid JSON.
    #[serde(default)]
    pub response_format: Option<crate::grammar::ResponseFormat>,
    /// Number of completions to generate. Engine supports only one; accepted
    /// for client compatibility (values > 1 are ignored).
    #[serde(default)]
    pub n: Option<usize>,
    /// OpenAI completions `logprobs`: per-token logprob of the sampled token
    /// plus the top-N alternatives (0–8; larger is rejected). Surfaced by the
    /// CUDA Qwen3.5/3.6 executor; other backends answer 501.
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Token-id → bias map added to the logits before sampling.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// End-user identifier for abuse monitoring. Accepted but unused.
    #[serde(default)]
    pub user: Option<String>,
}

impl CompletionRequest {
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        if self.prompt.is_empty() {
            return Err(ApiError::bad_request("prompt must not be empty"));
        }
        if let Some(n) = self.logprobs
            && n > MAX_LOGPROBS
        {
            return Err(ApiError::bad_request(format!(
                "logprobs must be at most {MAX_LOGPROBS}"
            )));
        }
        validate_common(
            self.max_tokens,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )
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
            self.logit_bias.clone(),
            self.n,
            self.logprobs.map(|n| n as usize),
        )
    }
}

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
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    /// Pass-through extra arguments for the checkpoint's Jinja chat template
    /// (vLLM / SGLang field name). The HF chat-template render receives these
    /// as top-level context variables — most notably `enable_thinking: bool`,
    /// which toggles the Qwen `{% if enable_thinking %}` reasoning branch.
    /// Absent (the default) renders byte-identically to before this field.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// OpenAI function-tool definitions. Empty (the default) = no tools, so the
    /// prompt/response wire shape stays byte-identical to a tool-less request.
    #[serde(default)]
    pub tools: Vec<chat::OpenAiToolDefinition>,
    /// OpenAI `tool_choice`: `"auto"` / `"none"` / `"required"` or a
    /// `{"type":"function","function":{"name":…}}` object. Parsed lazily into
    /// [`chat::ToolChoiceMode`] via [`Self::tool_choice_mode`].
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI structured output. `json_schema` constrains generation to the
    /// schema; `json_object` to any valid JSON.
    #[serde(default)]
    pub response_format: Option<crate::grammar::ResponseFormat>,
    /// Number of chat completion choices to generate. Engine supports only one;
    /// accepted for client compatibility (values > 1 are ignored).
    #[serde(default)]
    pub n: Option<usize>,
    /// Whether to return per-token logprobs. Surfaced (with `top_logprobs`
    /// alternatives) by the CUDA Qwen3.5/3.6 executor; other backends fall
    /// back to the legacy behavior-logprob shape.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top-k logprobs to return per token (0–8; larger is rejected).
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Token-id → bias map added to the logits before sampling.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// End-user identifier for abuse monitoring. Accepted but unused.
    #[serde(default)]
    pub user: Option<String>,
    /// Whether to allow parallel tool calls. Defaults to true in OpenAI; the
    /// engine emits one tool call at a time, so this is accepted but ignored.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI service tier. Accepted but unused.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Whether to store the completion for later retrieval. Accepted but unused.
    #[serde(default)]
    pub store: Option<bool>,
    /// Arbitrary metadata attached to the request. Accepted but unused.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

impl ChatCompletionRequest {
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        if self.messages.is_empty() {
            return Err(ApiError::bad_request(
                "messages must contain at least one message",
            ));
        }
        if let Some(n) = self.top_logprobs
            && n > MAX_LOGPROBS
        {
            return Err(ApiError::bad_request(format!(
                "top_logprobs must be at most {MAX_LOGPROBS}"
            )));
        }
        validate_common(
            self.max_tokens,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )
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
            self.logit_bias.clone(),
            self.n,
            self.logprobs
                .unwrap_or(false)
                .then(|| self.top_logprobs.unwrap_or(0) as usize),
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

    /// Map the OpenAI `tool_choice` wire value to [`chat::ToolChoiceMode`].
    /// Absent → `Auto`; a forced-function object collapses to `Function(name)`
    /// (treated as `Required` when the name is missing).
    pub(crate) fn tool_choice_mode(&self) -> chat::ToolChoiceMode {
        use chat::ToolChoiceMode;
        match self.tool_choice.as_ref() {
            None => ToolChoiceMode::Auto,
            Some(serde_json::Value::String(choice)) => match choice.as_str() {
                "none" => ToolChoiceMode::None,
                "required" => ToolChoiceMode::Required,
                _ => ToolChoiceMode::Auto,
            },
            Some(serde_json::Value::Object(object)) => object
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(serde_json::Value::as_str)
                .map_or(ToolChoiceMode::Required, |name| {
                    ToolChoiceMode::Function(name.to_string())
                }),
            Some(_) => ToolChoiceMode::Auto,
        }
    }

    /// Whether tools should be advertised to the model and parsed back out:
    /// tool definitions present AND `tool_choice` not `"none"`.
    #[must_use]
    pub(crate) fn wants_tools(&self) -> bool {
        !self.tools.is_empty() && !matches!(self.tool_choice_mode(), chat::ToolChoiceMode::None)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<ChatContent>,
    /// Assistant tool calls carried in the request history (round-trip form).
    #[serde(default)]
    pub tool_calls: Vec<chat::OpenAiToolCall>,
    /// `tool`-role reply linkage — the call id being answered.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// `tool`-role reply linkage — the tool name.
    #[serde(default)]
    pub name: Option<String>,
}

impl ChatMessage {
    #[must_use]
    pub fn content_text(&self) -> String {
        self.content
            .as_ref()
            .map_or_else(String::new, ChatContent::to_text)
    }

    /// Convert to the `chat` crate's OpenAI wire message so the shared
    /// renderers (DeepSeek-V4, ChatML+tools) consume one canonical shape.
    #[must_use]
    pub(crate) fn to_openai(&self) -> chat::OpenAiChatMessage {
        chat::OpenAiChatMessage {
            role: self.role.clone(),
            content: Some(chat::OpenAiChatContent::Text(self.content_text())),
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
            name: self.name.clone(),
        }
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

fn validate_common(
    max_tokens: Option<usize>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> Result<(), ApiError> {
    if max_tokens == Some(0) {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    // Rejected rather than clamped: a clamp silently answers a different
    // question than the one asked. `repetition_penalty` 0 is also a NaN hazard —
    // it maps a grammar-masked `-inf` to NaN, which outranks `+inf` in
    // `total_cmp` and would hand the argmax to a forbidden token.
    if let Some(p) = repetition_penalty
        && !(p > 0.0 && p <= 2.0)
    {
        return Err(ApiError::bad_request(
            "repetition_penalty must be in (0, 2]",
        ));
    }
    for (name, value) in [
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
    ] {
        // NaN is not contained in any range, so it is rejected here too.
        if let Some(p) = value
            && !(-2.0..=2.0).contains(&p)
        {
            return Err(ApiError::bad_request(format!("{name} must be in [-2, 2]")));
        }
    }
    Ok(())
}

/// Cap on requested logprobs alternatives (OpenAI allows up to 5 on
/// completions / 20 on chat; the capture is host-side O(vocab·n) per token).
pub(crate) const MAX_LOGPROBS: u32 = 8;

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
    logit_bias: Option<std::collections::HashMap<u32, f32>>,
    n: Option<usize>,
    top_logprobs: Option<usize>,
) -> SamplingParams {
    let default = SamplingParams::default();
    let serve = sampling_defaults();
    SamplingParams {
        temperature: temperature.unwrap_or(serve.temperature),
        top_k: top_k.unwrap_or(serve.top_k),
        top_p: top_p.unwrap_or(serve.top_p),
        min_p: min_p.unwrap_or(serve.min_p),
        repetition_penalty: repetition_penalty.unwrap_or(default.repetition_penalty),
        frequency_penalty: frequency_penalty.unwrap_or(default.frequency_penalty),
        presence_penalty: presence_penalty.unwrap_or(default.presence_penalty),
        ignore_eos: ignore_eos.unwrap_or(default.ignore_eos),
        stop_token_ids: stop_token_ids.unwrap_or_default(),
        seed,
        max_new_tokens: max_tokens,
        grammar_bitmask: None,
        logit_bias: {
            let mut v: Vec<(u32, f32)> = logit_bias.unwrap_or_default().into_iter().collect();
            v.sort_by_key(|&(tok, _)| tok);
            v
        },
        n: n.unwrap_or(1).max(1),
        top_logprobs,
        force_next_token: None,
        think_end_token_id: None,
        think_start_token_id: None,
        max_thinking_tokens: None,
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
    /// OpenAI `system_fingerprint` — a stable identifier for the model/backend
    /// configuration. Static for a given build.
    pub system_fingerprint: &'static str,
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
        prompt_token_ids: Option<Vec<u32>>,
        logprobs_value: Option<serde_json::Value>,
    ) -> Self {
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "text_completion",
            created: unix_time_secs(),
            model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: logprobs_value,
                finish_reason: finish_reason(finish).to_string(),
                token_ids,
                prompt_token_ids,
            }],
            usage: Usage::new(prompt_tokens, completion_tokens),
            system_fingerprint: SYSTEM_FINGERPRINT,
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
    pub build_identity: crate::BuildIdentity,
    pub scheduler: SchedulerStats,
    pub throughput: ThroughputStatsResponse,
    pub prefix_cache: PrefixCacheStatsResponse,
    pub kv_tier: KvTierStatsResponse,
    pub kv_system: KvSystemMetricsResponse,
    pub ssd_recall: SsdRecallStats,
    pub spec_decode: SpecDecodeStatsResponse,
    pub operator_dispatch: infer_seam::OperatorDispatchStats,
    pub op_timing: infer_seam::OpTimingStats,
    pub gpu: Option<infer_seam::GpuSample>,
}

impl StatsResponse {
    pub(crate) fn from_wire(mut w: WireStats) -> Self {
        let build_identity = std::mem::take(&mut w.build_identity);
        let operator_dispatch = std::mem::take(&mut w.operator_dispatch);
        Self::from_counters(w.into_counter_snapshot(), build_identity, operator_dispatch)
    }

    pub(crate) fn from_counters(
        counters: crate::execution::CounterSnapshot,
        build_identity: crate::BuildIdentity,
        operator_dispatch: infer_seam::OperatorDispatchStats,
    ) -> Self {
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
            || tier.slot_demote_failures > 0
            || tier.promoted_slots > 0
            || tier.slot_promote_failures > 0
            || system.host_demoted_pages > 0
            || system.disk_pages > 0;
        let ssd_available = system.disk_pages > 0 || system.reuse_hit_disk > 0;
        Self {
            build_identity,
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
                requests_succeeded: counters.throughput.requests_succeeded,
                requests_failed: counters.throughput.requests_failed,
                ttft_micros_total: counters.throughput.ttft_micros_total,
                ttft_count: counters.throughput.ttft_count,
                tpot_micros_total: counters.throughput.tpot_micros_total,
                tpot_count: counters.throughput.tpot_count,
                e2e_micros_total: counters.throughput.e2e_micros_total,
                e2e_count: counters.throughput.e2e_count,
                forward_busy_micros: counters.throughput.forward_busy_micros,
                prefill_forward_steps: counters.throughput.prefill_forward_steps,
                prefill_forward_busy_micros: counters.throughput.prefill_forward_busy_micros,
                decode_forward_steps: counters.throughput.decode_forward_steps,
                decode_forward_busy_micros: counters.throughput.decode_forward_busy_micros,
                mixed_forward_steps: counters.throughput.mixed_forward_steps,
                mixed_forward_busy_micros: counters.throughput.mixed_forward_busy_micros,
                decode_step_phase: StepPhaseStatsResponse {
                    steps: counters.throughput.decode_step_phase.steps,
                    poll_micros: counters.throughput.decode_step_phase.poll_micros,
                    apply_output_micros: counters.throughput.decode_step_phase.apply_output_micros,
                    poll_background_micros: counters
                        .throughput
                        .decode_step_phase
                        .poll_background_micros,
                    admit_micros: counters.throughput.decode_step_phase.admit_micros,
                    plan_micros: counters.throughput.decode_step_phase.plan_micros,
                    submit_micros: counters.throughput.decode_step_phase.submit_micros,
                },
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
                slot_demote_failures: tier.slot_demote_failures,
                promoted_slots: tier.promoted_slots,
                slot_promote_failures: tier.slot_promote_failures,
            },
            kv_system: KvSystemMetricsResponse {
                resident_pages: system.resident_pages,
                resident_evictable_pages: system.resident_evictable_pages,
                host_demoted_pages: system.host_demoted_pages,
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
                tier_io_mode: system.tier_io_mode,
                tier_io_useful_read_bytes: system.tier_io_useful_read_bytes,
                tier_io_useful_write_bytes: system.tier_io_useful_write_bytes,
                tier_io_submitted_read_bytes: system.tier_io_submitted_read_bytes,
                tier_io_submitted_write_bytes: system.tier_io_submitted_write_bytes,
                tier_io_metadata_write_bytes: system.tier_io_metadata_write_bytes,
                tier_io_failures: system.tier_io_failures,
                tier_io_completion_wait_ns: system.tier_io_completion_wait_ns,
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
            spec_decode: SpecDecodeStatsResponse {
                available: counters.spec_decode.chains > 0,
                chains: counters.spec_decode.chains,
                drafted: counters.spec_decode.drafted,
                accepted: counters.spec_decode.accepted,
                rejected: counters.spec_decode.rejected,
                partial_ctx_chains: counters.spec_decode.partial_ctx_chains,
                accept_rate: ratio(counters.spec_decode.accepted, counters.spec_decode.drafted),
            },
            operator_dispatch,
            op_timing: counters.op_timing,
            gpu: counters.gpu,
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
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub ttft_micros_total: u64,
    pub ttft_count: u64,
    pub tpot_micros_total: u64,
    pub tpot_count: u64,
    pub e2e_micros_total: u64,
    pub e2e_count: u64,
    pub forward_busy_micros: u64,
    pub prefill_forward_steps: u64,
    pub prefill_forward_busy_micros: u64,
    pub decode_forward_steps: u64,
    pub decode_forward_busy_micros: u64,
    pub mixed_forward_steps: u64,
    pub mixed_forward_busy_micros: u64,
    pub decode_step_phase: StepPhaseStatsResponse,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepPhaseStatsResponse {
    pub steps: u64,
    pub poll_micros: u64,
    pub apply_output_micros: u64,
    pub poll_background_micros: u64,
    pub admit_micros: u64,
    pub plan_micros: u64,
    pub submit_micros: u64,
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
    pub slot_demote_failures: u64,
    pub promoted_slots: u64,
    pub slot_promote_failures: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KvSystemMetricsResponse {
    pub resident_pages: usize,
    pub resident_evictable_pages: usize,
    pub host_demoted_pages: usize,
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
    pub tier_io_mode: infer_seam::KvTierIoMode,
    pub tier_io_useful_read_bytes: u64,
    pub tier_io_useful_write_bytes: u64,
    pub tier_io_submitted_read_bytes: u64,
    pub tier_io_submitted_write_bytes: u64,
    pub tier_io_metadata_write_bytes: u64,
    pub tier_io_failures: u64,
    pub tier_io_completion_wait_ns: u64,
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

/// Cumulative speculative-decode counters (MTP or DSpark). All zero (and
/// `available: false`) until a verified draft chain commits.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SpecDecodeStatsResponse {
    pub available: bool,
    pub chains: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub partial_ctx_chains: u64,
    pub accept_rate: Option<f64>,
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
    /// OpenAI completions logprobs object (`tokens` / `token_logprobs` /
    /// `top_logprobs` / `text_offset`), present when the request asked for it.
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
    /// The PROMPT's token ids — paired with `token_ids` under `return_token_ids`,
    /// so one turn-1 call yields both the prompt ids and the generated ids for an
    /// exact-token multi-turn feed-back. Omitted (default) → byte-identical wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// OpenAI `system_fingerprint` — a stable identifier for the model/backend
    /// configuration. Static for a given build.
    pub system_fingerprint: &'static str,
}

impl ChatCompletionResponse {
    pub(crate) fn from_parts(
        model: String,
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_tokens: usize,
        finish: Option<&FinishReason>,
        enable_thinking: bool,
        tool_calls: Vec<ResponseToolCall>,
        logprobs_value: Option<serde_json::Value>,
    ) -> Self {
        let (reasoning_content, content) = split_reasoning(&content, enable_thinking);
        // OpenAI semantics: any emitted tool call overrides the finish reason.
        let finish_reason = if tool_calls.is_empty() {
            finish_reason(finish).to_string()
        } else {
            "tool_calls".to_string()
        };
        let usage = if enable_thinking {
            Usage::with_reasoning(prompt_tokens, completion_tokens, reasoning_tokens)
        } else {
            Usage::new(prompt_tokens, completion_tokens)
        };
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
                    tool_calls,
                },
                logprobs: logprobs_value,
                finish_reason,
            }],
            usage,
            system_fingerprint: SYSTEM_FINGERPRINT,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    /// OpenAI chat logprobs object (`content` entries), present when the
    /// request asked for it.
    pub logprobs: Option<serde_json::Value>,
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
    /// Tool calls parsed from the model output. Empty (the default) is omitted so
    /// a tool-less response is byte-identical to before this field existed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ResponseToolCall>,
}

/// One OpenAI-format tool call in a chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    /// JSON-encoded argument object (OpenAI wire shape is a string, not object).
    pub arguments: String,
}

impl ResponseToolCall {
    /// Build a response tool call from a parsed [`chat::ToolCall`], minting a
    /// stable-per-response id from the call index plus a uuid tail.
    pub(crate) fn from_parsed(call: &chat::ToolCall, index: usize) -> Self {
        Self {
            id: format!("call_{index}_{}", uuid::Uuid::new_v4().simple()),
            call_type: "function",
            function: ResponseFunctionCall {
                name: call.name.clone(),
                arguments: serde_json::to_string(&call.arguments)
                    .unwrap_or_else(|_| "{}".to_string()),
            },
        }
    }
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// OpenAI `system_fingerprint` — a stable identifier for the model/backend
/// configuration. Baked into the binary; changes only with the build.
pub(crate) const SYSTEM_FINGERPRINT: &str = "arle_fp_1";

/// Split a thinking-model's generated text into `(reasoning_content, content)`.
///
/// State machine over `<think>` / `</think>` markers: text inside a thinking
/// block accumulates into `reasoning_content`, text outside into `content`.
/// Handles multi-segment thinking (`r1</think>c1<think>r2</think>c2`) and
/// models that re-open the block after answering.
///
/// Two entry shapes:
/// - **Thinking ON**: the chat template pre-fills `<think>\n` into the *prompt*,
///   so the model emits `reasoning</think>answer` (no opening tag).
/// - **Thinking OFF but model emits `<think>` anyway** (e.g. Qwen3.6):
///   output is `<think>reasoning</think>answer`.
///
/// When the model never closes the think block (hit max_tokens or a repetition
/// loop), the entire output is returned as `content` — the user gets the
/// model's best answer instead of an empty response.
///
/// Non-thinking outputs (no leading `<think>`, thinking off) pass through
/// byte-identically.
///
/// Empty reasoning collapses to `None` so the field is omitted.
// ponytail: chat SSE splits incrementally via
// `sse_util::StreamingReasoningSplitter` — keep the two policies in lockstep.
// pub(crate): also the canonical pre-split for the tools path
// (`coordinator::finalize_chat_content`) — the tool parser's paired-tag strip
// misses the prompt-prefilled `reasoning</think>` form.
pub(crate) fn split_reasoning(text: &str, enable_thinking: bool) -> (Option<String>, String) {
    let trimmed = text.trim_start();

    // Non-thinking output without a leading <think>: byte-identical passthrough.
    if !enable_thinking && !trimmed.starts_with(THINK_START) {
        return (None, text.to_string());
    }

    let mut reasoning = String::new();
    let mut content = String::new();
    let mut buf = trimmed;
    let mut in_thinking = true;

    // Strip the model's own <think> opener if present.
    if let Some(rest) = buf.strip_prefix(THINK_START) {
        buf = rest;
    }

    while !buf.is_empty() {
        if in_thinking {
            match buf.find(THINK_END) {
                Some(idx) => {
                    reasoning.push_str(&buf[..idx]);
                    buf = &buf[idx + THINK_END.len()..];
                    in_thinking = false;
                }
                None => {
                    // Model never closed the think block (max_tokens / loop).
                    return (None, text.to_string());
                }
            }
        } else {
            match buf.find(THINK_START) {
                Some(idx) => {
                    content.push_str(&buf[..idx]);
                    buf = &buf[idx + THINK_START.len()..];
                    in_thinking = true;
                }
                None => {
                    content.push_str(buf);
                    break;
                }
            }
        }
    }

    let reasoning = reasoning.trim();
    let reasoning_out = (!reasoning.is_empty()).then(|| reasoning.to_string());
    (reasoning_out, content.trim().to_string())
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Breakdown of prompt tokens. `cached_tokens` counts tokens served from
    /// the prefix cache (always 0 until the engine reports cache hits).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Breakdown of completion tokens. `reasoning_tokens` counts tokens spent
    /// in the thinking block (always 0 until the engine splits them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
}

impl Usage {
    pub(crate) fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            // The engine does not yet report prefix-cache hits to the API layer,
            // so `cached_tokens` is always 0. The field is present to match
            // OpenAI's wire shape.
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 0 }),
            completion_tokens_details: None,
        }
    }

    /// Build a usage block with the reasoning-token breakdown populated. Used
    /// by the chat path when thinking is enabled so `reasoning_tokens` reflects
    /// the split-out thinking length.
    pub(crate) fn with_reasoning(
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_tokens: usize,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 0 }),
            completion_tokens_details: Some(CompletionTokensDetails { reasoning_tokens }),
        }
    }
}

fn finish_reason(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) | None => "length",
        // OpenAI has no `abort` finish reason. A client disconnect yields no
        // response at all; an internal abort maps to `stop` (generation was
        // terminated) so strict OpenAI clients don't choke on an unknown value.
        Some(FinishReason::Abort) => "stop",
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

    pub(crate) fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
        }
    }

    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) fn into_message(self) -> String {
        self.message
    }

    /// OpenAI error `type` for the response's HTTP status.
    fn error_type(&self) -> &'static str {
        match self.status {
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            s if s.is_server_error() => "api_error",
            _ => "invalid_request_error",
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        let message = value.to_string();
        if message.starts_with("server is busy:") {
            return Self::too_many_requests(message);
        }
        // Internal failures (tokenizer decode, chat-template render, multimodal
        // extraction) are server errors, not client bad-request.
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error_type = self.error_type();
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": error_type,
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
    use super::ChatCompletionRequest;

    fn chat(penalties: &str) -> ChatCompletionRequest {
        serde_json::from_str(&format!(
            r#"{{"messages":[{{"role":"user","content":"hi"}}]{penalties}}}"#
        ))
        .expect("request parses")
    }

    /// An out-of-range penalty must be refused, not clamped and not dropped on
    /// the floor — the sampler's own `> 0.0` guard would silently ignore it.
    #[test]
    fn out_of_range_penalties_are_rejected() {
        assert!(chat("").validate().is_ok());
        assert!(chat(r#","repetition_penalty":1.5"#).validate().is_ok());
        assert!(chat(r#","frequency_penalty":-2.0"#).validate().is_ok());

        for body in [
            r#","repetition_penalty":0.0"#,
            r#","repetition_penalty":-1.0"#,
            r#","repetition_penalty":2.5"#,
            r#","frequency_penalty":2.1"#,
            r#","presence_penalty":-2.1"#,
        ] {
            assert!(chat(body).validate().is_err(), "{body} must be rejected");
        }
    }

    /// NaN passes every naive range check written as `p < lo || p > hi`.
    #[test]
    fn nan_penalties_are_rejected() {
        for body in [
            r#","repetition_penalty":"#,
            r#","frequency_penalty":"#,
            r#","presence_penalty":"#,
        ] {
            let mut req = chat("");
            let nan = Some(f32::NAN);
            match body {
                r#","repetition_penalty":"# => req.repetition_penalty = nan,
                r#","frequency_penalty":"# => req.frequency_penalty = nan,
                _ => req.presence_penalty = nan,
            }
            assert!(req.validate().is_err(), "NaN{body} must be rejected");
        }
    }

    #[test]
    fn split_reasoning_simple() {
        let (r, c) = super::split_reasoning("reasoning</think>answer", true);
        assert_eq!(r.as_deref(), Some("reasoning"));
        assert_eq!(c, "answer");
    }

    #[test]
    fn split_reasoning_multi_segment() {
        let (r, c) = super::split_reasoning("r1</think>c1<think>r2</think>c2", true);
        assert_eq!(r.as_deref(), Some("r1r2"));
        assert_eq!(c, "c1c2");
    }

    #[test]
    fn split_reasoning_no_close_returns_content() {
        let (r, c) = super::split_reasoning("stuck in a loop", true);
        assert_eq!(r, None);
        assert_eq!(c, "stuck in a loop");
    }

    #[test]
    fn split_reasoning_non_thinking_passthrough() {
        let (r, c) = super::split_reasoning("Hello world", false);
        assert_eq!(r, None);
        assert_eq!(c, "Hello world");
    }

    #[test]
    fn split_reasoning_model_emitted_think() {
        let (r, c) = super::split_reasoning("<think>r</think>a", false);
        assert_eq!(r.as_deref(), Some("r"));
        assert_eq!(c, "a");
    }
}
