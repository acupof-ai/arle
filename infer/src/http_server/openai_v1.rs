use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::ApiError;
use crate::server_engine::{CompletionOutput, CompletionStreamDelta};
use crate::types::SessionId;
use chat::{
    OpenAiChatContent, OpenAiChatMessage, OpenAiToolCall, OpenAiToolDefinition, ToolCall,
    openai_parse_tool_calls,
};

/// Normalize a raw string session hint from a client request. Empty / whitespace
/// ids are dropped so that "" and `null` behave identically.
fn normalize_session_id(raw: Option<&str>) -> Option<SessionId> {
    let trimmed = raw?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(SessionId::new(trimmed.to_string()))
    }
}

fn invalid_parameter(field: impl AsRef<str>, detail: impl Into<String>) -> ApiError {
    let field = field.as_ref();
    ApiError::bad_request(
        format!("Invalid `{field}`: {}", detail.into()),
        "invalid_parameter",
    )
    .with_param(field)
}

fn canonical_model_id(model: &str) -> &str {
    let trimmed = model.trim();
    trimmed
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(trimmed)
}

fn validate_requested_model(
    requested_model: Option<&str>,
    served_model_id: &str,
) -> Result<(), ApiError> {
    let Some(requested_model) = requested_model else {
        return Ok(());
    };
    let requested_model = requested_model.trim();
    if requested_model.is_empty() {
        return Err(invalid_parameter("model", "must not be empty"));
    }

    let served_key = canonical_model_id(served_model_id);
    let requested_key = canonical_model_id(requested_model);
    if requested_model.eq_ignore_ascii_case(served_model_id)
        || requested_key.eq_ignore_ascii_case(served_key)
    {
        return Ok(());
    }

    Err(ApiError::not_found(
        format!(
            "Model `{requested_model}` is not available on this server; loaded model is `{served_model_id}`"
        ),
        "model_not_found",
    ))
}

fn validate_max_tokens(value: Option<usize>, field: &'static str) -> Result<(), ApiError> {
    if matches!(value, Some(0)) {
        return Err(invalid_parameter(field, "must be at least 1"));
    }
    Ok(())
}

fn validate_non_empty_trimmed_string(value: &str, field: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(invalid_parameter(field, "must not be empty"));
    }
    Ok(())
}

fn validate_single_choice(value: Option<usize>, field: &'static str) -> Result<(), ApiError> {
    if let Some(value) = value {
        if value == 0 {
            return Err(invalid_parameter(field, "must be at least 1"));
        }
        if value != 1 {
            return Err(invalid_parameter(field, "only `1` is currently supported"));
        }
    }
    Ok(())
}

fn validate_non_negative_finite(value: Option<f32>, field: &'static str) -> Result<(), ApiError> {
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(invalid_parameter(field, "must be finite"));
        }
        if value < 0.0 {
            return Err(invalid_parameter(field, "must be >= 0.0"));
        }
    }
    Ok(())
}

fn validate_percentage_like(
    value: Option<f32>,
    field: &'static str,
    min: f32,
    max: f32,
    include_zero: bool,
) -> Result<(), ApiError> {
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(invalid_parameter(field, "must be finite"));
        }
        let lower_ok = if include_zero {
            value >= min
        } else {
            value > min
        };
        if !lower_ok || value > max {
            let lower = if include_zero {
                format!(">= {min}")
            } else {
                format!("> {min}")
            };
            return Err(invalid_parameter(
                field,
                format!("must be {lower} and <= {max}"),
            ));
        }
    }
    Ok(())
}

fn validate_top_k(value: Option<i32>) -> Result<(), ApiError> {
    if let Some(value) = value
        && value != -1
        && value < 1
    {
        return Err(invalid_parameter("top_k", "must be -1 (disabled) or >= 1"));
    }
    Ok(())
}

fn validate_penalty(value: Option<f32>, field: &'static str) -> Result<(), ApiError> {
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(invalid_parameter(field, "must be finite"));
        }
        if !(-2.0..=2.0).contains(&value) {
            return Err(invalid_parameter(field, "must be between -2.0 and 2.0"));
        }
    }
    Ok(())
}

fn validate_repetition_penalty(value: Option<f32>) -> Result<(), ApiError> {
    if let Some(value) = value {
        if !value.is_finite() {
            return Err(invalid_parameter("repetition_penalty", "must be finite"));
        }
        if value <= 0.0 {
            return Err(invalid_parameter("repetition_penalty", "must be > 0.0"));
        }
    }
    Ok(())
}

fn validate_stream_options(
    stream: Option<bool>,
    stream_options: Option<&StreamOptions>,
) -> Result<(), ApiError> {
    let Some(stream_options) = stream_options else {
        return Ok(());
    };
    if !stream.unwrap_or(false) {
        return Err(invalid_parameter(
            "stream_options",
            "requires `stream=true`",
        ));
    }
    if stream_options.continuous_usage_stats.unwrap_or(false)
        && !stream_options.include_usage.unwrap_or(false)
    {
        return Err(invalid_parameter(
            "stream_options.continuous_usage_stats",
            "requires `stream_options.include_usage=true`",
        ));
    }
    Ok(())
}

fn validate_logprobs(value: Option<u32>) -> Result<(), ApiError> {
    if matches!(value, Some(value) if value > 0) {
        return Err(invalid_parameter(
            "logprobs",
            "is not supported on this server yet",
        ));
    }
    Ok(())
}

fn validate_return_token_ids(
    stream: Option<bool>,
    return_token_ids: Option<bool>,
) -> Result<(), ApiError> {
    if stream.unwrap_or(false) && return_token_ids.unwrap_or(false) {
        return Err(invalid_parameter(
            "return_token_ids",
            "is only supported for non-streaming completions",
        ));
    }
    Ok(())
}

fn validate_text_only_content(content: &OpenAiChatContent, field: &str) -> Result<(), ApiError> {
    let OpenAiChatContent::Parts(parts) = content else {
        return Ok(());
    };

    for (index, part) in parts.iter().enumerate() {
        let part_field = format!("{field}[{index}]");
        let Some(part_type) = part.get("type").and_then(serde_json::Value::as_str) else {
            return Err(invalid_parameter(
                format!("{part_field}.type"),
                "content parts must include a string `type`",
            ));
        };
        if part_type != "text" {
            return Err(invalid_parameter(
                format!("{part_field}.type"),
                format!("content part type `{part_type}` is not supported on this server yet"),
            ));
        }
        if part
            .get("text")
            .and_then(serde_json::Value::as_str)
            .is_none()
        {
            return Err(invalid_parameter(
                format!("{part_field}.text"),
                "text content parts must include a string `text`",
            ));
        }
    }

    Ok(())
}

fn validate_tool_call(tool_call: &OpenAiToolCall, field: &str) -> Result<(), ApiError> {
    if tool_call.call_type != "function" {
        return Err(invalid_parameter(
            format!("{field}.type"),
            format!(
                "tool call type `{}` is not supported on this server yet",
                tool_call.call_type
            ),
        ));
    }
    if tool_call.function.name.trim().is_empty() {
        return Err(invalid_parameter(
            format!("{field}.function.name"),
            "must be a non-empty string",
        ));
    }
    if serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).is_err() {
        return Err(invalid_parameter(
            format!("{field}.function.arguments"),
            "must be a valid JSON-encoded string",
        ));
    }
    Ok(())
}

fn validate_supported_message(message: &OpenAiChatMessage, field: &str) -> Result<(), ApiError> {
    match message.role.as_str() {
        "system" | "user" | "assistant" | "tool" => {}
        other => {
            return Err(invalid_parameter(
                format!("{field}.role"),
                format!(
                    "role `{other}` is not supported on this server yet; expected system, user, assistant, or tool"
                ),
            ));
        }
    }

    if let Some(content) = &message.content {
        validate_text_only_content(content, &format!("{field}.content"))?;
    }

    if message.role == "assistant" {
        for (index, tool_call) in message.tool_calls.iter().enumerate() {
            validate_tool_call(tool_call, &format!("{field}.tool_calls[{index}]"))?;
        }
    } else if !message.tool_calls.is_empty() {
        return Err(invalid_parameter(
            format!("{field}.tool_calls"),
            "is only supported on assistant messages",
        ));
    }

    if message.role == "tool" {
        if message
            .tool_call_id
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(invalid_parameter(
                format!("{field}.tool_call_id"),
                "is required on tool messages",
            ));
        }
    } else if message.tool_call_id.is_some() {
        return Err(invalid_parameter(
            format!("{field}.tool_call_id"),
            "is only supported on tool messages",
        ));
    }

    Ok(())
}

fn validate_supported_messages(
    messages: &[OpenAiChatMessage],
    field: &str,
) -> Result<(), ApiError> {
    for (index, message) in messages.iter().enumerate() {
        validate_supported_message(message, &format!("{field}[{index}]"))?;
    }
    Ok(())
}

fn validate_supported_tool_definitions(
    tools: &[OpenAiToolDefinition],
    field: &str,
) -> Result<(), ApiError> {
    for (index, tool) in tools.iter().enumerate() {
        let tool_field = format!("{field}[{index}]");
        if tool.tool_type != "function" {
            return Err(invalid_parameter(
                format!("{tool_field}.type"),
                format!(
                    "tool type `{}` is not supported on this server yet",
                    tool.tool_type
                ),
            ));
        }
        if tool.function.name.trim().is_empty() {
            return Err(invalid_parameter(
                format!("{tool_field}.function.name"),
                "must be a non-empty string",
            ));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|value| !value.is_object())
        {
            return Err(invalid_parameter(
                format!("{tool_field}.function.parameters"),
                "must be a JSON object when provided",
            ));
        }
    }
    Ok(())
}

fn validate_supported_messages_and_tools(
    messages: &[OpenAiChatMessage],
    message_field: &str,
    tools: &[OpenAiToolDefinition],
    tools_field: &str,
) -> Result<(), ApiError> {
    validate_supported_messages(messages, message_field)?;
    validate_supported_tool_definitions(tools, tools_field)?;
    Ok(())
}

fn validate_common_sampling_fields(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> Result<(), ApiError> {
    validate_non_negative_finite(temperature, "temperature")?;
    validate_percentage_like(top_p, "top_p", 0.0, 1.0, false)?;
    validate_top_k(top_k)?;
    validate_percentage_like(min_p, "min_p", 0.0, 1.0, true)?;
    validate_repetition_penalty(repetition_penalty)?;
    validate_penalty(frequency_penalty, "frequency_penalty")?;
    validate_penalty(presence_penalty, "presence_penalty")?;
    Ok(())
}

fn sanitize_logprobs(values: &[f32]) -> Option<LogprobsResult> {
    let token_logprobs: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if token_logprobs.is_empty() {
        None
    } else {
        Some(LogprobsResult { token_logprobs })
    }
}

// ============================================================================
// /v1/models — list response
// ============================================================================

#[derive(Debug, Serialize)]
pub(super) struct ModelsListResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    /// DFlash speculative-decode sub-object. Omitted entirely when the
    /// runtime was not started with a DFlash draft, so existing consumers
    /// of `/v1/models` keep the same shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    dflash: Option<DflashStatusPayload>,
}

/// DFlash sub-object embedded under `ModelObject.dflash`. Serialised only
/// when the Metal runtime reports an active DFlash pairing.
#[derive(Debug, Serialize)]
pub(super) struct DflashStatusPayload {
    /// Whether a DFlash draft was loaded and is actively speculating.
    pub enabled: bool,
    /// HuggingFace-style id or local path of the draft model.
    pub draft: String,
    /// Speculative block size (tokens drafted per verification step).
    pub speculative_tokens: usize,
    /// Rolling acceptance rate over the last ~1000 blocks, in [0, 1].
    /// `None` until at least one speculative block has run this process.
    pub acceptance_rate: Option<f64>,
}

impl ModelsListResponse {
    /// Build the single-model response. `dflash` is `None` for the baseline
    /// shape and `Some(_)` only when the runtime reports active speculative
    /// decode — the field is `skip_serializing_if = "Option::is_none"` so
    /// existing consumers keep the same JSON output.
    pub(super) fn single(
        model_id: &str,
        created: u64,
        dflash: Option<DflashStatusPayload>,
    ) -> Self {
        Self {
            object: "list",
            data: vec![ModelObject {
                id: model_id.to_string(),
                object: "model",
                created,
                owned_by: "agent-infer",
                dflash,
            }],
        }
    }
}

// OpenAI-compatible /v1/completions request
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompletionRequest {
    #[allow(dead_code)]
    pub(super) model: Option<String>,
    pub(super) prompt: String,
    /// Maximum number of tokens to generate. Accepts the modern OpenAI
    /// spelling `max_completion_tokens` as an alias so bench tools that
    /// emit the new field (guidellm 0.6+, litellm, openai-python ≥ 1.40)
    /// do not silently fall back to the server default.
    #[serde(default, alias = "max_completion_tokens")]
    pub(super) max_tokens: Option<usize>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<i32>,
    pub(super) min_p: Option<f32>,
    pub(super) repetition_penalty: Option<f32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) presence_penalty: Option<f32>,
    #[allow(dead_code)]
    pub(super) n: Option<usize>,
    pub(super) stream: Option<bool>,
    pub(super) stream_options: Option<StreamOptions>,
    pub(super) stop: Option<Vec<String>>,
    pub(super) stop_token_ids: Option<Vec<u32>>,
    pub(super) ignore_eos: Option<bool>,
    pub(super) seed: Option<u64>,
    /// Return per-token logprobs. If set to a number > 0, returns logprobs.
    #[allow(dead_code)]
    pub(super) logprobs: Option<u32>,
    /// ARLE extension for correctness gates that need generated token IDs.
    #[serde(default)]
    pub(super) return_token_ids: Option<bool>,
    /// Optional client-supplied session/conversation identifier.
    ///
    /// When present, the scheduler uses it for sticky routing of subsequent
    /// turns of the same agent session to the slot that already holds their
    /// KV prefix (see
    /// `docs/projects/agent-first-architecture.md::A2`). Accepted as
    /// `session_id` with an `user` alias to match OpenAI's existing "client
    /// supplies a stable per-user token" field.
    #[serde(default, alias = "user")]
    pub(super) session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamOptions {
    pub(super) include_usage: Option<bool>,
    pub(super) continuous_usage_stats: Option<bool>,
}

impl CompletionRequest {
    pub(super) fn validate_for_model(&self, served_model_id: &str) -> Result<(), ApiError> {
        validate_requested_model(self.model.as_deref(), served_model_id)?;
        self.validate()
    }

    pub(super) fn validate(&self) -> Result<(), ApiError> {
        validate_non_empty_trimmed_string(&self.prompt, "prompt")?;
        validate_max_tokens(self.max_tokens, "max_tokens")?;
        validate_single_choice(self.n, "n")?;
        validate_stream_options(self.stream, self.stream_options.as_ref())?;
        validate_logprobs(self.logprobs)?;
        validate_return_token_ids(self.stream, self.return_token_ids)?;
        validate_common_sampling_fields(
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )?;
        Ok(())
    }

    pub(super) fn max_tokens_or_default(&self) -> usize {
        self.max_tokens.unwrap_or(16)
    }

    pub(super) fn stream_or_default(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub(super) fn include_usage_or_default(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    pub(super) fn continuous_usage_stats_or_default(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false)
    }

    pub(super) fn return_token_ids_or_default(&self) -> bool {
        self.return_token_ids.unwrap_or(false)
    }

    pub(super) fn session_id_parsed(&self) -> Option<SessionId> {
        normalize_session_id(self.session_id.as_deref())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    index: usize,
    logprobs: Option<LogprobsResult>,
    finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
struct LogprobsResult {
    token_logprobs: Vec<f32>,
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_field_names)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl From<crate::server_engine::TokenUsage> for Usage {
    fn from(value: crate::server_engine::TokenUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

impl CompletionResponse {
    pub(super) fn from_output(
        model: String,
        created: u64,
        output: CompletionOutput,
        return_token_ids: bool,
    ) -> Self {
        let CompletionOutput {
            text,
            finish_reason,
            usage,
            token_logprobs,
            response_token_ids,
            ..
        } = output;
        let logprobs = sanitize_logprobs(&token_logprobs);
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created,
            model,
            choices: vec![Choice {
                text,
                index: 0,
                logprobs,
                finish_reason: finish_reason.as_openai_str().to_string(),
                token_ids: return_token_ids.then_some(response_token_ids),
            }],
            usage: usage.into(),
        }
    }
}

// OpenAI-compatible SSE streaming chunk
#[derive(Debug, Serialize)]
pub(super) struct StreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    text: String,
    index: usize,
    logprobs: Option<LogprobsResult>,
    finish_reason: Option<String>,
}

impl StreamChunk {
    pub(super) fn from_delta(
        request_id: &str,
        created: u64,
        model: &str,
        delta: CompletionStreamDelta,
    ) -> Self {
        let logprobs = delta
            .logprob
            .filter(|value| value.is_finite())
            .map(|lp| LogprobsResult {
                token_logprobs: vec![lp],
            });
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                text: delta.text_delta,
                index: 0,
                logprobs,
                finish_reason: delta
                    .finish_reason
                    .map(|reason| reason.as_openai_str().to_string()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamUsageChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    usage: Usage,
}

impl StreamUsageChunk {
    pub(super) fn from_usage(
        request_id: &str,
        created: u64,
        model: &str,
        usage: crate::server_engine::TokenUsage,
    ) -> Self {
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            usage: usage.into(),
        }
    }
}

// ============================================================================
// /v1/chat/completions — request
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub(super) model: Option<String>,
    pub(super) messages: Vec<OpenAiChatMessage>,
    /// Maximum completion tokens. Accepts the modern OpenAI spelling
    /// `max_completion_tokens` as an alias; see `CompletionRequest`.
    #[serde(default, alias = "max_completion_tokens")]
    pub(super) max_tokens: Option<usize>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<i32>,
    pub(super) min_p: Option<f32>,
    pub(super) repetition_penalty: Option<f32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) presence_penalty: Option<f32>,
    pub(super) stream: Option<bool>,
    pub(super) stream_options: Option<StreamOptions>,
    pub(super) stop: Option<Vec<String>>,
    pub(super) stop_token_ids: Option<Vec<u32>>,
    pub(super) ignore_eos: Option<bool>,
    pub(super) seed: Option<u64>,
    /// Tool definitions (OpenAI format).
    #[serde(default)]
    pub(super) tools: Vec<OpenAiToolDefinition>,
    /// Optional client-supplied session/conversation identifier.
    ///
    /// See [`CompletionRequest::session_id`] for the routing contract.
    #[serde(default, alias = "user")]
    pub(super) session_id: Option<String>,
}

impl ChatCompletionRequest {
    pub(super) fn validate_for_model(&self, served_model_id: &str) -> Result<(), ApiError> {
        validate_requested_model(self.model.as_deref(), served_model_id)?;
        self.validate()
    }

    pub(super) fn validate(&self) -> Result<(), ApiError> {
        if self.messages.is_empty() {
            return Err(invalid_parameter(
                "messages",
                "must contain at least one message",
            ));
        }
        validate_max_tokens(self.max_tokens, "max_tokens")?;
        validate_stream_options(self.stream, self.stream_options.as_ref())?;
        validate_common_sampling_fields(
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )?;
        validate_supported_messages_and_tools(&self.messages, "messages", &self.tools, "tools")?;
        if self.stream_or_default() && !self.tools.is_empty() {
            return Err(invalid_parameter(
                "stream",
                "stream=true is not supported when tools are present; use non-streaming chat completions for tool calls",
            ));
        }
        Ok(())
    }

    pub(super) fn max_tokens_or_default(&self) -> usize {
        self.max_tokens.unwrap_or(16)
    }

    pub(super) fn stream_or_default(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub(super) fn include_usage_or_default(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false)
    }

    pub(super) fn continuous_usage_stats_or_default(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false)
    }

    pub(super) fn session_id_parsed(&self) -> Option<SessionId> {
        normalize_session_id(self.session_id.as_deref())
    }
}

// ============================================================================
// /v1/chat/completions — non-streaming response
// ============================================================================

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: AssistantMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct AssistantMessage {
    role: &'static str,
    /// `null` when there are tool calls and no text content.
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Serialize)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: ChatFunctionCall,
}

#[derive(Debug, Serialize)]
struct ChatFunctionCall {
    name: String,
    /// JSON-encoded arguments string (OpenAI wire format).
    arguments: String,
}

impl From<&ToolCall> for ChatToolCall {
    fn from(value: &ToolCall) -> Self {
        Self {
            id: format!("call_{}", Uuid::new_v4().simple()),
            call_type: "function",
            function: ChatFunctionCall {
                name: value.name.clone(),
                arguments: value.arguments.to_string(),
            },
        }
    }
}

impl ChatCompletionResponse {
    pub(super) fn from_output(model: String, created: u64, output: &CompletionOutput) -> Self {
        let (content, parsed_calls) = openai_parse_tool_calls(&output.text);
        let tool_calls: Vec<ChatToolCall> = parsed_calls.iter().map(ChatToolCall::from).collect();

        let message = AssistantMessage {
            role: "assistant",
            content: if tool_calls.is_empty() || !content.is_empty() {
                Some(content)
            } else {
                None
            },
            tool_calls,
        };

        let fr_str = if message.tool_calls.is_empty() {
            output.finish_reason.as_openai_str()
        } else {
            "tool_calls"
        };

        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion",
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: fr_str.to_string(),
            }],
            usage: output.usage.into(),
        }
    }
}

// ============================================================================
// /v1/chat/completions — streaming chunks
// ============================================================================

#[derive(Debug, Serialize)]
pub(super) struct ChatStreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Serialize)]
struct ChatStreamChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

/// Delta payload for a streaming chunk. Fields are `None` when not set.
#[derive(Debug, Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

impl ChatStreamChunk {
    /// First chunk — includes `role` field, empty content.
    pub(super) fn role_chunk(request_id: &str, created: u64, model: &str) -> Self {
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// Content delta chunk.
    pub(super) fn content_chunk(
        request_id: &str,
        created: u64,
        model: &str,
        delta: CompletionStreamDelta,
    ) -> Self {
        let finish_reason = delta.finish_reason.map(|r| r.as_openai_str().to_string());
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: if delta.text_delta.is_empty() && finish_reason.is_some() {
                        None
                    } else {
                        Some(delta.text_delta)
                    },
                },
                finish_reason,
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ChatStreamUsageChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    usage: Usage,
}

impl ChatStreamUsageChunk {
    pub(super) fn from_usage(
        request_id: &str,
        created: u64,
        model: &str,
        usage: crate::server_engine::TokenUsage,
    ) -> Self {
        Self {
            id: request_id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            usage: usage.into(),
        }
    }
}

// ============================================================================
// /v1/responses — request / response
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum ResponsesInput {
    Text(String),
    Messages(Vec<OpenAiChatMessage>),
    Message(OpenAiChatMessage),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesRequest {
    #[allow(dead_code)]
    pub(super) model: Option<String>,
    pub(super) input: ResponsesInput,
    pub(super) instructions: Option<String>,
    /// OpenAI Responses uses `max_output_tokens`, but some existing
    /// compatibility clients still send `max_tokens`.
    #[serde(default, alias = "max_tokens")]
    pub(super) max_output_tokens: Option<usize>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<i32>,
    pub(super) min_p: Option<f32>,
    pub(super) repetition_penalty: Option<f32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) presence_penalty: Option<f32>,
    pub(super) stream: Option<bool>,
    pub(super) stop: Option<Vec<String>>,
    pub(super) stop_token_ids: Option<Vec<u32>>,
    pub(super) ignore_eos: Option<bool>,
    pub(super) seed: Option<u64>,
    #[serde(default)]
    pub(super) tools: Vec<OpenAiToolDefinition>,
    #[serde(default, alias = "user")]
    pub(super) session_id: Option<String>,
}

impl ResponsesRequest {
    pub(super) fn validate_for_model(&self, served_model_id: &str) -> Result<(), ApiError> {
        validate_requested_model(self.model.as_deref(), served_model_id)?;
        self.validate()
    }

    pub(super) fn validate(&self) -> Result<(), ApiError> {
        validate_max_tokens(self.max_output_tokens, "max_output_tokens")?;
        validate_common_sampling_fields(
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
        )?;
        validate_supported_tool_definitions(&self.tools, "tools")?;
        match &self.input {
            ResponsesInput::Text(text) => validate_non_empty_trimmed_string(text, "input")?,
            ResponsesInput::Message(message) => {
                validate_supported_messages(std::slice::from_ref(message), "input")?;
            }
            ResponsesInput::Messages(messages) => {
                if messages.is_empty() {
                    return Err(invalid_parameter(
                        "input",
                        "must contain at least one message",
                    ));
                }
                validate_supported_messages(messages, "input")?;
            }
        }
        if self.stream_or_default() && !self.tools.is_empty() {
            return Err(invalid_parameter(
                "stream",
                "stream=true is not supported when tools are present; use non-streaming responses for tool calls",
            ));
        }
        Ok(())
    }

    pub(super) fn max_output_tokens_or_default(&self) -> usize {
        self.max_output_tokens.unwrap_or(16)
    }

    pub(super) fn stream_or_default(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub(super) fn session_id_parsed(&self) -> Option<SessionId> {
        normalize_session_id(self.session_id.as_deref())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    model: String,
    output: Vec<ResponseOutputItem>,
    output_text: String,
    usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<ResponseContentItem>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        status: &'static str,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponseContentItem {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        annotations: Vec<ResponseTextAnnotation>,
    },
}

#[derive(Debug, Serialize)]
struct ResponseTextAnnotation {}

#[derive(Debug, Serialize)]
struct ResponsesUsage {
    input_tokens: usize,
    output_tokens: usize,
    total_tokens: usize,
}

impl From<crate::server_engine::TokenUsage> for ResponsesUsage {
    fn from(value: crate::server_engine::TokenUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

impl ResponsesResponse {
    pub(super) fn from_output(model: String, created_at: u64, output: CompletionOutput) -> Self {
        Self::from_output_with_id(
            format!("resp_{}", Uuid::new_v4().simple()),
            model,
            created_at,
            output,
        )
    }

    pub(super) fn from_output_with_id(
        id: String,
        model: String,
        created_at: u64,
        output: CompletionOutput,
    ) -> Self {
        let (content, parsed_calls) = openai_parse_tool_calls(&output.text);
        let mut items = Vec::new();

        if !content.is_empty() || parsed_calls.is_empty() {
            items.push(ResponseOutputItem::Message {
                id: format!("msg_{}", Uuid::new_v4().simple()),
                status: "completed",
                role: "assistant",
                content: vec![ResponseContentItem::OutputText {
                    text: content.clone(),
                    annotations: Vec::new(),
                }],
            });
        }

        items.extend(
            parsed_calls
                .into_iter()
                .map(|call| ResponseOutputItem::FunctionCall {
                    id: format!("fc_{}", Uuid::new_v4().simple()),
                    call_id: format!("call_{}", Uuid::new_v4().simple()),
                    status: "completed",
                    name: call.name,
                    arguments: call.arguments.to_string(),
                }),
        );

        Self {
            id,
            object: "response",
            created_at,
            status: "completed",
            model,
            output: items,
            output_text: content,
            usage: output.usage.into(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesStreamCreatedEvent {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    model: String,
}

impl ResponsesStreamCreatedEvent {
    pub(super) fn new(id: String, created_at: u64, model: String) -> Self {
        Self {
            id,
            object: "response",
            created_at,
            status: "in_progress",
            model,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesStreamDeltaEvent {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    model: String,
    output_index: usize,
    content_index: usize,
    delta: String,
}

impl ResponsesStreamDeltaEvent {
    pub(super) fn new(id: String, created_at: u64, model: String, delta: String) -> Self {
        Self {
            id,
            object: "response",
            created_at,
            status: "in_progress",
            model,
            output_index: 0,
            content_index: 0,
            delta,
        }
    }
}

#[cfg(test)]
#[path = "openai_v1/tests.rs"]
mod tests;
