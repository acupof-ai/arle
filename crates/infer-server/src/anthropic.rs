//! Anthropic Messages API translation layer (COLD — fixed external contract).
//!
//! Pure wire-shape translation onto the existing OpenAI chat machinery so
//! Claude Code (via `ANTHROPIC_BASE_URL`) can drive the local serve:
//! `POST /v1/messages` requests map into [`ChatCompletionRequest`] (same
//! template render, sampling, tools gating), and the OpenAI-shaped result maps
//! back into the Anthropic message envelope / SSE event sequence. The route
//! handlers live in [`crate::coordinator`]; this file owns only the wire types,
//! the request/response mapping, and the streaming event encoder.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use infer_plan::FinishReason;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::schema::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, ChatContent, ChatMessage,
};

// ---------------------------------------------------------------------------
// Request wire shapes. Unknown fields are tolerated everywhere (no
// deny_unknown_fields): Claude Code sends `cache_control`, `metadata`,
// `thinking`, and evolving extras on every level.
// ---------------------------------------------------------------------------

/// `POST /v1/messages` body (also `count_tokens`, which ignores `max_tokens`
/// and `stream`).
#[derive(Debug, Deserialize)]
pub(crate) struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// Required by the Messages API; optional here so `count_tokens` (which
    /// omits it) shares the shape — [`Self::validate`] enforces presence.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    /// `{"type":"enabled","budget_tokens":N}` or `{"type":"disabled"}`. Maps to
    /// the internal `enable_thinking` chat-template kwarg; `budget_tokens` is
    /// not enforced (the server's `max_thinking_tokens` caps it).
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// End-user metadata. Accepted but not echoed (the engine has no storage).
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<Value>,
}

/// Anthropic `thinking` request parameter.
#[derive(Debug, Deserialize)]
pub(crate) struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: String,
    /// Thinking token budget. Accepted but not enforced (the server's
    /// `max_thinking_tokens` caps it).
    #[serde(default)]
    #[allow(dead_code)]
    pub budget_tokens: Option<usize>,
}

/// `system` is a plain string or a list of text blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl SystemPrompt {
    fn to_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => join_text_blocks(blocks, "\n\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// One content block. A plain struct keyed on `type` (not a tagged enum) so an
/// unknown block type deserializes instead of failing the whole request.
#[derive(Debug, Deserialize)]
pub(crate) struct ContentBlock {
    #[serde(default, rename = "type")]
    pub kind: String,
    /// `text` blocks.
    #[serde(default)]
    pub text: Option<String>,
    /// `tool_use` blocks (assistant history).
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    /// `tool_result` blocks (user messages).
    #[serde(default)]
    pub tool_use_id: Option<String>,
    #[serde(default)]
    pub content: Option<ToolResultContent>,
    #[serde(default)]
    pub is_error: Option<bool>,
}

/// `tool_result.content` is a plain string or a list of (text) blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl ToolResultContent {
    fn to_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => join_text_blocks(blocks, "\n"),
        }
    }
}

fn join_text_blocks(blocks: &[ContentBlock], separator: &str) -> String {
    blocks
        .iter()
        .filter(|block| block.kind == "text")
        .filter_map(|block| block.text.as_deref())
        .collect::<Vec<_>>()
        .join(separator)
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
}

/// `{"type":"auto"|"any"|"none"}` or `{"type":"tool","name":…}`.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolChoice {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
}

impl MessagesRequest {
    /// Full validation for `/v1/messages` (`max_tokens` is required there).
    pub(crate) fn validate(&self) -> Result<(), MessagesError> {
        self.validate_for_count()?;
        match self.max_tokens {
            None => Err(MessagesError::invalid_request("max_tokens is required")),
            Some(0) => Err(MessagesError::invalid_request(
                "max_tokens must be greater than zero",
            )),
            Some(_) => Ok(()),
        }
    }

    /// `count_tokens` validation: prompt shape only, no `max_tokens`.
    pub(crate) fn validate_for_count(&self) -> Result<(), MessagesError> {
        if self.messages.is_empty() {
            return Err(MessagesError::invalid_request(
                "messages must contain at least one message",
            ));
        }
        Ok(())
    }

    /// Map into the internal OpenAI chat request the existing machinery
    /// consumes (template render, sampling, tools gating). One Anthropic
    /// message may fan out into several internal messages (tool results).
    ///
    /// Claude Code sends `role:"system"` entries inside `messages[]`
    /// (mid-conversation system messages). A leading one (with no top-level
    /// `system`) becomes the system prompt; any other degrades to a user turn
    /// wrapped in `<system-reminder>` — strict checkpoint templates (Qwen)
    /// `raise_exception` on a non-first system message, and they tolerate
    /// consecutive user turns, so degraded reminders stay separate messages.
    pub(crate) fn to_chat_request(&self) -> ChatCompletionRequest {
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            let text = system.to_text();
            if !text.is_empty() {
                messages.push(text_message("system", text));
            }
        }
        for (position, message) in self.messages.iter().enumerate() {
            if message.role == "system" {
                let text = match &message.content {
                    MessageContent::Text(text) => text.clone(),
                    MessageContent::Blocks(blocks) => join_text_blocks(blocks, "\n\n"),
                };
                // First entry with no top-level `system` = THE system prompt.
                if position == 0 && messages.is_empty() {
                    messages.push(text_message("system", text));
                } else {
                    messages.push(text_message(
                        "user",
                        format!("<system-reminder>\n{text}\n</system-reminder>"),
                    ));
                }
                continue;
            }
            match &message.content {
                MessageContent::Text(text) => {
                    messages.push(text_message(&message.role, text.clone()));
                }
                MessageContent::Blocks(blocks) => fan_out_blocks(&mut messages, message, blocks),
            }
        }
        ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_token_ids: None,
            ignore_eos: None,
            seed: None,
            stream: self.stream,
            stream_options: None,
            stop: (!self.stop_sequences.is_empty()).then(|| self.stop_sequences.clone()),
            chat_template_kwargs: self.thinking.as_ref().map(|thinking| {
                let mut kwargs = serde_json::Map::new();
                kwargs.insert(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(thinking.kind != "disabled"),
                );
                kwargs
            }),
            tools: self.tools.iter().map(to_openai_tool).collect(),
            response_format: None,
            tool_choice: self.tool_choice.as_ref().map(to_openai_tool_choice),
            n: None,
            logprobs: None,
            top_logprobs: None,
            logit_bias: None,
            user: None,
            parallel_tool_calls: None,
            service_tier: None,
            store: None,
            metadata: None,
        }
    }
}

/// Offline entry for CC-trajectory tooling (`arle train cc-convert`): parse a
/// raw `/v1/messages` body and map it through the EXACT request path
/// [`MessagesRequest::to_chat_request`] the serve handler runs, so converted
/// records can never drift from the served prompt format.
pub fn messages_body_to_chat_request(body: Value) -> anyhow::Result<ChatCompletionRequest> {
    let request: MessagesRequest = serde_json::from_value(body)?;
    Ok(request.to_chat_request())
}

fn text_message(role: &str, text: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: Some(ChatContent::Text(text)),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    }
}

fn append_fragment(text: &mut String, fragment: &str) {
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(fragment);
}

/// Translate one Anthropic block-list message, preserving block order:
/// each `tool_result` becomes its own `tool`-role message (flushing any text
/// accumulated before it), `tool_use` blocks become assistant `tool_calls`,
/// `thinking` blocks are dropped, images degrade to a text placeholder.
fn fan_out_blocks(out: &mut Vec<ChatMessage>, message: &Message, blocks: &[ContentBlock]) {
    let mut text = String::new();
    let mut tool_calls: Vec<chat::OpenAiToolCall> = Vec::new();
    for block in blocks {
        match block.kind.as_str() {
            "text" => {
                if let Some(fragment) = &block.text {
                    append_fragment(&mut text, fragment);
                }
            }
            "image" => append_fragment(&mut text, "[image omitted]"),
            "tool_use" => tool_calls.push(chat::OpenAiToolCall {
                id: block.id.clone().unwrap_or_default(),
                call_type: "function".to_string(),
                function: chat::OpenAiFunctionCall {
                    name: block.name.clone().unwrap_or_default(),
                    arguments: block
                        .input
                        .as_ref()
                        .map_or_else(|| "{}".to_string(), Value::to_string),
                },
            }),
            "tool_result" => {
                if !text.is_empty() {
                    out.push(text_message(&message.role, std::mem::take(&mut text)));
                }
                let result = block
                    .content
                    .as_ref()
                    .map(ToolResultContent::to_text)
                    .unwrap_or_default();
                let result = if block.is_error.unwrap_or(false) {
                    format!("ERROR: {result}")
                } else {
                    result
                };
                out.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(ChatContent::Text(result)),
                    tool_calls: Vec::new(),
                    tool_call_id: block.tool_use_id.clone(),
                    name: None,
                });
            }
            // `thinking` / `redacted_thinking` / unknown: dropped silently.
            _ => {}
        }
    }
    if !text.is_empty() || !tool_calls.is_empty() {
        out.push(ChatMessage {
            role: message.role.clone(),
            content: Some(ChatContent::Text(text)),
            tool_calls,
            tool_call_id: None,
            name: None,
        });
    }
}

fn to_openai_tool(tool: &ToolDefinition) -> chat::OpenAiToolDefinition {
    chat::OpenAiToolDefinition {
        tool_type: "function".to_string(),
        function: chat::OpenAiFunctionDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        },
    }
}

/// auto→"auto", any→"required", none→"none", tool→forced-function object.
fn to_openai_tool_choice(choice: &ToolChoice) -> Value {
    match choice.kind.as_str() {
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => json!({
            "type": "function",
            "function": {"name": choice.name.clone().unwrap_or_default()},
        }),
        _ => json!("auto"),
    }
}

// ---------------------------------------------------------------------------
// Response wire shapes.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Anthropic `thinking` block — carries the model's reasoning content.
    /// Emitted before the `text` block, matching the API spec.
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Serialize)]
pub(crate) struct UsageTokens {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Non-streaming `POST /v1/messages` response envelope.
#[derive(Debug, Serialize)]
pub(crate) struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseBlock>,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: UsageTokens,
}

impl MessagesResponse {
    /// Map the OpenAI-shaped completion (built by the shared chat machinery,
    /// reasoning already split out) into the Anthropic envelope.
    pub(crate) fn from_chat(chat: &ChatCompletionResponse) -> Self {
        let choice = &chat.choices[0];
        let mut content = Vec::new();
        // Thinking block first (Anthropic convention), when the model produced
        // reasoning content and the request enabled thinking.
        if let Some(reasoning) = choice.message.reasoning_content.as_ref()
            && !reasoning.trim().is_empty()
        {
            content.push(ResponseBlock::Thinking {
                thinking: reasoning.clone(),
            });
        }
        // Whitespace-only text never becomes a block (the stream encoder's
        // held-whitespace rule — keep the two paths converged).
        if !choice.message.content.trim().is_empty() {
            content.push(ResponseBlock::Text {
                text: choice.message.content.clone(),
            });
        }
        for call in &choice.message.tool_calls {
            content.push(ResponseBlock::ToolUse {
                id: mint_tool_use_id(),
                name: call.function.name.clone(),
                input: arguments_to_input(&call.function.arguments),
            });
        }
        Self {
            id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            kind: "message",
            role: "assistant",
            model: chat.model.clone(),
            content,
            stop_reason: match choice.finish_reason.as_str() {
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                _ => "end_turn",
            },
            stop_sequence: None,
            usage: UsageTokens {
                input_tokens: chat.usage.prompt_tokens,
                output_tokens: chat.usage.completion_tokens,
            },
        }
    }
}

pub(crate) fn mint_tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// Normalize parsed tool-call arguments into the Anthropic `input` object;
/// non-object arguments are preserved under `{"_raw": …}`. The single
/// normalization for BOTH response paths (non-streaming via
/// [`arguments_to_input`], streaming via [`StreamEncoder::tool_use`]).
pub(crate) fn input_from_value(value: &Value) -> Value {
    if value.is_object() {
        value.clone()
    } else {
        json!({ "_raw": value.to_string() })
    }
}

/// Parse an OpenAI arguments-JSON string into the Anthropic `input` object;
/// unparseable arguments are preserved under `{"_raw": …}`.
pub(crate) fn arguments_to_input(raw: &str) -> Value {
    serde_json::from_str::<Value>(raw)
        .map_or_else(|_| json!({ "_raw": raw }), |value| input_from_value(&value))
}

/// Anthropic `stop_reason` from the engine finish reason. Any emitted tool
/// call wins (mirrors the OpenAI `tool_calls` override); `stop_sequence` never
/// occurs — the engine has no stop-string support (stop_sequences are accepted
/// but unenforced, same as the OpenAI `stop` field).
pub(crate) fn stop_reason(finish: Option<&FinishReason>, has_tool_use: bool) -> &'static str {
    if has_tool_use {
        return "tool_use";
    }
    match finish {
        Some(FinishReason::Stop | FinishReason::Abort) => "end_turn",
        Some(FinishReason::Length) | None => "max_tokens",
    }
}

// ---------------------------------------------------------------------------
// Streaming SSE encoder.
// ---------------------------------------------------------------------------

fn frame(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Encodes the Anthropic streaming event sequence (`message_start` →
/// per-block `content_block_start/delta/stop` → `message_delta` →
/// `message_stop`), tracking block indices. Text streams incrementally; each
/// tool call arrives fully parsed (the internal tool stream only surfaces
/// closed `<tool_call>` blocks), so its block emits one `input_json_delta`
/// carrying the whole arguments string.
pub(crate) struct StreamEncoder {
    id: String,
    model: String,
    next_index: usize,
    /// Index of the currently-open text block, if any.
    text_index: Option<usize>,
    /// Index of the currently-open thinking block, if any.
    thinking_index: Option<usize>,
    /// Whitespace held back while NO text block is open: emitted only if
    /// non-whitespace text follows, dropped at block/message boundaries — so a
    /// stray newline around a tool call never opens a whitespace-only text
    /// block (mirrors the non-streaming path's trim).
    pending_ws: String,
}

impl StreamEncoder {
    pub(crate) fn new(id: String, model: String) -> Self {
        Self {
            id,
            model,
            next_index: 0,
            text_index: None,
            thinking_index: None,
            pending_ws: String::new(),
        }
    }

    pub(crate) fn message_start(&self, input_tokens: usize) -> String {
        frame(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }),
        )
    }

    pub(crate) fn ping() -> String {
        frame("ping", &json!({"type": "ping"}))
    }

    pub(crate) fn error(message: &str) -> String {
        frame(
            "error",
            &json!({"type": "error", "error": {"type": "api_error", "message": message}}),
        )
    }

    /// Emit a text delta, opening a text content block first if none is open.
    /// Whitespace that would OPEN a block is held until non-whitespace follows.
    pub(crate) fn text_delta(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        if self.text_index.is_none() && text.chars().all(char::is_whitespace) {
            self.pending_ws.push_str(text);
            return String::new();
        }
        let merged = format!("{}{text}", std::mem::take(&mut self.pending_ws));
        let text = merged.as_str();
        let mut out = String::new();
        let index = match self.text_index {
            Some(index) => index,
            None => {
                let index = self.next_index;
                self.next_index += 1;
                self.text_index = Some(index);
                out.push_str(&frame(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    }),
                ));
                index
            }
        };
        out.push_str(&frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }),
        ));
        out
    }

    /// Emit a thinking delta, opening a thinking content block first if none is
    /// open. Thinking blocks always precede text blocks (Anthropic convention).
    pub(crate) fn thinking_delta(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        let index = match self.thinking_index {
            Some(index) => index,
            None => {
                let index = self.next_index;
                self.next_index += 1;
                self.thinking_index = Some(index);
                out.push_str(&frame(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }),
                ));
                index
            }
        };
        out.push_str(&frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": text},
            }),
        ));
        out
    }

    fn close_thinking(&mut self) -> String {
        self.thinking_index
            .take()
            .map_or_else(String::new, |index| {
                frame(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": index}),
                )
            })
    }

    fn close_text(&mut self) -> String {
        // Held whitespace never materialized into a block — drop it at the
        // boundary so it can't leak into a later block.
        self.pending_ws.clear();
        self.text_index.take().map_or_else(String::new, |index| {
            frame(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": index}),
            )
        })
    }

    /// Emit one complete tool_use block (start → whole-arguments
    /// `input_json_delta` → stop), closing any open text block first. The
    /// arguments normalize through [`input_from_value`] — same `input` object
    /// the non-streaming envelope builds.
    pub(crate) fn tool_use(&mut self, name: &str, arguments: &Value) -> String {
        let arguments_json = input_from_value(arguments).to_string();
        let mut out = self.close_thinking();
        out.push_str(&self.close_text());
        let index = self.next_index;
        self.next_index += 1;
        out.push_str(&frame(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": mint_tool_use_id(),
                    "name": name,
                    "input": {},
                },
            }),
        ));
        out.push_str(&frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": arguments_json},
            }),
        ));
        out.push_str(&frame(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        out
    }

    /// Close any open block and terminate: `message_delta` + `message_stop`.
    pub(crate) fn finish(&mut self, stop_reason: &str, output_tokens: usize) -> String {
        let mut out = self.close_thinking();
        out.push_str(&self.close_text());
        out.push_str(&frame(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": output_tokens},
            }),
        ));
        out.push_str(&frame("message_stop", &json!({"type": "message_stop"})));
        out
    }
}

// ---------------------------------------------------------------------------
// Error shape: `{"type":"error","error":{"type":…,"message":…}}`.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct MessagesError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl MessagesError {
    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }
}

impl From<ApiError> for MessagesError {
    fn from(error: ApiError) -> Self {
        let status = error.status();
        Self {
            status,
            kind: if status.is_server_error() {
                "api_error"
            } else {
                "invalid_request_error"
            },
            message: error.into_message(),
        }
    }
}

impl IntoResponse for MessagesError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "type": "error",
                "error": {"type": self.kind, "message": self.message},
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ResponseToolCall;

    fn parse(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("deserialize MessagesRequest")
    }

    /// A realistic Claude-Code-shaped request: system array with cache_control,
    /// tools with extras, assistant tool_use history, a user message mixing
    /// tool_result + text, and unknown fields at every level.
    #[test]
    fn claude_code_shaped_request_maps_to_chat() {
        let request = parse(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["<END>"],
            "metadata": {"user_id": "u-1"},
            "thinking": {"type": "disabled"},
            "system": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be concise."}
            ],
            "tools": [
                {"name": "bash", "description": "Run a command",
                 "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                 "strict": true, "cache_control": {"type": "ephemeral"}}
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "messages": [
                {"role": "user", "content": "list the files"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me run ls", "signature": "sig"},
                    {"type": "text", "text": "Running ls."},
                    {"type": "tool_use", "id": "toolu_abc", "name": "bash",
                     "input": {"command": "ls"}, "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "is_error": true,
                     "content": [{"type": "text", "text": "ls: not found"}]},
                    {"type": "text", "text": "try again"},
                    {"type": "image", "source": {"type": "base64", "data": "…"}}
                ]}
            ]
        }));
        let chat = request.to_chat_request();

        // Sampling / control fields.
        assert_eq!(chat.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(chat.max_tokens, Some(4096));
        assert_eq!(chat.temperature, Some(0.2));
        assert_eq!(chat.top_p, Some(0.9));
        assert_eq!(chat.top_k, Some(40));
        assert_eq!(chat.stop.as_deref(), Some(&["<END>".to_string()][..]));
        assert!(chat.wants_tools());
        assert_eq!(chat.tools.len(), 1);
        assert_eq!(chat.tools[0].tool_type, "function");
        assert_eq!(chat.tools[0].function.name, "bash");
        assert_eq!(
            chat.tools[0].function.parameters.as_ref().unwrap()["type"],
            "object"
        );
        assert_eq!(chat.tool_choice_mode(), chat::ToolChoiceMode::Auto);

        // Message fan-out: system, user, assistant(+tool_calls), tool, user.
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"]);
        assert_eq!(
            chat.messages[0].content_text(),
            "You are Claude Code.\n\nBe concise."
        );
        // Assistant: thinking dropped, text kept, tool_use -> tool_calls.
        assert_eq!(chat.messages[2].content_text(), "Running ls.");
        assert_eq!(chat.messages[2].tool_calls.len(), 1);
        assert_eq!(chat.messages[2].tool_calls[0].id, "toolu_abc");
        assert_eq!(chat.messages[2].tool_calls[0].function.name, "bash");
        assert_eq!(
            chat.messages[2].tool_calls[0].function.arguments,
            r#"{"command":"ls"}"#
        );
        // tool_result precedes the text block => tool message first, ERROR-prefixed.
        assert_eq!(chat.messages[3].tool_call_id.as_deref(), Some("toolu_abc"));
        assert_eq!(chat.messages[3].content_text(), "ERROR: ls: not found");
        // Trailing text + image placeholder become the closing user message.
        assert_eq!(
            chat.messages[4].content_text(),
            "try again\n[image omitted]"
        );
    }

    #[test]
    fn tool_choice_matrix_maps_to_openai_modes() {
        use chat::ToolChoiceMode;
        let mode = |choice: Value| {
            parse(json!({
                "max_tokens": 16,
                "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
                "tool_choice": choice,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .to_chat_request()
            .tool_choice_mode()
        };
        assert_eq!(mode(json!({"type": "auto"})), ToolChoiceMode::Auto);
        assert_eq!(mode(json!({"type": "any"})), ToolChoiceMode::Required);
        assert_eq!(mode(json!({"type": "none"})), ToolChoiceMode::None);
        assert_eq!(
            mode(json!({"type": "tool", "name": "bash"})),
            ToolChoiceMode::Function("bash".to_string())
        );
        // Absent tool_choice defaults to Auto; tool_choice none gates tools off.
        let absent = parse(json!({
            "max_tokens": 16,
            "tools": [{"name": "bash", "input_schema": {}}],
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .to_chat_request();
        assert_eq!(absent.tool_choice_mode(), ToolChoiceMode::Auto);
        assert!(absent.wants_tools());
        let none = parse(json!({
            "max_tokens": 16,
            "tools": [{"name": "bash", "input_schema": {}}],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .to_chat_request();
        assert!(!none.wants_tools());
    }

    #[test]
    fn validate_requires_max_tokens_only_for_messages() {
        let no_max = parse(json!({"messages": [{"role": "user", "content": "hi"}]}));
        assert!(no_max.validate().is_err(), "max_tokens is required");
        assert!(no_max.validate_for_count().is_ok(), "count_tokens skips it");
        let zero = parse(json!({"max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]}));
        assert!(zero.validate().is_err(), "max_tokens=0 rejected");
        let empty = parse(json!({"max_tokens": 16, "messages": []}));
        assert!(empty.validate().is_err(), "empty messages rejected");
    }

    /// Claude Code sends `role:"system"` entries inside `messages[]`. A
    /// leading one (no top-level `system`) is THE system prompt; mid-list ones
    /// degrade to separate `<system-reminder>` user turns, and the result must
    /// survive a strict Qwen-style template that rejects non-first system
    /// messages (consecutive user turns are tolerated).
    #[test]
    fn mid_conversation_system_messages_degrade_to_reminders() {
        let chat = parse(json!({
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "terse mode"},
                {"role": "user", "content": "and?"},
            ],
        }))
        .to_chat_request();
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "user", "user"]);
        assert_eq!(
            chat.messages[1].content_text(),
            "<system-reminder>\nterse mode\n</system-reminder>"
        );
        // Render gate: a strict template raising on non-first system messages
        // (the Qwen rule) accepts the mapped conversation.
        const STRICT: &str = "{%- for message in messages %}\
            {%- if message.role == 'system' and not loop.first %}\
            {{ raise_exception('System message must be at the beginning.') }}\
            {%- endif %}\
            {{- '<|' + message.role + '|>' + message.content }}\
            {%- endfor %}";
        let out =
            crate::tokenizer::render_jinja(STRICT, "", "", &chat.messages, None, &[]).unwrap();
        assert_eq!(
            out,
            "<|user|>hi<|user|><system-reminder>\nterse mode\n</system-reminder><|user|>and?"
        );

        // Leading system entry with no top-level `system` = the system prompt.
        let leading = parse(json!({
            "max_tokens": 16,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "be terse"}]},
                {"role": "user", "content": "hi"},
            ],
        }))
        .to_chat_request();
        let roles: Vec<&str> = leading.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user"]);
        assert_eq!(leading.messages[0].content_text(), "be terse");

        // With a top-level `system`, even a leading entry degrades to a reminder.
        let both = parse(json!({
            "max_tokens": 16,
            "system": "top-level prompt",
            "messages": [
                {"role": "system", "content": "extra system"},
                {"role": "user", "content": "hi"},
            ],
        }))
        .to_chat_request();
        let roles: Vec<&str> = both.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "user"]);
        assert_eq!(both.messages[0].content_text(), "top-level prompt");
        assert_eq!(
            both.messages[1].content_text(),
            "<system-reminder>\nextra system\n</system-reminder>"
        );
    }

    #[test]
    fn unknown_block_types_are_tolerated_and_dropped() {
        let chat = parse(json!({
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "server_tool_use_2030", "payload": {"x": 1}},
                {"type": "text", "text": "hello"}
            ]}],
        }))
        .to_chat_request();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].content_text(), "hello");
    }

    #[test]
    fn response_mapping_builds_anthropic_envelope() {
        let calls = vec![ResponseToolCall::from_parsed(
            &chat::ToolCall::new("get_weather", json!({"city": "Paris"})),
            0,
        )];
        let chat_response = ChatCompletionResponse::from_parts(
            "claude-sonnet-4-5".to_string(),
            "Checking the weather.".to_string(),
            12,
            7,
            Some(&FinishReason::Stop),
            false,
            calls,
        );
        let response = MessagesResponse::from_chat(&chat_response);
        let v = serde_json::to_value(&response).expect("serialize");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["model"], "claude-sonnet-4-5");
        assert!(v["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(v["stop_reason"], "tool_use");
        assert!(v["stop_sequence"].is_null());
        assert_eq!(v["usage"]["input_tokens"], 12);
        assert_eq!(v["usage"]["output_tokens"], 7);
        let blocks = v["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Checking the weather.");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "get_weather");
        assert_eq!(blocks[1]["input"], json!({"city": "Paris"}));
        assert!(blocks[1]["id"].as_str().unwrap().starts_with("toolu_"));

        // Tool-less stop / length map to end_turn / max_tokens.
        let plain = MessagesResponse::from_chat(&ChatCompletionResponse::from_parts(
            "m".to_string(),
            "hi".to_string(),
            1,
            1,
            Some(&FinishReason::Stop),
            false,
            Vec::new(),
        ));
        assert_eq!(plain.stop_reason, "end_turn");
        let truncated = MessagesResponse::from_chat(&ChatCompletionResponse::from_parts(
            "m".to_string(),
            "hi".to_string(),
            1,
            1,
            Some(&FinishReason::Length),
            false,
            Vec::new(),
        ));
        assert_eq!(truncated.stop_reason, "max_tokens");
    }

    #[test]
    fn arguments_to_input_keeps_raw_on_parse_failure() {
        assert_eq!(
            arguments_to_input(r#"{"city":"Paris"}"#),
            json!({"city": "Paris"})
        );
        assert_eq!(arguments_to_input("not json"), json!({"_raw": "not json"}));
        // Parseable but non-object arguments also fall back to _raw.
        assert_eq!(arguments_to_input("[1,2]"), json!({"_raw": "[1,2]"}));
        // The streaming lane normalizes through the same Value-level helper.
        assert_eq!(
            input_from_value(&json!({"city": "Paris"})),
            json!({"city": "Paris"})
        );
        assert_eq!(input_from_value(&json!([1, 2])), json!({"_raw": "[1,2]"}));
    }

    /// Whitespace-only content never becomes a text block — the non-streaming
    /// twin of the stream encoder's held-whitespace rule.
    #[test]
    fn whitespace_only_content_is_suppressed_in_envelope() {
        let chat = ChatCompletionResponse::from_parts(
            "m".to_string(),
            " \n".to_string(),
            1,
            1,
            Some(&FinishReason::Stop),
            false,
            Vec::new(),
        );
        let response = MessagesResponse::from_chat(&chat);
        assert!(response.content.is_empty());
    }

    #[test]
    fn stop_reason_maps_finish_reasons() {
        assert_eq!(stop_reason(Some(&FinishReason::Stop), false), "end_turn");
        assert_eq!(
            stop_reason(Some(&FinishReason::Length), false),
            "max_tokens"
        );
        assert_eq!(stop_reason(None, false), "max_tokens");
        assert_eq!(stop_reason(Some(&FinishReason::Stop), true), "tool_use");
    }

    fn parse_frames(raw: &str) -> Vec<(String, Value)> {
        raw.split("\n\n")
            .filter(|frame| !frame.is_empty())
            .map(|frame| {
                let (event_line, data_line) = frame.split_once('\n').expect("two-line frame");
                (
                    event_line
                        .strip_prefix("event: ")
                        .expect("event line")
                        .to_string(),
                    serde_json::from_str(data_line.strip_prefix("data: ").expect("data line"))
                        .expect("frame data is JSON"),
                )
            })
            .collect()
    }

    /// Synthetic two-block completion (text then tool_use): the encoder must
    /// emit the exact Anthropic event sequence with correct block indices.
    #[test]
    fn sse_two_block_stream_sequence() {
        let mut encoder = StreamEncoder::new("msg_test".to_string(), "claude-x".to_string());
        let mut raw = encoder.message_start(12);
        raw.push_str(&StreamEncoder::ping());
        raw.push_str(&encoder.text_delta("Hel"));
        raw.push_str(&encoder.text_delta("lo"));
        raw.push_str(&encoder.tool_use("bash", &json!({"command": "ls"})));
        raw.push_str(&encoder.finish("tool_use", 7));

        let frames = parse_frames(&raw);
        let events: Vec<&str> = frames.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(
            events,
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start = &frames[0].1;
        assert_eq!(start["message"]["id"], "msg_test");
        assert_eq!(start["message"]["model"], "claude-x");
        assert_eq!(start["message"]["content"], json!([]));
        assert_eq!(start["message"]["usage"]["input_tokens"], 12);
        assert_eq!(start["message"]["usage"]["output_tokens"], 0);
        // Text block at index 0.
        assert_eq!(frames[2].1["index"], 0);
        assert_eq!(frames[2].1["content_block"]["type"], "text");
        assert_eq!(frames[3].1["delta"]["type"], "text_delta");
        assert_eq!(frames[3].1["delta"]["text"], "Hel");
        assert_eq!(frames[4].1["delta"]["text"], "lo");
        assert_eq!(frames[5].1["index"], 0);
        // Tool block at index 1 with the whole arguments as one json delta.
        assert_eq!(frames[6].1["index"], 1);
        assert_eq!(frames[6].1["content_block"]["type"], "tool_use");
        assert_eq!(frames[6].1["content_block"]["name"], "bash");
        assert_eq!(frames[6].1["content_block"]["input"], json!({}));
        assert!(
            frames[6].1["content_block"]["id"]
                .as_str()
                .unwrap()
                .starts_with("toolu_")
        );
        assert_eq!(frames[7].1["delta"]["type"], "input_json_delta");
        assert_eq!(frames[7].1["delta"]["partial_json"], r#"{"command":"ls"}"#);
        assert_eq!(frames[8].1["index"], 1);
        // Terminal events.
        assert_eq!(frames[9].1["delta"]["stop_reason"], "tool_use");
        assert!(frames[9].1["delta"]["stop_sequence"].is_null());
        assert_eq!(frames[9].1["usage"]["output_tokens"], 7);
        assert_eq!(frames[10].1["type"], "message_stop");
    }

    /// Whitespace-only text never opens a block: a stray newline around a tool
    /// call is dropped (like the non-streaming trim), while whitespace followed
    /// by real text is preserved into the block's first delta.
    #[test]
    fn sse_block_opening_whitespace_is_held() {
        let mut encoder = StreamEncoder::new("msg_t".to_string(), "m".to_string());
        let mut raw = encoder.tool_use("bash", &json!({}));
        assert_eq!(encoder.text_delta("\n"), "", "trailing newline held");
        raw.push_str(&encoder.finish("tool_use", 3));
        let events: Vec<String> = parse_frames(&raw)
            .into_iter()
            .map(|(event, _)| event)
            .collect();
        assert_eq!(
            events,
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
            "no whitespace-only text block after the tool block"
        );

        // Held whitespace flushes into the first real delta of the next block.
        let mut encoder = StreamEncoder::new("msg_t".to_string(), "m".to_string());
        assert_eq!(encoder.text_delta("\n\n"), "");
        let frames = parse_frames(&encoder.text_delta("Done."));
        assert_eq!(frames[1].1["delta"]["text"], "\n\nDone.");
    }

    /// Finishing with an open text block closes it before message_delta.
    #[test]
    fn sse_finish_closes_open_text_block() {
        let mut encoder = StreamEncoder::new("msg_t".to_string(), "m".to_string());
        let mut raw = encoder.text_delta("hi");
        raw.push_str(&encoder.finish("end_turn", 1));
        let events: Vec<String> = parse_frames(&raw)
            .into_iter()
            .map(|(event, _)| event)
            .collect();
        assert_eq!(
            events,
            [
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }
}
