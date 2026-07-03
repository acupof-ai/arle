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
    pub(crate) fn to_chat_request(&self) -> ChatCompletionRequest {
        let mut messages = Vec::new();
        if let Some(system) = &self.system {
            let text = system.to_text();
            if !text.is_empty() {
                messages.push(text_message("system", text));
            }
        }
        for message in &self.messages {
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
            stop: (!self.stop_sequences.is_empty()).then(|| self.stop_sequences.clone()),
            chat_template_kwargs: None,
            tools: self.tools.iter().map(to_openai_tool).collect(),
            tool_choice: self.tool_choice.as_ref().map(to_openai_tool_choice),
        }
    }
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
        if !choice.message.content.is_empty() {
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

/// Parse an OpenAI arguments-JSON string into the Anthropic `input` object;
/// non-object / unparseable arguments are preserved under `{"_raw": …}`.
pub(crate) fn arguments_to_input(raw: &str) -> Value {
    serde_json::from_str::<Value>(raw)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({ "_raw": raw }))
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
}

impl StreamEncoder {
    pub(crate) fn new(id: String, model: String) -> Self {
        Self {
            id,
            model,
            next_index: 0,
            text_index: None,
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
    pub(crate) fn text_delta(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
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

    fn close_text(&mut self) -> String {
        self.text_index.take().map_or_else(String::new, |index| {
            frame(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": index}),
            )
        })
    }

    /// Emit one complete tool_use block (start → whole-arguments
    /// `input_json_delta` → stop), closing any open text block first.
    pub(crate) fn tool_use(&mut self, name: &str, arguments_json: &str) -> String {
        let mut out = self.close_text();
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
        let mut out = self.close_text();
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
