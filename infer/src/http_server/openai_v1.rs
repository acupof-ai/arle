use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::server_engine::{CompletionOutput, CompletionStreamDelta};
use crate::types::SessionId;
use infer_chat::{OpenAiChatMessage, OpenAiToolDefinition, ToolCall, openai_parse_tool_calls};

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
}

impl ModelsListResponse {
    pub(super) fn single(model_id: &str, created: u64) -> Self {
        Self {
            object: "list",
            data: vec![ModelObject {
                id: model_id.to_string(),
                object: "model",
                created,
                owned_by: "agent-infer",
            }],
        }
    }
}

// OpenAI-compatible /v1/completions request
#[derive(Debug, Deserialize)]
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
pub(super) struct StreamOptions {
    pub(super) include_usage: Option<bool>,
}

impl CompletionRequest {
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
    pub(super) fn from_output(model: String, created: u64, output: CompletionOutput) -> Self {
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created,
            model,
            choices: vec![Choice {
                text: output.text,
                index: 0,
                logprobs: if output.token_logprobs.is_empty() {
                    None
                } else {
                    Some(LogprobsResult {
                        token_logprobs: output.token_logprobs,
                    })
                },
                finish_reason: output.finish_reason.as_openai_str().to_string(),
            }],
            usage: output.usage.into(),
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
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                text: delta.text_delta,
                index: 0,
                logprobs: delta.logprob.map(|lp| LogprobsResult {
                    token_logprobs: vec![lp],
                }),
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
pub(super) struct ResponsesRequest {
    #[allow(dead_code)]
    pub(super) model: Option<String>,
    pub(super) input: ResponsesInput,
    pub(super) instructions: Option<String>,
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
mod tests {
    use super::*;

    #[test]
    fn completion_request_accepts_session_id() {
        let raw = r#"{"prompt":"hi","session_id":"agent-42"}"#;
        let req: CompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id_parsed().unwrap().as_str(), "agent-42");
    }

    #[test]
    fn completion_request_missing_session_id_is_none() {
        let raw = r#"{"prompt":"hi"}"#;
        let req: CompletionRequest = serde_json::from_str(raw).unwrap();
        assert!(req.session_id_parsed().is_none());
    }

    #[test]
    fn completion_request_empty_and_whitespace_session_id_is_none() {
        let empty: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hi","session_id":""}"#).unwrap();
        assert!(empty.session_id_parsed().is_none());

        let whitespace: CompletionRequest =
            serde_json::from_str(r#"{"prompt":"hi","session_id":"   "}"#).unwrap();
        assert!(whitespace.session_id_parsed().is_none());
    }

    #[test]
    fn completion_request_accepts_user_alias() {
        // OpenAI's canonical "user" field is the standard per-user identifier;
        // we accept it as an alias so existing clients opt into sticky routing
        // without changing their payloads.
        let raw = r#"{"prompt":"hi","user":"client-9"}"#;
        let req: CompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id_parsed().unwrap().as_str(), "client-9");
    }

    #[test]
    fn chat_completion_request_accepts_session_id_and_trims() {
        let raw = r#"{"messages":[{"role":"user","content":"hi"}],"session_id":"  sess-1  "}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id_parsed().unwrap().as_str(), "sess-1");
    }

    #[test]
    fn responses_request_accepts_string_input_and_user_alias() {
        let raw = r#"{"input":"hi","user":"agent-7"}"#;
        let req: ResponsesRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id_parsed().unwrap().as_str(), "agent-7");
        assert!(matches!(req.input, ResponsesInput::Text(_)));
    }

    #[test]
    fn responses_response_exposes_output_text_and_function_calls() {
        let response = ResponsesResponse::from_output(
            "Qwen3-4B".to_string(),
            1,
            CompletionOutput {
                text: "Let me check.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>".to_string(),
                finish_reason: crate::server_engine::FinishReason::Stop,
                usage: crate::server_engine::TokenUsage {
                    prompt_tokens: 2,
                    completion_tokens: 3,
                    total_tokens: 5,
                },
                token_logprobs: Vec::new(),
            },
        );

        let payload = serde_json::to_value(response).unwrap();
        assert_eq!(payload["object"], "response");
        assert_eq!(payload["usage"]["input_tokens"], 2);
        assert_eq!(payload["usage"]["output_tokens"], 3);
        assert_eq!(
            payload["output"][1]["type"],
            serde_json::Value::String("function_call".to_string())
        );
    }
}
