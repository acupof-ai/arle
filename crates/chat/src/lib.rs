//! OpenAI-compatible chat request/response types layered on top of the shared
//! chat/tool-call protocol helpers in [`crate::protocol`].
//!
//! Naming convention:
//! - Types and functions prefixed `OpenAi` / `openai_` are the OpenAI wire
//!   format (HTTP request/response bodies).
//! - Unprefixed types and functions (`ChatMessage`, `ToolCall`, `ToolDefinition`,
//!   `messages_to_prompt`, `parse_tool_calls`) are the internal canonical
//!   protocol format, re-exported from [`crate::protocol`].

pub mod protocol;

pub use protocol::{
    ChatMessage, ChatMlMessage, ChatMlSpan, ChatRole, ParsedAssistantResponse, RenderedChatMl,
    ToolCall, ToolDefinition, build_tool_block, messages_to_prompt, parse_tool_calls,
    render_chatml, render_chatml_with_spans, render_structured_chatml_with_spans,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// A single message in a chat conversation, in OpenAI wire format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChatMessage {
    pub role: String,
    /// Text content. `None` when the assistant message contains only tool calls.
    #[serde(default)]
    pub content: Option<OpenAiChatContent>,
    /// Tool calls emitted by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAiToolCall>,
    /// Present on `tool` role messages — the call id being responded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on `tool` role messages — the tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message content as sent by OAI-compatible clients.
///
/// Modern tools (OpenAI SDK, vllm-project/guidellm, LiteLLM, LangChain's
/// openai adapter) always send `content` as a **part array**
/// (`[{"type":"text","text":"..."}, ...]`) to leave room for multimodal
/// inputs, while older tools still send a plain string. Our server is
/// text-only so we accept both and flatten via [`Self::to_text`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenAiChatContent {
    /// Legacy plain-text form: `"content": "hello"`.
    Text(String),
    /// Modern part-array form: `"content": [{"type":"text","text":"hello"}, ...]`.
    /// Kept as untyped `Value` so unsupported part types (image_url, audio,
    /// etc.) do not fail deserialization — they are simply ignored by
    /// [`Self::to_text`].
    Parts(Vec<Value>),
}

impl OpenAiChatContent {
    /// Flatten to a plain text string. For the Parts form, concatenates
    /// every `{"type":"text","text":"..."}` part in order; parts whose type
    /// is not `text` are silently skipped (we are a text-only server).
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            out.push_str(text);
                        }
                    }
                }
                out
            }
        }
    }
}

impl From<String> for OpenAiChatContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for OpenAiChatContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

/// OpenAI-format tool call object embedded in an assistant message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunctionCall,
}

/// Function name + JSON-encoded arguments.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    /// JSON string (not parsed object) — matches OpenAI wire format.
    pub arguments: String,
}

/// Tool definition passed in a chat completion request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiFunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

impl From<&OpenAiToolCall> for ToolCall {
    fn from(tool_call: &OpenAiToolCall) -> Self {
        let arguments = serde_json::from_str::<Value>(&tool_call.function.arguments)
            .unwrap_or_else(|_| Value::String(tool_call.function.arguments.clone()));
        Self::new(tool_call.function.name.clone(), arguments)
    }
}

impl From<&OpenAiChatMessage> for ChatMessage {
    fn from(message: &OpenAiChatMessage) -> Self {
        let tool_calls = message.tool_calls.iter().map(ToolCall::from).collect();

        Self {
            role: ChatRole::from(message.role.as_str()),
            content: message
                .content
                .as_ref()
                .map(OpenAiChatContent::to_text)
                .unwrap_or_default(),
            tool_calls,
        }
    }
}

impl From<&OpenAiToolDefinition> for ToolDefinition {
    fn from(tool: &OpenAiToolDefinition) -> Self {
        Self::new(
            tool.function.name.clone(),
            tool.function.description.clone().unwrap_or_default(),
            tool.function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({})),
        )
    }
}

/// Convert an OpenAI messages array + optional tool definitions into a
/// ChatML prompt string ready for inference.
pub fn openai_messages_to_prompt(
    messages: &[OpenAiChatMessage],
    tools: &[OpenAiToolDefinition],
) -> String {
    let protocol_messages = messages.iter().map(ChatMessage::from).collect::<Vec<_>>();
    let protocol_tools = tools.iter().map(ToolDefinition::from).collect::<Vec<_>>();
    messages_to_prompt(&protocol_messages, &protocol_tools)
}

/// Parse `<tool_call>...</tool_call>` blocks from model output.
/// Returns `(visible_content, tool_calls)` where `visible_content` has the
/// tool call blocks and `<think>` blocks stripped.
pub fn openai_parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let ParsedAssistantResponse {
        content,
        tool_calls,
    } = parse_tool_calls(text);
    (content, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_user_message() {
        let msgs = vec![OpenAiChatMessage {
            role: "user".into(),
            content: Some("hello".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = openai_messages_to_prompt(&msgs, &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn tool_definition_injected_into_system() {
        let tools = vec![OpenAiToolDefinition {
            tool_type: "function".into(),
            function: OpenAiFunctionDefinition {
                name: "shell".into(),
                description: Some("Run a shell command".into()),
                parameters: None,
            },
        }];
        let msgs = vec![OpenAiChatMessage {
            role: "user".into(),
            content: Some("hi".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = openai_messages_to_prompt(&msgs, &tools);
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains(r#""name":"shell""#));
    }

    #[test]
    fn parse_tool_call_basic() {
        let text = r#"Let me check that.
<tool_call>
{"name":"shell","arguments":{"command":"pwd"}}
</tool_call>"#;
        let (content, calls) = openai_parse_tool_calls(text);
        assert_eq!(content, "Let me check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn invalid_openai_tool_arguments_fall_back_to_string() {
        let prompt = openai_messages_to_prompt(
            &[OpenAiChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: vec![OpenAiToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: OpenAiFunctionCall {
                        name: "shell".into(),
                        arguments: "not-json".into(),
                    },
                }],
                tool_call_id: None,
                name: None,
            }],
            &[],
        );

        assert!(prompt.contains(r#""arguments":"not-json""#));
    }
}
