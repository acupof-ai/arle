//! OpenAI-compatible chat request/response types layered on top of the shared
//! chat/tool-call protocol helpers in [`crate::protocol`].

pub mod protocol;

pub use protocol::{
    ChatMessage as ProtocolChatMessage, ChatRole, ParsedAssistantResponse, ToolCall,
    ToolCall as ProtocolToolCall, ToolDefinition as ProtocolToolDefinition, build_tool_block,
    messages_to_prompt as protocol_messages_to_prompt,
    parse_tool_calls as parse_protocol_tool_calls,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// A single message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// Text content. `None` when the assistant message contains only tool calls.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls emitted by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallObject>,
    /// Present on `tool` role messages — the call id being responded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on `tool` role messages — the tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI-format tool call object embedded in an assistant message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallObject {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Function name + JSON-encoded arguments.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON string (not parsed object) — matches OpenAI wire format.
    pub arguments: String,
}

/// Tool definition passed in a chat completion request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

pub type ParsedToolCall = ToolCall;

impl From<&ToolCallObject> for ToolCall {
    fn from(tool_call: &ToolCallObject) -> Self {
        let arguments = serde_json::from_str::<Value>(&tool_call.function.arguments)
            .unwrap_or_else(|_| Value::String(tool_call.function.arguments.clone()));
        Self::new(tool_call.function.name.clone(), arguments)
    }
}

impl From<&ChatMessage> for ProtocolChatMessage {
    fn from(message: &ChatMessage) -> Self {
        let tool_calls = message.tool_calls.iter().map(ToolCall::from).collect();

        Self {
            role: ChatRole::from(message.role.as_str()),
            content: message.content.clone().unwrap_or_default(),
            tool_calls,
        }
    }
}

impl From<&ToolDefinition> for ProtocolToolDefinition {
    fn from(tool: &ToolDefinition) -> Self {
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
pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let protocol_messages = messages
        .iter()
        .map(ProtocolChatMessage::from)
        .collect::<Vec<_>>();
    let protocol_tools = tools
        .iter()
        .map(ProtocolToolDefinition::from)
        .collect::<Vec<_>>();
    protocol_messages_to_prompt(&protocol_messages, &protocol_tools)
}

/// Parse `<tool_call>...</tool_call>` blocks from model output.
/// Returns `(visible_content, tool_calls)` where `visible_content` has the
/// tool call blocks and `<think>` blocks stripped.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let ParsedAssistantResponse {
        content,
        tool_calls,
    } = parse_protocol_tool_calls(text);
    (content, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_user_message() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some("hello".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = messages_to_prompt(&msgs, &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn tool_definition_injected_into_system() {
        let tools = vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "shell".into(),
                description: Some("Run a shell command".into()),
                parameters: None,
            },
        }];
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: Some("hi".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = messages_to_prompt(&msgs, &tools);
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains(r#""name":"shell""#));
    }

    #[test]
    fn parse_tool_call_basic() {
        let text = r#"Let me check that.
<tool_call>
{"name":"shell","arguments":{"command":"pwd"}}
</tool_call>"#;
        let (content, calls) = parse_tool_calls(text);
        assert_eq!(content, "Let me check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn invalid_openai_tool_arguments_fall_back_to_string() {
        let prompt = messages_to_prompt(
            &[ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: vec![ToolCallObject {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
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
