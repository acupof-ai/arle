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
    ToolCall, ToolDefinition, VisibleTextStream, build_tool_block, messages_to_prompt,
    parse_tool_calls, render_chatml, render_chatml_with_spans, render_structured_chatml_with_spans,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DSV4_BOS: &str = "<｜begin▁of▁sentence｜>";
const DSV4_EOS: &str = "<｜end▁of▁sentence｜>";
const DSV4_USER: &str = "<｜User｜>";
const DSV4_ASSISTANT: &str = "<｜Assistant｜>";
const DSV4_THINK_START: &str = "<think>";
const DSV4_THINK_END: &str = "</think>";
const DSV4_DSML: &str = "｜DSML｜";

const DSV4_REASONING_EFFORT_MAX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
);

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

/// Image input embedded in an OpenAI-style chat content part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiImageInput {
    pub url: String,
    pub detail: Option<String>,
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

    /// Extract `image_url` parts without changing text prompt rendering.
    ///
    /// Supports both OpenAI's object form
    /// `{"type":"image_url","image_url":{"url":"..."}}` and the string
    /// shortcut accepted by several compatible clients.
    pub fn image_inputs(&self) -> Vec<OpenAiImageInput> {
        let Self::Parts(parts) = self else {
            return Vec::new();
        };

        parts.iter().filter_map(parse_image_input_part).collect()
    }
}

fn parse_image_input_part(part: &Value) -> Option<OpenAiImageInput> {
    if part.get("type").and_then(Value::as_str) != Some("image_url") {
        return None;
    }
    match part.get("image_url")? {
        Value::String(url) => Some(OpenAiImageInput {
            url: url.clone(),
            detail: None,
        }),
        Value::Object(image_url) => {
            let url = image_url.get("url")?.as_str()?.to_owned();
            let detail = image_url
                .get("detail")
                .and_then(Value::as_str)
                .map(str::to_owned);
            Some(OpenAiImageInput { url, detail })
        }
        _ => None,
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeepSeekV4ChatTemplateOptions {
    /// `false` renders the non-thinking chat prefix:
    /// `<｜Assistant｜></think>`. `true` renders `<｜Assistant｜><think>`.
    pub thinking: bool,
    /// Currently only `"max"` emits the official long reasoning-effort prefix.
    pub reasoning_effort: Option<String>,
}

/// Convert OpenAI messages to the DeepSeek-V4 encoding format.
///
/// DeepSeek-V4-Flash does not ship a Jinja `chat_template`; the model repo
/// provides a dedicated `encoding/encoding_dsv4.py` reference. This renderer
/// implements the serving subset used by OpenAI-compatible chat requests:
/// system/user/assistant/tool messages, OpenAI function-tool definitions, and
/// non-thinking or thinking assistant generation prefixes.
pub fn openai_messages_to_deepseek_v4_prompt(
    messages: &[OpenAiChatMessage],
    tools: &[OpenAiToolDefinition],
    options: &DeepSeekV4ChatTemplateOptions,
) -> String {
    let mut out = String::from(DSV4_BOS);
    if options.thinking && options.reasoning_effort.as_deref() == Some("max") {
        out.push_str(DSV4_REASONING_EFFORT_MAX);
    }

    let assistant_prefix = if options.thinking {
        DSV4_THINK_START
    } else {
        DSV4_THINK_END
    };
    let mut rendered_tools = false;

    if !tools.is_empty() && !messages.iter().any(|m| m.role == "system") {
        out.push_str(&render_deepseek_v4_tools(tools));
        rendered_tools = true;
    }

    for (idx, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "system" => {
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
                if !rendered_tools && !tools.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(&render_deepseek_v4_tools(tools));
                    rendered_tools = true;
                }
            }
            "user" => {
                out.push_str(DSV4_USER);
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
            }
            "assistant" => {
                out.push_str(DSV4_ASSISTANT);
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
                if !message.tool_calls.is_empty() {
                    out.push_str(&render_deepseek_v4_tool_calls(&message.tool_calls));
                }
                out.push_str(DSV4_EOS);
            }
            "tool" => {
                out.push_str(DSV4_USER);
                out.push_str("<tool_result>");
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
                out.push_str("</tool_result>");
            }
            "developer" => {
                out.push_str(DSV4_USER);
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
            }
            _ => {
                out.push_str(DSV4_USER);
                if let Some(content) = message.content.as_ref() {
                    out.push_str(&content.to_text());
                }
            }
        }

        let is_last = idx + 1 == messages.len();
        if is_last
            && matches!(
                message.role.as_str(),
                "system" | "user" | "tool" | "developer"
            )
        {
            out.push_str(DSV4_ASSISTANT);
            out.push_str(assistant_prefix);
        }
    }

    out
}

fn render_deepseek_v4_tools(tools: &[OpenAiToolDefinition]) -> String {
    let schemas = tools
        .iter()
        .map(|tool| {
            serde_json::to_string(&json!({
                "name": tool.function.name,
                "description": tool.function.description.clone().unwrap_or_default(),
                "parameters": tool.function.parameters.clone().unwrap_or_else(|| json!({})),
            }))
            .expect("tool schema serialization")
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "## Tools\n\n\
You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{DSV4_DSML}tool_calls>\" block like the following:\n\n\
<{DSV4_DSML}tool_calls>\n\
<{DSV4_DSML}invoke name=\"$TOOL_NAME\">\n\
<{DSV4_DSML}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSV4_DSML}parameter>\n\
...\n\
</{DSV4_DSML}invoke>\n\
</{DSV4_DSML}tool_calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by {DSV4_THINK_START}), you MUST output your complete reasoning inside {DSV4_THINK_START}...{DSV4_THINK_END} BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after {DSV4_THINK_END} with tool calls or final response.\n\n\
### Available Tool Schemas\n\n\
{schemas}\n\n\
You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls."
    )
}

fn render_deepseek_v4_tool_calls(tool_calls: &[OpenAiToolCall]) -> String {
    if tool_calls.is_empty() {
        return String::new();
    }

    let calls = tool_calls
        .iter()
        .map(|call| {
            format!(
                "<{DSV4_DSML}invoke name=\"{}\">\n{}\n</{DSV4_DSML}invoke>",
                call.function.name,
                render_deepseek_v4_tool_arguments(&call.function.arguments)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("\n\n<{DSV4_DSML}tool_calls>\n{calls}\n</{DSV4_DSML}tool_calls>")
}

fn render_deepseek_v4_tool_arguments(arguments: &str) -> String {
    let parsed = serde_json::from_str::<Value>(arguments)
        .unwrap_or_else(|_| json!({ "arguments": arguments }));
    let Some(obj) = parsed.as_object() else {
        return format!(
            "<{DSV4_DSML}parameter name=\"arguments\" string=\"false\">{parsed}</{DSV4_DSML}parameter>"
        );
    };

    obj.iter()
        .map(|(key, value)| {
            let is_string = value.is_string();
            let rendered = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            format!(
                "<{DSV4_DSML}parameter name=\"{key}\" string=\"{}\">{rendered}</{DSV4_DSML}parameter>",
                if is_string { "true" } else { "false" }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
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
    fn deepseek_v4_chat_prompt_uses_official_prefixes() {
        let msgs = vec![
            OpenAiChatMessage {
                role: "system".into(),
                content: Some("You are a helpful assistant.".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            OpenAiChatMessage {
                role: "user".into(),
                content: Some("What is 2+2?".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
        ];
        let prompt = openai_messages_to_deepseek_v4_prompt(
            &msgs,
            &[],
            &DeepSeekV4ChatTemplateOptions::default(),
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn deepseek_v4_thinking_prompt_opens_think_block() {
        let msgs = vec![OpenAiChatMessage {
            role: "user".into(),
            content: Some("hello".into()),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }];
        let prompt = openai_messages_to_deepseek_v4_prompt(
            &msgs,
            &[],
            &DeepSeekV4ChatTemplateOptions {
                thinking: true,
                reasoning_effort: Some("max".into()),
            },
        );

        assert!(prompt.starts_with(DSV4_BOS));
        assert!(prompt.contains("Reasoning Effort: Absolute maximum"));
        assert!(prompt.ends_with("<｜User｜>hello<｜Assistant｜><think>"));
    }

    #[test]
    fn deepseek_v4_history_marks_assistant_turns_once() {
        let msgs = vec![
            OpenAiChatMessage {
                role: "user".into(),
                content: Some("one".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            OpenAiChatMessage {
                role: "assistant".into(),
                content: Some("two".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
            OpenAiChatMessage {
                role: "user".into(),
                content: Some("three".into()),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
            },
        ];
        let prompt = openai_messages_to_deepseek_v4_prompt(
            &msgs,
            &[],
            &DeepSeekV4ChatTemplateOptions::default(),
        );

        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>one<｜Assistant｜>two<｜end▁of▁sentence｜><｜User｜>three<｜Assistant｜></think>"
        );
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

    #[test]
    fn parts_text_rendering_ignores_image_parts() {
        let content = OpenAiChatContent::Parts(vec![
            json!({"type": "text", "text": "look"}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}),
            json!({"type": "text", "text": " now"}),
        ]);

        assert_eq!(content.to_text(), "look now");
    }

    #[test]
    fn image_inputs_extract_object_and_string_forms() {
        let content = OpenAiChatContent::Parts(vec![
            json!({"type": "text", "text": "look"}),
            json!({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AAAA",
                    "detail": "high"
                }
            }),
            json!({
                "type": "image_url",
                "image_url": "https://example.test/image.jpg"
            }),
        ]);

        assert_eq!(
            content.image_inputs(),
            vec![
                OpenAiImageInput {
                    url: "data:image/png;base64,AAAA".to_owned(),
                    detail: Some("high".to_owned()),
                },
                OpenAiImageInput {
                    url: "https://example.test/image.jpg".to_owned(),
                    detail: None,
                },
            ]
        );
    }
}
