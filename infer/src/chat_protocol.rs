//! Shared chat/tool-call protocol helpers used by both the `infer` HTTP layer
//! and the root agent loop.

use serde_json::{Map, Value, json};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

#[derive(Clone, Copy)]
struct TaggedBlock {
    open: &'static str,
    close: &'static str,
}

impl TaggedBlock {
    fn strip_and_collect<T>(
        self,
        text: &str,
        mut parse_block: impl FnMut(&str) -> Option<T>,
    ) -> (String, Vec<T>) {
        let mut parsed = Vec::new();
        let mut remaining = text;
        let mut stripped = String::with_capacity(text.len());

        while let Some(start) = remaining.find(self.open) {
            stripped.push_str(&remaining[..start]);
            remaining = &remaining[start + self.open.len()..];

            if let Some(end) = remaining.find(self.close) {
                let block = remaining[..end].trim();
                if let Some(item) = parse_block(block) {
                    parsed.push(item);
                }
                remaining = &remaining[end + self.close.len()..];
            } else {
                stripped.push_str(remaining);
                remaining = "";
            }
        }

        stripped.push_str(remaining);
        (stripped, parsed)
    }

    fn strip_all(self, text: &str) -> String {
        let (stripped, _) = self.strip_and_collect::<()>(text, |_| None);
        stripped
    }
}

const TOOL_CALL_BLOCK: TaggedBlock = TaggedBlock {
    open: "<tool_call>",
    close: "</tool_call>",
};

const THINK_BLOCK: TaggedBlock = TaggedBlock {
    open: "<think>",
    close: "</think>",
};

struct PromptRenderer<'a> {
    prompt: String,
    tool_block: &'a str,
    system_injected: bool,
}

impl<'a> PromptRenderer<'a> {
    fn new(tool_block: &'a str) -> Self {
        Self {
            prompt: String::new(),
            tool_block,
            system_injected: false,
        }
    }

    fn push_message(&mut self, message: &ChatMessage) {
        match &message.role {
            ChatRole::System => self.push_system(&message.content),
            ChatRole::User => self.push_user(&message.content),
            ChatRole::Assistant => self.push_assistant(&message.content, &message.tool_calls),
            ChatRole::Tool => self.push_tool(&message.content),
            ChatRole::Other(role) => self.push_other(role, &message.content),
        }
    }

    fn finish(mut self) -> String {
        self.prompt.push_str("<|im_start|>assistant\n");
        self.prompt
    }

    fn ensure_default_system_message(&mut self) {
        if self.system_injected || self.tool_block.is_empty() {
            return;
        }

        self.start_message(ChatRole::System.as_str());
        self.prompt.push_str(DEFAULT_SYSTEM_PROMPT);
        self.prompt.push_str(self.tool_block);
        self.end_message();
        self.system_injected = true;
    }

    fn start_message(&mut self, role: &str) {
        self.prompt.push_str("<|im_start|>");
        self.prompt.push_str(role);
        self.prompt.push('\n');
    }

    fn end_message(&mut self) {
        self.prompt.push_str("<|im_end|>\n");
    }

    fn push_system(&mut self, content: &str) {
        self.start_message(ChatRole::System.as_str());
        self.prompt.push_str(content);
        if !self.tool_block.is_empty() {
            self.prompt.push_str(self.tool_block);
        }
        self.end_message();
        self.system_injected = true;
    }

    fn push_user(&mut self, content: &str) {
        self.ensure_default_system_message();
        self.start_message(ChatRole::User.as_str());
        self.prompt.push_str(content);
        self.end_message();
    }

    fn push_assistant(&mut self, content: &str, tool_calls: &[ToolCall]) {
        self.start_message(ChatRole::Assistant.as_str());
        self.prompt.push_str(content);
        for tool_call in tool_calls {
            self.prompt.push_str("\n<tool_call>\n");
            self.prompt.push_str(&tool_call.prompt_payload());
            self.prompt.push_str("\n</tool_call>");
        }
        self.end_message();
    }

    fn push_tool(&mut self, content: &str) {
        self.start_message(ChatRole::Tool.as_str());
        self.prompt.push_str("<tool_response>\n");
        self.prompt.push_str(content);
        self.prompt.push_str("\n</tool_response>");
        self.end_message();
    }

    fn push_other(&mut self, role: &str, content: &str) {
        self.start_message(role);
        self.prompt.push_str(content);
        self.end_message();
    }
}

/// Structured tool definition used for prompt injection.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    fn prompt_schema(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "arguments": compact_parameters(&self.parameters),
        })
    }
}

fn compact_parameters(parameters: &Value) -> Value {
    let Some(object) = parameters.as_object() else {
        return parameters.clone();
    };

    let Some(properties) = object.get("properties").and_then(Value::as_object) else {
        return parameters.clone();
    };

    let mut compact = Map::new();
    for (name, schema) in properties {
        let ty = schema
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("value")
            .to_string();
        compact.insert(name.clone(), Value::String(ty));
    }

    if let Some(required) = object.get("required").and_then(Value::as_array)
        && !required.is_empty()
    {
        compact.insert("required".to_string(), Value::Array(required.clone()));
    }

    Value::Object(compact)
}

/// Structured tool call emitted by the model or embedded in an assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    fn prompt_payload(&self) -> String {
        serde_json::to_string(&json!({
            "name": self.name,
            "arguments": self.arguments,
        }))
        .expect("tool call serialization")
    }
}

/// Role tags used by the shared ChatML formatter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Other(String),
}

impl ChatRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Other(role) => role.as_str(),
        }
    }
}

impl From<&str> for ChatRole {
    fn from(role: &str) -> Self {
        match role {
            "system" => Self::System,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Shared message shape used for prompt construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: ChatRole::System,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: ChatRole::User,
            content: content.to_string(),
            tool_calls: vec![],
        }
    }

    pub fn assistant(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.to_string(),
            tool_calls,
        }
    }

    pub fn tool_result(_tool_name: &str, result: &str) -> Self {
        Self {
            role: ChatRole::Tool,
            content: result.to_string(),
            tool_calls: vec![],
        }
    }
}

/// Parsed assistant output with tool calls stripped from visible content.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAssistantResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Build the tool definitions block for injection into a system prompt.
pub fn build_tool_block(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n<tools>");

    for tool in tools {
        out.push('\n');
        out.push_str(
            &serde_json::to_string(&tool.prompt_schema()).expect("tool schema serialization"),
        );
    }

    out.push_str("\n</tools>\nUse <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>.");
    out
}

/// Convert structured messages + tool definitions into a ChatML prompt.
pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let tool_block = build_tool_block(tools);
    let mut renderer = PromptRenderer::new(&tool_block);
    for message in messages {
        renderer.push_message(message);
    }
    renderer.finish()
}

/// Parse `<tool_call>...</tool_call>` blocks from raw assistant output.
pub fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    let (stripped, tool_calls) = TOOL_CALL_BLOCK.strip_and_collect(text, |json_str| {
        let value = serde_json::from_str::<Value>(json_str).ok()?;
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Map::default()));
        Some(ToolCall::new(name, arguments))
    });

    ParsedAssistantResponse {
        content: THINK_BLOCK.strip_all(&stripped).trim().to_string(),
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_user_message() {
        let prompt = messages_to_prompt(&[ChatMessage::user("hello")], &[]);
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn system_and_user_messages() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::system("You are helpful."),
                ChatMessage::user("hi"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn tool_definition_injected_into_system_prompt() {
        let prompt = messages_to_prompt(
            &[ChatMessage::user("list files")],
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
        );

        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("<tools>"));
    }

    #[test]
    fn tool_block_uses_compact_argument_shape() {
        let block = build_tool_block(&[ToolDefinition::new(
            "shell",
            "Run a shell command",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )]);

        assert!(block.contains(r#""arguments":{"command":"string","required":["command"]}"#));
        assert!(!block.contains(r#""type":"function""#));
        assert!(!block.contains(r#""properties""#));
        assert!(!block.contains("You may call one or more functions"));
    }

    #[test]
    fn assistant_tool_calls_render_as_xml_blocks() {
        let prompt = messages_to_prompt(
            &[ChatMessage::assistant(
                "Checking.",
                vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
            )],
            &[],
        );

        assert!(prompt.contains("Checking."));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains(r#""name":"shell""#));
        assert!(prompt.contains(r#""command":"pwd""#));
    }

    #[test]
    fn parse_tool_call_basic() {
        let parsed = parse_tool_calls(
            "Sure.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}\n</tool_call>",
        );

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["cmd"], "ls");
        assert_eq!(parsed.content, "Sure.");
    }

    #[test]
    fn parse_strips_think_blocks() {
        let parsed = parse_tool_calls("<think>\nI should check.\n</think>\nHere is the answer.");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "Here is the answer.");
    }

    #[test]
    fn multi_turn_conversation() {
        let prompt = messages_to_prompt(
            &[
                ChatMessage::user("first"),
                ChatMessage::assistant("response", vec![]),
                ChatMessage::user("second"),
            ],
            &[],
        );

        assert!(prompt.contains("<|im_start|>assistant\nresponse<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
