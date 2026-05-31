//! Shared chat/tool-call protocol helpers used by both the `infer` HTTP layer
//! and the root agent loop.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};
use std::ops::Range;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HiddenBlock {
    ToolCall,
    Think,
}

/// Incremental text filter for streamed assistant output.
///
/// This keeps user-visible text while stripping `<tool_call>...</tool_call>`
/// and `<think>...</think>` blocks across chunk boundaries.
#[derive(Default)]
pub struct VisibleTextStream {
    pending: String,
    hidden: Option<HiddenBlock>,
}

impl VisibleTextStream {
    pub fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> String {
        let mut visible = String::new();

        loop {
            match self.hidden {
                None => {
                    const VISIBLE_TAGS: [&str; 4] = [
                        TOOL_CALL_BLOCK.open,
                        TOOL_CALL_BLOCK.close,
                        THINK_BLOCK.open,
                        THINK_BLOCK.close,
                    ];
                    let Some((idx, tag)) = find_first_tag(&self.pending, &VISIBLE_TAGS) else {
                        if flush {
                            visible.push_str(&self.pending);
                            self.pending.clear();
                        } else {
                            let keep = longest_tag_prefix_suffix(&self.pending, &VISIBLE_TAGS);
                            let emit_len = self.pending.len().saturating_sub(keep);
                            visible.push_str(&self.pending[..emit_len]);
                            self.pending.drain(..emit_len);
                        }
                        break;
                    };

                    visible.push_str(&self.pending[..idx]);
                    self.pending.drain(..idx + tag.len());
                    self.hidden = match tag {
                        tag if tag == TOOL_CALL_BLOCK.open => Some(HiddenBlock::ToolCall),
                        tag if tag == THINK_BLOCK.open => Some(HiddenBlock::Think),
                        _ => None,
                    };
                }
                Some(HiddenBlock::ToolCall) => {
                    if let Some(idx) = self.pending.find(TOOL_CALL_BLOCK.close) {
                        self.pending.drain(..idx + TOOL_CALL_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep =
                            longest_tag_prefix_suffix(&self.pending, &[TOOL_CALL_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
                Some(HiddenBlock::Think) => {
                    if let Some(idx) = self.pending.find(THINK_BLOCK.close) {
                        self.pending.drain(..idx + THINK_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(&self.pending, &[THINK_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
            }
        }

        visible
    }
}

/// Incremental filter that mirrors [`VisibleTextStream`]'s hiding of
/// `<think>...</think>` and `<tool_call>...</tool_call>` blocks, but also
/// captures the completed tool calls instead of discarding them.
///
/// Use this on the streaming path when the request carries tool definitions:
/// it emits user-visible text exactly as `VisibleTextStream` would while
/// surfacing each closed `<tool_call>` block as a parsed [`ToolCall`].
#[derive(Default)]
pub struct StreamingToolCalls {
    pending: String,
    hidden: Option<HiddenBlock>,
    tool_buf: String,
}

impl StreamingToolCalls {
    /// Feed a chunk; returns `(visible_text_to_emit, newly_completed_tool_calls)`.
    pub fn push(&mut self, chunk: &str) -> (String, Vec<ToolCall>) {
        self.pending.push_str(chunk);
        self.drain(false)
    }

    /// Flush remaining buffered text. An unterminated `<tool_call>` block is
    /// dropped (no partial tool call is emitted).
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> (String, Vec<ToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();

        loop {
            match self.hidden {
                None => {
                    const VISIBLE_TAGS: [&str; 4] = [
                        TOOL_CALL_BLOCK.open,
                        TOOL_CALL_BLOCK.close,
                        THINK_BLOCK.open,
                        THINK_BLOCK.close,
                    ];
                    let Some((idx, tag)) = find_first_tag(&self.pending, &VISIBLE_TAGS) else {
                        if flush {
                            visible.push_str(&self.pending);
                            self.pending.clear();
                        } else {
                            let keep = longest_tag_prefix_suffix(&self.pending, &VISIBLE_TAGS);
                            let emit_len = self.pending.len().saturating_sub(keep);
                            visible.push_str(&self.pending[..emit_len]);
                            self.pending.drain(..emit_len);
                        }
                        break;
                    };

                    visible.push_str(&self.pending[..idx]);
                    self.pending.drain(..idx + tag.len());
                    self.hidden = match tag {
                        tag if tag == TOOL_CALL_BLOCK.open => Some(HiddenBlock::ToolCall),
                        tag if tag == THINK_BLOCK.open => Some(HiddenBlock::Think),
                        _ => None,
                    };
                }
                Some(HiddenBlock::ToolCall) => {
                    if let Some(idx) = self.pending.find(TOOL_CALL_BLOCK.close) {
                        self.tool_buf.push_str(&self.pending[..idx]);
                        if let Some(call) = parse_tool_call_block(self.tool_buf.trim()) {
                            calls.push(call);
                        }
                        self.tool_buf.clear();
                        self.pending.drain(..idx + TOOL_CALL_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        // Unterminated tool_call block: drop it.
                        self.pending.clear();
                        self.tool_buf.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep =
                            longest_tag_prefix_suffix(&self.pending, &[TOOL_CALL_BLOCK.close]);
                        let take_len = self.pending.len().saturating_sub(keep);
                        self.tool_buf.push_str(&self.pending[..take_len]);
                        self.pending.drain(..take_len);
                        break;
                    }
                }
                Some(HiddenBlock::Think) => {
                    if let Some(idx) = self.pending.find(THINK_BLOCK.close) {
                        self.pending.drain(..idx + THINK_BLOCK.close.len());
                        self.hidden = None;
                    } else if flush {
                        self.pending.clear();
                        self.hidden = None;
                        break;
                    } else {
                        let keep = longest_tag_prefix_suffix(&self.pending, &[THINK_BLOCK.close]);
                        let drop_len = self.pending.len().saturating_sub(keep);
                        self.pending.drain(..drop_len);
                        break;
                    }
                }
            }
        }

        (visible, calls)
    }
}

/// Parse a single `<tool_call>` inner JSON payload into a [`ToolCall`].
///
/// Shares the exact decoding contract used by [`parse_tool_calls`].
fn parse_tool_call_block(json_str: &str) -> Option<ToolCall> {
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
}

fn find_first_tag<'a>(text: &str, tags: &'a [&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|idx| (idx, *tag)))
        .min_by_key(|(idx, _)| *idx)
}

fn longest_tag_prefix_suffix(text: &str, tags: &[&str]) -> usize {
    let text = text.as_bytes();
    let max_len = tags
        .iter()
        .map(|tag| tag.len())
        .max()
        .unwrap_or(0)
        .min(text.len());

    (1..=max_len)
        .rev()
        .find(|&len| {
            let suffix = &text[text.len() - len..];
            tags.iter().any(|tag| tag.as_bytes().starts_with(suffix))
        })
        .unwrap_or(0)
}

struct PromptRenderer<'a> {
    prompt: String,
    tool_block: &'a str,
    system_injected: bool,
}

/// Borrowed ChatML message used by callers that only need raw role/content
/// rendering without tool or default-system handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMlMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Byte spans for a rendered ChatML turn.
///
/// `turn` covers the full `<|im_start|>role\ncontent<|im_end|>\n` slice.
/// `supervised` covers the slice that should receive labels. Different
/// renderers may choose slightly different supervision boundaries, but the
/// span always excludes the trailing newline after `<|im_end|>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMlSpan {
    pub turn: Range<usize>,
    pub supervised: Range<usize>,
}

/// Fully rendered ChatML prompt plus per-turn byte spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedChatMl {
    pub prompt: String,
    pub spans: Vec<ChatMlSpan>,
}

fn append_chatml_message_with_span(prompt: &mut String, role: &str, content: &str) -> ChatMlSpan {
    let turn_start = prompt.len();
    prompt.push_str("<|im_start|>");
    prompt.push_str(role);
    prompt.push('\n');

    let supervised_start = prompt.len();
    prompt.push_str(content);
    prompt.push_str("<|im_end|>");
    let supervised_end = prompt.len();
    prompt.push('\n');

    ChatMlSpan {
        turn: turn_start..prompt.len(),
        supervised: supervised_start..supervised_end,
    }
}

fn append_structured_chatml_message_with_span(
    prompt: &mut String,
    message: &ChatMessage,
) -> ChatMlSpan {
    let turn_start = prompt.len();
    prompt.push_str("<|im_start|>");
    let rendered_role = match message.role {
        ChatRole::Tool => ChatRole::User.as_str(),
        _ => message.role.as_str(),
    };
    prompt.push_str(rendered_role);
    prompt.push('\n');

    let supervised_start = prompt.len();
    match &message.role {
        ChatRole::System | ChatRole::User | ChatRole::Other(_) => {
            prompt.push_str(&message.content);
        }
        ChatRole::Assistant => {
            append_assistant_content(prompt, &message.content, &message.tool_calls);
        }
        ChatRole::Tool => {
            prompt.push_str("<tool_response>\n");
            prompt.push_str(&message.content);
            prompt.push_str("\n</tool_response>");
        }
    }

    prompt.push_str("<|im_end|>\n");
    let supervised_end = if matches!(&message.role, ChatRole::Assistant) {
        prompt.len() - 1
    } else {
        supervised_start
    };

    ChatMlSpan {
        turn: turn_start..prompt.len(),
        supervised: supervised_start..supervised_end,
    }
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
        append_assistant_content(&mut self.prompt, content, tool_calls);
        self.end_message();
    }

    fn push_tool(&mut self, content: &str) {
        // Qwen's official tool-calling template feeds tool results back as
        // special user messages rather than a dedicated `tool` role.
        self.start_message(ChatRole::User.as_str());
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

fn append_assistant_content(prompt: &mut String, content: &str, tool_calls: &[ToolCall]) {
    prompt.push_str(content);
    for tool_call in tool_calls {
        append_tool_call_block(prompt, tool_call);
    }
}

fn append_tool_call_block(prompt: &mut String, tool_call: &ToolCall) {
    if !prompt.ends_with('\n') {
        prompt.push('\n');
    }
    prompt.push_str(TOOL_CALL_BLOCK.open);
    prompt.push('\n');
    prompt.push_str(&tool_call.prompt_payload());
    prompt.push('\n');
    prompt.push_str(TOOL_CALL_BLOCK.close);
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
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

impl From<ChatRole> for String {
    fn from(role: ChatRole) -> Self {
        role.as_str().to_string()
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

impl From<String> for ChatRole {
    fn from(role: String) -> Self {
        Self::from(role.as_str())
    }
}

/// Shared message shape used for prompt construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default, deserialize_with = "deserialize_string_or_empty")]
    pub content: String,
    #[serde(
        default,
        deserialize_with = "deserialize_tool_calls_or_empty",
        skip_serializing_if = "Vec::is_empty"
    )]
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

/// OpenAI `tool_choice` semantics applied during prompt construction.
///
/// Mirrors the OpenAI wire format: `Auto` lets the model decide, `None`
/// suppresses tool emission entirely, `Required` forces some tool call, and
/// `Function` forces one named tool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoiceMode {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

/// Build the tool definitions block for injection into a system prompt.
///
/// Thin `Auto` wrapper over [`build_tool_block_with_choice`].
pub fn build_tool_block(tools: &[ToolDefinition]) -> String {
    build_tool_block_with_choice(tools, &ToolChoiceMode::Auto)
}

/// Build the tool definitions block honoring an OpenAI `tool_choice` hint.
///
/// `None` returns an empty block (no tools are advertised, so the model
/// answers as plain text). `Required` and `Function` append a directive that
/// forces tool emission.
pub fn build_tool_block_with_choice(tools: &[ToolDefinition], choice: &ToolChoiceMode) -> String {
    if matches!(choice, ToolChoiceMode::None) {
        return String::new();
    }
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

    out.push_str(
        "\n</tools>\nUse <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>. \
Never echo <tools>, <tool_call>, or <tool_response> in the final user-facing answer.",
    );

    match choice {
        ToolChoiceMode::Required => {
            out.push_str(
                "\nYou MUST respond by emitting a <tool_call> for one of the available tools; \
do not answer in plain prose.",
            );
        }
        ToolChoiceMode::Function(name) => {
            out.push_str(&format!(
                "\nYou MUST call the `{name}` tool via <tool_call> and no other tool; \
do not answer in plain prose."
            ));
        }
        ToolChoiceMode::Auto | ToolChoiceMode::None => {}
    }

    out
}

/// Convert structured messages + tool definitions into a ChatML prompt.
pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    messages_to_prompt_with_tool_choice(messages, tools, &ToolChoiceMode::Auto)
}

/// Convert structured messages + tool definitions into a ChatML prompt,
/// honoring an OpenAI `tool_choice` hint via [`build_tool_block_with_choice`].
pub fn messages_to_prompt_with_tool_choice(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    choice: &ToolChoiceMode,
) -> String {
    let tool_block = build_tool_block_with_choice(tools, choice);
    let mut renderer = PromptRenderer::new(&tool_block);
    for message in messages {
        renderer.push_message(message);
    }
    renderer.finish()
}

/// Render structured chat messages and return byte spans for each turn.
pub fn render_structured_chatml_with_spans(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> RenderedChatMl {
    let mut prompt = String::new();
    let mut spans = Vec::with_capacity(messages.len());

    for message in messages {
        spans.push(append_structured_chatml_message_with_span(
            &mut prompt,
            message,
        ));
    }
    if add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }

    RenderedChatMl { prompt, spans }
}

/// Parse `<tool_call>...</tool_call>` blocks from raw assistant output.
pub fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    let (stripped, tool_calls) = extract_tool_calls(text);

    ParsedAssistantResponse {
        content: THINK_BLOCK.strip_all(&stripped).trim().to_string(),
        tool_calls,
    }
}

/// Extract `<tool_call>` blocks tolerantly. Handles the canonical
/// `<tool_call>{json}</tool_call>` AND the Qwen3.6 quirk of dropping the
/// `</tool_call>` close tag — the JSON object is located by brace-matching after
/// the open tag, so a missing close no longer leaves the raw JSON in the visible
/// content (the old `strip_and_collect` else-branch did exactly that, which made
/// the agent leak a tool call as its "final answer" and give up). A truncated /
/// unbalanced JSON after an open tag is dropped to end-of-input rather than
/// leaked — a partial tool call can't be executed anyway. Returns the text with
/// every tool-call span removed, plus the parsed calls.
fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    if !text.contains(TOOL_CALL_BLOCK.open) {
        return (text.to_string(), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_CALL_BLOCK.open) {
        out.push_str(&rest[..start]);
        let after = &rest[start + TOOL_CALL_BLOCK.open.len()..];
        match after
            .find('{')
            .and_then(|b| json_object_len(&after[b..]).map(|len| (b, len)))
        {
            Some((b, len)) => {
                if let Some(call) = parse_tool_call_block(after[b..b + len].trim()) {
                    calls.push(call);
                }
                let mut consumed = b + len;
                // Swallow a trailing `</tool_call>` when present (only whitespace
                // between the JSON and the close tag) so it doesn't leak.
                let tail = &after[consumed..];
                if let Some(close_at) = tail.find(TOOL_CALL_BLOCK.close)
                    && tail[..close_at].trim().is_empty()
                {
                    consumed += close_at + TOOL_CALL_BLOCK.close.len();
                }
                rest = &after[consumed..];
            }
            None => {
                // No balanced JSON after the open tag (truncated mid-emission or a
                // non-JSON tool-call form) — drop the remainder so nothing leaks.
                rest = "";
            }
        }
    }
    out.push_str(rest);
    (out, calls)
}

/// Byte length of the balanced `{...}` object at the start of `s`, or `None` if
/// it never balances (truncated). String contents and `\`-escapes are respected
/// so braces inside JSON strings don't affect the depth count.
fn json_object_len(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &c) in bytes.iter().enumerate() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Render a raw ChatML message list using the canonical `<|im_start|>...`
/// layout without tool or default-system injection.
pub fn render_chatml(messages: &[ChatMlMessage<'_>], add_generation_prompt: bool) -> String {
    render_chatml_with_spans(messages, add_generation_prompt).prompt
}

/// Render ChatML and return byte spans for each turn.
pub fn render_chatml_with_spans(
    messages: &[ChatMlMessage<'_>],
    add_generation_prompt: bool,
) -> RenderedChatMl {
    let mut prompt = String::new();
    let mut spans = Vec::with_capacity(messages.len());

    for message in messages {
        spans.push(append_chatml_message_with_span(
            &mut prompt,
            message.role,
            message.content,
        ));
    }
    if add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }

    RenderedChatMl { prompt, spans }
}

fn deserialize_string_or_empty<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_tool_calls_or_empty<'de, D>(deserializer: D) -> Result<Vec<ToolCall>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<Vec<ToolCall>>::deserialize(deserializer)?.unwrap_or_default())
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
    fn render_chatml_single_message() {
        let prompt = render_chatml(
            &[ChatMlMessage {
                role: "user",
                content: "hello",
            }],
            true,
        );

        assert_eq!(
            prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_chatml_with_spans_tracks_body_range() {
        let rendered = render_chatml_with_spans(
            &[ChatMlMessage {
                role: "assistant",
                content: "\nhello",
            }],
            false,
        );

        assert_eq!(
            rendered.prompt,
            "<|im_start|>assistant\n\nhello<|im_end|>\n"
        );
        assert_eq!(rendered.spans.len(), 1);
        assert_eq!(rendered.spans[0].turn, 0..39);
        assert_eq!(rendered.spans[0].supervised, 22..38);
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
        assert!(prompt.contains("Checking.\n<tool_call>\n"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains(r#""name":"shell""#));
        assert!(prompt.contains(r#""command":"pwd""#));
    }

    #[test]
    fn empty_assistant_tool_call_matches_generation_prompt_continuation() {
        let user = ChatMessage::user("session w4 base prefix");
        let tool_call = ToolCall::new("retrieve_context", json!({ "query": "session-context" }));

        let warmup_prompt = messages_to_prompt(std::slice::from_ref(&user), &[]);
        let mut completed_warmup_prefix = warmup_prompt;
        completed_warmup_prefix.push_str(TOOL_CALL_BLOCK.open);
        completed_warmup_prefix.push('\n');
        completed_warmup_prefix.push_str(&tool_call.prompt_payload());
        completed_warmup_prefix.push('\n');
        completed_warmup_prefix.push_str(TOOL_CALL_BLOCK.close);
        completed_warmup_prefix.push_str("<|im_end|>\n");

        let resume_prompt = messages_to_prompt(
            &[
                user,
                ChatMessage::assistant("", vec![tool_call]),
                ChatMessage::tool_result("retrieve_context", "tool result payload"),
            ],
            &[],
        );

        assert!(resume_prompt.starts_with(&completed_warmup_prefix));
        assert!(!resume_prompt.contains("<|im_start|>assistant\n\n<tool_call>"));
    }

    #[test]
    fn chat_message_deserializes_tool_calls_and_null_content() {
        let message = serde_json::from_str::<ChatMessage>(
            r#"{"role":"assistant","content":null,"tool_calls":[{"name":"shell","arguments":{"command":"pwd"}}]}"#,
        )
        .expect("chat message should deserialize");

        assert_eq!(message.role, ChatRole::Assistant);
        assert_eq!(message.content, "");
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].name, "shell");
        assert_eq!(message.tool_calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn structured_render_only_labels_assistant_turns() {
        let rendered = render_structured_chatml_with_spans(
            &[
                ChatMessage::user("first"),
                ChatMessage::assistant(
                    "",
                    vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
                ),
                ChatMessage::tool_result("shell", "cwd"),
                ChatMessage::assistant("done", vec![]),
            ],
            false,
        );

        assert!(rendered.prompt.contains("<tool_call>"));
        assert!(rendered.prompt.contains("<tool_response>"));
        assert!(rendered.spans[0].supervised.is_empty());
        assert!(!rendered.spans[1].supervised.is_empty());
        assert!(rendered.spans[2].supervised.is_empty());
        assert!(!rendered.spans[3].supervised.is_empty());
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
    fn parse_tool_call_missing_close_tag() {
        // Qwen3.6 quirk: drops `</tool_call>`, sometimes puts `arguments` first.
        let parsed = parse_tool_calls(
            "<tool_call>\n\n{\"arguments\":{\"command\":\"ls -la\"},\"name\":\"shell\"}",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "ls -la");
        // The raw JSON must NOT leak into the visible content.
        assert_eq!(parsed.content, "");
        assert!(!parsed.content.contains("arguments"));
    }

    #[test]
    fn parse_tool_call_truncated_is_dropped_not_leaked() {
        // Generation cut off mid-JSON (unbalanced) — must hide, not leak.
        let parsed =
            parse_tool_calls("<tool_call>\n{\"arguments\":{\"command\":\"ls crates/cli/src/\"}");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "");
        assert!(!parsed.content.contains("arguments"));
    }

    #[test]
    fn parse_tool_call_text_after_close_preserved() {
        let parsed = parse_tool_calls(
            "<tool_call>{\"name\":\"shell\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call> done now.",
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.content, "done now.");
    }

    #[test]
    fn parse_strips_think_blocks() {
        let parsed = parse_tool_calls("<think>\nI should check.\n</think>\nHere is the answer.");
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content, "Here is the answer.");
    }

    #[test]
    fn visible_text_stream_strips_hidden_blocks_across_chunk_boundaries() {
        let mut stream = VisibleTextStream::default();
        let mut visible = String::new();

        for chunk in [
            "Hello<th",
            "ink>secret</th",
            "ink> world<tool",
            "_call>{\"name\":\"shell\"}</tool_call>!",
        ] {
            visible.push_str(&stream.push(chunk));
        }
        visible.push_str(&stream.finish());

        assert_eq!(visible, "Hello world!");
    }

    #[test]
    fn visible_text_stream_keeps_partial_tag_prefix_hidden_until_resolved() {
        let mut stream = VisibleTextStream::default();

        assert_eq!(stream.push("abc<th"), "abc");
        assert_eq!(stream.push("ink>secret</think>def"), "def");
        assert_eq!(stream.finish(), "");
    }

    #[test]
    fn visible_text_stream_handles_multibyte_text_before_partial_tag() {
        let mut stream = VisibleTextStream::default();

        assert_eq!(stream.push("User:** \"你<th"), "User:** \"你");
        assert_eq!(stream.push("ink>secret</think>好"), "好");
        assert_eq!(stream.finish(), "");
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

    fn drive_streaming(
        stream: &mut StreamingToolCalls,
        chunks: &[&str],
    ) -> (String, Vec<ToolCall>) {
        let mut visible = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            let (text, new_calls) = stream.push(chunk);
            visible.push_str(&text);
            calls.extend(new_calls);
        }
        let (text, new_calls) = stream.finish();
        visible.push_str(&text);
        calls.extend(new_calls);
        (visible, calls)
    }

    #[test]
    fn streaming_tool_calls_single_call_split_across_chunks() {
        let mut stream = StreamingToolCalls::default();
        // The `<tool_call>` open tag is split mid-tag across chunk boundaries.
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Looking<tool",
                "_call>{\"name\":\"shell\",\"argum",
                "ents\":{\"command\":\"pwd\"}}</tool_call> done",
            ],
        );

        assert_eq!(visible, "Looking done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
    }

    #[test]
    fn streaming_tool_calls_two_sequential_calls() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call>",
                "<tool_call>{\"name\":\"b\",\"arguments\":{\"y\":2}}</tool_call>",
            ],
        );

        assert_eq!(visible, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].arguments["x"], 1);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments["y"], 2);
    }

    #[test]
    fn streaming_tool_calls_hides_think_and_keeps_visible_text() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(
            &mut stream,
            &[
                "Hello <think>private</think>world ",
                "<tool_call>{\"name\":\"shell\",\"arguments\":{}}</tool_call>",
                "!",
            ],
        );

        assert_eq!(visible, "Hello world !");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn streaming_tool_calls_plain_text_passes_through() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) = drive_streaming(&mut stream, &["just ", "plain ", "text answer"]);

        assert_eq!(visible, "just plain text answer");
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_tool_calls_drops_unterminated_block() {
        let mut stream = StreamingToolCalls::default();
        let (visible, calls) =
            drive_streaming(&mut stream, &["text <tool_call>{\"name\":\"shell\",\"argu"]);

        assert_eq!(visible, "text ");
        assert!(calls.is_empty());
    }

    #[test]
    fn build_tool_block_with_choice_none_is_empty() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::None,
        );
        assert!(block.is_empty());
    }

    #[test]
    fn build_tool_block_with_choice_required_appends_directive() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::Required,
        );
        assert!(block.contains("<tools>"));
        assert!(block.contains("You MUST respond by emitting a <tool_call>"));
    }

    #[test]
    fn build_tool_block_with_choice_function_names_the_tool() {
        let block = build_tool_block_with_choice(
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
            &ToolChoiceMode::Function("shell".to_string()),
        );
        assert!(block.contains("You MUST call the `shell` tool via <tool_call>"));
    }

    #[test]
    fn build_tool_block_required_with_no_tools_is_empty() {
        let block = build_tool_block_with_choice(&[], &ToolChoiceMode::Required);
        assert!(block.is_empty());
    }
}
