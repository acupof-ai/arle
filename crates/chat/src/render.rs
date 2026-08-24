use super::tool::{TOOL_CALL_BLOCK, ToolCall, ToolDefinition};
use super::{ChatMessage, ChatRole};
use std::ops::Range;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedChatMl {
    pub prompt: String,
    pub spans: Vec<ChatMlSpan>,
}

/// OpenAI `tool_choice` semantics applied during prompt construction.
///
/// Mirrors the OpenAI wire format: `Auto` lets the model decide, `None`
/// suppresses tool emission entirely.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoiceMode {
    #[default]
    Auto,
    None,
}

struct PromptRenderer<'a> {
    prompt: String,
    tool_block: &'a str,
    system_injected: bool,
    /// Set when any message contains the Qwen `/no_think` soft-switch; the
    /// generation prompt then gets a pre-closed `<think></think>` block so the
    /// model skips reasoning and emits its action directly (mirrors the Qwen
    /// `enable_thinking=False` chat template). Scoped: callers that never pass
    /// `/no_think` (e.g. the interactive agent) keep thinking enabled.
    no_think: bool,
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
            no_think: false,
        }
    }

    fn push_message(&mut self, message: &ChatMessage) {
        if !self.no_think && message.content.contains("/no_think") {
            self.no_think = true;
        }
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
        if self.no_think {
            self.prompt.push_str("<think>\n\n</think>\n\n");
        }
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

fn build_tool_block(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut out =
        String::from("\n# Tools\n\nYou have access to the following functions:\n\n<tools>");

    for tool in tools {
        out.push('\n');
        out.push_str(
            &serde_json::to_string(&tool.prompt_schema()).expect("tool schema serialization"),
        );
    }

    out.push_str(
        "\n</tools>\n\n\
If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n\
<function=example_function_name>\n\
<parameter=example_parameter_1>\n\
value_1\n\
</parameter>\n\
<parameter=example_parameter_2>\n\
This is the value for the second parameter\n\
that can span\n\
multiple lines\n\
</parameter>\n\
</function>\n\
</tool_call>\n\n\
<IMPORTANT>\n\
Reminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>",
    );

    out
}

pub fn messages_to_prompt(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let tool_block = build_tool_block(tools);
    let mut renderer = PromptRenderer::new(&tool_block);
    for message in messages {
        renderer.push_message(message);
    }
    renderer.finish()
}

pub fn render_structured_chatml_with_spans(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> RenderedChatMl {
    let mut prompt = String::new();
    let spans: Vec<_> = messages
        .iter()
        .map(|message| append_structured_chatml_message_with_span(&mut prompt, message))
        .collect();
    if add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }

    RenderedChatMl { prompt, spans }
}
