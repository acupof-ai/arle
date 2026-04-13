use std::path::Path;

use anyhow::{Context, Result};
use infer_chat::{
    ChatRole, ParsedAssistantResponse, ProtocolChatMessage, ProtocolToolCall,
    ProtocolToolDefinition,
};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::json;

use infer_tools::{Tool, execute_tool_call};

pub type Message = ProtocolChatMessage;
#[cfg_attr(not(test), allow(dead_code))]
pub type ToolCall = infer_chat::ToolCall;

#[derive(Clone, Debug, PartialEq)]
pub struct AgentCompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Option<Vec<String>>,
    pub logprobs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentFinishReason {
    Length,
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentCompleteOutput {
    pub text: String,
    pub finish_reason: AgentFinishReason,
    pub usage: AgentUsage,
    pub token_logprobs: Vec<f32>,
}

pub trait AgentEngine {
    fn model_id(&self) -> &str;
    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput>;
}

fn format_prompt(messages: &[Message], tools: &[Tool]) -> String {
    let protocol_tools: Vec<ProtocolToolDefinition> =
        tools.iter().map(Tool::to_definition).collect();
    infer_chat::protocol_messages_to_prompt(messages, &protocol_tools)
}

fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    infer_chat::parse_protocol_tool_calls(text)
}

const SYSTEM_PROMPT: &str = r#"You are a local CLI assistant with shell and python tools.
Answer briefly.
If a tool is needed, emit a <tool_call> block immediately with no explanation first.
After tool results, answer directly.
If the user asks for an exact format, output exactly that.
Do not expose chain-of-thought.
Prefer python for arithmetic and structured computation."#;

#[derive(Clone, Copy, Debug)]
pub struct AgentSettings {
    pub max_turns: usize,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTurnResult {
    pub text: String,
    pub tool_calls_executed: usize,
    pub max_turns_reached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionStats {
    pub conversation_messages: usize,
    pub user_messages: usize,
    pub assistant_messages: usize,
    pub tool_messages: usize,
    pub tool_calls: usize,
    pub content_chars: usize,
}

#[derive(Debug, Clone)]
pub struct AgentSession {
    messages: Vec<Message>,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    pub fn new() -> Self {
        Self {
            messages: vec![Message::system(SYSTEM_PROMPT)],
        }
    }

    pub fn reset(&mut self) {
        self.messages.truncate(1);
    }

    pub fn stats(&self) -> AgentSessionStats {
        let mut stats = AgentSessionStats {
            conversation_messages: self.messages.len().saturating_sub(1),
            user_messages: 0,
            assistant_messages: 0,
            tool_messages: 0,
            tool_calls: 0,
            content_chars: 0,
        };

        for message in self.messages.iter().skip(1) {
            stats.content_chars += message.content.len();
            match &message.role {
                ChatRole::User => stats.user_messages += 1,
                ChatRole::Assistant => {
                    stats.assistant_messages += 1;
                    stats.tool_calls += message.tool_calls.len();
                }
                ChatRole::Tool => stats.tool_messages += 1,
                _ => {}
            }
        }

        stats
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let snapshot = SessionSnapshot::from_messages(&self.messages);
        let payload = serde_json::to_vec_pretty(&snapshot)?;
        std::fs::write(path, payload)
            .with_context(|| format!("failed to write session file {}", path.display()))
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let payload = std::fs::read(path)
            .with_context(|| format!("failed to read session file {}", path.display()))?;
        let snapshot: SessionSnapshot = serde_json::from_slice(&payload)
            .with_context(|| format!("failed to parse session file {}", path.display()))?;
        Ok(Self {
            messages: snapshot.into_messages()?,
        })
    }

    pub fn replace_from_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        *self = Self::load_from_path(path)?;
        Ok(())
    }

    #[cfg(test)]
    fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn run_turn<E: AgentEngine + ?Sized>(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[Tool],
        settings: AgentSettings,
    ) -> Result<AgentTurnResult> {
        self.messages.push(Message::user(user_input));

        let mut tool_calls_executed = 0usize;
        let mut last_tool_name = None::<String>;
        let mut last_tool_scalar_result = None::<String>;

        if let Some(initial) = extract_tool_calls_from_user_request(user_input, tools) {
            info!("Recovered tool call(s) from user request before first model turn");
            self.messages
                .push(Message::assistant("", initial.tool_calls.clone()));
            execute_tool_calls(
                &initial.tool_calls,
                &mut self.messages,
                &mut tool_calls_executed,
                &mut last_tool_name,
                &mut last_tool_scalar_result,
            );

            if let Some(text) = finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                return Ok(AgentTurnResult {
                    text,
                    tool_calls_executed,
                    max_turns_reached: false,
                });
            }
        }

        for turn in 0..settings.max_turns {
            let prompt = format_prompt(&self.messages, tools);
            info!(
                "Agent turn {}/{}: prompt length = {} chars",
                turn + 1,
                settings.max_turns,
                prompt.len()
            );

            let output = engine.complete(AgentCompleteRequest {
                prompt,
                max_tokens: settings.max_tokens,
                temperature: settings.temperature,
                stop: Some(vec!["<|im_end|>".to_string()]),
                logprobs: false,
            })?;

            info!(
                "Generated {} chars, finish_reason={:?}",
                output.text.len(),
                output.finish_reason
            );

            let mut parsed = parse_tool_calls(&output.text);
            if parsed.tool_calls.is_empty() && tool_calls_executed == 0 && !tools.is_empty() {
                if let Some(recovered) = extract_tool_calls_from_user_request(user_input, tools)
                    .or_else(|| extract_tool_calls_from_draft(&output.text, tools))
                {
                    info!("Recovered tool call(s) via deterministic extraction");
                    parsed = recovered;
                } else if maybe_needs_tool_repair(&parsed.content)
                    && let Some(repaired) =
                        repair_tool_calls(engine, &self.messages, tools, settings, &output.text)?
                {
                    info!("Recovered tool call(s) via repair turn");
                    parsed = repaired;
                }
            }

            let content = finalize_response_text(
                user_input,
                parsed.content,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
                tool_calls_executed,
            );
            let tool_calls = parsed.tool_calls;

            self.messages
                .push(Message::assistant(&content, tool_calls.clone()));

            if tool_calls.is_empty() {
                return Ok(AgentTurnResult {
                    text: content,
                    tool_calls_executed,
                    max_turns_reached: false,
                });
            }

            if !content.is_empty() {
                println!("\x1b[2m{}\x1b[0m", content);
            }

            execute_tool_calls(
                &tool_calls,
                &mut self.messages,
                &mut tool_calls_executed,
                &mut last_tool_name,
                &mut last_tool_scalar_result,
            );

            if let Some(text) = finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                return Ok(AgentTurnResult {
                    text,
                    tool_calls_executed,
                    max_turns_reached: false,
                });
            }
        }

        Ok(AgentTurnResult {
            text: "(max turns reached - agent stopped)".to_string(),
            tool_calls_executed,
            max_turns_reached: true,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionSnapshot {
    version: u32,
    messages: Vec<StoredMessage>,
}

impl SessionSnapshot {
    const VERSION: u32 = 1;

    fn from_messages(messages: &[Message]) -> Self {
        Self {
            version: Self::VERSION,
            messages: messages.iter().map(StoredMessage::from_message).collect(),
        }
    }

    fn into_messages(self) -> Result<Vec<Message>> {
        if self.version != Self::VERSION {
            anyhow::bail!(
                "unsupported session version {} (expected {})",
                self.version,
                Self::VERSION
            );
        }

        if self.messages.is_empty() {
            anyhow::bail!("session file does not contain any messages");
        }

        let messages = self
            .messages
            .into_iter()
            .map(StoredMessage::into_message)
            .collect::<Result<Vec<_>>>()?;

        if messages[0].role != ChatRole::System {
            anyhow::bail!("session file must start with a system message");
        }

        Ok(messages)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Vec<StoredToolCall>,
}

impl StoredMessage {
    fn from_message(message: &Message) -> Self {
        Self {
            role: message.role.as_str().to_string(),
            content: message.content.clone(),
            tool_calls: message
                .tool_calls
                .iter()
                .map(StoredToolCall::from_tool_call)
                .collect(),
        }
    }

    fn into_message(self) -> Result<Message> {
        let role = ChatRole::from(self.role.as_str());
        if role == ChatRole::Tool && !self.tool_calls.is_empty() {
            anyhow::bail!("tool result messages cannot contain tool_calls");
        }

        let tool_calls = self
            .tool_calls
            .into_iter()
            .map(StoredToolCall::into_tool_call)
            .collect::<Result<Vec<_>>>()?;

        Ok(Message {
            role,
            content: self.content,
            tool_calls,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredToolCall {
    name: String,
    arguments: serde_json::Value,
}

impl StoredToolCall {
    fn from_tool_call(tool_call: &ProtocolToolCall) -> Self {
        Self {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
        }
    }

    fn into_tool_call(self) -> Result<ProtocolToolCall> {
        if self.name.trim().is_empty() {
            anyhow::bail!("tool call name cannot be empty");
        }

        Ok(ProtocolToolCall::new(self.name, self.arguments))
    }
}

fn maybe_needs_tool_repair(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "tool",
        "function",
        "python",
        "shell",
        "execute",
        "run the code",
        "call the",
        "use the",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn extract_tool_calls_from_user_request(
    user_input: &str,
    tools: &[Tool],
) -> Option<ParsedAssistantResponse> {
    if tool_available(tools, "python") && mentions_python_tool(user_input) {
        if let Some(code) = extract_python_code(user_input) {
            return Some(single_tool_response("python", json!({ "code": code })));
        }
        if let Some(expr) = extract_arithmetic_expression(user_input) {
            return Some(single_tool_response(
                "python",
                json!({ "code": format!("print({expr})") }),
            ));
        }
    }

    if tool_available(tools, "shell")
        && mentions_shell_tool(user_input)
        && let Some(command) = extract_shell_command(user_input)
    {
        return Some(single_tool_response("shell", json!({ "command": command })));
    }

    if tool_available(tools, "shell") && asks_for_file_listing(user_input) {
        return Some(single_tool_response(
            "shell",
            json!({ "command": default_directory_listing_command() }),
        ));
    }

    None
}

fn extract_tool_calls_from_draft(draft: &str, tools: &[Tool]) -> Option<ParsedAssistantResponse> {
    if tool_available(tools, "python")
        && let Some(code) = extract_python_code(draft)
    {
        return Some(single_tool_response("python", json!({ "code": code })));
    }

    if tool_available(tools, "shell")
        && mentions_shell_tool(draft)
        && let Some(command) = extract_shell_command(draft)
    {
        return Some(single_tool_response("shell", json!({ "command": command })));
    }

    None
}

fn finalize_response_text(
    user_input: &str,
    content: String,
    last_tool_name: Option<&str>,
    last_tool_scalar_result: Option<&str>,
    tool_calls_executed: usize,
) -> String {
    if tool_calls_executed == 0 {
        return content;
    }

    if last_tool_name == Some("shell") && asks_for_file_listing(user_input) {
        return "Listed the current directory above.".to_string();
    }

    let Some(tool_result) = last_tool_scalar_result else {
        return content;
    };

    if content.trim().is_empty() || asks_for_exact_scalar_output(user_input) {
        return tool_result.to_string();
    }

    content
}

fn finalize_after_tool_execution(
    user_input: &str,
    last_tool_name: Option<&str>,
    last_tool_scalar_result: Option<&str>,
) -> Option<String> {
    if last_tool_name == Some("shell") && asks_for_file_listing(user_input) {
        return Some("Listed the current directory above.".to_string());
    }

    if asks_for_exact_scalar_output(user_input)
        && let Some(result) = last_tool_scalar_result
    {
        return Some(result.to_string());
    }

    None
}

fn single_tool_response(name: &str, arguments: serde_json::Value) -> ParsedAssistantResponse {
    ParsedAssistantResponse {
        content: String::new(),
        tool_calls: vec![ProtocolToolCall::new(name, arguments)],
    }
}

fn execute_tool_calls(
    tool_calls: &[ProtocolToolCall],
    messages: &mut Vec<Message>,
    tool_calls_executed: &mut usize,
    last_tool_name: &mut Option<String>,
    last_tool_scalar_result: &mut Option<String>,
) {
    println!();
    for tool_call in tool_calls {
        *tool_calls_executed += 1;
        println!(
            "\x1b[33m[tool: {}]\x1b[0m {}",
            tool_call.name,
            serde_json::to_string(&tool_call.arguments).unwrap_or_default()
        );

        let result = execute_tool_call(tool_call);
        let display_result = if result.len() > 500 {
            format!("{}... ({} chars total)", &result[..500], result.len())
        } else {
            result.clone()
        };

        println!("\x1b[36m{}\x1b[0m", display_result);
        println!();

        *last_tool_scalar_result = scalar_tool_result(&result);
        *last_tool_name = Some(tool_call.name.clone());
        messages.push(Message::tool_result(&tool_call.name, &result));
    }
}

fn tool_available(tools: &[Tool], name: &str) -> bool {
    tools.iter().any(|tool| tool.name == name)
}

fn mentions_python_tool(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("python tool")
        || lower.contains("python function")
        || lower.contains("use python")
        || lower.contains("run python")
}

fn mentions_shell_tool(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("shell tool")
        || lower.contains("shell command")
        || lower.contains("use shell")
        || lower.contains("run shell")
}

fn asks_for_file_listing(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "list files",
        "show files",
        "what files",
        "which files",
        "current directory",
        "local files",
        "files here",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || [
            "哪些文件",
            "有哪些文件",
            "有什么文件",
            "列出文件",
            "当前目录",
            "本地文件",
            "目录下",
        ]
        .iter()
        .any(|needle| text.contains(needle))
}

#[cfg(target_os = "windows")]
fn default_directory_listing_command() -> &'static str {
    "cd && dir /a"
}

#[cfg(not(target_os = "windows"))]
fn default_directory_listing_command() -> &'static str {
    "pwd && ls -la"
}

fn extract_python_code(text: &str) -> Option<String> {
    extract_fenced_code_block(text, &["python", "py"])
        .or_else(|| extract_balanced_call(text, "print("))
}

fn extract_shell_command(text: &str) -> Option<String> {
    extract_fenced_code_block(text, &["bash", "sh", "shell"]).or_else(|| {
        extract_backticked_snippet(text).and_then(|snippet| {
            if snippet.contains('\n') || snippet.trim().is_empty() {
                None
            } else {
                Some(snippet)
            }
        })
    })
}

fn extract_fenced_code_block(text: &str, languages: &[&str]) -> Option<String> {
    let mut remaining = text;
    while let Some(start) = remaining.find("```") {
        remaining = &remaining[start + 3..];
        let Some(end) = remaining.find("```") else {
            break;
        };

        let block = &remaining[..end];
        let (first_line, rest) = block.split_once('\n').unwrap_or((block, ""));
        let language = first_line.trim().to_ascii_lowercase();
        if languages.iter().any(|candidate| language == *candidate) {
            let code = rest.trim();
            if !code.is_empty() {
                return Some(code.to_string());
            }
        }

        remaining = &remaining[end + 3..];
    }

    None
}

fn extract_backticked_snippet(text: &str) -> Option<String> {
    let start = text.find('`')?;
    let rest = &text[start + 1..];
    let end = rest.find('`')?;
    let snippet = rest[..end].trim();
    if snippet.is_empty() {
        None
    } else {
        Some(snippet.to_string())
    }
}

fn extract_balanced_call(text: &str, start_pattern: &str) -> Option<String> {
    let start = text.find(start_pattern)?;
    let mut depth = 1usize;

    for (offset, ch) in text[start + start_pattern.len()..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + start_pattern.len() + offset + ch.len_utf8();
                    let snippet = text[start..end]
                        .trim_matches(|c| matches!(c, '`' | '"' | '\''))
                        .trim();
                    if !snippet.is_empty() {
                        return Some(snippet.to_string());
                    }
                    return None;
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_arithmetic_expression(text: &str) -> Option<String> {
    let mut best = String::new();
    let mut current = String::new();
    let mut has_digit = false;
    let mut has_operator = false;

    for ch in text.chars().chain(std::iter::once('\n')) {
        let allowed = ch.is_ascii_digit()
            || ch.is_ascii_whitespace()
            || matches!(ch, '+' | '-' | '*' | '/' | '%' | '(' | ')');
        if allowed {
            current.push(ch);
            has_digit |= ch.is_ascii_digit();
            has_operator |= matches!(ch, '+' | '-' | '*' | '/' | '%');
            continue;
        }

        let candidate = current.trim();
        if has_digit && has_operator && candidate.len() > best.len() {
            best = candidate.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        current.clear();
        has_digit = false;
        has_operator = false;
    }

    if best.is_empty() { None } else { Some(best) }
}

fn asks_for_exact_scalar_output(user_input: &str) -> bool {
    let lower = user_input.to_ascii_lowercase();
    [
        "answer with just",
        "reply with just",
        "nothing else",
        "the token only",
        "the word only",
        "just the integer",
        "integer only",
        "number only",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn scalar_tool_result(result: &str) -> Option<String> {
    if result.contains("[stderr]") {
        return None;
    }

    let mut lines = result
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    if lines.len() != 1 {
        return None;
    }

    let line = lines.remove(0);
    if line.len() > 120 {
        return None;
    }

    Some(line.to_string())
}

fn repair_tool_calls<E: AgentEngine + ?Sized>(
    engine: &mut E,
    messages: &[Message],
    tools: &[Tool],
    settings: AgentSettings,
    assistant_draft: &str,
) -> Result<Option<ParsedAssistantResponse>> {
    let mut repair_messages = messages.to_vec();
    repair_messages.push(Message::assistant(assistant_draft, vec![]));
    repair_messages.push(Message::user(
        "Rewrite your previous assistant message using the tool-call protocol. \
If a tool is needed, output only valid <tool_call> blocks and no other text. \
If no tool is needed, output exactly NO_TOOL.",
    ));

    let repair_prompt = format_prompt(&repair_messages, tools);
    let repair_output = engine.complete(AgentCompleteRequest {
        prompt: repair_prompt,
        max_tokens: settings.max_tokens.min(128),
        temperature: 0.0,
        stop: Some(vec!["<|im_end|>".to_string()]),
        logprobs: false,
    })?;

    let repaired = parse_tool_calls(&repair_output.text);
    if !repaired.tool_calls.is_empty() {
        return Ok(Some(repaired));
    }

    if repaired.content.trim() == "NO_TOOL" {
        return Ok(None);
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::{Result, anyhow};
    use infer_tools::Tool;
    use serde_json::json;

    use super::{
        AgentCompleteOutput, AgentCompleteRequest, AgentEngine, AgentFinishReason, AgentSession,
        AgentSettings, AgentUsage, Message, ToolCall,
    };

    struct FakeEngine {
        outputs: VecDeque<String>,
        prompts: Vec<String>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<&str>) -> Self {
            Self {
                outputs: outputs.into_iter().map(str::to_string).collect(),
                prompts: Vec::new(),
            }
        }
    }

    impl AgentEngine for FakeEngine {
        fn model_id(&self) -> &str {
            "fake"
        }

        fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput> {
            self.prompts.push(req.prompt);
            let text = self
                .outputs
                .pop_front()
                .ok_or_else(|| anyhow!("fake engine exhausted"))?;
            Ok(AgentCompleteOutput {
                usage: AgentUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                text,
                finish_reason: AgentFinishReason::Stop,
                token_logprobs: Vec::new(),
            })
        }
    }

    fn settings() -> AgentSettings {
        AgentSettings {
            max_turns: 4,
            max_tokens: 128,
            temperature: 0.0,
        }
    }

    fn python_tool() -> Tool {
        Tool {
            name: "python".into(),
            description: "Run Python".into(),
            parameters: json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            }),
        }
    }

    fn python_tool_available() -> bool {
        infer_tools::execute_tool("python", &json!({ "code": "print(1)" }))
            .trim()
            .eq("1")
    }

    fn shell_tool() -> Tool {
        Tool {
            name: "shell".into(),
            description: "Run shell".into(),
            parameters: json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        }
    }

    fn temp_session_path(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "agent-infer-{test_name}-{}-{nanos}.json",
            std::process::id()
        ))
    }

    #[test]
    fn parse_tool_calls_uses_shared_protocol_parser() {
        let parsed = super::parse_tool_calls(
            "Let me check.\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>",
        );

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert_eq!(parsed.tool_calls[0].arguments["command"], "pwd");
        assert_eq!(parsed.content, "Let me check.");
    }

    #[test]
    fn format_prompt_uses_shared_protocol_formatter() {
        let prompt = super::format_prompt(
            &[
                Message::system("You are helpful."),
                Message::assistant(
                    "Checking.",
                    vec![ToolCall::new("shell", json!({ "command": "pwd" }))],
                ),
            ],
            &[Tool {
                name: "shell".into(),
                description: "Run a shell command".into(),
                parameters: json!({}),
            }],
        );

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains(r#""name":"shell""#));
        assert!(prompt.contains(r#""command":"pwd""#));
    }

    #[test]
    fn session_persists_conversation_history_across_turns() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["alpha", "beta"]);

        let first = session
            .run_turn(&mut engine, "remember alpha", &[], settings())
            .expect("first turn");
        assert_eq!(first.text, "alpha");

        let second = session
            .run_turn(&mut engine, "what did I say before?", &[], settings())
            .expect("second turn");
        assert_eq!(second.text, "beta");

        let second_prompt = &engine.prompts[1];
        assert!(second_prompt.contains("remember alpha"));
        assert!(second_prompt.contains("alpha"));
    }

    #[test]
    fn tool_call_messages_are_not_duplicated_in_followup_prompt() {
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }

        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "Checking.\n<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(2 + 2)\"}}\n</tool_call>",
            "The answer is 4.",
        ]);
        let tools = vec![python_tool()];

        let result = session
            .run_turn(&mut engine, "compute 2+2", &tools, settings())
            .expect("tool turn");

        assert_eq!(result.text, "The answer is 4.");
        assert_eq!(result.tool_calls_executed, 1);

        let followup_prompt = &engine.prompts[1];
        assert_eq!(followup_prompt.matches("print(2 + 2)").count(), 1);
        assert!(followup_prompt.contains("Checking."));
        assert!(followup_prompt.contains("4"));
    }

    #[test]
    fn reset_clears_messages_but_keeps_system_prompt() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["hello"]);

        session
            .run_turn(&mut engine, "hi", &[], settings())
            .expect("turn");
        session.reset();

        assert_eq!(session.messages().len(), 1);
        assert_eq!(session.messages()[0].role, infer_chat::ChatRole::System);
    }

    #[test]
    fn stats_report_conversation_shape() {
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }

        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "Checking.\n<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(2 + 2)\"}}\n</tool_call>",
            "4",
        ]);

        session
            .run_turn(
                &mut engine,
                "Use the python tool to compute 2 + 2. After the tool returns, answer with just the integer.",
                &[python_tool()],
                settings(),
            )
            .expect("tool turn");

        let stats = session.stats();
        assert_eq!(stats.conversation_messages, 3);
        assert_eq!(stats.user_messages, 1);
        assert_eq!(stats.assistant_messages, 1);
        assert_eq!(stats.tool_messages, 1);
        assert_eq!(stats.tool_calls, 1);
        assert!(stats.content_chars > 0);
    }

    #[test]
    fn session_can_round_trip_via_disk_snapshot() {
        let path = temp_session_path("session-roundtrip");
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "Checking.\n<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(2 + 2)\"}}\n</tool_call>",
            "The answer is 4.",
        ]);

        session
            .run_turn(&mut engine, "compute 2+2", &[python_tool()], settings())
            .expect("tool turn");
        session.save_to_path(&path).expect("save session");

        let loaded = AgentSession::load_from_path(&path).expect("load session");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages(), session.messages());
        assert_eq!(loaded.stats(), session.stats());
    }

    #[test]
    fn maybe_needs_tool_repair_detects_tool_intent() {
        assert!(super::maybe_needs_tool_repair(
            "I should use the python tool to compute this."
        ));
        assert!(!super::maybe_needs_tool_repair("RIVER"));
    }

    #[test]
    fn explicit_python_request_can_skip_model_generation() {
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }

        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["I'll help with that.", "The result is 56088."]);

        let result = session
            .run_turn(
                &mut engine,
                "Use the python tool to compute 123 * 456. Do not do the math mentally. After the tool returns, answer with just the integer.",
                &[python_tool()],
                settings(),
            )
            .expect("tool recovery turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "56088");
        assert_eq!(engine.prompts.len(), 0);
    }

    #[test]
    fn python_tool_is_recovered_from_natural_language_draft() {
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }

        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "I should use the Python tool here. I can run print(7 * 8).",
            "The answer is 56.",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "What is 7 * 8? After the tool returns, answer with just the integer.",
                &[python_tool()],
                settings(),
            )
            .expect("draft recovery turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "56");
        assert_eq!(engine.prompts.len(), 1);
    }

    #[test]
    fn extract_arithmetic_expression_finds_inline_math() {
        assert_eq!(
            super::extract_arithmetic_expression("Use python to compute 123 * 456 right now."),
            Some("123 * 456".to_string())
        );
    }

    #[test]
    fn chinese_file_listing_request_is_recovered_with_shell_tool() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "I don't have a tool for listing files.",
            "I still don't have a suitable tool for this.",
        ]);

        let result = session
            .run_turn(&mut engine, "本地有哪些文件", &[shell_tool()], settings())
            .expect("file listing turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "Listed the current directory above.");
        assert_eq!(engine.prompts.len(), 0);
    }
}
