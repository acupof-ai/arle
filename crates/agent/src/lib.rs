use std::path::Path;

use anyhow::{Context, Result};
use chat::{ChatMessage, ChatRole, ParsedAssistantResponse, ToolCall, ToolDefinition};
use infer::sampler::SamplingParams;
use infer::server_engine::{CompletionRequest, InferenceEngine};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type Message = ChatMessage;

pub trait ToolExecutor {
    fn execute(&self, tool_call: &ToolCall) -> String;
}

pub trait ToolPolicy {
    fn recover_tool_calls_from_user_request(
        &self,
        _user_input: &str,
        _tools: &[ToolDefinition],
    ) -> Option<ParsedAssistantResponse> {
        None
    }

    fn recover_tool_calls_from_draft(
        &self,
        _draft: &str,
        _tools: &[ToolDefinition],
    ) -> Option<ParsedAssistantResponse> {
        None
    }

    fn should_repair_tool_calls(&self, _text: &str) -> bool {
        false
    }

    fn finalize_response_text(
        &self,
        _user_input: &str,
        content: String,
        _last_tool_name: Option<&str>,
        _last_tool_scalar_result: Option<&str>,
        _tool_calls_executed: usize,
    ) -> String {
        content
    }

    fn finalize_after_tool_execution(
        &self,
        _user_input: &str,
        _last_tool_name: Option<&str>,
        _last_tool_scalar_result: Option<&str>,
    ) -> Option<String> {
        None
    }
}

fn format_prompt(messages: &[Message], tools: &[ToolDefinition]) -> String {
    chat::messages_to_prompt(messages, tools)
}

fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    chat::parse_tool_calls(text)
}

const DEFAULT_SYSTEM_PROMPT: &str = r"You are a local CLI assistant.
Answer briefly.
If a tool is needed, emit a <tool_call> block immediately with no explanation first.
After tool results, answer directly.
If the user asks for an exact format, output exactly that.
Do not expose chain-of-thought.";

#[derive(Clone, Copy, Debug)]
pub struct AgentSettings {
    pub max_turns: usize,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentTraceEvent {
    AssistantNote(String),
    ToolCall {
        name: String,
        arguments: Value,
        result: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentTurnResult {
    pub text: String,
    pub tool_calls_executed: usize,
    pub max_turns_reached: bool,
    pub trace_events: Vec<AgentTraceEvent>,
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
        Self::with_system_prompt(DEFAULT_SYSTEM_PROMPT)
    }

    pub fn with_system_prompt(system_prompt: impl Into<String>) -> Self {
        let system_prompt = system_prompt.into();
        Self {
            messages: vec![Message::system(&system_prompt)],
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

    pub fn run_turn<E: InferenceEngine + ?Sized, X: ToolExecutor, P: ToolPolicy>(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[ToolDefinition],
        tool_executor: &X,
        tool_policy: &P,
        settings: AgentSettings,
    ) -> Result<AgentTurnResult> {
        self.messages.push(Message::user(user_input));

        let mut tool_calls_executed = 0usize;
        let mut last_tool_name = None::<String>;
        let mut last_tool_scalar_result = None::<String>;
        let mut trace_events = Vec::new();

        if let Some(initial) = tool_policy.recover_tool_calls_from_user_request(user_input, tools) {
            info!("Recovered tool call(s) from user request before first model turn");
            self.messages
                .push(Message::assistant("", initial.tool_calls.clone()));
            execute_tool_calls(
                &initial.tool_calls,
                tool_executor,
                &mut self.messages,
                &mut tool_calls_executed,
                &mut last_tool_name,
                &mut last_tool_scalar_result,
                &mut trace_events,
            );

            if let Some(text) = tool_policy.finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                return Ok(AgentTurnResult {
                    text,
                    tool_calls_executed,
                    max_turns_reached: false,
                    trace_events,
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

            let output = engine.complete(CompletionRequest {
                prompt,
                max_tokens: settings.max_tokens,
                sampling: SamplingParams {
                    temperature: settings.temperature,
                    ..SamplingParams::default()
                },
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
                if let Some(recovered) = tool_policy
                    .recover_tool_calls_from_user_request(user_input, tools)
                    .or_else(|| tool_policy.recover_tool_calls_from_draft(&output.text, tools))
                {
                    info!("Recovered tool call(s) via deterministic extraction");
                    parsed = recovered;
                } else if tool_policy.should_repair_tool_calls(&parsed.content)
                    && let Some(repaired) =
                        repair_tool_calls(engine, &self.messages, tools, settings, &output.text)?
                {
                    info!("Recovered tool call(s) via repair turn");
                    parsed = repaired;
                }
            }

            let content = tool_policy.finalize_response_text(
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
                    trace_events,
                });
            }

            if !content.is_empty() {
                trace_events.push(AgentTraceEvent::AssistantNote(content));
            }

            execute_tool_calls(
                &tool_calls,
                tool_executor,
                &mut self.messages,
                &mut tool_calls_executed,
                &mut last_tool_name,
                &mut last_tool_scalar_result,
                &mut trace_events,
            );

            if let Some(text) = tool_policy.finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                return Ok(AgentTurnResult {
                    text,
                    tool_calls_executed,
                    max_turns_reached: false,
                    trace_events,
                });
            }
        }

        Ok(AgentTurnResult {
            text: "(max turns reached - agent stopped)".to_string(),
            tool_calls_executed,
            max_turns_reached: true,
            trace_events,
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
    fn from_tool_call(tool_call: &ToolCall) -> Self {
        Self {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
        }
    }

    fn into_tool_call(self) -> Result<ToolCall> {
        if self.name.trim().is_empty() {
            anyhow::bail!("tool call name cannot be empty");
        }

        Ok(ToolCall::new(self.name, self.arguments))
    }
}

fn execute_tool_calls(
    tool_calls: &[ToolCall],
    tool_executor: &dyn ToolExecutor,
    messages: &mut Vec<Message>,
    tool_calls_executed: &mut usize,
    last_tool_name: &mut Option<String>,
    last_tool_scalar_result: &mut Option<String>,
    trace_events: &mut Vec<AgentTraceEvent>,
) {
    for tool_call in tool_calls {
        *tool_calls_executed += 1;
        let result = tool_executor.execute(tool_call);

        *last_tool_scalar_result = scalar_tool_result(&result);
        *last_tool_name = Some(tool_call.name.clone());
        trace_events.push(AgentTraceEvent::ToolCall {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
            result: result.clone(),
        });
        messages.push(Message::tool_result(&tool_call.name, &result));
    }
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

fn repair_tool_calls<E: InferenceEngine + ?Sized>(
    engine: &mut E,
    messages: &[Message],
    tools: &[ToolDefinition],
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
    let repair_output = engine.complete(CompletionRequest {
        prompt: repair_prompt,
        max_tokens: settings.max_tokens.min(128),
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
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
    use chat::{ParsedAssistantResponse, ToolCall, ToolDefinition};
    use infer::server_engine::{
        CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason, InferenceEngine,
        TokenUsage,
    };
    use serde_json::json;
    use tokio::sync::mpsc::UnboundedSender;
    use tools::BuiltinToolPolicyHooks;

    use super::{AgentSession, AgentSettings, Message, ToolExecutor, ToolPolicy};

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

    impl InferenceEngine for FakeEngine {
        fn model_id(&self) -> &str {
            "fake"
        }

        fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
            self.prompts.push(req.prompt);
            let text = self
                .outputs
                .pop_front()
                .ok_or_else(|| anyhow!("fake engine exhausted"))?;
            Ok(CompletionOutput {
                usage: TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                text,
                finish_reason: FinishReason::Stop,
                token_logprobs: Vec::new(),
            })
        }

        fn complete_stream(
            &mut self,
            req: CompletionRequest,
            tx: UnboundedSender<CompletionStreamDelta>,
        ) -> Result<()> {
            let output = self.complete(req)?;
            if !output.text.is_empty() {
                let _ = tx.send(CompletionStreamDelta {
                    text_delta: output.text.clone(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                });
            }
            let _ = tx.send(CompletionStreamDelta {
                text_delta: String::new(),
                finish_reason: Some(output.finish_reason),
                usage: Some(output.usage),
                logprob: None,
            });
            Ok(())
        }
    }

    fn settings() -> AgentSettings {
        AgentSettings {
            max_turns: 4,
            max_tokens: 128,
            temperature: 0.0,
        }
    }

    struct TestToolExecutor;

    impl ToolExecutor for TestToolExecutor {
        fn execute(&self, tool_call: &ToolCall) -> String {
            tools::execute_tool_call(tool_call)
        }
    }

    fn tool_executor() -> TestToolExecutor {
        TestToolExecutor
    }

    struct TestToolPolicy;

    impl ToolPolicy for TestToolPolicy {
        fn recover_tool_calls_from_user_request(
            &self,
            user_input: &str,
            tools: &[ToolDefinition],
        ) -> Option<ParsedAssistantResponse> {
            BuiltinToolPolicyHooks.recover_tool_calls_from_user_request(user_input, tools)
        }

        fn recover_tool_calls_from_draft(
            &self,
            draft: &str,
            tools: &[ToolDefinition],
        ) -> Option<ParsedAssistantResponse> {
            BuiltinToolPolicyHooks.recover_tool_calls_from_draft(draft, tools)
        }

        fn should_repair_tool_calls(&self, text: &str) -> bool {
            BuiltinToolPolicyHooks.should_repair_tool_calls(text)
        }

        fn finalize_response_text(
            &self,
            user_input: &str,
            content: String,
            last_tool_name: Option<&str>,
            last_tool_scalar_result: Option<&str>,
            tool_calls_executed: usize,
        ) -> String {
            BuiltinToolPolicyHooks.finalize_response_text(
                user_input,
                content,
                last_tool_name,
                last_tool_scalar_result,
                tool_calls_executed,
            )
        }

        fn finalize_after_tool_execution(
            &self,
            user_input: &str,
            last_tool_name: Option<&str>,
            last_tool_scalar_result: Option<&str>,
        ) -> Option<String> {
            BuiltinToolPolicyHooks.finalize_after_tool_execution(
                user_input,
                last_tool_name,
                last_tool_scalar_result,
            )
        }
    }

    fn tool_policy() -> TestToolPolicy {
        TestToolPolicy
    }

    fn python_tool() -> ToolDefinition {
        ToolDefinition::new(
            "python",
            "Run Python",
            json!({
                "type": "object",
                "properties": { "code": { "type": "string" } },
                "required": ["code"]
            }),
        )
    }

    fn python_tool_available() -> bool {
        tools::execute_tool("python", &json!({ "code": "print(1)" }))
            .trim()
            .eq("1")
    }

    fn shell_tool() -> ToolDefinition {
        ToolDefinition::new(
            "shell",
            "Run shell",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )
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
            &[ToolDefinition::new(
                "shell",
                "Run a shell command",
                json!({}),
            )],
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
            .run_turn(
                &mut engine,
                "remember alpha",
                &[],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("first turn");
        assert_eq!(first.text, "alpha");

        let second = session
            .run_turn(
                &mut engine,
                "what did I say before?",
                &[],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
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
            .run_turn(
                &mut engine,
                "compute 2+2",
                &tools,
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
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
            .run_turn(
                &mut engine,
                "hi",
                &[],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("turn");
        session.reset();

        assert_eq!(session.messages().len(), 1);
        assert_eq!(session.messages()[0].role, chat::ChatRole::System);
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
                &tool_executor(),
                &tool_policy(),
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
            .run_turn(
                &mut engine,
                "compute 2+2",
                &[python_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("tool turn");
        session.save_to_path(&path).expect("save session");

        let loaded = AgentSession::load_from_path(&path).expect("load session");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages(), session.messages());
        assert_eq!(loaded.stats(), session.stats());
    }

    #[test]
    fn maybe_needs_tool_repair_detects_tool_intent() {
        assert!(
            BuiltinToolPolicyHooks
                .should_repair_tool_calls("I should use the python tool to compute this.")
        );
        assert!(!BuiltinToolPolicyHooks.should_repair_tool_calls("RIVER"));
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
                &tool_executor(),
                &tool_policy(),
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
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("draft recovery turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "56");
        assert_eq!(engine.prompts.len(), 1);
    }

    #[test]
    fn extract_arithmetic_expression_finds_inline_math() {
        let parsed = BuiltinToolPolicyHooks
            .recover_tool_calls_from_user_request(
                "Use python to compute 123 * 456 right now.",
                &[python_tool()],
            )
            .expect("recover tool call");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "python");
        assert_eq!(parsed.tool_calls[0].arguments["code"], "print(123 * 456)");
    }

    #[test]
    fn chinese_file_listing_request_is_recovered_with_shell_tool() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "I don't have a tool for listing files.",
            "I still don't have a suitable tool for this.",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "本地有哪些文件",
                &[shell_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("file listing turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "Listed the current directory above.");
        assert_eq!(engine.prompts.len(), 0);
    }
}
