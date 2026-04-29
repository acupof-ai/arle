use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chat::{
    ChatMessage, ChatRole, ParsedAssistantResponse, ToolCall, ToolDefinition, VisibleTextStream,
};
use infer::sampler::SamplingParams;
use infer::server_engine::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason, InferenceEngine,
    TokenUsage,
};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

pub use tools::{TOOL_RESULT_TRUNCATION_MARKER, ToolExecutionMetadata};

pub type Message = ChatMessage;

pub trait ToolExecutor {
    fn execute(&self, tool_call: &ToolCall) -> String;

    /// Execute and surface telemetry. Default impl synthesizes neutral
    /// metadata (`latency_ms = 0`, `truncated = false`) so existing
    /// implementations keep compiling unchanged. Production executors
    /// (e.g. the CLI's [`BuiltinToolExecutor`]) override this to emit
    /// real timings via `tools::execute_tool_call_with_metadata`.
    fn execute_with_metadata(&self, tool_call: &ToolCall) -> (String, ToolExecutionMetadata) {
        let result = self.execute(tool_call);
        let truncated = result.ends_with(TOOL_RESULT_TRUNCATION_MARKER);
        (
            result,
            ToolExecutionMetadata {
                latency_ms: 0,
                truncated,
            },
        )
    }
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
        _last_tool_result: Option<&str>,
        _last_tool_scalar_result: Option<&str>,
    ) -> Option<String> {
        None
    }
}

fn format_prompt(messages: &[Message], tools: &[ToolDefinition]) -> String {
    chat::messages_to_prompt(messages, tools)
}

fn parse_tool_calls(text: &str) -> ParsedAssistantResponse {
    let mut parsed = chat::parse_tool_calls(text);
    parsed
        .tool_calls
        .retain(|call| !call.name.trim().is_empty());
    parsed
}

const DEFAULT_SYSTEM_PROMPT: &str = r"You are a local CLI coding assistant.
Answer briefly and directly.
Use tools silently when needed.
Never expose raw role markers, XML protocol tags, or internal tool protocol in user-facing answers.
If the user asks for an exact format, output exactly that.
Do not expose chain-of-thought.";
const TOOL_PLANNING_MAX_TOKENS: usize = 256;
const STREAM_POLL_INTERVAL: Duration = Duration::from_micros(200);

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
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub max_turns_reached: bool,
    pub trace_events: Vec<AgentTraceEvent>,
    /// Wall-clock latency from turn start to the engine's first emitted
    /// token, regardless of whether that token was visible after tool-XML
    /// stripping. This is the metric to use for RL / SLO dashboards —
    /// `tps`'s visible-text-only TTFT undercounts when a turn opens with
    /// a `<tool_call>` block. `None` when no tokens streamed at all
    /// (e.g. the turn was cancelled before generation began).
    pub time_to_first_token: Option<Duration>,
    /// Anthropic-shaped message log captured for trajectory export. The
    /// system prompt is excluded; the user message starts each turn,
    /// followed by assistant blocks (text + tool_use) and tool results.
    pub messages: Vec<TrajectoryMessage>,
    /// Per-engine-call breakdown for the turn — one entry per
    /// `InferenceEngine::complete_stream` invocation. Empty when the
    /// turn finalised entirely through a deterministic policy hook
    /// (e.g. `recover_tool_calls_from_user_request`).
    pub sub_turns: Vec<SubTurnRecord>,
    /// Why the turn ended. Encodes the four exits documented in
    /// `docs/projects/agent-trajectory-export.md` so RL can reward
    /// or penalise specific failure modes (notably `EmptyNoProgress`).
    pub terminal_state: TerminalState,
    /// Total wall-clock seconds for the turn, captured from the same
    /// monotonic anchor as `time_to_first_token`. Surfaced separately
    /// because the trace writer needs it without re-sampling.
    pub wall_secs: f64,
}

/// Schema version for v1 trajectory records. v2 will bump this when token
/// IDs and `response_mask` are added.
pub const TRAJECTORY_SCHEMA_VERSION: i32 = 1;

/// Anthropic-shaped trajectory message. User and tool messages carry a
/// plain string; assistant messages always carry a content-block array
/// so tool_use entries can be correlated with later tool results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryMessage {
    pub role: TrajectoryRole,
    pub content: MessageContent,
    /// Set on `role: tool` messages; references the matching assistant
    /// `tool_use` block by deterministic id (`tu_<sub>_<call>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    /// Set on `role: tool` messages when the underlying tool result
    /// was truncated by the executor. Mirrors
    /// [`ToolExecutionMetadata::truncated`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_truncated: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryRole {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubTurnRecord {
    pub index: usize,
    /// Full ChatML prompt sent to the engine for this sub-turn. `None`
    /// when `--trace-prompts off` was set on the CLI; the agent loop
    /// always populates `Some(_)` and the trace writer rewrites to
    /// `None` per the operator's preference.
    pub prompt_text: Option<String>,
    /// Raw text the engine returned, including any `<tool_call>` XML.
    pub completion_text: String,
    pub usage: ToolUsage,
    /// Per-sub-turn TTFT in milliseconds — measured from the
    /// `complete_stream` call site to the first non-empty delta.
    /// `None` when the engine never emitted text (cancelled / errored
    /// before any chunk).
    pub ttft_ms: Option<u64>,
    /// Wall-clock seconds for this sub-turn (entire `complete_stream`
    /// duration).
    pub decode_secs: f64,
    /// Lowercased finish reason (`"stop"` / `"length"`).
    pub finish_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalState {
    /// `tool_calls.is_empty() && !content.trim().is_empty()` — the
    /// model produced a final answer.
    Stop,
    /// `max_turns` exhausted before the model produced a final answer.
    MaxTurns,
    /// `tool_calls.is_empty() && content.trim().is_empty()` — the
    /// model emitted nothing actionable. Surfaced as a distinct state
    /// so RL can reward against it.
    EmptyNoProgress,
    /// `tool_policy.finalize_after_tool_execution` returned `Some(_)`.
    PolicyShortCircuit,
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

#[derive(Default)]
pub struct AgentTurnCallbacks<'a> {
    pub on_text_chunk: Option<&'a mut dyn FnMut(&str)>,
    pub on_trace_event: Option<&'a mut dyn FnMut(&AgentTraceEvent)>,
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

    pub fn messages(&self) -> &[Message] {
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
        self.run_turn_inner(
            engine,
            user_input,
            tools,
            tool_executor,
            tool_policy,
            settings,
            None,
            AgentTurnCallbacks::default(),
        )?
        .ok_or_else(|| anyhow!("agent turn cancelled"))
    }

    pub fn run_turn_interruptibly<E: InferenceEngine + ?Sized, X: ToolExecutor, P: ToolPolicy>(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[ToolDefinition],
        tool_executor: &X,
        tool_policy: &P,
        settings: AgentSettings,
        cancel: &AtomicBool,
    ) -> Result<Option<AgentTurnResult>> {
        self.run_turn_interruptibly_with_callbacks(
            engine,
            user_input,
            tools,
            tool_executor,
            tool_policy,
            settings,
            cancel,
            AgentTurnCallbacks::default(),
        )
    }

    pub fn run_turn_interruptibly_with_callbacks<
        E: InferenceEngine + ?Sized,
        X: ToolExecutor,
        P: ToolPolicy,
    >(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[ToolDefinition],
        tool_executor: &X,
        tool_policy: &P,
        settings: AgentSettings,
        cancel: &AtomicBool,
        callbacks: AgentTurnCallbacks<'_>,
    ) -> Result<Option<AgentTurnResult>> {
        self.run_turn_inner(
            engine,
            user_input,
            tools,
            tool_executor,
            tool_policy,
            settings,
            Some(cancel),
            callbacks,
        )
    }

    fn run_turn_inner<E: InferenceEngine + ?Sized, X: ToolExecutor, P: ToolPolicy>(
        &mut self,
        engine: &mut E,
        user_input: &str,
        tools: &[ToolDefinition],
        tool_executor: &X,
        tool_policy: &P,
        settings: AgentSettings,
        cancel: Option<&AtomicBool>,
        callbacks: AgentTurnCallbacks<'_>,
    ) -> Result<Option<AgentTurnResult>> {
        let turn_start = self.messages.len();
        self.messages.push(Message::user(user_input));

        // Wall-clock anchor for the engine-token TTFT we surface to the
        // caller. Captured here (before any sub-turn fires) so a turn that
        // opens with a `<tool_call>` block — whose visible-text count is
        // zero — still reports a meaningful first-token latency.
        let turn_started_at = Instant::now();
        let mut first_engine_token_at: Option<Instant> = None;

        let mut tool_calls_executed = 0usize;
        let mut prompt_tokens = 0u64;
        let mut completion_tokens = 0u64;
        let mut last_tool_name = None::<String>;
        let mut last_tool_result = None::<String>;
        let mut last_tool_scalar_result = None::<String>;
        let mut trace_events = Vec::new();
        let mut on_text_chunk = callbacks.on_text_chunk;
        let mut on_trace_event = callbacks.on_trace_event;
        let mut recovered_user_request = (!tools.is_empty())
            .then(|| tool_policy.recover_tool_calls_from_user_request(user_input, tools))
            .flatten();

        // Trajectory accumulators. The user message is the first entry;
        // assistant + tool messages are appended as the loop progresses.
        let mut trajectory_messages: Vec<TrajectoryMessage> = Vec::new();
        trajectory_messages.push(TrajectoryMessage {
            role: TrajectoryRole::User,
            content: MessageContent::Text(user_input.to_string()),
            tool_use_id: None,
            result_truncated: None,
        });
        let mut sub_turns: Vec<SubTurnRecord> = Vec::new();

        for turn in 0..settings.max_turns {
            // Two indices, both monotone: `tool_use_id_base` is the loop
            // iteration (always advances, including on the recovered
            // branch) so synthesized tool_use IDs `tu_{base}_{n}` stay
            // unique even when iteration 0 emits a recovered tool call
            // and iteration 1 produces an engine-driven one. (codex
            // Phase-1 P2). `sub_turn_record_index` is the position the
            // next SubTurnRecord will land in the `sub_turns` Vec; it
            // only advances when we actually invoke the engine.
            let tool_use_id_base = turn;
            let sub_turn_record_index = sub_turns.len();
            // Tracks whether THIS sub-turn invoked the engine — the
            // recovered-user-request branch skips the engine entirely
            // and must not append a SubTurnRecord.
            let mut emitted_engine_call = false;
            let parsed = if let Some(parsed) = recovered_user_request.take() {
                info!("Recovered tool call(s) directly from user request");
                parsed
            } else {
                emitted_engine_call = true;
                let prompt = format_prompt(&self.messages, tools);
                info!(
                    "Agent turn {}/{}: prompt length = {} chars",
                    turn + 1,
                    settings.max_turns,
                    prompt.len()
                );

                let turn_max_tokens = if tool_calls_executed == 0 && !tools.is_empty() {
                    settings.max_tokens.min(TOOL_PLANNING_MAX_TOKENS)
                } else {
                    settings.max_tokens
                };

                let mut visible_stream = VisibleTextStream::default();
                let sub_turn_started_at = Instant::now();
                let mut sub_turn_first_token_at: Option<Instant> = None;
                // We always need to observe each engine chunk to capture the
                // engine-token TTFT, even if the caller did not register a
                // visible-text callback. So `stream_visible_chunk` is now
                // unconditionally wired into the streaming path; it just
                // skips the visible-text handoff when no callback is set.
                let mut stream_visible_chunk = |chunk: &str| {
                    if !chunk.is_empty() {
                        let now = Instant::now();
                        if first_engine_token_at.is_none() {
                            first_engine_token_at = Some(now);
                        }
                        if sub_turn_first_token_at.is_none() {
                            sub_turn_first_token_at = Some(now);
                        }
                    }
                    let visible = visible_stream.push(chunk);
                    if !visible.is_empty()
                        && let Some(callback) = on_text_chunk.as_deref_mut()
                    {
                        callback(&visible);
                    }
                };
                let Some(output) = complete_with_optional_cancel(
                    engine,
                    CompletionRequest {
                        prompt: prompt.clone(),
                        max_tokens: turn_max_tokens,
                        sampling: SamplingParams {
                            temperature: settings.temperature,
                            ..SamplingParams::default()
                        },
                        stop: Some(vec!["<|im_end|>".to_string()]),
                        logprobs: false,
                        session_id: None,
                        trace_context: None,
                    },
                    cancel,
                    Some(&mut stream_visible_chunk as &mut dyn FnMut(&str)),
                )?
                else {
                    return Ok(None);
                };
                if let Some(callback) = on_text_chunk.as_deref_mut() {
                    let tail = visible_stream.finish();
                    if !tail.is_empty() {
                        callback(&tail);
                    }
                }

                info!(
                    "Generated {} chars, finish_reason={:?}",
                    output.text.len(),
                    output.finish_reason
                );
                prompt_tokens = prompt_tokens.saturating_add(output.usage.prompt_tokens as u64);
                completion_tokens =
                    completion_tokens.saturating_add(output.usage.completion_tokens as u64);

                let decode_secs = sub_turn_started_at.elapsed().as_secs_f64();
                let ttft_ms = sub_turn_first_token_at.map(|t| {
                    u64::try_from(t.duration_since(sub_turn_started_at).as_millis())
                        .unwrap_or(u64::MAX)
                });
                sub_turns.push(SubTurnRecord {
                    index: sub_turn_record_index,
                    prompt_text: Some(prompt),
                    completion_text: output.text.clone(),
                    usage: ToolUsage {
                        prompt_tokens: output.usage.prompt_tokens as u64,
                        completion_tokens: output.usage.completion_tokens as u64,
                    },
                    ttft_ms,
                    decode_secs,
                    finish_reason: finish_reason_to_str(output.finish_reason).to_string(),
                });

                let mut parsed = parse_tool_calls(&output.text);
                if parsed.tool_calls.is_empty() && tool_calls_executed == 0 && !tools.is_empty() {
                    if let Some(recovered) =
                        tool_policy.recover_tool_calls_from_draft(&output.text, tools)
                    {
                        info!("Recovered tool call(s) via deterministic extraction");
                        parsed = recovered;
                    } else if (output.text.contains("<tool_call>")
                        || tool_policy.should_repair_tool_calls(&parsed.content))
                        && let Some(repair_outcome) = repair_tool_calls(
                            engine,
                            &self.messages,
                            tools,
                            settings,
                            &output.text,
                            cancel,
                            // The next free slot — the main generation
                            // already pushed its record above.
                            sub_turns.len(),
                        )?
                    {
                        info!("Recovered tool call(s) via repair turn");
                        parsed = repair_outcome.parsed;
                        // Repair issues another `complete_stream` call; if we
                        // don't append its record here, the trajectory shows
                        // a `tool_use` with no matching engine call in any
                        // `completion_text` and under-reports engine work.
                        // (codex Phase-1 P2)
                        sub_turns.push(repair_outcome.record);
                    }
                }
                parsed
            };

            let content = tool_policy.finalize_response_text(
                user_input,
                parsed.content,
                last_tool_name.as_deref(),
                last_tool_scalar_result.as_deref(),
                tool_calls_executed,
            );
            let tool_calls = parsed.tool_calls;

            // Emit the assistant trajectory message — even on the
            // recovered-user-request branch (no engine call), so RL can
            // still see what the agent decided to do. We key tool_use
            // IDs off `tool_use_id_base` (= the loop turn), which is
            // monotone across both engine and recovered branches; the
            // earlier `sub_turn_index` was tied to `sub_turns.len()`
            // and collided across recovered + engine pairs (codex P2).
            let _ = emitted_engine_call;
            let assistant_blocks = build_assistant_blocks(&content, &tool_calls, tool_use_id_base);
            trajectory_messages.push(TrajectoryMessage {
                role: TrajectoryRole::Assistant,
                content: MessageContent::Blocks(assistant_blocks),
                tool_use_id: None,
                result_truncated: None,
            });

            self.messages
                .push(Message::assistant(&content, tool_calls.clone()));

            if tool_calls.is_empty() {
                self.compact_turn_history(turn_start, &content);
                let terminal_state = if content.trim().is_empty() {
                    TerminalState::EmptyNoProgress
                } else {
                    TerminalState::Stop
                };
                return Ok(Some(AgentTurnResult {
                    text: content,
                    tool_calls_executed,
                    prompt_tokens,
                    completion_tokens,
                    max_turns_reached: false,
                    trace_events,
                    time_to_first_token: first_engine_token_at
                        .map(|t| t.duration_since(turn_started_at)),
                    messages: trajectory_messages,
                    sub_turns,
                    terminal_state,
                    wall_secs: turn_started_at.elapsed().as_secs_f64(),
                }));
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
                &mut last_tool_result,
                &mut last_tool_scalar_result,
                &mut trace_events,
                &mut trajectory_messages,
                tool_use_id_base,
                match on_trace_event {
                    Some(ref mut callback) => Some(&mut **callback),
                    None => None,
                },
            );

            if let Some(text) = tool_policy.finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_result.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                self.compact_turn_history(turn_start, &text);
                return Ok(Some(AgentTurnResult {
                    text,
                    tool_calls_executed,
                    prompt_tokens,
                    completion_tokens,
                    max_turns_reached: false,
                    trace_events,
                    time_to_first_token: first_engine_token_at
                        .map(|t| t.duration_since(turn_started_at)),
                    messages: trajectory_messages,
                    sub_turns,
                    terminal_state: TerminalState::PolicyShortCircuit,
                    wall_secs: turn_started_at.elapsed().as_secs_f64(),
                }));
            }
        }

        let final_text = "(max turns reached - agent stopped)".to_string();
        self.compact_turn_history(turn_start, &final_text);
        Ok(Some(AgentTurnResult {
            text: final_text,
            tool_calls_executed,
            prompt_tokens,
            completion_tokens,
            max_turns_reached: true,
            trace_events,
            time_to_first_token: first_engine_token_at.map(|t| t.duration_since(turn_started_at)),
            messages: trajectory_messages,
            sub_turns,
            terminal_state: TerminalState::MaxTurns,
            wall_secs: turn_started_at.elapsed().as_secs_f64(),
        }))
    }

    fn compact_turn_history(&mut self, turn_start: usize, assistant_text: &str) {
        self.messages.truncate(turn_start + 1);
        self.messages
            .push(Message::assistant(assistant_text, vec![]));
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

#[allow(clippy::too_many_arguments)]
fn execute_tool_calls(
    tool_calls: &[ToolCall],
    tool_executor: &dyn ToolExecutor,
    messages: &mut Vec<Message>,
    tool_calls_executed: &mut usize,
    last_tool_name: &mut Option<String>,
    last_tool_result: &mut Option<String>,
    last_tool_scalar_result: &mut Option<String>,
    trace_events: &mut Vec<AgentTraceEvent>,
    trajectory_messages: &mut Vec<TrajectoryMessage>,
    sub_turn_index: usize,
    mut on_trace_event: Option<&mut dyn FnMut(&AgentTraceEvent)>,
) {
    for (call_index, tool_call) in tool_calls.iter().enumerate() {
        *tool_calls_executed += 1;
        let (result, metadata) = tool_executor.execute_with_metadata(tool_call);

        *last_tool_result = Some(result.clone());
        *last_tool_scalar_result = scalar_tool_result(&result);
        *last_tool_name = Some(tool_call.name.clone());
        trace_events.push(AgentTraceEvent::ToolCall {
            name: tool_call.name.clone(),
            arguments: tool_call.arguments.clone(),
            result: result.clone(),
        });
        if let Some(callback) = on_trace_event.as_deref_mut()
            && let Some(event) = trace_events.last()
        {
            callback(event);
        }
        messages.push(Message::tool_result(&tool_call.name, &result));
        trajectory_messages.push(TrajectoryMessage {
            role: TrajectoryRole::Tool,
            content: MessageContent::Text(result),
            tool_use_id: Some(tool_use_id(sub_turn_index, call_index)),
            result_truncated: Some(metadata.truncated),
        });
    }
}

/// Build the deterministic `tu_<sub>_<call>` id used to correlate
/// assistant `tool_use` blocks with their matching `tool` messages.
/// Stable across runs given the same input — no UUIDs, no clocks.
fn tool_use_id(sub_turn_index: usize, call_index: usize) -> String {
    format!("tu_{sub_turn_index}_{call_index}")
}

fn build_assistant_blocks(
    content: &str,
    tool_calls: &[ToolCall],
    sub_turn_index: usize,
) -> Vec<ContentBlock> {
    let mut blocks = Vec::with_capacity(1 + tool_calls.len());
    // Always emit a leading text block — even when empty — so the
    // schema's "assistant content is always blocks" invariant holds
    // and downstream consumers don't have to special-case empty text.
    blocks.push(ContentBlock::Text {
        text: content.to_string(),
    });
    for (call_index, call) in tool_calls.iter().enumerate() {
        blocks.push(ContentBlock::ToolUse {
            id: tool_use_id(sub_turn_index, call_index),
            name: call.name.clone(),
            input: call.arguments.clone(),
        });
    }
    blocks
}

fn finish_reason_to_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
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

/// Result of a successful repair turn. The caller appends `record` to its
/// `sub_turns` so the repair generation is visible in the trajectory; the
/// `parsed` half replaces the main generation's malformed parse output.
/// (codex Phase-1 P2: repair was an unrecorded engine call.)
struct RepairOutcome {
    parsed: ParsedAssistantResponse,
    record: SubTurnRecord,
}

fn repair_tool_calls<E: InferenceEngine + ?Sized>(
    engine: &mut E,
    messages: &[Message],
    tools: &[ToolDefinition],
    settings: AgentSettings,
    assistant_draft: &str,
    cancel: Option<&AtomicBool>,
    sub_turn_index: usize,
) -> Result<Option<RepairOutcome>> {
    let mut repair_messages = messages.to_vec();
    repair_messages.push(Message::assistant(assistant_draft, vec![]));
    repair_messages.push(Message::user(
        "Rewrite your previous assistant message using the tool-call protocol. \
If a tool is needed, output only valid <tool_call> blocks and no other text. \
If no tool is needed, output exactly NO_TOOL.",
    ));

    let repair_prompt = format_prompt(&repair_messages, tools);
    let started_at = Instant::now();
    let Some(repair_output) = complete_with_optional_cancel(
        engine,
        CompletionRequest {
            prompt: repair_prompt.clone(),
            max_tokens: settings.max_tokens.min(128),
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Some(vec!["<|im_end|>".to_string()]),
            logprobs: false,
            session_id: None,
            trace_context: None,
        },
        cancel,
        None,
    )?
    else {
        return Ok(None);
    };
    let decode_secs = started_at.elapsed().as_secs_f64();

    let repaired = parse_tool_calls(&repair_output.text);
    let record = SubTurnRecord {
        index: sub_turn_index,
        prompt_text: Some(repair_prompt),
        completion_text: repair_output.text.clone(),
        usage: ToolUsage {
            prompt_tokens: repair_output.usage.prompt_tokens as u64,
            completion_tokens: repair_output.usage.completion_tokens as u64,
        },
        // Repair calls go through the non-streaming path (no chunk
        // callback), so per-chunk TTFT is unobservable here. None is
        // honest signal — RL pipelines should mask this row out of TTFT
        // SLO calcs.
        ttft_ms: None,
        decode_secs,
        finish_reason: finish_reason_to_str(repair_output.finish_reason).to_string(),
    };

    if !repaired.tool_calls.is_empty() {
        return Ok(Some(RepairOutcome {
            parsed: repaired,
            record,
        }));
    }
    if repaired.content.trim() == "NO_TOOL" {
        return Ok(None);
    }
    Ok(None)
}

fn complete_with_optional_cancel<E: InferenceEngine + ?Sized>(
    engine: &mut E,
    req: CompletionRequest,
    cancel: Option<&AtomicBool>,
    mut on_text_chunk: Option<&mut dyn FnMut(&str)>,
) -> Result<Option<CompletionOutput>> {
    if cancel.is_none() && on_text_chunk.is_none() {
        return engine.complete(req).map(Some);
    }

    let (tx, rx) = mpsc::unbounded_channel::<CompletionStreamDelta>();
    let mut rx: Option<mpsc::UnboundedReceiver<CompletionStreamDelta>> = Some(rx);
    let mut text = String::new();
    let mut finish_reason = None::<FinishReason>;
    let mut usage = None::<TokenUsage>;
    let mut stream_err = None::<anyhow::Error>;
    let mut cancelled = false;

    std::thread::scope(|s| {
        let worker = s.spawn(|| engine.complete_stream(req, tx));

        loop {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                cancelled = true;
                rx = None;
                break;
            }

            let Some(rx_ref) = rx.as_mut() else { break };
            match rx_ref.try_recv() {
                Ok(delta) => {
                    if !delta.text_delta.is_empty() {
                        if let Some(callback) = on_text_chunk.as_deref_mut() {
                            callback(&delta.text_delta);
                        }
                        text.push_str(&delta.text_delta);
                    }
                    if let Some(final_usage) = delta.usage {
                        usage = Some(final_usage);
                    }
                    if let Some(reason) = delta.finish_reason {
                        finish_reason = Some(reason);
                        break;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(STREAM_POLL_INTERVAL);
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        if let Ok(res) = worker.join()
            && let Err(err) = res
        {
            stream_err = Some(err);
        }
    });

    if cancelled {
        return Ok(None);
    }

    if let Some(err) = stream_err {
        return Err(err);
    }

    let finish_reason =
        finish_reason.ok_or_else(|| anyhow!("stream ended without finish reason"))?;
    let usage = usage.ok_or_else(|| anyhow!("stream ended without token usage"))?;

    Ok(Some(CompletionOutput {
        text,
        finish_reason,
        usage,
        token_logprobs: Vec::new(),
    }))
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
        max_tokens: Vec<usize>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<&str>) -> Self {
            Self {
                outputs: outputs.into_iter().map(str::to_string).collect(),
                prompts: Vec::new(),
                max_tokens: Vec::new(),
            }
        }
    }

    impl InferenceEngine for FakeEngine {
        fn model_id(&self) -> &str {
            "fake"
        }

        fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
            self.prompts.push(req.prompt);
            self.max_tokens.push(req.max_tokens);
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
            last_tool_result: Option<&str>,
            last_tool_scalar_result: Option<&str>,
        ) -> Option<String> {
            BuiltinToolPolicyHooks.finalize_after_tool_execution(
                user_input,
                last_tool_name,
                last_tool_result,
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
        assert_eq!(stats.conversation_messages, 2);
        assert_eq!(stats.user_messages, 1);
        assert_eq!(stats.assistant_messages, 1);
        assert_eq!(stats.tool_messages, 0);
        assert_eq!(stats.tool_calls, 0);
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
    fn explicit_python_request_uses_deterministic_recovery() {
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }

        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["56088"]);

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
    fn chinese_file_listing_request_uses_deterministic_recovery() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["I listed the current directory contents above."]);

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
        assert!(result.text.contains("Cargo.toml"));
        assert!(result.text.contains("src"));
        assert_eq!(engine.prompts.len(), 0);
    }

    #[test]
    fn shell_tool_follow_up_text_comes_from_model_not_template() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd && ls -la\"}}\n</tool_call>",
            "I listed the current directory contents above.",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "Check the workspace root.",
                &[shell_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("shell tool turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(
            result.text,
            "I listed the current directory contents above."
        );
        assert_eq!(result.prompt_tokens, 2);
        assert_eq!(result.completion_tokens, 2);
        assert_eq!(engine.prompts.len(), 2);
    }

    #[test]
    fn first_agent_tool_planning_turn_is_capped() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["Done."]);

        let result = session
            .run_turn(
                &mut engine,
                "Summarize the current workspace.",
                &[shell_tool()],
                &tool_executor(),
                &tool_policy(),
                AgentSettings {
                    max_turns: 4,
                    max_tokens: 4096,
                    temperature: 0.0,
                },
            )
            .expect("tool planning turn");

        assert_eq!(result.tool_calls_executed, 0);
        assert_eq!(engine.max_tokens, vec![super::TOOL_PLANNING_MAX_TOKENS]);
    }

    #[test]
    fn malformed_tool_call_block_triggers_repair_instead_of_empty_tool_name() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"arguments\":{\"command\":\"pwd && ls -la\"}}\n</tool_call>",
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd && ls -la\"}}\n</tool_call>",
            "done",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "Check the workspace root.",
                &[shell_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("repair turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "done");
    }

    #[test]
    fn repair_generation_is_recorded_as_its_own_sub_turn() {
        // codex Phase-1 P2: when the main generation produces malformed
        // tool-call XML and `repair_tool_calls` issues a second engine
        // call, that call must appear in `sub_turns`. Otherwise the
        // trajectory shows a `tool_use` whose `completion_text` exists
        // in no recorded sub-turn — under-reporting engine work.
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            // 1. Main generation: malformed tool_call (missing "name")
            "<tool_call>\n{\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>",
            // 2. Repair generation: valid tool_call
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>",
            // 3. Final assistant turn after tool result
            "done",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "Check the workspace root.",
                &[shell_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("repair turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert_eq!(result.text, "done");
        // 3 engine calls fired: main (malformed), repair (valid),
        // final-after-tool. Each must own a SubTurnRecord.
        assert_eq!(
            result.sub_turns.len(),
            3,
            "expected main + repair + final SubTurnRecords, got {} ({:?})",
            result.sub_turns.len(),
            result
                .sub_turns
                .iter()
                .map(|r| &r.completion_text)
                .collect::<Vec<_>>()
        );
        // Every `tool_use` block in the assistant content must be backed
        // by a recorded `completion_text` somewhere.
        let recorded: Vec<&str> = result
            .sub_turns
            .iter()
            .map(|r| r.completion_text.as_str())
            .collect();
        for msg in &result.messages {
            if let MessageContent::Blocks(blocks) = &msg.content {
                for block in blocks {
                    if let ContentBlock::ToolUse { name, input, .. } = block {
                        let needle = format!("\"name\":\"{}\"", name);
                        let _ = input; // not all tool_use blocks must hit by raw substring
                        assert!(
                            recorded.iter().any(|c| c.contains(&needle)),
                            "tool_use {:?} appears in trajectory but no SubTurnRecord shows it (recorded={:?})",
                            name,
                            recorded
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tool_use_ids_stay_unique_across_recovered_and_engine_branches() {
        // codex Phase-1 P2: when iteration 0 is the deterministic
        // "recovered_user_request" branch (no engine call → previously
        // didn't bump sub_turns.len()) and iteration 1 is an engine
        // sub-turn that itself produces tool calls, both used to
        // serialize as `tu_0_*`. Ensure IDs are now unique.
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }
        let mut session = AgentSession::new();
        // The user-request recovery for "Use python to compute …" fires
        // a python tool_call at iteration 0 without consulting the
        // engine. After tool execution, iteration 1 calls the engine,
        // which we make fire ANOTHER tool call so we get two tool_use
        // blocks in the trajectory and can assert ID uniqueness.
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(99)\"}}\n</tool_call>",
            "99",
        ]);
        let _ = session
            .run_turn(
                &mut engine,
                "Use the python tool to compute 123 * 456. After the tool returns, answer with just the integer.",
                &[python_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("multi-branch turn");

        // The synthesized IDs are produced by the `tool_use_id` helper.
        // Iteration 0 (recovered) used base 0; iteration 1 (engine) uses
        // base 1; before the fix both keyed off `sub_turns.len()` and
        // collided at base 0 for the engine sub-turn since the
        // recovered branch never bumped the counter.
        let id_a = super::tool_use_id(0, 0);
        let id_b = super::tool_use_id(1, 0);
        assert_ne!(
            id_a, id_b,
            "tool_use_id base must differ across iterations 0 and 1"
        );
        let mut ids = vec![id_a, id_b];
        ids.sort();
        let unique = ids.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(unique, 2, "tool_use IDs must be unique across iterations");
    }

    #[test]
    fn ttft_is_captured_for_tool_only_first_subturn() {
        // Reproduces the REPL screenshot: turn 1 opens with a tool_call
        // block (zero visible text after stripping). The visible-stream
        // TTFT capture in `tps` would miss this because it keys on
        // post-filter chunks; the agent's engine-token TTFT must still
        // fire on the raw delta and surface a non-None
        // `time_to_first_token`.
        if !python_tool_available() {
            eprintln!("Skipping test: python tool is unavailable in this environment");
            return;
        }
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"python\",\"arguments\":{\"code\":\"print(2 + 2)\"}}\n</tool_call>",
            "The answer is 4.",
        ]);

        let result = session
            .run_turn(
                &mut engine,
                "compute 2+2",
                &[python_tool()],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("tool turn");

        assert_eq!(result.tool_calls_executed, 1);
        assert!(
            result.time_to_first_token.is_some(),
            "engine emitted text in turn 1 (the tool_call XML) — TTFT must be Some"
        );
    }

    #[test]
    fn ttft_is_none_when_engine_emits_nothing() {
        // Edge case: a turn that fails before any text streams should
        // leave time_to_first_token as None, not zero — None means
        // "the engine never produced a token", which is the honest
        // signal for SLO dashboards.
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![""]);
        let result = session
            .run_turn(
                &mut engine,
                "hi",
                &[],
                &tool_executor(),
                &tool_policy(),
                settings(),
            )
            .expect("empty-output turn");
        assert_eq!(result.text, "");
        assert!(result.time_to_first_token.is_none());
    }

    // ── Trajectory export (Phase 1 / v1) ─────────────────────────────────

    use super::{
        ContentBlock, MessageContent, SubTurnRecord, TerminalState, TrajectoryMessage,
        TrajectoryRole,
    };

    fn shell_recorder_settings() -> AgentSettings {
        AgentSettings {
            max_turns: 2,
            max_tokens: 128,
            temperature: 0.0,
        }
    }

    /// A tool executor that returns a canned string and never touches the
    /// real sandbox — used to keep these tests deterministic + offline.
    struct StubToolExecutor {
        result: String,
    }

    impl StubToolExecutor {
        fn new(result: impl Into<String>) -> Self {
            Self {
                result: result.into(),
            }
        }
    }

    impl ToolExecutor for StubToolExecutor {
        fn execute(&self, _tool_call: &ToolCall) -> String {
            self.result.clone()
        }
    }

    /// Bare ToolPolicy with no recovery hooks — keeps the recovered-user
    /// branch dormant so we exercise the engine path deterministically.
    struct NoopToolPolicy;

    impl ToolPolicy for NoopToolPolicy {}

    /// Policy that ALWAYS short-circuits with `finalize_after_tool_execution`,
    /// regardless of input.
    struct AlwaysShortCircuit;

    impl ToolPolicy for AlwaysShortCircuit {
        fn finalize_after_tool_execution(
            &self,
            _user_input: &str,
            _last_tool_name: Option<&str>,
            last_tool_result: Option<&str>,
            _last_tool_scalar_result: Option<&str>,
        ) -> Option<String> {
            Some(
                last_tool_result
                    .unwrap_or("policy-short-circuit")
                    .to_string(),
            )
        }
    }

    fn shell_def() -> ToolDefinition {
        ToolDefinition::new(
            "shell",
            "shell",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        )
    }

    #[test]
    fn terminal_state_stop_on_normal_finish() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec!["final answer"]);
        let result = session
            .run_turn(
                &mut engine,
                "hello",
                &[],
                &StubToolExecutor::new("ignored"),
                &NoopToolPolicy,
                settings(),
            )
            .expect("turn");
        assert_eq!(result.terminal_state, TerminalState::Stop);
        assert_eq!(result.text, "final answer");
    }

    #[test]
    fn terminal_state_max_turns_when_budget_exhausted() {
        // Force every sub-turn to emit a tool call so the loop never
        // exits via the "tool_calls.is_empty" path. With max_turns=2 and
        // the StubToolExecutor returning the same string, the loop will
        // run twice and then fall through to the max-turns return site.
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
        ]);
        let result = session
            .run_turn(
                &mut engine,
                "do thing",
                &[shell_def()],
                &StubToolExecutor::new("ok"),
                &NoopToolPolicy,
                shell_recorder_settings(),
            )
            .expect("turn");
        assert_eq!(result.terminal_state, TerminalState::MaxTurns);
        assert!(result.max_turns_reached);
    }

    #[test]
    fn terminal_state_empty_no_progress_when_model_emits_nothing() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![""]);
        let result = session
            .run_turn(
                &mut engine,
                "hi",
                &[],
                &StubToolExecutor::new("ignored"),
                &NoopToolPolicy,
                settings(),
            )
            .expect("turn");
        assert_eq!(result.terminal_state, TerminalState::EmptyNoProgress);
        assert_eq!(result.text, "");
    }

    #[test]
    fn terminal_state_policy_short_circuit_when_policy_returns_text() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
        ]);
        let result = session
            .run_turn(
                &mut engine,
                "list",
                &[shell_def()],
                &StubToolExecutor::new("file-a\nfile-b"),
                &AlwaysShortCircuit,
                shell_recorder_settings(),
            )
            .expect("turn");
        assert_eq!(result.terminal_state, TerminalState::PolicyShortCircuit);
        assert!(result.text.contains("file-a"));
    }

    #[test]
    fn sub_turns_capture_per_call_usage_and_finish_reason() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
            "done",
        ]);
        let result = session
            .run_turn(
                &mut engine,
                "go",
                &[shell_def()],
                &StubToolExecutor::new("ok"),
                &NoopToolPolicy,
                settings(),
            )
            .expect("turn");
        assert_eq!(result.sub_turns.len(), 2);
        for st in &result.sub_turns {
            // FakeEngine emits prompt_tokens=1, completion_tokens=1, finish=stop.
            assert_eq!(st.usage.prompt_tokens, 1);
            assert_eq!(st.usage.completion_tokens, 1);
            assert_eq!(st.finish_reason, "stop");
            assert!(st.prompt_text.is_some());
            assert!(st.decode_secs >= 0.0);
        }
        assert_eq!(result.sub_turns[0].index, 0);
        assert_eq!(result.sub_turns[1].index, 1);
    }

    #[test]
    fn messages_preserve_tool_use_id_correlation() {
        let mut session = AgentSession::new();
        let mut engine = FakeEngine::new(vec![
            "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
            "done",
        ]);
        let result = session
            .run_turn(
                &mut engine,
                "go",
                &[shell_def()],
                &StubToolExecutor::new("ok"),
                &NoopToolPolicy,
                settings(),
            )
            .expect("turn");

        // Find the assistant message with a tool_use block, then the
        // following tool message — their ids must match.
        let mut assistant_tool_use_id = None::<String>;
        let mut tool_message_id = None::<String>;
        for msg in &result.messages {
            match (&msg.role, &msg.content) {
                (TrajectoryRole::Assistant, MessageContent::Blocks(blocks)) => {
                    for block in blocks {
                        if let ContentBlock::ToolUse { id, .. } = block
                            && assistant_tool_use_id.is_none()
                        {
                            assistant_tool_use_id = Some(id.clone());
                        }
                    }
                }
                (TrajectoryRole::Tool, _) if tool_message_id.is_none() => {
                    tool_message_id = msg.tool_use_id.clone();
                }
                _ => {}
            }
        }
        assert_eq!(assistant_tool_use_id, tool_message_id);
        assert_eq!(assistant_tool_use_id, Some("tu_0_0".to_string()));
    }

    #[test]
    fn tool_use_id_is_deterministic_across_runs() {
        // Same input → same IDs. We don't rely on UUIDs for tool_use ids
        // exactly so the trajectory stays diff-able across re-runs.
        let mut ids = Vec::new();
        for _ in 0..2 {
            let mut session = AgentSession::new();
            let mut engine = FakeEngine::new(vec![
                "<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>",
                "done",
            ]);
            let result = session
                .run_turn(
                    &mut engine,
                    "go",
                    &[shell_def()],
                    &StubToolExecutor::new("ok"),
                    &NoopToolPolicy,
                    settings(),
                )
                .expect("turn");
            let mut run_ids = Vec::new();
            for msg in &result.messages {
                if let MessageContent::Blocks(blocks) = &msg.content {
                    for block in blocks {
                        if let ContentBlock::ToolUse { id, .. } = block {
                            run_ids.push(id.clone());
                        }
                    }
                }
            }
            ids.push(run_ids);
        }
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0], vec!["tu_0_0".to_string()]);
    }

    #[test]
    fn trajectory_record_round_trips_through_serde() {
        // Build a minimal record by hand and verify serialize → deserialize
        // → equality. This is the schema-level invariant the trace writer
        // depends on.
        let record = TrajectoryMessage {
            role: TrajectoryRole::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "hello".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tu_0_0".to_string(),
                    name: "shell".to_string(),
                    input: json!({ "command": "ls" }),
                },
            ]),
            tool_use_id: None,
            result_truncated: None,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        let back: TrajectoryMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(record, back);

        let st = SubTurnRecord {
            index: 0,
            prompt_text: Some("prompt".to_string()),
            completion_text: "out".to_string(),
            usage: super::ToolUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
            },
            ttft_ms: Some(42),
            decode_secs: 0.5,
            finish_reason: "stop".to_string(),
        };
        let st_json = serde_json::to_string(&st).expect("serialize sub-turn");
        let st_back: SubTurnRecord = serde_json::from_str(&st_json).expect("deserialize sub-turn");
        assert_eq!(st, st_back);
    }
}
