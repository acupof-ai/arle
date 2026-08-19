use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chat::{
    ChatMessage, ChatRole, ParsedAssistantResponse, ToolCall, ToolDefinition, VisibleTextStream,
};
use infer_api::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason, InferenceEngine,
    SamplingParams, TokenUsage,
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
        let truncated = result.contains(TOOL_RESULT_TRUNCATION_MARKER);
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
const TOOL_PLANNING_MAX_TOKENS: usize = 49_984; // ~50k, 64-aligned (781 * 64)
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
    /// Why the turn ended. Encodes the four exits so RL can reward
    /// or penalise specific failure modes (notably `EmptyNoProgress`).
    pub terminal_state: TerminalState,
    /// Total wall-clock seconds for the turn, captured from the same
    /// monotonic anchor as `time_to_first_token`. Surfaced separately
    /// because the trace writer needs it without re-sampling.
    pub wall_secs: f64,
    /// Phase 2 trajectory token layer. `Some(TokensRecord)` only when
    /// every component of the turn's token IDs was available — empty
    /// `prompt_token_ids` from the engine, empty `response_token_ids`
    /// from any sub-turn, or a tokenize failure on a tool result all
    /// downgrade to `None`. Honest-`None` lets RL pipelines mask the
    /// turn out instead of training on partial / lying data.
    pub tokens: Option<TokensRecord>,
}

/// Token-level RL-trainer-friendly view of a turn's trajectory.
///
/// `response_ids` interleaves LLM-generated tokens AND tool-result
/// tokens in the order they entered the model's context.
/// `response_mask` is `1` for LLM tokens, `0` for tool tokens —
/// matches verl's `AgentLoopOutput` semantics so an RL loss can mask
/// environment tokens out of the policy gradient.
///
/// `prompt_ids` is the tokenized prompt for the FIRST engine sub-turn
/// (the original user prompt + system); subsequent sub-turns' prompts
/// are reconstructable from this plus `response_ids` + `response_mask`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokensRecord {
    pub prompt_ids: Vec<u32>,
    pub response_ids: Vec<u32>,
    pub response_mask: Vec<u8>,
}

/// Trajectory schema version. Bumped to `2` when the token layer
/// (`tokens.{prompt_ids,response_ids,response_mask}`) started populating;
/// the rule is "version bumps when a meaningful new payload starts
/// populating". Records can
/// still carry `tokens: null` on backends that haven't wired
/// `tokenize()`, but the format-version contract is v2 either way so
/// v1-only readers refuse early instead of silently misreading.
pub const TRAJECTORY_SCHEMA_VERSION: i32 = 2;

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

    pub fn append_plain_turn(&mut self, user_input: &str, assistant_text: &str) {
        self.messages.push(Message::user(user_input));
        self.messages
            .push(Message::assistant(assistant_text, vec![]));
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
        cancel: Arc<AtomicBool>,
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
        cancel: Option<Arc<AtomicBool>>,
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

        // Phase 2 trajectory token layer. `prompt_ids` is set on the
        // FIRST engine sub-turn from `output.prompt_token_ids`; if that
        // is empty, or any later component is unavailable, we set
        // `tokens_aborted = true` and surface `tokens = None` from the
        // ultimate return — never partial / lying data.
        let mut prompt_ids: Option<Vec<u32>> = None;
        let mut response_ids: Vec<u32> = Vec::new();
        let mut response_mask: Vec<u8> = Vec::new();
        let mut tokens_aborted: bool = false;

        for turn in 0..settings.max_turns {
            // Two indices, both monotone: `tool_use_id_base` is the loop
            // iteration (always advances, including on the recovered
            // branch) so synthesized tool_use IDs `tu_{base}_{n}` stay
            // unique even when iteration 0 emits a recovered tool call
            // and iteration 1 produces an engine-driven one.
            // `sub_turn_record_index` is the position the
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
                    },
                    cancel.clone(),
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

                // Phase 2 token layer: assemble per-sub-turn deltas.
                //
                // First engine sub-turn fixes `prompt_ids` from the
                // engine's view of the original prompt. Subsequent
                // sub-turns get a longer `prompt_token_ids` because
                // ChatML re-renders prior assistant messages and
                // appends tool results + the next assistant prefix —
                // the tail past `prompt_ids.len() + response_ids.len()`
                // is the env-side delta (mask=0). Then we append the
                // sub-turn's actual generated tokens (mask=1).
                //
                // We do NOT byte-match the prior response — ChatML's
                // re-rendering can add framing bytes (a leading `\n`
                // before `<tool_call>`, the trailing `<|im_end|>`)
                // that don't appear in the engine's raw response_token_
                // ids. A strict prefix check would abort virtually
                // every multi-sub-turn trace; we accept that the first
                // few env-delta bytes may include the chatml frame
                // around the previously-generated content. RL trainers
                // that need byte-perfect token streams must use the
                // continuous-token-stream architecture (verl-style),
                // which is a separate bigger refactor; ARLE's current
                // re-render-per-sub-turn loop fundamentally can't
                // promise more than this.
                //
                // Honest-None: when the engine returns shorter
                // prompt_token_ids than what we already accumulated
                // (impossible under the contract — the prompt only
                // grows), or empty IDs at all, abort to None.
                if !tokens_aborted {
                    if let Some(existing_prompt_ids) = prompt_ids.as_ref() {
                        let prev_prompt_len = existing_prompt_ids.len();
                        let expected_offset = prev_prompt_len + response_ids.len();
                        // Only abort on extreme shrinkage that suggests a broken
                        // tokenizer or malformed engine response. ChatML re-rendering
                        // in repair scenarios can cause moderate length changes.
                        // Threshold: if the prompt shrank to less than half of the
                        // original size, that's likely a tokenizer bug.
                        if output.prompt_token_ids.len() < prev_prompt_len / 2 {
                            tokens_aborted = true;
                        } else if output.prompt_token_ids.len() >= expected_offset {
                            // Normal case: prompt grew or stayed same, extract env delta.
                            let env_delta = &output.prompt_token_ids[expected_offset..];
                            if !env_delta.is_empty() {
                                response_ids.extend_from_slice(env_delta);
                                response_mask.extend(std::iter::repeat_n(0u8, env_delta.len()));
                            }
                        }
                        // If expected_offset > len but len >= prev_prompt_len/2,
                        // just skip env delta extraction (lossy contract).
                    } else if output.prompt_token_ids.is_empty() {
                        tokens_aborted = true;
                    } else {
                        prompt_ids = Some(output.prompt_token_ids.clone());
                    }
                    if !tokens_aborted {
                        if output.response_token_ids.is_empty() {
                            tokens_aborted = true;
                        } else {
                            response_ids.extend(output.response_token_ids.iter());
                            response_mask
                                .extend(std::iter::repeat_n(1u8, output.response_token_ids.len()));
                        }
                    }
                }

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
                            cancel.clone(),
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
                        sub_turns.push(repair_outcome.record);

                        // Phase 2 token layer: repair was a real engine
                        // call. Same prompt-delta accounting as the
                        // main path — first engine call fixes
                        // prompt_ids; subsequent calls' prompt_token_ids
                        // includes the prior context and any wrappers
                        // the engine added between (in repair's case:
                        // the malformed-draft assistant message + the
                        // "rewrite using protocol" user nudge). That
                        // delta is mask=0; the repair's generated
                        // tokens are mask=1.
                        if !tokens_aborted {
                            if let Some(existing_prompt_ids) = prompt_ids.as_ref() {
                                // For repair, the prompt includes: original prompt + malformed response + repair instruction.
                                // We need to find where the new content starts relative to what we've already accumulated.
                                let prev_prompt_len = existing_prompt_ids.len();
                                let expected_offset = prev_prompt_len + response_ids.len();
                                // Only abort on extreme shrinkage that suggests broken tokenizer.
                                if repair_outcome.prompt_token_ids.len() < prev_prompt_len / 2 {
                                    tokens_aborted = true;
                                } else if repair_outcome.prompt_token_ids.len() >= expected_offset {
                                    // Normal case: prompt grew or stayed same, extract env delta.
                                    let env_delta =
                                        &repair_outcome.prompt_token_ids[expected_offset..];
                                    if !env_delta.is_empty() {
                                        response_ids.extend_from_slice(env_delta);
                                        response_mask
                                            .extend(std::iter::repeat_n(0u8, env_delta.len()));
                                    }
                                }
                                // If expected_offset > len but len >= prev_prompt_len/2,
                                // just skip env delta extraction (lossy contract).
                            } else if repair_outcome.prompt_token_ids.is_empty() {
                                tokens_aborted = true;
                            } else {
                                prompt_ids = Some(repair_outcome.prompt_token_ids.clone());
                            }
                            if !tokens_aborted {
                                if repair_outcome.response_token_ids.is_empty() {
                                    tokens_aborted = true;
                                } else {
                                    let n = repair_outcome.response_token_ids.len();
                                    response_ids.extend(repair_outcome.response_token_ids);
                                    response_mask.extend(std::iter::repeat_n(1u8, n));
                                }
                            }
                        }
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
            // and collided across recovered + engine pairs.
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
                let tokens = build_tokens_record(
                    tokens_aborted,
                    prompt_ids.clone(),
                    response_ids.clone(),
                    response_mask.clone(),
                );
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
                    tokens,
                }));
            }

            if !content.is_empty() {
                trace_events.push(AgentTraceEvent::AssistantNote(content));
            }

            let _tool_results_text = execute_tool_calls(
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

            // Note: tool-result tokens are NOT tokenized here anymore.
            // The next engine sub-turn's `prompt_token_ids` already
            // contains the model's view of the full context (system +
            // user + assistant tool_call + ChatML tool wrappers + tool
            // result + next assistant prompt prefix). The prompt-delta
            // logic at the top of each engine sub-turn captures those
            // tokens with mask=0. Tokenizing
            // bare tool result strings missed the wrappers and yielded
            // a reconstruction the model never actually saw.

            if let Some(text) = tool_policy.finalize_after_tool_execution(
                user_input,
                last_tool_name.as_deref(),
                last_tool_result.as_deref(),
                last_tool_scalar_result.as_deref(),
            ) {
                self.compact_turn_history(turn_start, &text);
                let tokens = build_tokens_record(
                    tokens_aborted,
                    prompt_ids.clone(),
                    response_ids.clone(),
                    response_mask.clone(),
                );
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
                    tokens,
                }));
            }
        }

        let final_text = "(max turns reached - agent stopped)".to_string();
        self.compact_turn_history(turn_start, &final_text);
        let tokens = build_tokens_record(tokens_aborted, prompt_ids, response_ids, response_mask);
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
            tokens,
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
) -> Vec<String> {
    // Returns the per-call tool result strings in execution order so
    // the caller can tokenize them into the trajectory's `response_ids`
    // (with mask=0 — env tokens RL must mask out of the policy loss).
    let mut results = Vec::with_capacity(tool_calls.len());
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
            content: MessageContent::Text(result.clone()),
            tool_use_id: Some(tool_use_id(sub_turn_index, call_index)),
            result_truncated: Some(metadata.truncated),
        });
        results.push(result);
    }
    results
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
    // Always emit a leading text block — even when empty — so the
    // schema's "assistant content is always blocks" invariant holds
    // and downstream consumers don't have to special-case empty text.
    std::iter::once(ContentBlock::Text {
        text: content.to_string(),
    })
    .chain(
        tool_calls
            .iter()
            .enumerate()
            .map(|(call_index, call)| ContentBlock::ToolUse {
                id: tool_use_id(sub_turn_index, call_index),
                name: call.name.clone(),
                input: call.arguments.clone(),
            }),
    )
    .collect()
}

fn finish_reason_to_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

/// Phase 2 trajectory token layer assembler. Returns `Some(record)` ONLY
/// when every required component was available — `tokens_aborted` is
/// the kill switch that fires the moment any sub-turn or tool result
/// produced an empty / errored token list. `prompt_ids` must also be
/// `Some` and non-empty (no engine sub-turn ever fired ⇒ no prompt
/// ⇒ no record). The contract: never ship a partial / lying mask.
fn build_tokens_record(
    tokens_aborted: bool,
    prompt_ids: Option<Vec<u32>>,
    response_ids: Vec<u32>,
    response_mask: Vec<u8>,
) -> Option<TokensRecord> {
    if tokens_aborted {
        return None;
    }
    let prompt_ids = prompt_ids?;
    if prompt_ids.is_empty() {
        return None;
    }
    // Defensive: enforce the docs invariant `len == len`.
    if response_ids.len() != response_mask.len() {
        return None;
    }
    Some(TokensRecord {
        prompt_ids,
        response_ids,
        response_mask,
    })
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
/// (Repair was previously an unrecorded engine call.)
struct RepairOutcome {
    parsed: ParsedAssistantResponse,
    record: SubTurnRecord,
    /// Phase 2 trajectory: the engine's tokenized response (mask=1) for
    /// this repair sub-turn. Empty when the engine didn't surface ids
    /// — caller should abort token tracking.
    response_token_ids: Vec<u32>,
    /// The repair sub-turn's tokenized prompt. Used only when no prior
    /// engine sub-turn established `prompt_ids` yet.
    prompt_token_ids: Vec<u32>,
}

fn repair_tool_calls<E: InferenceEngine + ?Sized>(
    engine: &mut E,
    messages: &[Message],
    tools: &[ToolDefinition],
    settings: AgentSettings,
    assistant_draft: &str,
    cancel: Option<Arc<AtomicBool>>,
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
            response_token_ids: repair_output.response_token_ids,
            prompt_token_ids: repair_output.prompt_token_ids,
        }));
    }
    // No tool calls parsed (including an explicit "NO_TOOL" reply): nothing to
    // repair into, so the turn yields no tool invocation.
    Ok(None)
}

fn complete_with_optional_cancel<E: InferenceEngine + ?Sized>(
    engine: &mut E,
    req: CompletionRequest,
    cancel: Option<Arc<AtomicBool>>,
    mut on_text_chunk: Option<&mut dyn FnMut(&str)>,
) -> Result<Option<CompletionOutput>> {
    if cancel.is_none() && on_text_chunk.is_none() {
        return engine.complete(req).map(Some);
    }

    // Phase 2 trajectory: snapshot the prompt's tokenized form before
    // the worker takes ownership of `req`. Empty Vec on failure — the
    // agent loop treats empty as "unavailable" and downgrades
    // `tokens = None`.
    let prompt_token_ids = engine.tokenize(&req.prompt).unwrap_or_default();

    let (tx, rx) = mpsc::unbounded_channel::<CompletionStreamDelta>();
    let mut rx: Option<mpsc::UnboundedReceiver<CompletionStreamDelta>> = Some(rx);
    let mut text = String::new();
    let mut finish_reason = None::<FinishReason>;
    let mut usage = None::<TokenUsage>;
    let mut response_token_ids: Vec<u32> = Vec::new();
    let mut stream_err = None::<anyhow::Error>;
    let mut cancelled = false;

    std::thread::scope(|s| {
        let worker = s.spawn(|| engine.complete_stream(req, tx));

        loop {
            if cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                cancelled = true;
                rx = None;
                break;
            }

            let Some(rx_ref) = rx.as_mut() else { break };
            match rx_ref.try_recv() {
                Ok(delta) => {
                    let CompletionStreamDelta {
                        text_delta,
                        finish_reason: delta_finish_reason,
                        usage: delta_usage,
                        token_ids,
                        error,
                    } = delta;
                    if let Some(error) = error {
                        stream_err = Some(error.into_anyhow());
                        break;
                    }
                    if !text_delta.is_empty() {
                        if let Some(callback) = on_text_chunk.as_deref_mut() {
                            callback(&text_delta);
                        }
                        text.push_str(&text_delta);
                    }
                    response_token_ids.extend(token_ids);
                    if let Some(final_usage) = delta_usage {
                        usage = Some(final_usage);
                    }
                    if let Some(reason) = delta_finish_reason {
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
        prompt_token_ids,
        response_token_ids,
    }))
}
