//! Phase 1 trajectory writer for the ARLE CLI agent loop.
//!
//! See `docs/projects/agent-trajectory-export.md` for the canonical
//! schema. v1 captures the message log + per-sub-turn telemetry; v2
//! (token IDs + response_mask) is deferred until the engine surface
//! exposes per-token state.
//!
//! All IO failures are logged and dropped. A REPL turn must NEVER
//! crash because the trace file rotated, the disk filled, or the path
//! went read-only — the run is the source of truth, the trace is a
//! best-effort sidecar.

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::fs::OpenOptions;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::Write;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::path::{Path, PathBuf};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::Mutex;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use agent::{
    AgentTurnResult, SubTurnRecord, TRAJECTORY_SCHEMA_VERSION, TerminalState, TrajectoryMessage,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use crate::repl::format_iso8601_utc_secs;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AgentTrajectoryRecord {
    pub schema_version: i32,
    pub ts: String,
    pub turn_id: String,
    pub model_id: String,
    pub backend: String,
    pub user_input: String,
    pub messages: Vec<TrajectoryMessage>,
    pub sub_turns: Vec<SubTurnRecord>,
    /// Phase 1 placeholder. Populated in v2 with `{prompt_ids,
    /// response_ids, response_mask}`.
    pub tokens: Option<serde_json::Value>,
    pub result: TrajectoryResult,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrajectoryResult {
    pub text: String,
    pub terminal_state: TerminalState,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub wall_secs: f64,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) struct TraceWriter {
    path: PathBuf,
    keep_prompts: bool,
    /// `Mutex` rather than a bare `RefCell` so `&TraceWriter` can be
    /// shared across the REPL's interactive + piped paths without a
    /// `RefCell::borrow_mut` panic if both ever fire concurrently.
    file: Mutex<std::fs::File>,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl TraceWriter {
    pub(crate) fn open(path: impl AsRef<Path>, keep_prompts: bool) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            keep_prompts,
            file: Mutex::new(file),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Build a trajectory record from an agent turn and append it to
    /// the JSONL file. IO failures are logged at warn level and
    /// dropped — never bubbled back up to the caller.
    pub(crate) fn write_turn(
        &self,
        model_id: &str,
        backend: &str,
        user_input: &str,
        result: &AgentTurnResult,
    ) {
        let record = self.build_record(model_id, backend, user_input, result);
        match serde_json::to_string(&record) {
            Ok(line) => {
                let mut guard = match self.file.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if let Err(err) = writeln!(*guard, "{line}") {
                    log::warn!(
                        "trajectory write failed for {}: {err} (dropped, turn unaffected)",
                        self.path.display()
                    );
                    return;
                }
                if let Err(err) = guard.flush() {
                    log::warn!(
                        "trajectory flush failed for {}: {err} (dropped)",
                        self.path.display()
                    );
                }
            }
            Err(err) => {
                log::warn!("trajectory serialization failed: {err} (dropped, turn unaffected)");
            }
        }
    }

    pub(crate) fn build_record(
        &self,
        model_id: &str,
        backend: &str,
        user_input: &str,
        result: &AgentTurnResult,
    ) -> AgentTrajectoryRecord {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut sub_turns = result.sub_turns.clone();
        if !self.keep_prompts {
            for st in &mut sub_turns {
                st.prompt_text = None;
            }
        }
        AgentTrajectoryRecord {
            schema_version: TRAJECTORY_SCHEMA_VERSION,
            ts: format_iso8601_utc_secs(secs),
            turn_id: uuid::Uuid::new_v4().to_string(),
            model_id: model_id.to_string(),
            backend: backend.to_string(),
            user_input: user_input.to_string(),
            messages: result.messages.clone(),
            sub_turns,
            tokens: None,
            result: TrajectoryResult {
                text: result.text.clone(),
                terminal_state: result.terminal_state,
                total_prompt_tokens: result.prompt_tokens,
                total_completion_tokens: result.completion_tokens,
                wall_secs: result.wall_secs,
            },
        }
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "metal", feature = "cpu")))]
mod tests {
    use super::*;
    use agent::{
        ContentBlock, MessageContent, SubTurnRecord, TerminalState, ToolUsage, TrajectoryMessage,
        TrajectoryRole,
    };

    fn synthetic_turn() -> AgentTurnResult {
        AgentTurnResult {
            text: "final".to_string(),
            tool_calls_executed: 1,
            prompt_tokens: 10,
            completion_tokens: 4,
            max_turns_reached: false,
            trace_events: vec![],
            time_to_first_token: None,
            messages: vec![
                TrajectoryMessage {
                    role: TrajectoryRole::User,
                    content: MessageContent::Text("hi".to_string()),
                    tool_use_id: None,
                    result_truncated: None,
                },
                TrajectoryMessage {
                    role: TrajectoryRole::Assistant,
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Text {
                            text: "thinking".to_string(),
                        },
                        ContentBlock::ToolUse {
                            id: "tu_0_0".to_string(),
                            name: "shell".to_string(),
                            input: serde_json::json!({"command": "ls"}),
                        },
                    ]),
                    tool_use_id: None,
                    result_truncated: None,
                },
                TrajectoryMessage {
                    role: TrajectoryRole::Tool,
                    content: MessageContent::Text("ok".to_string()),
                    tool_use_id: Some("tu_0_0".to_string()),
                    result_truncated: Some(false),
                },
            ],
            sub_turns: vec![SubTurnRecord {
                index: 0,
                prompt_text: Some("PROMPT".to_string()),
                completion_text: "raw".to_string(),
                usage: ToolUsage {
                    prompt_tokens: 10,
                    completion_tokens: 4,
                },
                ttft_ms: Some(100),
                decode_secs: 0.4,
                finish_reason: "stop".to_string(),
            }],
            terminal_state: TerminalState::Stop,
            wall_secs: 0.5,
        }
    }

    #[test]
    fn write_turn_emits_one_line_per_call() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let writer = TraceWriter::open(&path, true).expect("open");

        let turn = synthetic_turn();
        writer.write_turn("fake-model", "metal", "hi", &turn);
        writer.write_turn("fake-model", "metal", "again", &turn);

        let body = std::fs::read_to_string(&path).expect("read trace");
        let lines: Vec<&str> = body.split_terminator('\n').collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).expect("valid json");
            assert_eq!(parsed["schema_version"], 1);
            assert_eq!(parsed["model_id"], "fake-model");
            assert_eq!(parsed["backend"], "metal");
            assert_eq!(parsed["sub_turns"][0]["prompt_text"], "PROMPT");
            assert_eq!(parsed["result"]["terminal_state"], "stop");
            assert!(parsed["tokens"].is_null());
        }
    }

    #[test]
    fn trace_prompts_off_blanks_prompt_text_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let writer = TraceWriter::open(&path, false).expect("open");
        let turn = synthetic_turn();
        writer.write_turn("fake-model", "metal", "hi", &turn);

        let body = std::fs::read_to_string(&path).expect("read trace");
        let parsed: serde_json::Value = serde_json::from_str(body.trim()).expect("json");
        assert!(
            parsed["sub_turns"][0]["prompt_text"].is_null(),
            "prompt_text should serialize as null when --trace-prompts off"
        );
    }

    /// Mirrors the `FakeEngine` fixture pattern from `crates/agent/src/lib.rs`
    /// — drives a real `AgentSession::run_turn` end-to-end so the trace
    /// reflects the schema the production REPL would emit, not the
    /// hand-built `synthetic_turn`.
    #[test]
    fn end_to_end_run_emits_valid_jsonl_via_fake_engine() {
        use agent::{AgentSession, AgentSettings, ToolExecutor, ToolPolicy};
        use anyhow::Result;
        use chat::ToolCall;
        use infer::server_engine::{
            CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason,
            InferenceEngine, TokenUsage,
        };
        use std::collections::VecDeque;
        use tokio::sync::mpsc::UnboundedSender;

        struct FakeEngine {
            outputs: VecDeque<String>,
        }

        impl InferenceEngine for FakeEngine {
            fn model_id(&self) -> &str {
                "fake-model"
            }

            fn complete(&mut self, _req: CompletionRequest) -> Result<CompletionOutput> {
                let text = self
                    .outputs
                    .pop_front()
                    .expect("FakeEngine outputs exhausted");
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

        struct StubExecutor;
        impl ToolExecutor for StubExecutor {
            fn execute(&self, _: &ToolCall) -> String {
                "ok".to_string()
            }
        }

        struct NoopPolicy;
        impl ToolPolicy for NoopPolicy {}

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let writer = TraceWriter::open(&path, true).expect("open");

        let mut session = AgentSession::new();
        let mut engine = FakeEngine {
            outputs: VecDeque::from(vec!["hello world".to_string()]),
        };
        let result = session
            .run_turn(
                &mut engine,
                "say hi",
                &[],
                &StubExecutor,
                &NoopPolicy,
                AgentSettings {
                    max_turns: 4,
                    max_tokens: 32,
                    temperature: 0.0,
                },
            )
            .expect("run_turn");

        writer.write_turn(engine.model_id(), "fake-backend", "say hi", &result);

        let body = std::fs::read_to_string(&path).expect("read trace");
        let lines: Vec<&str> = body.split_terminator('\n').collect();
        assert_eq!(lines.len(), 1, "expected exactly one JSONL record");
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).expect("valid json");
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["result"]["terminal_state"], "stop");
        assert_eq!(parsed["model_id"], "fake-model");
        assert_eq!(parsed["backend"], "fake-backend");
        assert_eq!(parsed["user_input"], "say hi");
        // First message is the user prompt.
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert_eq!(parsed["messages"][0]["content"], "say hi");
        // Second message is the assistant block array carrying the text.
        assert_eq!(parsed["messages"][1]["role"], "assistant");
        let blocks = parsed["messages"][1]["content"].as_array().expect("blocks");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "hello world");
        // No tool messages because the turn took the no-tool path.
        assert!(
            parsed["messages"]
                .as_array()
                .expect("array")
                .iter()
                .all(|m| m["role"] != "tool")
        );
        // tokens placeholder must serialize as JSON null in v1.
        assert!(parsed["tokens"].is_null());
        // schema_version is 1 (i32) — already asserted above; this also
        // pins the field's name as a smoke for casual schema rename.
        assert!(parsed.get("schema_version").is_some());
    }
}
