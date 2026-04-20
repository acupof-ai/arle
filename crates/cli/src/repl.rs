#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::path::{Path, PathBuf};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::Arc;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::OnceLock;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use agent::{
    AgentSession, AgentSessionStats, AgentSettings, AgentTraceEvent, ToolExecutor, ToolPolicy,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use chat::{ChatMessage, ParsedAssistantResponse, ToolCall, ToolDefinition};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::sampler::SamplingParams;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::{CompletionRequest, CompletionStreamDelta, InferenceEngine};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::DefaultEditor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::error::ReadlineError;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use tokio::sync::mpsc;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use tools::{BuiltinToolPolicyHooks, builtin_tools, execute_tool_call};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const SYSTEM_PROMPT_CHAT: &str =
    "You are a helpful AI assistant. Be concise unless the user asks for detail.";

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplMode {
    Chat,
    Agent,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl ReplMode {
    fn label(self) -> &'static str {
        match self {
            ReplMode::Chat => "chat",
            ReplMode::Agent => "agent",
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReplCommand {
    Help,
    Reset,
    Tools,
    Model,
    Stats,
    Save(String),
    Load(String),
    SwitchChat,
    SwitchAgent,
    /// `/models` with no arg lists discovered hub snapshots; `/models N` is
    /// reserved for engine hot-swap which is currently unsupported (owner
    /// pattern requires an owned `LoadedInferenceEngine` instead of the
    /// `&mut dyn InferenceEngine` threaded through `run_repl`). When N is
    /// supplied we print a friendly pointer to restart with `--model-path`.
    Models(Option<usize>),
    /// `/export` → default path; `/export <path>` → explicit path/dir.
    Export(String),
    /// `/retry` — drop last assistant turn, re-run the prior user turn.
    Retry,
    Exit,
    Unknown(String),
}

/// Per-session rolling accumulators for `/stats` enrichment. Populated by
/// streaming chat turns. Agent-mode turns do not contribute tokens here
/// because agent-mode uses a different code path that doesn't surface
/// streaming usage deltas to the REPL.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Default, Clone, Copy)]
struct SessionStats {
    /// Number of chat turns (one user + one assistant reply each).
    turn_count: usize,
    /// Sum of prompt_tokens reported by each turn's final TokenUsage, when
    /// populated. Falls back to chars/4 for the most recent user message
    /// if usage was unset (e.g. backend without usage reporting).
    prompt_tokens: u64,
    /// Sum of completion_tokens reported by each turn's final TokenUsage,
    /// when populated. Falls back to the rolling TpsMeter count.
    completion_tokens: u64,
    /// Weighted TPS accumulator: sum of (tokens · tokens/elapsed) across
    /// turns; divided by total tokens for the weighted average.
    /// Stored separately rather than as a running avg because a simple mean
    /// over turns would treat a 1-token turn and a 1000-token turn equally.
    weighted_rate_numer: f64,
    /// Sum of elapsed seconds across turns (the /stats "wall time" row).
    total_secs: f64,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl SessionStats {
    fn record_turn(&mut self, prompt_tokens: u64, completion_tokens: u64, elapsed: Duration) {
        self.turn_count = self.turn_count.saturating_add(1);
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
        let secs = elapsed.as_secs_f64();
        self.total_secs += secs;
        if secs > f64::EPSILON {
            // Weight each turn's rate by its token count so long turns
            // dominate, matching what an operator intuitively means by
            // "avg throughput".
            let rate = completion_tokens as f64 / secs;
            self.weighted_rate_numer += rate * completion_tokens as f64;
        }
    }

    fn avg_tps(&self) -> f64 {
        if self.completion_tokens == 0 {
            0.0
        } else {
            self.weighted_rate_numer / self.completion_tokens as f64
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
struct BuiltinToolExecutor;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
struct BuiltinToolPolicy;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl ToolExecutor for BuiltinToolExecutor {
    fn execute(&self, tool_call: &ToolCall) -> String {
        execute_tool_call(tool_call)
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl ToolPolicy for BuiltinToolPolicy {
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

/// Process-wide cancel flag for an in-flight generation. Flipped by SIGINT;
/// the streaming loop polls it and short-circuits on cancel.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
static CANCEL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Unix-millis timestamp of the last Ctrl-C. Used to detect double-tap exit.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
static LAST_SIGINT_MS: OnceLock<Arc<AtomicU64>> = OnceLock::new();

/// Set when two Ctrl-C taps land within 2 seconds — REPL drains and exits.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
static EXIT_REQUESTED: OnceLock<Arc<AtomicBool>> = OnceLock::new();

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn install_ctrlc_handler() -> (Arc<AtomicBool>, Arc<AtomicBool>) {
    let cancel = CANCEL_FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();
    let last_ms = LAST_SIGINT_MS
        .get_or_init(|| Arc::new(AtomicU64::new(0)))
        .clone();
    let exit = EXIT_REQUESTED
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();

    let cancel_h = cancel.clone();
    let last_ms_h = last_ms.clone();
    let exit_h = exit.clone();

    // ctrlc::set_handler errors if a handler is already installed (e.g. test
    // re-entry). That's harmless here — the existing handler still flips the
    // shared statics above.
    let _ = ctrlc::set_handler(move || {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let prev = last_ms_h.swap(now_ms, Ordering::Relaxed);
        let was_active = cancel_h.swap(true, Ordering::Relaxed);
        // Double-tap within 2s, OR Ctrl-C while no generation is in flight,
        // means "exit". The REPL loop checks EXIT_REQUESTED after each prompt.
        if !was_active && prev != 0 && now_ms.saturating_sub(prev) <= 2_000 {
            exit_h.store(true, Ordering::Relaxed);
        }
    });

    (cancel, exit)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    start_in_tools_mode: bool,
) -> Result<()> {
    let initial_mode = if start_in_tools_mode {
        ReplMode::Agent
    } else {
        ReplMode::Chat
    };

    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_interactive_repl(
            engine,
            backend_name,
            max_turns,
            max_tokens,
            temperature,
            initial_mode,
        );
    }

    run_piped_repl(
        engine,
        backend_name,
        max_turns,
        max_tokens,
        temperature,
        initial_mode,
    )
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_banner(
    engine: &dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    mode: ReplMode,
) {
    println!();
    println!("=== agent-infer REPL ===");
    println!("Model: {}", engine.model_id());
    println!("Backend: {}", backend_name);
    println!("Mode: {} (use /chat or /agent to switch)", mode.label());
    println!(
        "Tools: {}",
        tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Max turns: {}, Max tokens: {}, Temperature: {}",
        max_turns, max_tokens, temperature
    );
    println!("End a line with `\\` to continue input on the next line.");
    println!("Press Ctrl-C to cancel a running generation; press it twice within 2s to exit.");
    println!("Type '/help' for commands, '/quit' or '/exit' to leave.");
    println!();
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".agent-infer-history"))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn prompt_for(mode: ReplMode) -> &'static str {
    match mode {
        ReplMode::Chat => "\x1b[1;32m> \x1b[0m",
        ReplMode::Agent => "\x1b[1;35m> \x1b[0m",
    }
}

/// Read one logical input from rustyline. Lines ending with `\` are joined
/// with `\n` until a line without a trailing backslash is entered.
///
/// Returns `Ok(Some(_))` for a complete input, `Ok(None)` for EOF, and
/// propagates `ReadlineError::Interrupted` upward (REPL turns it into ^C).
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_multiline(
    editor: &mut DefaultEditor,
    mode: ReplMode,
) -> Result<Option<String>, ReadlineError> {
    let mut accumulated: Option<String> = None;
    let primary = prompt_for(mode);
    let cont = "\x1b[2m… \x1b[0m";

    loop {
        let prompt = if accumulated.is_some() { cont } else { primary };
        let line = editor.readline(prompt)?;
        if let Some(stripped) = line.strip_suffix('\\') {
            let acc = accumulated.get_or_insert_with(String::new);
            if !acc.is_empty() {
                acc.push('\n');
            }
            acc.push_str(stripped);
            continue;
        }
        let result = if let Some(mut acc) = accumulated.take() {
            if !acc.is_empty() {
                acc.push('\n');
            }
            acc.push_str(&line);
            acc
        } else {
            line
        };
        return Ok(Some(result));
    }
}

/// Same `\` continuation, but for piped stdin.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_multiline_piped<R: BufRead>(reader: &mut R, prompt: &str) -> io::Result<Option<String>> {
    let mut stdout = io::stdout();
    print!("{prompt}");
    stdout.flush()?;

    let mut accumulated: Option<String> = None;
    loop {
        let mut buf = String::new();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            return Ok(if accumulated.is_some() {
                accumulated
            } else {
                None
            });
        }
        // strip a trailing `\n` (and `\r` on CRLF) without touching internal whitespace
        let line = buf.trim_end_matches(['\n', '\r']);
        if let Some(stripped) = line.strip_suffix('\\') {
            let acc = accumulated.get_or_insert_with(String::new);
            if !acc.is_empty() {
                acc.push('\n');
            }
            acc.push_str(stripped);
            print!("{prompt}");
            stdout.flush()?;
            continue;
        }
        let result = if let Some(mut acc) = accumulated.take() {
            if !acc.is_empty() {
                acc.push('\n');
            }
            acc.push_str(line);
            acc
        } else {
            line.to_string()
        };
        return Ok(Some(result));
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_interactive_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    initial_mode: ReplMode,
) -> Result<()> {
    let tools = builtin_tools()
        .into_iter()
        .map(|tool| tool.to_definition())
        .collect::<Vec<_>>();
    let mut session = AgentSession::new();
    let mut chat_history: Vec<ChatMessage> = vec![ChatMessage::system(SYSTEM_PROMPT_CHAT)];
    let mut session_stats = SessionStats::default();
    let mut mode = initial_mode;
    let mut editor = DefaultEditor::new()?;
    let history = history_path();

    if let Some(path) = history.as_ref()
        && let Err(err) = editor.load_history(path)
    {
        log::debug!("Skipping history load from {}: {err}", path.display());
    }

    let (cancel, exit) = install_ctrlc_handler();
    // Reset any stale state from a prior session.
    cancel.store(false, Ordering::Relaxed);
    exit.store(false, Ordering::Relaxed);

    print_repl_banner(
        engine,
        backend_name,
        &tools,
        max_turns,
        max_tokens,
        temperature,
        mode,
    );

    loop {
        // Make sure the cancel flag is clear at each prompt — it may have
        // been left set by a Ctrl-C that landed at a blank prompt.
        cancel.store(false, Ordering::Relaxed);

        match read_multiline(&mut editor, mode) {
            Ok(Some(line)) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                editor.add_history_entry(input)?;
                if !handle_repl_input(
                    engine,
                    backend_name,
                    &tools,
                    &mut session,
                    &mut chat_history,
                    &mut session_stats,
                    &mut mode,
                    input,
                    max_turns,
                    max_tokens,
                    temperature,
                    cancel.clone(),
                )? {
                    break;
                }
                if exit.load(Ordering::Relaxed) {
                    println!();
                    break;
                }
            }
            Ok(None) => {
                println!();
                break;
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C at the prompt: reset cancel state, print marker,
                // continue (or exit on double-tap).
                cancel.store(false, Ordering::Relaxed);
                println!("^C");
                if exit.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(e) => {
                return Err(e.into());
            }
        }
    }

    if let Some(path) = history.as_ref()
        && let Err(err) = editor.save_history(path)
    {
        log::debug!("Skipping history save to {}: {err}", path.display());
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_piped_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    initial_mode: ReplMode,
) -> Result<()> {
    let tools = builtin_tools()
        .into_iter()
        .map(|tool| tool.to_definition())
        .collect::<Vec<_>>();
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut session = AgentSession::new();
    let mut chat_history: Vec<ChatMessage> = vec![ChatMessage::system(SYSTEM_PROMPT_CHAT)];
    let mut session_stats = SessionStats::default();
    let mut mode = initial_mode;

    let (cancel, exit) = install_ctrlc_handler();
    cancel.store(false, Ordering::Relaxed);
    exit.store(false, Ordering::Relaxed);

    print_repl_banner(
        engine,
        backend_name,
        &tools,
        max_turns,
        max_tokens,
        temperature,
        mode,
    );

    loop {
        cancel.store(false, Ordering::Relaxed);
        match read_multiline_piped(&mut reader, "\x1b[1;32m> \x1b[0m")? {
            Some(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if !handle_repl_input(
                    engine,
                    backend_name,
                    &tools,
                    &mut session,
                    &mut chat_history,
                    &mut session_stats,
                    &mut mode,
                    input,
                    max_turns,
                    max_tokens,
                    temperature,
                    cancel.clone(),
                )? {
                    break;
                }
                if exit.load(Ordering::Relaxed) {
                    println!();
                    break;
                }
            }
            None => {
                println!();
                break;
            }
        }
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn handle_repl_input(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    chat_history: &mut Vec<ChatMessage>,
    session_stats: &mut SessionStats,
    mode: &mut ReplMode,
    input: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) -> Result<bool> {
    if input == "quit" || input == "exit" {
        return Ok(false);
    }
    if let Some(command) = parse_repl_command(input) {
        return Ok(execute_repl_command(
            command,
            engine,
            backend_name,
            tools,
            session,
            chat_history,
            session_stats,
            mode,
            max_turns,
            max_tokens,
            temperature,
            cancel,
        ));
    }

    match *mode {
        ReplMode::Chat => {
            run_chat_turn(
                engine,
                chat_history,
                session_stats,
                input,
                max_tokens,
                temperature,
                cancel,
            )?;
        }
        ReplMode::Agent => {
            run_agent_turn(
                engine,
                tools,
                session,
                input,
                max_turns,
                max_tokens,
                temperature,
                cancel,
            );
        }
    }

    Ok(true)
}

/// Streaming chat turn. Builds a ChatML prompt from `history`, kicks off
/// `engine.complete_stream` on a worker thread, drains deltas on the main
/// thread, prints them with line-buffered flushes, and appends the final
/// assistant message to history.
///
/// Cancellation: `cancel` is polled every `try_recv` iteration. On cancel,
/// the partial assistant message is still appended (so context stays
/// consistent), and the receiver is dropped so the worker's next
/// `tx.send` fails — `generate_inner` observes `SinkControl::ConsumerDropped`
/// and exits the sampling loop before the next token, so `worker.join()`
/// returns promptly instead of waiting for `max_tokens` of decoding
/// (see `server_engine.rs` `on_token` returning `false` on send failure).
/// Closes the 76ea6ce codex review High finding: Ctrl-C used to ACK
/// the keystroke on the UI but block the REPL until the full completion
/// finished; now it actually short-circuits the engine.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn run_chat_turn(
    engine: &mut dyn InferenceEngine,
    history: &mut Vec<ChatMessage>,
    session_stats: &mut SessionStats,
    user_input: &str,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    // `/retry` pushes the user message itself and calls this variant so the
    // history is not duplicated. The common path still builds history here.
    history.push(ChatMessage::user(user_input));
    run_chat_turn_with_history(
        engine,
        history,
        session_stats,
        user_input,
        max_tokens,
        temperature,
        cancel,
    )
}

/// Same as `run_chat_turn` but assumes the caller has already appended the
/// user message to `history`. Used by `/retry` to avoid a duplicate user
/// turn after popping the prior assistant reply.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn run_chat_turn_with_history(
    engine: &mut dyn InferenceEngine,
    history: &mut Vec<ChatMessage>,
    session_stats: &mut SessionStats,
    user_input: &str,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    let turn_start = Instant::now();
    let prompt = chat::messages_to_prompt(history, &[]);
    let req = CompletionRequest {
        prompt,
        max_tokens,
        sampling: SamplingParams {
            temperature,
            ..SamplingParams::default()
        },
        stop: Some(vec!["<|im_end|>".to_string()]),
        logprobs: false,
    };

    let (tx, rx) = mpsc::unbounded_channel::<CompletionStreamDelta>();
    // `rx` is wrapped in Option so the cancel branch can `take()` + drop it
    // mid-scope. Dropping the receiver disconnects the unbounded channel,
    // so the next `tx.send` in the worker fails, `on_token` returns
    // `false`, and `generate_inner` exits via `SinkControl::ConsumerDropped`
    // (see `server_engine.rs` — `complete_stream`'s `on_token` treats a
    // send error as "consumer gone, stop sampling").
    let mut rx: Option<mpsc::UnboundedReceiver<CompletionStreamDelta>> = Some(rx);
    let mut accumulated = String::new();
    // Buffer for incomplete UTF-8 sequences arriving across deltas.
    let mut partial_bytes: Vec<u8> = Vec::new();
    let mut cancelled = false;
    let mut stream_err: Option<anyhow::Error> = None;
    let mut tps_meter = crate::tps::TpsMeter::new();
    // Completion tokens from the final delta's TokenUsage, if populated.
    // Both CUDA and Metal backends set this at finish_reason; we prefer it
    // over the rolling counter for the final summary.
    let mut final_completion_tokens: Option<u64> = None;
    // Prompt tokens reported by the final delta (exact from the backend).
    let mut final_prompt_tokens: Option<u64> = None;

    // Assistant-text styling: dim green on TTY, nothing on piped stdout
    // (matches the brief — "just the assistant streamed output"). We only
    // print the opening SGR here; the close SGR lands at end-of-turn.
    let color_on = io::stdout().is_terminal();
    if color_on {
        // 0;32 = normal-weight green. Readable on both light and dark
        // terminals. (Bright green 1;32 ends up washed-out on light bg.)
        print!("\x1b[32m");
        let _ = io::stdout().flush();
    }

    std::thread::scope(|s| {
        let worker = s.spawn(|| engine.complete_stream(req, tx));

        loop {
            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                // Drop the receiver NOW so the worker short-circuits on its
                // next `tx.send`. Without this, `worker.join()` below would
                // block until `max_tokens` of decoding finished, negating
                // the Ctrl-C (76ea6ce codex review High finding).
                rx = None;
                break;
            }
            // Only the cancel branch above ever takes `rx`; if we reach
            // here after it was taken, break out defensively rather than
            // unwrap-panicking in the inner try_recv.
            let Some(rx_ref) = rx.as_mut() else { break };
            match rx_ref.try_recv() {
                Ok(delta) => {
                    if !delta.text_delta.is_empty() {
                        let chunk = decode_chunk(&mut partial_bytes, delta.text_delta.as_bytes());
                        if !chunk.is_empty() {
                            // Erase the live TPS line (if visible) before
                            // the next token chunk lands on stdout — keeps
                            // streamed text uncorrupted on the same row.
                            tps_meter.hide_before_chunk();
                            accumulated.push_str(&chunk);
                            print!("{}", chunk);
                            let _ = io::stdout().flush();
                            tps_meter.record_chunk(chunk.len());
                        }
                    }
                    if let Some(usage) = delta.usage {
                        final_completion_tokens = Some(usage.completion_tokens as u64);
                        final_prompt_tokens = Some(usage.prompt_tokens as u64);
                    }
                    if delta.finish_reason.is_some() {
                        break;
                    }
                    tps_meter.maybe_refresh();
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    tps_meter.maybe_refresh();
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        // On cancel, `rx` was already dropped above, so the worker's next
        // `tx.send` returns Err → `generate_inner` observes
        // ConsumerDropped and exits the sampling loop before the next
        // token. `worker.join()` therefore returns promptly rather than
        // blocking the REPL for another `max_tokens` of decoding. On the
        // non-cancel path, the worker has already finished and this is
        // a no-wait reap.
        if let Ok(res) = worker.join()
            && let Err(e) = res
        {
            stream_err = Some(e);
        }
    });

    // Flush any remaining undecoded bytes as lossy UTF-8 (last resort).
    if !partial_bytes.is_empty() {
        let tail = String::from_utf8_lossy(&partial_bytes).to_string();
        if !tail.is_empty() {
            accumulated.push_str(&tail);
            print!("{}", tail);
            let _ = io::stdout().flush();
        }
        partial_bytes.clear();
    }

    if color_on {
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    if cancelled {
        println!();
        println!("\x1b[2m^C (generation cancelled)\x1b[0m");
        // Reset cancel flag now that we've handled it.
        cancel.store(false, Ordering::Relaxed);
    } else {
        println!();
    }
    // Live line clear + final summary (tokens / elapsed / tok-s). Prints to
    // stderr; on non-TTY stderr the live refresh was already skipped and
    // this just prints plain text.
    tps_meter.print_final(final_completion_tokens);
    println!();

    // Always record the (possibly partial) assistant turn so subsequent
    // turns see consistent context.
    history.push(ChatMessage::assistant(&accumulated, vec![]));

    // Accumulate into session-level /stats. Prefer the backend's exact
    // usage; fall back to the chars/4 heuristic for prompt tokens when
    // the stream closed without a finish_reason (cancelled, stream_err).
    let prompt_tokens = final_prompt_tokens.unwrap_or_else(|| approx_tokens(user_input));
    let completion_tokens = final_completion_tokens.unwrap_or(0);
    session_stats.record_turn(prompt_tokens, completion_tokens, turn_start.elapsed());

    if let Some(err) = stream_err {
        eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
        println!();
    }

    Ok(())
}

/// Rough chars/4 estimator. Used as a fallback input-token count when the
/// backend didn't report an exact value (e.g. streaming closed before the
/// final usage delta). Standard industry heuristic.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn approx_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

/// Decode bytes into a String, buffering an incomplete trailing multi-byte
/// UTF-8 sequence into `partial`. Returns whatever portion is now valid.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn decode_chunk(partial: &mut Vec<u8>, new_bytes: &[u8]) -> String {
    partial.extend_from_slice(new_bytes);
    match std::str::from_utf8(partial) {
        Ok(s) => {
            let out = s.to_string();
            partial.clear();
            out
        }
        Err(e) => {
            let valid_up_to = e.valid_up_to();
            // Safe: valid_up_to is by definition a valid UTF-8 prefix length.
            let head = std::str::from_utf8(&partial[..valid_up_to])
                .expect("valid_up_to bounds")
                .to_string();
            // Keep the trailing bytes for the next chunk; if they are
            // unambiguously invalid (not just incomplete), drop them as
            // lossy U+FFFD so we don't grow the buffer forever.
            let remainder = partial[valid_up_to..].to_vec();
            partial.clear();
            if e.error_len().is_some() {
                // Definite invalid byte — emit replacement chars and skip.
                let lossy = String::from_utf8_lossy(&remainder).into_owned();
                head + &lossy
            } else {
                // Incomplete sequence — buffer and wait for more bytes.
                partial.extend_from_slice(&remainder);
                head
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_agent_turn(
    engine: &mut dyn InferenceEngine,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    input: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) {
    let start = Instant::now();
    match session.run_turn_interruptibly(
        engine,
        input,
        tools,
        &BuiltinToolExecutor,
        &BuiltinToolPolicy,
        AgentSettings {
            max_turns,
            max_tokens,
            temperature,
        },
        cancel.as_ref(),
    ) {
        Ok(Some(result)) => {
            print_trace_events(&result.trace_events);
            println!();
            println!("\x1b[1;34m{}\x1b[0m", result.text);
            if result.max_turns_reached {
                println!("\x1b[2m(agent stopped after reaching max turns)\x1b[0m");
            }
            println!();
            let elapsed = start.elapsed();
            println!("\x1b[2m({:.1}s)\x1b[0m", elapsed.as_secs_f64());
            println!();
        }
        Ok(None) => {
            println!();
            println!("\x1b[2m^C (generation cancelled)\x1b[0m");
            cancel.store(false, Ordering::Relaxed);
            println!();
        }
        Err(e) => {
            eprintln!("\x1b[1;31mError: {e:#}\x1b[0m");
            println!();
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn parse_repl_command(input: &str) -> Option<ReplCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let (command, rest) = trimmed
        .split_once(char::is_whitespace)
        .unwrap_or((trimmed, ""));
    let arg = rest.trim();

    match command {
        "/help" => Some(ReplCommand::Help),
        "/reset" | "/clear" => Some(ReplCommand::Reset),
        "/tools" => Some(ReplCommand::Tools),
        "/model" => Some(ReplCommand::Model),
        "/models" => {
            // /models → list; /models <N> → switch attempt (currently a
            // no-op with a pointer to --model-path). Non-numeric args fall
            // through to Unknown so the user sees a clear error rather than
            // silently listing.
            if arg.is_empty() {
                Some(ReplCommand::Models(None))
            } else {
                match arg.parse::<usize>() {
                    Ok(n) => Some(ReplCommand::Models(Some(n))),
                    Err(_) => Some(ReplCommand::Unknown(format!("/models {arg}"))),
                }
            }
        }
        "/stats" => Some(ReplCommand::Stats),
        "/save" => Some(ReplCommand::Save(arg.to_string())),
        "/load" => Some(ReplCommand::Load(arg.to_string())),
        "/export" => Some(ReplCommand::Export(arg.to_string())),
        "/retry" => Some(ReplCommand::Retry),
        "/chat" => Some(ReplCommand::SwitchChat),
        "/agent" => Some(ReplCommand::SwitchAgent),
        "/quit" | "/exit" => Some(ReplCommand::Exit),
        _ => Some(ReplCommand::Unknown(command.to_string())),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn execute_repl_command(
    command: ReplCommand,
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    chat_history: &mut Vec<ChatMessage>,
    session_stats: &mut SessionStats,
    mode: &mut ReplMode,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) -> bool {
    match command {
        ReplCommand::Help => {
            print_repl_help();
            println!();
            true
        }
        ReplCommand::Reset => {
            session.reset();
            chat_history.truncate(1);
            *session_stats = SessionStats::default();
            println!("\x1b[2m(conversation reset)\x1b[0m");
            println!();
            true
        }
        ReplCommand::Tools => {
            print_tools_help(tools);
            println!();
            true
        }
        ReplCommand::Model => {
            println!("\x1b[2mbackend: {backend_name}\x1b[0m");
            println!("\x1b[2mmodel: {}\x1b[0m", engine.model_id());
            println!("\x1b[2mmode: {}\x1b[0m", mode.label());
            println!();
            true
        }
        ReplCommand::Models(maybe_idx) => {
            handle_models_command(maybe_idx);
            println!();
            true
        }
        ReplCommand::Stats => {
            print_session_stats(
                engine.model_id(),
                *mode,
                *session_stats,
                session.stats(),
                chat_history.len().saturating_sub(1),
                max_turns,
                max_tokens,
                temperature,
                tools.len(),
            );
            println!();
            true
        }
        ReplCommand::Save(path) => {
            if path.is_empty() {
                eprintln!("\x1b[1;31mError: usage: /save <path>\x1b[0m");
                println!();
                return true;
            }
            match session.save_to_path(&path) {
                Ok(()) => {
                    println!("\x1b[2m(saved session to {path})\x1b[0m");
                    println!();
                }
                Err(err) => {
                    eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
                    println!();
                }
            }
            true
        }
        ReplCommand::Load(path) => {
            if path.is_empty() {
                eprintln!("\x1b[1;31mError: usage: /load <path>\x1b[0m");
                println!();
                return true;
            }
            match session.replace_from_path(&path) {
                Ok(()) => {
                    println!("\x1b[2m(loaded session from {path})\x1b[0m");
                    println!();
                }
                Err(err) => {
                    eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
                    println!();
                }
            }
            true
        }
        ReplCommand::Export(path_arg) => {
            handle_export_command(engine.model_id(), chat_history, &path_arg);
            println!();
            true
        }
        ReplCommand::Retry => {
            match *mode {
                ReplMode::Chat => handle_retry_command(
                    engine,
                    chat_history,
                    session_stats,
                    max_tokens,
                    temperature,
                    cancel,
                ),
                ReplMode::Agent => {
                    println!("\x1b[2m(/retry only works in chat mode)\x1b[0m");
                    println!();
                }
            }
            true
        }
        ReplCommand::SwitchChat => {
            *mode = ReplMode::Chat;
            println!("\x1b[2m(switched to chat mode — streaming, no tools)\x1b[0m");
            println!();
            true
        }
        ReplCommand::SwitchAgent => {
            *mode = ReplMode::Agent;
            println!("\x1b[2m(switched to agent mode — tool-calling loop, no streaming)\x1b[0m");
            println!();
            true
        }
        ReplCommand::Exit => false,
        ReplCommand::Unknown(command) => {
            eprintln!("\x1b[1;31mError: unknown command {command}. Type /help.\x1b[0m");
            println!();
            true
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// /models — listing + (descoped) switch pointer.
//
// Hot-swap notes (2026-04-20 wave-2 UX polish):
// Mid-REPL engine reinit would require owning the `LoadedInferenceEngine`
// inside the REPL rather than taking `&mut dyn InferenceEngine`. Threading
// the concrete type through the existing API would cascade into
// `lib.rs`/`startup.rs` — outside the "crates/cli/ only" scope guard in the
// wave-2 brief. Ship listing today; track hot-swap separately.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn handle_models_command(maybe_idx: Option<usize>) {
    let snapshots = crate::hub_discovery::discover_hub_snapshots();
    if snapshots.is_empty() {
        println!("No local models found. Try --model-path or ./scripts/run_dflash.sh serve.");
        return;
    }

    if let Some(idx_one_based) = maybe_idx {
        if idx_one_based == 0 || idx_one_based > snapshots.len() {
            eprintln!(
                "\x1b[1;31mError: /models {idx_one_based} out of range (1..={})\x1b[0m",
                snapshots.len()
            );
            return;
        }
        let picked = &snapshots[idx_one_based - 1];
        println!("\x1b[2m(mid-REPL model hot-swap is not yet supported.)\x1b[0m");
        println!(
            "\x1b[2m Restart with:  agent-infer --model-path {}\x1b[0m",
            picked.path.display()
        );
        return;
    }

    // Listing mode.
    println!("Local models (from HuggingFace hub cache):");
    for (i, snap) in snapshots.iter().enumerate() {
        let size = approx_dir_size_gb(&snap.path);
        let family = detect_family(&snap.model_id);
        let size_str = match size {
            Some(gb) => format!("{:>5.1} GB", gb),
            None => "     ?".to_string(),
        };
        println!(
            "  {:>2}. {:<42}  {}  {}",
            i + 1,
            snap.model_id,
            size_str,
            family,
        );
    }
    println!();
    println!(
        "\x1b[2m /models <N>   would switch — currently unsupported; restart with --model-path.\x1b[0m"
    );
}

/// Sum of top-level file sizes under `path`, in GB. Best-effort — returns
/// `None` on any IO error rather than failing the listing.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn approx_dir_size_gb(path: &Path) -> Option<f64> {
    let mut total: u64 = 0;
    // HuggingFace snapshot dirs are typically flat (symlinks into blobs/),
    // so walking one level is sufficient.
    for entry in std::fs::read_dir(path).ok()?.flatten() {
        if let Ok(md) = entry.metadata() {
            total = total.saturating_add(md.len());
        }
    }
    Some((total as f64) / (1024.0 * 1024.0 * 1024.0))
}

/// Infer model family from id substring — mirrors `hub_discovery`'s
/// supported-families list. Returns a short label for the /models table.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn detect_family(model_id: &str) -> &'static str {
    let lc = model_id.to_ascii_lowercase();
    // Order matters — qwen3.5 and qwen2.5 must outrank the bare qwen3 match.
    if lc.contains("qwen3.5") {
        "qwen3.5"
    } else if lc.contains("qwen2.5") {
        "qwen2.5"
    } else if lc.contains("qwen3") {
        "qwen3"
    } else if lc.contains("glm4") || lc.contains("glm-4") {
        "glm4"
    } else {
        "other"
    }
}

// ─────────────────────────────────────────────────────────────────────────
// /export — markdown conversation dump.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn handle_export_command(model_id: &str, history: &[ChatMessage], path_arg: &str) {
    // Count only real turns — system prompt at index 0 is excluded.
    let turns = count_export_turns(history);
    if turns == 0 {
        println!("nothing to export (0 turns)");
        return;
    }

    let md = render_history_markdown(model_id, history);
    let out_path = resolve_export_path(path_arg);
    match std::fs::write(&out_path, md) {
        Ok(()) => {
            let abs = out_path.canonicalize().unwrap_or_else(|_| out_path.clone());
            println!("exported {turns} turns → {}", abs.display());
        }
        Err(err) => {
            eprintln!(
                "\x1b[1;31mError: could not write {}: {err}\x1b[0m",
                out_path.display()
            );
        }
    }
}

/// Count "turns" in the export-spec sense — one user message OR one
/// assistant message. The system prompt at history[0] is excluded.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn count_export_turns(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .skip(1)
        .filter(|m| matches!(m.role, chat::ChatRole::User | chat::ChatRole::Assistant))
        .count()
}

/// Resolve the destination path:
/// - empty arg    → `./agent-infer-<ts>.md` in CWD
/// - dir path     → `<dir>/agent-infer-<ts>.md`
/// - file path    → used verbatim
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn resolve_export_path(path_arg: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let default_name = format!("agent-infer-{ts}.md");

    if path_arg.is_empty() {
        return PathBuf::from(&default_name);
    }
    let p = PathBuf::from(path_arg);
    if p.is_dir() {
        return p.join(&default_name);
    }
    p
}

/// Build the markdown body per the wave-2 brief.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn render_history_markdown(model_id: &str, history: &[ChatMessage]) -> String {
    let ts = iso8601_utc_now();
    let turns = count_export_turns(history);

    let mut out = String::new();
    out.push_str(&format!("# agent-infer conversation — {model_id}\n\n"));
    out.push_str(&format!("> Started: {ts}\n"));
    out.push_str("> Mode: chat\n");
    out.push_str(&format!("> Turns: {turns}\n\n"));

    for msg in history.iter().skip(1) {
        match &msg.role {
            chat::ChatRole::User => {
                out.push_str("## You\n\n");
                out.push_str(msg.content.trim_end());
                out.push_str("\n\n");
            }
            chat::ChatRole::Assistant => {
                out.push_str("## Assistant\n\n");
                out.push_str(msg.content.trim_end());
                out.push_str("\n\n");
            }
            _ => { /* system / tool / other — skip */ }
        }
    }
    out
}

/// Minimal RFC3339 UTC timestamp, no external deps.
/// Format: `YYYY-MM-DDThh:mm:ssZ`. Uses Unix epoch arithmetic. This is
/// adequate for a conversation banner — we don't need leap-second
/// accuracy or subsecond precision.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn iso8601_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_iso8601_utc(secs)
}

/// Pure function separated out for testing — format Unix seconds as
/// `YYYY-MM-DDThh:mm:ssZ`. Handles the 1970-… range with proleptic
/// Gregorian calendar math.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn format_iso8601_utc(unix_secs: u64) -> String {
    // Days since 1970-01-01.
    let days = unix_secs / 86_400;
    let rem = unix_secs % 86_400;
    let h = rem / 3_600;
    let m = (rem % 3_600) / 60;
    let s = rem % 60;

    // Civil-date from day-number algorithm (Howard Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };

    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

// ─────────────────────────────────────────────────────────────────────────
// /retry — drop last assistant reply, re-run prior user turn.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn handle_retry_command(
    engine: &mut dyn InferenceEngine,
    history: &mut Vec<ChatMessage>,
    session_stats: &mut SessionStats,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) {
    // history[0] is the system prompt; anything past that is conversational.
    let last = history.last();
    let Some(last_msg) = last else {
        println!("nothing to retry");
        return;
    };
    match last_msg.role {
        chat::ChatRole::Assistant => {
            // Pop the assistant reply. The prior user message remains in
            // history — `run_chat_turn_with_history` re-runs it in place.
            history.pop();
            // The prior turn's stats were already accumulated. We let the
            // re-run add another turn rather than subtract the prior —
            // simpler and more honest (time was genuinely spent).
            let Some(last_user) = history
                .iter()
                .rev()
                .find(|m| matches!(m.role, chat::ChatRole::User))
            else {
                // Consistency guard: assistant without a preceding user is
                // an invariant violation, but fail soft rather than panic.
                println!("nothing to retry");
                return;
            };
            let user_text = last_user.content.clone();
            if let Err(err) = run_chat_turn_with_history(
                engine,
                history,
                session_stats,
                &user_text,
                max_tokens,
                temperature,
                cancel,
            ) {
                eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
                println!();
            }
        }
        _ => {
            // System or user as last entry — nothing to retry yet.
            println!("nothing to retry");
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_help() {
    println!("Commands:");
    println!("  /help            Show this help");
    println!("  /chat            Switch to streaming chat mode (default)");
    println!("  /agent           Switch to tool-calling agent mode");
    println!("  /reset, /clear   Clear conversation history (both modes)");
    println!("  /tools           Show available tools");
    println!("  /model           Show active model, backend, and current mode");
    println!("  /models [N]      List local models; /models <N> shows switch hint");
    println!("  /stats           Show session token/throughput rollup");
    println!("  /save <path>     Save the current agent session to JSON");
    println!("  /load <path>     Load a saved agent session JSON");
    println!("  /export [path]   Dump chat history to markdown (default: ./agent-infer-<ts>.md)");
    println!("  /retry           Re-run the last user turn (chat mode only)");
    println!("  /quit, /exit     Leave the REPL");
    println!();
    println!("Input:");
    println!("  End a line with `\\` to continue input on the next line.");
    println!("  Ctrl-C cancels a generation in progress; press twice within 2s to exit.");
    println!("  Ctrl-D or /quit exits the REPL.");
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_tools_help(tools: &[ToolDefinition]) {
    println!("Tools:");
    for tool in tools {
        println!("  {}: {}", tool.name, tool.description);
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_trace_events(events: &[AgentTraceEvent]) {
    for event in events {
        match event {
            AgentTraceEvent::AssistantNote(content) => {
                println!("\x1b[2m{}\x1b[0m", content);
            }
            AgentTraceEvent::ToolCall {
                name,
                arguments,
                result,
            } => {
                println!();
                println!(
                    "\x1b[33m[tool: {}]\x1b[0m {}",
                    name,
                    serde_json::to_string(arguments).unwrap_or_default()
                );
                let display_result = if result.len() > 500 {
                    format!("{}... ({} chars total)", &result[..500], result.len())
                } else {
                    result.clone()
                };
                println!("\x1b[36m{}\x1b[0m", display_result);
                println!();
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn print_session_stats(
    model_id: &str,
    mode: ReplMode,
    session: SessionStats,
    agent_stats: AgentSessionStats,
    chat_messages: usize,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tool_count: usize,
) {
    println!("Session stats:");
    println!("  Model:      {model_id}");
    println!("  Mode:       {}", mode.label());
    println!("  Turns:      {}", session.turn_count);
    println!("  Tokens in:  {}", session.prompt_tokens);
    println!("  Tokens out: {}", session.completion_tokens);
    println!("  Avg TPS:    {:.1}", session.avg_tps());
    println!("  Wall time:  {:.1}s", session.total_secs);
    println!();
    println!("Agent session:");
    println!("  messages:    {}", agent_stats.conversation_messages);
    println!("  users:       {}", agent_stats.user_messages);
    println!("  assistants:  {}", agent_stats.assistant_messages);
    println!("  tool results:{}", agent_stats.tool_messages);
    println!("  tool calls:  {}", agent_stats.tool_calls);
    println!("  chars:       {}", agent_stats.content_chars);
    println!("  chat turns:  {}", chat_messages);
    println!("Runtime:");
    println!("  tools enabled: {}", tool_count);
    println!("  max turns: {}", max_turns);
    println!("  max tokens: {}", max_tokens);
    println!("  temperature: {}", temperature);
}

#[cfg(test)]
mod tests {
    use super::{
        ReplCommand, SessionStats, count_export_turns, decode_chunk, detect_family,
        format_iso8601_utc, handle_export_command, handle_models_command, parse_repl_command,
        render_history_markdown, resolve_export_path,
    };
    use chat::ChatMessage;
    use std::time::Duration;

    #[test]
    fn parse_repl_command_supports_aliases_and_paths() {
        assert_eq!(parse_repl_command("/help"), Some(ReplCommand::Help));
        assert_eq!(parse_repl_command("/clear"), Some(ReplCommand::Reset));
        assert_eq!(parse_repl_command("/quit"), Some(ReplCommand::Exit));
        assert_eq!(
            parse_repl_command("/save /tmp/session with spaces.json"),
            Some(ReplCommand::Save(
                "/tmp/session with spaces.json".to_string()
            ))
        );
        assert_eq!(
            parse_repl_command("/load ./snapshots/foo.json"),
            Some(ReplCommand::Load("./snapshots/foo.json".to_string()))
        );
        assert_eq!(
            parse_repl_command("/wat"),
            Some(ReplCommand::Unknown("/wat".to_string()))
        );
    }

    #[test]
    fn parse_repl_command_supports_mode_toggles() {
        assert_eq!(parse_repl_command("/chat"), Some(ReplCommand::SwitchChat));
        assert_eq!(parse_repl_command("/agent"), Some(ReplCommand::SwitchAgent));
    }

    #[test]
    fn parse_repl_command_ignores_normal_chat_input() {
        assert_eq!(parse_repl_command("list files"), None);
    }

    #[test]
    fn decode_chunk_passes_clean_ascii() {
        let mut partial = Vec::new();
        let s = decode_chunk(&mut partial, b"hello");
        assert_eq!(s, "hello");
        assert!(partial.is_empty());
    }

    #[test]
    fn decode_chunk_buffers_incomplete_utf8() {
        // "你" = E4 BD A0; deliver the first two bytes, then the third.
        let mut partial = Vec::new();
        let s1 = decode_chunk(&mut partial, &[0xE4, 0xBD]);
        assert_eq!(s1, "");
        assert_eq!(partial, vec![0xE4, 0xBD]);
        let s2 = decode_chunk(&mut partial, &[0xA0]);
        assert_eq!(s2, "你");
        assert!(partial.is_empty());
    }

    #[test]
    fn decode_chunk_drops_definitively_invalid_bytes() {
        let mut partial = Vec::new();
        // 0xFF is never valid in UTF-8.
        let s = decode_chunk(&mut partial, &[b'a', 0xFF, b'b']);
        assert!(s.starts_with('a'));
        assert!(s.ends_with('b'));
        assert!(partial.is_empty());
    }

    // ── Wave-2 polish: new command parse coverage ───────────────────────

    #[test]
    fn parse_retry_command() {
        assert_eq!(parse_repl_command("/retry"), Some(ReplCommand::Retry));
        assert_eq!(
            parse_repl_command("  /retry  "),
            Some(ReplCommand::Retry),
            "leading/trailing whitespace should not disturb parsing"
        );
    }

    #[test]
    fn parse_models_command_with_and_without_index() {
        assert_eq!(
            parse_repl_command("/models"),
            Some(ReplCommand::Models(None))
        );
        assert_eq!(
            parse_repl_command("/models 3"),
            Some(ReplCommand::Models(Some(3)))
        );
        // Non-numeric -> Unknown, not silently a listing.
        assert_eq!(
            parse_repl_command("/models foo"),
            Some(ReplCommand::Unknown("/models foo".to_string()))
        );
    }

    #[test]
    fn parse_export_command() {
        assert_eq!(
            parse_repl_command("/export"),
            Some(ReplCommand::Export("".to_string()))
        );
        assert_eq!(
            parse_repl_command("/export /tmp/out.md"),
            Some(ReplCommand::Export("/tmp/out.md".to_string()))
        );
    }

    // ── /export — markdown dump ─────────────────────────────────────────

    #[test]
    fn export_markdown_empty_history() {
        // Empty = only the system prompt with no user/assistant turns.
        let history = vec![ChatMessage::system("You are helpful.")];
        assert_eq!(count_export_turns(&history), 0);

        // Exporting to a nonexistent path should NOT create the file —
        // 0 turns short-circuits the write. Use a path we can sanity-
        // check post-call.
        let tmp = std::env::temp_dir().join(format!(
            "agent-infer-export-empty-{}.md",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        handle_export_command("dummy/model", &history, tmp.to_str().unwrap());
        assert!(!tmp.exists(), "no file should be written for empty history");
    }

    #[test]
    fn export_markdown_writes_turns() {
        let history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there", vec![]),
            ChatMessage::user("bye"),
            ChatMessage::assistant("see ya", vec![]),
        ];
        assert_eq!(count_export_turns(&history), 4);

        let md = render_history_markdown("Qwen/Qwen3-4B", &history);
        assert!(md.starts_with("# agent-infer conversation — Qwen/Qwen3-4B"));
        assert!(md.contains("> Turns: 4"));
        assert!(md.contains("## You\n\nhello"));
        assert!(md.contains("## Assistant\n\nhi there"));
        assert!(md.contains("## You\n\nbye"));
        assert!(md.contains("## Assistant\n\nsee ya"));
        // System prompt must NOT appear in the body.
        assert!(!md.contains("\nsys\n"));

        // Round-trip through handle_export_command to a temp file.
        let tmp = std::env::temp_dir().join(format!(
            "agent-infer-export-writes-{}.md",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        handle_export_command("Qwen/Qwen3-4B", &history, tmp.to_str().unwrap());
        let written = std::fs::read_to_string(&tmp).expect("file written");
        assert!(written.contains("## Assistant\n\nsee ya"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn export_default_path_in_cwd_with_timestamp() {
        let p = resolve_export_path("");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("agent-infer-") && name.ends_with(".md"),
            "default filename should be agent-infer-<ts>.md, got {name}"
        );
        assert!(
            p.parent()
                .map(|pp| pp.as_os_str().is_empty())
                .unwrap_or(true),
            "default path should be relative to CWD: {}",
            p.display()
        );
    }

    #[test]
    fn export_resolves_dir_path_to_dir_slash_default() {
        // Use the system temp dir, which is guaranteed to be a directory.
        let tmp = std::env::temp_dir();
        let p = resolve_export_path(tmp.to_str().unwrap());
        assert!(
            p.parent() == Some(tmp.as_path()),
            "dir arg should prepend the dir: {}",
            p.display()
        );
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("agent-infer-") && name.ends_with(".md"));
    }

    // ── /stats session accumulator ──────────────────────────────────────

    #[test]
    fn stats_accumulator_sums_across_turns() {
        let mut s = SessionStats::default();
        s.record_turn(10, 20, Duration::from_secs(1));
        s.record_turn(5, 40, Duration::from_secs(2));
        assert_eq!(s.turn_count, 2);
        assert_eq!(s.prompt_tokens, 15);
        assert_eq!(s.completion_tokens, 60);
        assert!((s.total_secs - 3.0).abs() < 1e-9);
        // Turn 1: 20 tok / 1s = 20 tok/s; weight 20 → numer contrib 400
        // Turn 2: 40 tok / 2s = 20 tok/s; weight 40 → numer contrib 800
        // avg = (400 + 800) / 60 = 20
        assert!((s.avg_tps() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn stats_avg_tps_weighted_not_plain_mean() {
        // A 1-tok 1s turn (1 tok/s) should be dominated by a 1000-tok 10s
        // turn (100 tok/s). Plain mean would be ~50.5; weighted ≈ 99.9.
        let mut s = SessionStats::default();
        s.record_turn(1, 1, Duration::from_secs(1));
        s.record_turn(1, 1000, Duration::from_secs(10));
        let avg = s.avg_tps();
        assert!(
            avg > 99.0,
            "weighted avg should lean toward the big turn, got {avg}"
        );
    }

    #[test]
    fn stats_accumulator_handles_zero_elapsed() {
        // A turn with zero elapsed time (should never happen at runtime
        // but the math must not NaN out). weighted_rate_numer stays 0.
        let mut s = SessionStats::default();
        s.record_turn(5, 10, Duration::ZERO);
        assert_eq!(s.turn_count, 1);
        assert_eq!(s.avg_tps(), 0.0);
    }

    // ── /models offline fallback ────────────────────────────────────────

    #[test]
    fn models_command_offline_message() {
        // Point HF cache at an empty temp dir so discover_hub_snapshots
        // returns zero entries. handle_models_command should then emit
        // the offline hint without touching the filesystem further.
        let tmp =
            std::env::temp_dir().join(format!("agent-infer-empty-hub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Use HUGGINGFACE_HUB_CACHE which hub_discovery::hub_cache_root
        // honours with highest priority.
        //
        // Safety: this test is a single-threaded unit test; setting an env
        // var from this process scope is safe because no other test in
        // this file reads HUGGINGFACE_HUB_CACHE concurrently.
        let prior = std::env::var_os("HUGGINGFACE_HUB_CACHE");
        unsafe {
            std::env::set_var("HUGGINGFACE_HUB_CACHE", &tmp);
        }

        // Smoke-test: should return without panicking or writing a file.
        handle_models_command(None);
        handle_models_command(Some(1));

        unsafe {
            match prior {
                Some(v) => std::env::set_var("HUGGINGFACE_HUB_CACHE", v),
                None => std::env::remove_var("HUGGINGFACE_HUB_CACHE"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Misc helpers ────────────────────────────────────────────────────

    #[test]
    fn detect_family_matches_known_ids() {
        assert_eq!(detect_family("Qwen/Qwen3-4B"), "qwen3");
        assert_eq!(detect_family("Qwen/Qwen3.5-4B"), "qwen3.5");
        assert_eq!(detect_family("Qwen/Qwen2.5-7B"), "qwen2.5");
        assert_eq!(detect_family("THUDM/glm-4-9b-chat"), "glm4");
        assert_eq!(detect_family("something-else"), "other");
    }

    #[test]
    fn format_iso8601_utc_known_epochs() {
        // Unix epoch.
        assert_eq!(format_iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2026-04-20 00:00:00 UTC.
        assert_eq!(format_iso8601_utc(1_776_643_200), "2026-04-20T00:00:00Z");
        // Pre-2000 sanity: 1999-12-31 23:59:59 UTC = 946_684_799.
        assert_eq!(format_iso8601_utc(946_684_799), "1999-12-31T23:59:59Z");
    }
}
