#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::cell::RefCell;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::{self, BufRead, IsTerminal, Read, Write};
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
use crate::args::RunArgs;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use agent::{
    AgentSession, AgentSessionStats, AgentSettings, AgentTraceEvent, AgentTurnCallbacks,
    ToolExecutor, ToolPolicy,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use chat::{ChatMessage, ParsedAssistantResponse, ToolCall, ToolDefinition};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::InferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::DefaultEditor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::error::ReadlineError;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use serde::Serialize;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use tools::{BuiltinToolPolicyHooks, builtin_tools, execute_tool_call};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const REPL_PROMPT: &str = "\x1b[1;35m> \x1b[0m";

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Serialize)]
struct OneShotOutput {
    model_id: String,
    backend: String,
    text: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    tool_calls_executed: usize,
    max_turns_reached: bool,
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
    /// `/models` with no arg lists discovered hub snapshots; `/models N` is
    /// reserved for engine hot-swap which is currently unsupported (owner
    /// pattern requires an owned `LoadedInferenceEngine` instead of the
    /// `&mut dyn InferenceEngine` threaded through `run_repl`). When N is
    /// supplied we print a friendly pointer to restart with `--model-path`.
    Models(Option<usize>),
    /// `/export` → default path; `/export <path>` → explicit path/dir.
    Export(String),
    Exit,
    Unknown(String),
}

/// Per-session rolling accumulators for `/stats` enrichment.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Default, Clone, Copy)]
struct SessionStats {
    /// Number of completed turns.
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
    /// Number of tool calls executed during the session.
    tool_calls: usize,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl SessionStats {
    fn record_turn(
        &mut self,
        prompt_tokens: u64,
        completion_tokens: u64,
        tool_calls: usize,
        elapsed: Duration,
    ) {
        self.turn_count = self.turn_count.saturating_add(1);
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
        self.tool_calls = self.tool_calls.saturating_add(tool_calls);
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn tool_definitions(enabled: bool) -> Vec<ToolDefinition> {
    if enabled {
        builtin_tools()
            .into_iter()
            .map(|tool| tool.to_definition())
            .collect()
    } else {
        Vec::new()
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
    tools_enabled: bool,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_interactive_repl(
            engine,
            backend_name,
            max_turns,
            max_tokens,
            temperature,
            tools_enabled,
        );
    }

    run_piped_repl(
        engine,
        backend_name,
        max_turns,
        max_tokens,
        temperature,
        tools_enabled,
    )
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_one_shot(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    run_args: &RunArgs,
    tools_enabled: bool,
) -> Result<()> {
    let prompt = resolve_one_shot_prompt(run_args)?;
    anyhow::ensure!(
        !prompt.trim().is_empty(),
        "one-shot prompt is empty; pass --prompt or pipe non-empty stdin to --stdin"
    );

    let tools = tool_definitions(tools_enabled);
    let mut session = AgentSession::new();
    let result = session.run_turn(
        engine,
        &prompt,
        &tools,
        &BuiltinToolExecutor,
        &BuiltinToolPolicy,
        AgentSettings {
            max_turns,
            max_tokens,
            temperature,
        },
    )?;

    let output = OneShotOutput {
        model_id: engine.model_id().to_string(),
        backend: backend_name.to_string(),
        text: result.text,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        total_tokens: result
            .prompt_tokens
            .saturating_add(result.completion_tokens),
        tool_calls_executed: result.tool_calls_executed,
        max_turns_reached: result.max_turns_reached,
    };

    if run_args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("{}", output.text);
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_banner(
    engine: &dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) {
    println!();
    println!("=== ARLE REPL ===");
    println!("Model: {}", engine.model_id());
    println!("Backend: {}", backend_name);
    println!("Mode: agent");
    println!("Tools available: {}", tools.len());
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
    Some(PathBuf::from(home).join(".arle-history"))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn legacy_history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".agent-infer-history"))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn migrate_legacy_history(path: &Path) {
    if path.exists() {
        return;
    }
    let Some(legacy_path) = legacy_history_path() else {
        return;
    };
    if !legacy_path.exists() {
        return;
    }
    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        log::debug!("Skipping history migration to {}: {err}", path.display());
        return;
    }
    if let Err(err) = std::fs::copy(&legacy_path, path) {
        log::debug!(
            "Skipping history migration from {} to {}: {err}",
            legacy_path.display(),
            path.display()
        );
    }
}

/// Read one logical input from rustyline. Lines ending with `\` are joined
/// with `\n` until a line without a trailing backslash is entered.
///
/// Returns `Ok(Some(_))` for a complete input, `Ok(None)` for EOF, and
/// propagates `ReadlineError::Interrupted` upward (REPL turns it into ^C).
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn read_multiline(editor: &mut DefaultEditor) -> Result<Option<String>, ReadlineError> {
    let mut accumulated: Option<String> = None;
    let cont = "\x1b[2m… \x1b[0m";

    loop {
        let prompt = if accumulated.is_some() {
            cont
        } else {
            REPL_PROMPT
        };
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
fn read_multiline_piped<R: BufRead>(
    reader: &mut R,
    prompt: Option<&str>,
) -> io::Result<Option<String>> {
    let mut stdout = io::stdout();
    if let Some(prompt) = prompt {
        print!("{prompt}");
        stdout.flush()?;
    }

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
            if let Some(prompt) = prompt {
                print!("{prompt}");
                stdout.flush()?;
            }
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
    tools_enabled: bool,
) -> Result<()> {
    let tools = tool_definitions(tools_enabled);
    let mut session = AgentSession::new();
    let mut session_stats = SessionStats::default();
    let mut editor = DefaultEditor::new()?;
    let history = history_path();

    if let Some(path) = history.as_ref() {
        migrate_legacy_history(path);
    }
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
    );

    loop {
        // Make sure the cancel flag is clear at each prompt — it may have
        // been left set by a Ctrl-C that landed at a blank prompt.
        cancel.store(false, Ordering::Relaxed);

        match read_multiline(&mut editor) {
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
                    &mut session_stats,
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
    tools_enabled: bool,
) -> Result<()> {
    let tools = tool_definitions(tools_enabled);
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut session = AgentSession::new();
    let mut session_stats = SessionStats::default();

    let (cancel, exit) = install_ctrlc_handler();
    cancel.store(false, Ordering::Relaxed);
    exit.store(false, Ordering::Relaxed);

    loop {
        cancel.store(false, Ordering::Relaxed);
        match read_multiline_piped(&mut reader, None)? {
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
                    &mut session_stats,
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
fn resolve_one_shot_prompt(run_args: &RunArgs) -> Result<String> {
    match (&run_args.prompt, run_args.stdin) {
        (Some(prompt), false) => Ok(prompt.clone()),
        (None, true) => {
            let mut input = String::new();
            io::stdin().read_to_string(&mut input)?;
            Ok(input)
        }
        _ => anyhow::bail!("run one-shot mode requires exactly one of --prompt or --stdin"),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn handle_repl_input(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    session_stats: &mut SessionStats,
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
            session_stats,
            max_turns,
            max_tokens,
            temperature,
        ));
    }

    run_agent_turn(
        engine,
        tools,
        session,
        session_stats,
        input,
        max_turns,
        max_tokens,
        temperature,
        cancel,
    );

    Ok(true)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_agent_turn(
    engine: &mut dyn InferenceEngine,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    session_stats: &mut SessionStats,
    input: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) {
    let start = Instant::now();
    let color_on = io::stdout().is_terminal();
    let render_state = RefCell::new((false, false));
    let tps_meter = RefCell::new(crate::tps::TpsMeter::new());
    let mut on_text_chunk = |chunk: &str| {
        if chunk.is_empty() {
            return;
        }
        let mut state = render_state.borrow_mut();
        tps_meter.borrow_mut().hide_before_chunk();
        if color_on && !state.0 {
            print!("\x1b[1;34m");
            let _ = io::stdout().flush();
            state.0 = true;
        }
        print!("{chunk}");
        let _ = io::stdout().flush();
        tps_meter.borrow_mut().record_chunk(chunk.len());
        state.1 = true;
    };
    let mut on_trace_event = |event: &AgentTraceEvent| {
        let AgentTraceEvent::ToolCall {
            name,
            arguments,
            result,
        } = event
        else {
            // AssistantNote text was already streamed via on_text_chunk —
            // re-emitting it here would just duplicate.
            return;
        };
        let mut state = render_state.borrow_mut();
        tps_meter.borrow_mut().hide_before_chunk();
        if color_on && state.0 {
            print!("\x1b[0m");
            state.0 = false;
        }
        if state.1 {
            println!();
            state.1 = false;
        }
        let line = format_tool_call_line(name, arguments, result);
        println!("{line}");
        let _ = io::stdout().flush();
    };
    match session.run_turn_interruptibly_with_callbacks(
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
        AgentTurnCallbacks {
            on_text_chunk: Some(&mut on_text_chunk),
            on_trace_event: Some(&mut on_trace_event),
        },
    ) {
        Ok(Some(result)) => {
            let (color_open, streamed_any) = *render_state.borrow();
            if color_on && color_open {
                print!("\x1b[0m");
                let _ = io::stdout().flush();
            }
            if streamed_any {
                println!();
            }
            if !streamed_any {
                println!("\x1b[1;34m{}\x1b[0m", result.text);
            }
            if result.max_turns_reached {
                println!("\x1b[2m(agent stopped after reaching max turns)\x1b[0m");
            }
            let elapsed = start.elapsed();
            tps_meter
                .borrow_mut()
                .print_final(Some(result.completion_tokens));
            session_stats.record_turn(
                result.prompt_tokens,
                result.completion_tokens,
                result.tool_calls_executed,
                elapsed,
            );
            println!();
        }
        Ok(None) => {
            let (color_open, streamed_any) = *render_state.borrow();
            if color_on && color_open {
                print!("\x1b[0m");
                let _ = io::stdout().flush();
            }
            if streamed_any {
                println!();
            }
            println!();
            println!("\x1b[2m^C (generation cancelled)\x1b[0m");
            cancel.store(false, Ordering::Relaxed);
            println!();
        }
        Err(e) => {
            let (color_open, streamed_any) = *render_state.borrow();
            if color_on && color_open {
                print!("\x1b[0m");
                let _ = io::stdout().flush();
            }
            if streamed_any {
                println!();
            }
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
    session_stats: &mut SessionStats,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> bool {
    match command {
        ReplCommand::Help => {
            print_repl_help();
            println!();
            true
        }
        ReplCommand::Reset => {
            session.reset();
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
                *session_stats,
                session.stats(),
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
            handle_export_command(engine.model_id(), session.messages(), &path_arg);
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
// Inline tool-call rendering — one dim line per call:
//   `  ⏵ name(args) → result`
// Args/result are aggressively truncated so the line stays scannable.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const TOOL_ARGS_MAX: usize = 60;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const TOOL_RESULT_MAX: usize = 80;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn format_tool_call_line(name: &str, arguments: &serde_json::Value, result: &str) -> String {
    let args = brief_tool_args(arguments);
    let result = brief_tool_result(result);
    format!("\x1b[2m  ⏵ {name}({args}) → {result}\x1b[0m")
}

/// Pull a one-line argument summary out of a tool call's JSON arguments.
/// Prefers a single dominant scalar field (`command`, `code`, `path`, …)
/// when one is present; falls back to compact JSON otherwise.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn brief_tool_args(arguments: &serde_json::Value) -> String {
    const PREFERRED: &[&str] = &[
        "command", "code", "path", "file", "query", "pattern", "url", "input",
    ];
    let raw = match arguments {
        serde_json::Value::Object(map) => PREFERRED
            .iter()
            .find_map(|key| map.get(*key).and_then(scalar_to_string))
            .or_else(|| {
                if map.len() == 1 {
                    map.values().next().and_then(scalar_to_string)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| serde_json::to_string(arguments).unwrap_or_default()),
        serde_json::Value::Null => String::new(),
        other => scalar_to_string(other).unwrap_or_else(|| other.to_string()),
    };
    truncate_one_line(&raw, TOOL_ARGS_MAX)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn brief_tool_result(result: &str) -> String {
    let trimmed = result.trim();
    if trimmed.is_empty() {
        return "(empty)".to_string();
    }
    truncate_one_line(trimmed, TOOL_RESULT_MAX)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn scalar_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Collapse newlines/tabs to spaces, squeeze runs of whitespace, and cap at
/// `max` chars (counting Unicode scalars, not bytes — emoji-safe).
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn truncate_one_line(s: &str, max: usize) -> String {
    let mut out = String::with_capacity(s.len().min(max + 1));
    let mut prev_space = false;
    for ch in s.chars() {
        let c = if ch.is_whitespace() { ' ' } else { ch };
        if c == ' ' {
            if prev_space {
                continue;
            }
            prev_space = true;
        } else {
            prev_space = false;
        }
        out.push(c);
    }
    let trimmed = out.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
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
            "\x1b[2m Restart with:  arle --model-path {}\x1b[0m",
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
/// - empty arg    → `./arle-<ts>.md` in CWD
/// - dir path     → `<dir>/arle-<ts>.md`
/// - file path    → used verbatim
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn resolve_export_path(path_arg: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let default_name = format!("arle-{ts}.md");

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
    out.push_str(&format!("# ARLE conversation — {model_id}\n\n"));
    out.push_str(&format!("> Started: {ts}\n"));
    out.push_str("> Mode: agent\n");
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_help() {
    println!("Commands:");
    println!("  /help            Show this help");
    println!("  /reset, /clear   Clear conversation history");
    println!("  /tools           Show available built-in tools");
    println!("  /model           Show the active model and backend");
    println!("  /models [N]      List local models; /models <N> shows switch hint");
    println!("  /stats           Show session token/throughput rollup");
    println!("  /save <path>     Save the current agent session to JSON");
    println!("  /load <path>     Load a saved agent session JSON");
    println!("  /export [path]   Dump the conversation to markdown (default: ./arle-<ts>.md)");
    println!("  /quit, /exit     Leave the REPL");
    println!();
    println!("Input:");
    println!("  End a line with `\\` to continue input on the next line.");
    println!("  Ctrl-C cancels a generation in progress; press twice within 2s to exit.");
    println!("  Ctrl-D or /quit exits the REPL.");
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_tools_help(tools: &[ToolDefinition]) {
    println!("Built-in tools:");
    for tool in tools {
        println!("  {}: {}", tool.name, tool.description);
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn print_session_stats(
    model_id: &str,
    session: SessionStats,
    agent_stats: AgentSessionStats,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tool_count: usize,
) {
    println!("Session stats:");
    println!("  Model:      {model_id}");
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
    println!("  chars:       {}", agent_stats.content_chars);
    println!("Runtime:");
    println!("  tools: {}", tool_count);
    println!("  tool calls: {}", session.tool_calls);
    println!("  max turns: {}", max_turns);
    println!("  max tokens: {}", max_tokens);
    println!("  temperature: {}", temperature);
}

#[cfg(test)]
mod tests {
    use super::{
        ReplCommand, SessionStats, brief_tool_args, brief_tool_result, count_export_turns,
        detect_family, format_iso8601_utc, format_tool_call_line, handle_export_command,
        handle_models_command, parse_repl_command, render_history_markdown, resolve_export_path,
        truncate_one_line,
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
    fn parse_repl_command_ignores_normal_input() {
        assert_eq!(parse_repl_command("list files"), None);
    }

    #[test]
    fn parse_removed_mode_commands_as_unknown() {
        assert_eq!(
            parse_repl_command("/chat"),
            Some(ReplCommand::Unknown("/chat".to_string()))
        );
        assert_eq!(
            parse_repl_command("/agent"),
            Some(ReplCommand::Unknown("/agent".to_string()))
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
        let tmp = std::env::temp_dir().join(format!("arle-export-empty-{}.md", std::process::id()));
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
        assert!(md.starts_with("# ARLE conversation — Qwen/Qwen3-4B"));
        assert!(md.contains("> Mode: agent"));
        assert!(md.contains("> Turns: 4"));
        assert!(md.contains("## You\n\nhello"));
        assert!(md.contains("## Assistant\n\nhi there"));
        assert!(md.contains("## You\n\nbye"));
        assert!(md.contains("## Assistant\n\nsee ya"));
        // System prompt must NOT appear in the body.
        assert!(!md.contains("\nsys\n"));

        // Round-trip through handle_export_command to a temp file.
        let tmp =
            std::env::temp_dir().join(format!("arle-export-writes-{}.md", std::process::id()));
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
            name.starts_with("arle-") && name.ends_with(".md"),
            "default filename should be arle-<ts>.md, got {name}"
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
        assert!(name.starts_with("arle-") && name.ends_with(".md"));
    }

    // ── /stats session accumulator ──────────────────────────────────────

    #[test]
    fn stats_accumulator_sums_across_turns() {
        let mut s = SessionStats::default();
        s.record_turn(10, 20, 0, Duration::from_secs(1));
        s.record_turn(5, 40, 0, Duration::from_secs(2));
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
        s.record_turn(1, 1, 0, Duration::from_secs(1));
        s.record_turn(1, 1000, 0, Duration::from_secs(10));
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
        s.record_turn(5, 10, 0, Duration::ZERO);
        assert_eq!(s.turn_count, 1);
        assert_eq!(s.avg_tps(), 0.0);
    }

    // ── /models offline fallback ────────────────────────────────────────

    #[test]
    fn models_command_offline_message() {
        // Point HF cache at an empty temp dir so discover_hub_snapshots
        // returns zero entries. handle_models_command should then emit
        // the offline hint without touching the filesystem further.
        let tmp = std::env::temp_dir().join(format!("arle-empty-hub-{}", std::process::id()));
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

    // ── Inline tool-call rendering ──────────────────────────────────────

    #[test]
    fn brief_tool_args_prefers_known_scalar_field() {
        let args = serde_json::json!({ "command": "git status", "cwd": "/repo" });
        assert_eq!(brief_tool_args(&args), "git status");
    }

    #[test]
    fn brief_tool_args_falls_back_to_compact_json_for_unknown_shape() {
        let args = serde_json::json!({ "alpha": 1, "beta": 2 });
        let out = brief_tool_args(&args);
        // Unknown shape with multiple non-preferred fields → compact JSON.
        assert!(out.contains("\"alpha\""));
        assert!(out.contains("\"beta\""));
    }

    #[test]
    fn brief_tool_args_truncates_long_command() {
        let long = "echo ".to_string() + &"x".repeat(200);
        let args = serde_json::json!({ "command": long });
        let out = brief_tool_args(&args);
        assert!(out.ends_with('…'), "expected ellipsis suffix, got: {out}");
        assert!(out.chars().count() <= 60);
    }

    #[test]
    fn brief_tool_result_collapses_multiline_output() {
        let raw = "On branch main\n\nmodified: foo.rs\nmodified: bar.rs\n";
        let out = brief_tool_result(raw);
        assert!(!out.contains('\n'));
        assert!(out.starts_with("On branch main"));
    }

    #[test]
    fn brief_tool_result_handles_empty() {
        assert_eq!(brief_tool_result("   \n\t "), "(empty)");
    }

    #[test]
    fn truncate_one_line_squeezes_repeated_whitespace() {
        assert_eq!(truncate_one_line("a   b\t\n c", 32), "a b c");
    }

    #[test]
    fn truncate_one_line_caps_at_unicode_chars_not_bytes() {
        // 4 emoji = 4 chars, 16 bytes — must cap by char count.
        let out = truncate_one_line("🚀🚀🚀🚀🚀🚀", 4);
        assert_eq!(out.chars().count(), 4);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn format_tool_call_line_has_dim_style_and_indent() {
        let args = serde_json::json!({ "command": "ls" });
        let line = format_tool_call_line("shell", &args, "Cargo.toml\nsrc\n");
        // Dim ANSI prefix, two-space indent, arrow markers.
        assert!(line.starts_with("\x1b[2m  ⏵ shell(ls) → "));
        assert!(line.ends_with("\x1b[0m"));
        assert!(line.contains("Cargo.toml"));
        // Result was multi-line — must be flattened.
        assert!(!line.contains("Cargo.toml\nsrc"));
    }
}
