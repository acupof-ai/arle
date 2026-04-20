#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::path::PathBuf;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::Arc;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::OnceLock;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    Exit,
    Unknown(String),
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
            mode,
            max_turns,
            max_tokens,
            temperature,
        ));
    }

    match *mode {
        ReplMode::Chat => {
            run_chat_turn(engine, chat_history, input, max_tokens, temperature, cancel)?;
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
/// consistent), and the worker thread is left to drain into a dropped
/// receiver — safe for `UnboundedSender` and matches the e2e pattern.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_chat_turn(
    engine: &mut dyn InferenceEngine,
    history: &mut Vec<ChatMessage>,
    user_input: &str,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
) -> Result<()> {
    history.push(ChatMessage::user(user_input));

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

    let (tx, mut rx) = mpsc::unbounded_channel::<CompletionStreamDelta>();
    let start = Instant::now();
    let mut accumulated = String::new();
    // Buffer for incomplete UTF-8 sequences arriving across deltas.
    let mut partial_bytes: Vec<u8> = Vec::new();
    let mut cancelled = false;
    let mut stream_err: Option<anyhow::Error> = None;

    // Print a faint marker so the user sees we're generating before the
    // first token arrives. Cleared on first delta.
    print!("\x1b[34m");
    let _ = io::stdout().flush();

    std::thread::scope(|s| {
        let worker = s.spawn(|| engine.complete_stream(req, tx));

        loop {
            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                break;
            }
            match rx.try_recv() {
                Ok(delta) => {
                    if !delta.text_delta.is_empty() {
                        let chunk = decode_chunk(&mut partial_bytes, delta.text_delta.as_bytes());
                        if !chunk.is_empty() {
                            accumulated.push_str(&chunk);
                            print!("{}", chunk);
                            let _ = io::stdout().flush();
                        }
                    }
                    if delta.finish_reason.is_some() {
                        break;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        // Worker may still be running on cancel; we don't try to interrupt
        // the engine mid-decode (no API for that). Wait for it to finish so
        // the &mut engine borrow returns cleanly.
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

    print!("\x1b[0m");
    let _ = io::stdout().flush();

    if cancelled {
        println!();
        println!("\x1b[2m^C (generation cancelled)\x1b[0m");
        // Reset cancel flag now that we've handled it.
        cancel.store(false, Ordering::Relaxed);
    } else {
        println!();
        let elapsed = start.elapsed();
        println!("\x1b[2m({:.1}s)\x1b[0m", elapsed.as_secs_f64());
    }
    println!();

    // Always record the (possibly partial) assistant turn so subsequent
    // turns see consistent context.
    history.push(ChatMessage::assistant(&accumulated, vec![]));

    if let Some(err) = stream_err {
        eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
        println!();
    }

    Ok(())
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
) {
    let start = Instant::now();
    match session.run_turn(
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
    ) {
        Ok(result) => {
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
        "/stats" => Some(ReplCommand::Stats),
        "/save" => Some(ReplCommand::Save(arg.to_string())),
        "/load" => Some(ReplCommand::Load(arg.to_string())),
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
    mode: &mut ReplMode,
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
            chat_history.truncate(1);
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
        ReplCommand::Stats => {
            print_session_stats(
                session.stats(),
                chat_history.len().saturating_sub(1),
                max_turns,
                max_tokens,
                temperature,
                tools.len(),
                *mode,
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_help() {
    println!("Commands:");
    println!("  /help            Show this help");
    println!("  /chat            Switch to streaming chat mode (default)");
    println!("  /agent           Switch to tool-calling agent mode");
    println!("  /reset, /clear   Clear conversation history (both modes)");
    println!("  /tools           Show available tools");
    println!("  /model           Show active model, backend, and current mode");
    println!("  /stats           Show session and runtime settings");
    println!("  /save <path>     Save the current agent session to JSON");
    println!("  /load <path>     Load a saved agent session JSON");
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
    stats: AgentSessionStats,
    chat_messages: usize,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tool_count: usize,
    mode: ReplMode,
) {
    println!("Session:");
    println!("  mode: {}", mode.label());
    println!("  agent messages: {}", stats.conversation_messages);
    println!("  agent users: {}", stats.user_messages);
    println!("  agent assistants: {}", stats.assistant_messages);
    println!("  agent tool results: {}", stats.tool_messages);
    println!("  agent tool calls: {}", stats.tool_calls);
    println!("  agent content chars: {}", stats.content_chars);
    println!("  chat messages: {}", chat_messages);
    println!("Runtime:");
    println!("  tools enabled: {}", tool_count);
    println!("  max turns: {}", max_turns);
    println!("  max tokens: {}", max_tokens);
    println!("  temperature: {}", temperature);
}

#[cfg(test)]
mod tests {
    use super::{ReplCommand, decode_chunk, parse_repl_command};

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
}
