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
    AgentTurnResult, ContentBlock, MessageContent, SubTurnRecord, TerminalState, TokensRecord,
    ToolExecutionMetadata, ToolExecutor, ToolPolicy, ToolUsage, TrajectoryMessage, TrajectoryRole,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use anyhow::{Context, Result};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use chat::{ChatMessage, ParsedAssistantResponse, ToolCall, ToolDefinition};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer_api::{
    ChatPromptImage, ChatPromptMessage, CompletionRequest, FinishReason, InferenceEngine,
    MultimodalChatRequest, SamplingParams,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::DefaultEditor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::error::ReadlineError;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use serde::Serialize;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use tools::{
    BuiltinToolPolicyHooks, builtin_tools, execute_tool_call, execute_tool_call_with_metadata,
};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use crate::trace::TraceWriter;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const REPL_PROMPT: &str = "\x1b[1;35m> \x1b[0m";

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const MAX_CLI_IMAGE_BYTES: usize = 32 * 1024 * 1024;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Serialize)]
struct OneShotOutput {
    model_id: String,
    backend: String,
    text: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    image_count: usize,
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
    Image(String),
    Export(String),
    Exit,
    Unknown(String),
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[derive(Debug, Default, Clone, Copy)]
struct SessionStats {
    turn_count: usize,
    /// Falls back to chars/4 for the most recent user message when usage was
    /// unset (e.g. backend without usage reporting).
    prompt_tokens: u64,
    /// Falls back to the rolling TpsMeter count when usage was unset.
    completion_tokens: u64,
    /// Weighted TPS accumulator: sum of (tokens · tokens/elapsed) across
    /// turns; divided by total tokens for the weighted average.
    /// Stored separately rather than as a running avg because a simple mean
    /// over turns would treat a 1-token turn and a 1000-token turn equally.
    weighted_rate_numer: f64,
    total_secs: f64,
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

    fn execute_with_metadata(&self, tool_call: &ToolCall) -> (String, ToolExecutionMetadata) {
        execute_tool_call_with_metadata(tool_call)
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
static CANCEL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Last Ctrl-C timestamp, for double-tap exit detection.
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tools_enabled: bool,
    trace: Option<&TraceWriter>,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_interactive_repl(
            engine,
            backend_name,
            max_turns,
            max_tokens,
            temperature,
            tools_enabled,
            trace,
        );
    }

    run_piped_repl(
        engine,
        backend_name,
        max_turns,
        max_tokens,
        temperature,
        tools_enabled,
        trace,
    )
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_one_shot(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    run_args: &RunArgs,
    tools_enabled: bool,
    trace: Option<&TraceWriter>,
) -> Result<()> {
    let prompt = resolve_one_shot_prompt(run_args)?;
    anyhow::ensure!(
        !prompt.trim().is_empty(),
        "one-shot prompt is empty; pass --prompt or pipe non-empty stdin to --stdin"
    );
    anyhow::ensure!(
        run_args.image.is_empty() || supports_cli_images(backend_name),
        "--image requires a VLM chat backend; current backend is {backend_name}"
    );
    let images = resolve_run_images(run_args)?;

    let tools = tool_definitions(tools_enabled);
    let mut session = if is_direct_chat_backend(backend_name) {
        AgentSession::with_system_prompt("")
    } else {
        AgentSession::new()
    };
    let result = if is_direct_chat_backend(backend_name) {
        run_direct_chat_completion(
            engine,
            &mut session,
            &prompt,
            &images,
            max_tokens,
            temperature,
        )?
    } else {
        anyhow::ensure!(
            images.is_empty(),
            "--image requires a direct VLM chat backend; current backend is {backend_name}"
        );
        session.run_turn(
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
        )?
    };

    if let Some(writer) = trace {
        writer.write_turn(engine.model_id(), backend_name, &prompt, &result);
    }

    let output = OneShotOutput {
        model_id: engine.model_id().to_string(),
        backend: backend_name.to_string(),
        text: result.text,
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        total_tokens: result
            .prompt_tokens
            .saturating_add(result.completion_tokens),
        image_count: images.len(),
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

/// One dim config line for the REPL. Model / backend / commands / Ctrl-C are
/// already covered by the loaded line and the welcome banner, so this only
/// surfaces the runtime knobs that aren't shown elsewhere. `/help` re-prints
/// the full reference on demand.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_banner(
    mode_name: &str,
    tools: &[ToolDefinition],
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) {
    println!(
        "\x1b[2mmode \x1b[0;1m{mode_name}\x1b[0;2m · tools \x1b[0;1m{}\x1b[0;2m · \
         turns \x1b[0;1m{max_turns}\x1b[0;2m · tokens \x1b[0;1m{max_tokens}\x1b[0;2m · \
         temp \x1b[0;1m{temperature}\x1b[0m",
        tools.len()
    );
    println!();
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".arle-history"))
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
#[allow(clippy::too_many_arguments)]
fn run_interactive_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tools_enabled: bool,
    trace: Option<&TraceWriter>,
) -> Result<()> {
    let direct_chat = is_direct_chat_backend(backend_name);
    let tools = if direct_chat {
        Vec::new()
    } else {
        tool_definitions(tools_enabled)
    };
    let mut session = if direct_chat {
        AgentSession::with_system_prompt("")
    } else {
        AgentSession::new()
    };
    let mut pending_images = Vec::new();
    let mut session_stats = SessionStats::default();
    let mut editor = DefaultEditor::new()?;
    let history = history_path();

    if let Some(path) = history.as_ref()
        && let Err(err) = editor.load_history(path)
    {
        log::debug!("Skipping history load from {}: {err}", path.display());
    }

    let (cancel, exit) = install_ctrlc_handler();
    cancel.store(false, Ordering::Relaxed);
    exit.store(false, Ordering::Relaxed);

    print_repl_banner(
        if direct_chat { "chat" } else { "agent" },
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
                    &mut pending_images,
                    &mut session_stats,
                    input,
                    max_turns,
                    max_tokens,
                    temperature,
                    cancel.clone(),
                    trace,
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
#[allow(clippy::too_many_arguments)]
fn run_piped_repl(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tools_enabled: bool,
    trace: Option<&TraceWriter>,
) -> Result<()> {
    let direct_chat = is_direct_chat_backend(backend_name);
    let tools = if direct_chat {
        Vec::new()
    } else {
        tool_definitions(tools_enabled)
    };
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut session = if direct_chat {
        AgentSession::with_system_prompt("")
    } else {
        AgentSession::new()
    };
    let mut pending_images = Vec::new();
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
                    &mut pending_images,
                    &mut session_stats,
                    input,
                    max_turns,
                    max_tokens,
                    temperature,
                    cancel.clone(),
                    trace,
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
fn resolve_run_images(run_args: &RunArgs) -> Result<Vec<ChatPromptImage>> {
    run_args
        .image
        .iter()
        .map(|source| load_cli_image(source))
        .collect()
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn load_cli_image(source: &str) -> Result<ChatPromptImage> {
    let source = source.trim();
    anyhow::ensure!(!source.is_empty(), "image source must not be empty");
    if source.starts_with("http://") || source.starts_with("https://") {
        return load_remote_cli_image(source);
    }
    let path = source.strip_prefix("file://").unwrap_or(source);
    load_local_cli_image(source, Path::new(path))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn load_remote_cli_image(source: &str) -> Result<ChatPromptImage> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("arle-cli/0.1")
        .build()
        .context("build HTTP client failed")?;
    let response = client
        .get(source)
        .send()
        .with_context(|| format!("fetch image {source} failed"))?
        .error_for_status()
        .with_context(|| format!("fetch image {source} returned an error status"))?;
    if let Some(len) = response.content_length() {
        anyhow::ensure!(
            len <= MAX_CLI_IMAGE_BYTES as u64,
            "image {source} is too large: {len} bytes > {MAX_CLI_IMAGE_BYTES}"
        );
    }
    let mime_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
        .filter(|value| !value.is_empty());
    let data = response
        .bytes()
        .with_context(|| format!("read image {source} response failed"))?
        .to_vec();
    anyhow::ensure!(
        data.len() <= MAX_CLI_IMAGE_BYTES,
        "image {source} is too large: {} bytes > {MAX_CLI_IMAGE_BYTES}",
        data.len()
    );
    Ok(ChatPromptImage::new(source.to_string(), data).with_mime_type(mime_type))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn load_local_cli_image(source: &str, path: &Path) -> Result<ChatPromptImage> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("stat image {} failed", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "image {} is not a regular file",
        path.display()
    );
    anyhow::ensure!(
        metadata.len() <= MAX_CLI_IMAGE_BYTES as u64,
        "image {} is too large: {} bytes > {MAX_CLI_IMAGE_BYTES}",
        path.display(),
        metadata.len()
    );
    let data =
        std::fs::read(path).with_context(|| format!("read image {} failed", path.display()))?;
    let mime_type = guess_image_mime(path).map(str::to_string);
    Ok(ChatPromptImage::new(source.to_string(), data).with_mime_type(mime_type))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn guess_image_mime(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn handle_repl_input(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
    session: &mut AgentSession,
    pending_images: &mut Vec<ChatPromptImage>,
    session_stats: &mut SessionStats,
    input: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    cancel: Arc<AtomicBool>,
    trace: Option<&TraceWriter>,
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
            pending_images,
            session_stats,
            max_turns,
            max_tokens,
            temperature,
        ));
    }

    if is_direct_chat_backend(backend_name) {
        if !pending_images.is_empty() && !supports_cli_images(backend_name) {
            eprintln!(
                "\x1b[1;31mError: pending images require a VLM chat backend; current backend is {backend_name}\x1b[0m"
            );
            println!();
            return Ok(true);
        }
        let images = pending_images.clone();
        pending_images.clear();
        run_direct_chat_turn(
            engine,
            backend_name,
            session,
            session_stats,
            input,
            &images,
            max_tokens,
            temperature,
            trace,
        );
    } else {
        if !pending_images.is_empty() {
            eprintln!(
                "\x1b[1;31mError: pending images require a direct VLM chat backend; current backend is {backend_name}\x1b[0m"
            );
            println!();
            return Ok(true);
        }
        run_agent_turn(
            engine,
            backend_name,
            tools,
            session,
            session_stats,
            input,
            max_turns,
            max_tokens,
            temperature,
            cancel,
            trace,
        );
    }

    Ok(true)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn is_direct_chat_backend(backend_name: &str) -> bool {
    matches!(backend_name, "metal-deepseek-ocr")
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn supports_cli_images(backend_name: &str) -> bool {
    matches!(backend_name, "metal-deepseek-ocr")
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn session_prompt_messages(
    session: &AgentSession,
    user_input: &str,
    images: &[ChatPromptImage],
) -> Vec<ChatPromptMessage> {
    let mut messages: Vec<ChatPromptMessage> = session
        .messages()
        .iter()
        .filter(|m| {
            let role = m.role.as_str();
            role != "tool"
                && m.tool_calls.is_empty()
                && !(role == "system" && m.content.trim().is_empty())
        })
        .map(|m| ChatPromptMessage::new(m.role.as_str(), m.content.clone()))
        .collect();
    messages.push(ChatPromptMessage::user_with_images(
        user_input,
        images.to_vec(),
    ));
    messages
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_direct_chat_completion(
    engine: &mut dyn InferenceEngine,
    session: &mut AgentSession,
    input: &str,
    images: &[ChatPromptImage],
    max_tokens: usize,
    temperature: f32,
) -> Result<AgentTurnResult> {
    let prompt_messages = session_prompt_messages(session, input, images);
    let prompt = engine.render_chat_prompt(&prompt_messages)?;
    let started_at = Instant::now();
    let sampling = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };
    let output = if images.is_empty() {
        engine.complete(CompletionRequest {
            prompt: prompt.clone(),
            max_tokens,
            sampling,
            // The checkpoint template and diffusion config own EOS handling. A
            // ChatML stop string here is wrong for DiffusionGemma.
            stop: None,
        })?
    } else {
        engine.complete_multimodal_chat(MultimodalChatRequest {
            messages: prompt_messages.clone(),
            max_tokens,
            sampling,
        })?
    };
    let decode_secs = started_at.elapsed().as_secs_f64();
    let prompt_tokens = output.usage.prompt_tokens as u64;
    let completion_tokens = output.usage.completion_tokens as u64;
    let terminal_state = if output.text.trim().is_empty() {
        TerminalState::EmptyNoProgress
    } else {
        TerminalState::Stop
    };
    let tokens = if output.prompt_token_ids.is_empty() || output.response_token_ids.is_empty() {
        None
    } else {
        Some(TokensRecord {
            prompt_ids: output.prompt_token_ids.clone(),
            response_ids: output.response_token_ids.clone(),
            response_mask: std::iter::repeat_n(1u8, output.response_token_ids.len()).collect(),
        })
    };
    let messages = vec![
        TrajectoryMessage {
            role: TrajectoryRole::User,
            content: MessageContent::Text(input.to_string()),
            tool_use_id: None,
            result_truncated: None,
        },
        TrajectoryMessage {
            role: TrajectoryRole::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: output.text.clone(),
            }]),
            tool_use_id: None,
            result_truncated: None,
        },
    ];
    let sub_turns = vec![SubTurnRecord {
        index: 0,
        prompt_text: Some(prompt),
        completion_text: output.text.clone(),
        usage: ToolUsage {
            prompt_tokens,
            completion_tokens,
        },
        ttft_ms: None,
        decode_secs,
        finish_reason: finish_reason_label(output.finish_reason).to_string(),
    }];
    session.append_plain_turn(input, &output.text);
    Ok(AgentTurnResult {
        text: output.text,
        tool_calls_executed: 0,
        prompt_tokens,
        completion_tokens,
        max_turns_reached: false,
        trace_events: Vec::new(),
        time_to_first_token: None,
        messages,
        sub_turns,
        terminal_state,
        wall_secs: decode_secs,
        tokens,
    })
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_direct_chat_turn(
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    session: &mut AgentSession,
    session_stats: &mut SessionStats,
    input: &str,
    images: &[ChatPromptImage],
    max_tokens: usize,
    temperature: f32,
    trace: Option<&TraceWriter>,
) {
    let start = Instant::now();
    let color_on = io::stdout().is_terminal();
    let mut spinner = ThinkingSpinner::new_with_label(io::stderr().is_terminal(), "denoising");
    let mut tps_meter = crate::tps::TpsMeter::new();
    spinner.start();
    let result =
        run_direct_chat_completion(engine, session, input, images, max_tokens, temperature);
    spinner.stop();
    match result {
        Ok(result) => {
            let trimmed = result.text.trim();
            if trimmed.is_empty() {
                println!(
                    "\x1b[2m(chat model produced an empty reply — finish_reason={}, completion_tokens={})\x1b[0m",
                    result
                        .sub_turns
                        .last()
                        .map(|turn| turn.finish_reason.as_str())
                        .unwrap_or("unknown"),
                    result.completion_tokens
                );
            } else if color_on {
                println!("\x1b[1;34m{}\x1b[0m", trimmed);
            } else {
                println!("{trimmed}");
            }
            let elapsed = start.elapsed();
            tps_meter.print_final(result.prompt_tokens, Some(result.completion_tokens), None);
            session_stats.record_turn(result.prompt_tokens, result.completion_tokens, 0, elapsed);
            if let Some(writer) = trace {
                writer.write_turn(engine.model_id(), backend_name, input, &result);
            }
            println!();
        }
        Err(e) => {
            eprintln!("\x1b[1;31mError: {e:#}\x1b[0m");
            println!();
        }
    }
}

/// Animated "thinking…" spinner shown on stderr while the model is computing
/// and nothing visible has been printed yet. A complete no-op when stderr is
/// not a TTY — that keeps `run_one_shot --json` and piped output clean (and
/// `run_one_shot` doesn't route through `run_agent_turn` anyway, so its stdout
/// JSON path can never see this).
///
/// indicatif draws to stderr by default, which is what we want: the actual
/// answer tokens go to stdout, so the spinner never collides with them.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
struct ThinkingSpinner {
    bar: Option<indicatif::ProgressBar>,
    enabled: bool,
    label: &'static str,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl ThinkingSpinner {
    /// `enabled` should be `io::stderr().is_terminal()`. When false every
    /// method is inert and `bar` stays `None`.
    fn new(enabled: bool) -> Self {
        Self::new_with_label(enabled, "thinking")
    }

    fn new_with_label(enabled: bool, label: &'static str) -> Self {
        Self {
            bar: None,
            enabled,
            label,
        }
    }

    /// Create + steady-tick a spinner if enabled and none is currently active.
    /// Idempotent: a second call while one is live is a no-op.
    fn start(&mut self) {
        if !self.enabled || self.bar.is_some() {
            return;
        }
        let bar = indicatif::ProgressBar::new_spinner();
        let template = format!("{{spinner:.cyan}} {}… {{elapsed}}", self.label);
        bar.set_style(
            indicatif::ProgressStyle::with_template(&template)
                .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
        );
        bar.enable_steady_tick(Duration::from_millis(80));
        self.bar = Some(bar);
    }

    /// Clear the spinner line and drop the bar. Idempotent / safe when none
    /// is active.
    fn stop(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
#[allow(clippy::too_many_arguments)]
fn run_agent_turn(
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
    trace: Option<&TraceWriter>,
) {
    let start = Instant::now();
    let color_on = io::stdout().is_terminal();
    let render_state = RefCell::new((false, false));
    let tps_meter = RefCell::new(crate::tps::TpsMeter::new());
    let spinner = RefCell::new(ThinkingSpinner::new(io::stderr().is_terminal()));
    let mut on_text_chunk = |chunk: &str| {
        if chunk.is_empty() {
            return;
        }
        // Suppress leading whitespace-only output at the start of a turn so the
        // reply (or first tool line) begins right after the prompt instead of
        // after blank lines the model emits while "thinking". Once real text
        // has streamed, internal whitespace is preserved.
        let to_print = if render_state.borrow().1 {
            chunk
        } else {
            chunk.trim_start()
        };
        if to_print.is_empty() {
            return;
        }
        // Clear the spinner the instant *visible* output starts — before any
        // stdout write and before tps_meter erases its own status line, so
        // no leftover spinner glyph collides with the answer tokens.
        spinner.borrow_mut().stop();
        let mut state = render_state.borrow_mut();
        tps_meter.borrow_mut().hide_before_chunk();
        if color_on && !state.0 {
            print!("\x1b[1;34m");
            let _ = io::stdout().flush();
            state.0 = true;
        }
        print!("{to_print}");
        let _ = io::stdout().flush();
        tps_meter.borrow_mut().record_chunk(to_print.len());
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
        spinner.borrow_mut().stop();
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
        // The model now computes its next step — show the spinner again until
        // the next visible output (text chunk or tool line) arrives.
        spinner.borrow_mut().start();
    };
    // Show the spinner while the model computes before any output appears.
    // The callbacks stop() it the instant visible output starts.
    spinner.borrow_mut().start();
    let turn = session.run_turn_interruptibly_with_callbacks(
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
        Arc::clone(&cancel),
        AgentTurnCallbacks {
            on_text_chunk: Some(&mut on_text_chunk),
            on_trace_event: Some(&mut on_trace_event),
        },
    );
    // Ensure no spinner survives the turn — including the cancelled / error
    // paths, and turns that finished without ever streaming. Must precede any
    // stdout token or tps_meter.print_final() below. stop() is idempotent, so
    // the arms that already stopped it via the callbacks are unaffected.
    spinner.borrow_mut().stop();
    match turn {
        Ok(Some(result)) => {
            let (color_open, streamed_any) = *render_state.borrow();
            if color_on && color_open {
                print!("\x1b[0m");
                let _ = io::stdout().flush();
            }
            if streamed_any {
                println!();
            }
            // When the agent terminates because it ran out of turn budget,
            // the engine fills `result.text` with a placeholder ("max turns
            // reached - agent stopped") and sets `max_turns_reached`. Showing
            // both the blue placeholder and a dim status marker is double-
            // talk — keep just the dim marker, and include the hint so the
            // user can raise the budget without having to dig.
            if result.max_turns_reached {
                println!(
                    "\x1b[2m(agent stopped — max turns reached; pass --max-turns N to allow more tool steps)\x1b[0m"
                );
            } else if !streamed_any {
                // The agent finished without streaming any visible text —
                // either it produced an empty `result.text`, or it spent
                // every sub-turn on tool_call XML that VisibleTextStream
                // stripped. Without this branch the user sees only the
                // status line and assumes the REPL hung ("没回复"). Print
                // a dim diagnostic in the empty case so the silent hang
                // is now an honest "no visible answer was generated"
                // signal — including how many tool calls fired, so the
                // user can tell a 0-turn flop from a 250-turn loop.
                let trimmed = result.text.trim();
                if trimmed.is_empty() {
                    println!(
                        "\x1b[2m(agent finished without a visible reply — {} tool call(s) executed; \
                         try a more specific prompt, or `/reset` to clear context)\x1b[0m",
                        result.tool_calls_executed
                    );
                } else {
                    println!("\x1b[1;34m{}\x1b[0m", trimmed);
                }
            }
            let elapsed = start.elapsed();
            tps_meter.borrow_mut().print_final(
                result.prompt_tokens,
                Some(result.completion_tokens),
                result.time_to_first_token,
            );
            session_stats.record_turn(
                result.prompt_tokens,
                result.completion_tokens,
                result.tool_calls_executed,
                elapsed,
            );
            if let Some(writer) = trace {
                writer.write_turn(engine.model_id(), backend_name, input, &result);
            }
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
        "/image" => Some(ReplCommand::Image(arg.to_string())),
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
    pending_images: &mut Vec<ChatPromptImage>,
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
        ReplCommand::Image(arg) => {
            handle_image_command(backend_name, pending_images, &arg);
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn handle_image_command(backend_name: &str, pending_images: &mut Vec<ChatPromptImage>, arg: &str) {
    match arg.trim() {
        "" => {
            eprintln!("\x1b[1;31mError: usage: /image <path-or-url>|list|clear\x1b[0m");
        }
        "clear" => {
            pending_images.clear();
            println!("\x1b[2m(cleared pending images)\x1b[0m");
        }
        "list" => {
            if pending_images.is_empty() {
                println!("\x1b[2m(no pending images)\x1b[0m");
            } else {
                for (idx, image) in pending_images.iter().enumerate() {
                    let mime = image.mime_type.as_deref().unwrap_or("unknown");
                    println!(
                        "\x1b[2m({}: {} bytes, {mime}) {}\x1b[0m",
                        idx + 1,
                        image.data.len(),
                        image.source
                    );
                }
            }
        }
        source => {
            if !supports_cli_images(backend_name) {
                eprintln!(
                    "\x1b[1;31mError: /image requires a VLM chat backend; current backend is {backend_name}\x1b[0m"
                );
                return;
            }
            match load_cli_image(source) {
                Ok(image) => {
                    let mime = image.mime_type.as_deref().unwrap_or("unknown").to_string();
                    let bytes = image.data.len();
                    let source = image.source.clone();
                    pending_images.push(image);
                    println!(
                        "\x1b[2m(attached image {}: {bytes} bytes, {mime}; it will be sent with the next message)\x1b[0m",
                        source
                    );
                }
                Err(err) => {
                    eprintln!("\x1b[1;31mError: {err:#}\x1b[0m");
                }
            }
        }
    }
}

// Inline tool-call rendering — one dim line per call:
//   `  ⏵ name(args) → result`
// Args/result are aggressively truncated so the line stays scannable.

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const TOOL_NAME_MAX: usize = 32;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const TOOL_ARGS_MAX: usize = 60;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
const TOOL_RESULT_MAX: usize = 80;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn format_tool_call_line(name: &str, arguments: &serde_json::Value, result: &str) -> String {
    // Cap + flatten the name too — malformed model output can ship a `name`
    // with embedded newlines or pathological length that would break the
    // single-line invariant just as easily as a bad args/result.
    let name = truncate_one_line(name, TOOL_NAME_MAX);
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

// /models — listing + (descoped) switch pointer.
//
// Hot-swap is descoped: mid-REPL engine reinit would require owning the
// `LoadedInferenceEngine` inside the REPL rather than taking `&mut dyn
// InferenceEngine`. Threading the concrete type through the existing API would
// cascade into `lib.rs`/`startup.rs`. Ship listing today; track hot-swap
// separately.

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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn render_history_markdown(model_id: &str, history: &[ChatMessage]) -> String {
    let ts = iso8601_utc_now();
    let turns = count_export_turns(history);
    let body: String = history
        .iter()
        .skip(1)
        .filter_map(|msg| match &msg.role {
            chat::ChatRole::User => Some(format!("## You\n\n{}\n\n", msg.content.trim_end())),
            chat::ChatRole::Assistant => {
                Some(format!("## Assistant\n\n{}\n\n", msg.content.trim_end()))
            }
            _ => None,
        })
        .collect();
    format!(
        "# ARLE conversation — {model_id}\n\n> Started: {ts}\n> Mode: agent\n> Turns: {turns}\n\n{body}"
    )
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
///
/// Re-exported as `format_iso8601_utc_secs` for the trace writer; the
/// markdown export path keeps its private alias for now.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn format_iso8601_utc_secs(unix_secs: u64) -> String {
    format_iso8601_utc(unix_secs)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn format_iso8601_utc(unix_secs: u64) -> String {
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
    println!("  /image <src>     Attach an image path or URL to the next chat turn");
    println!("  /image list|clear  Show or clear pending images");
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
