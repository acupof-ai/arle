#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::path::PathBuf;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer_agent::{AgentEngine, AgentSession, AgentSessionStats, AgentSettings};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer_engine::{LoadedAgentEngine, init_default_logging, resolve_model_source};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer_tools::{Tool, builtin_tools};
#[cfg(all(not(feature = "cuda"), any(feature = "metal", feature = "cpu")))]
use log::warn;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::DefaultEditor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::error::ReadlineError;

#[derive(Parser)]
#[command(name = "agent-infer", about = "Local LLM agent with tool use")]
struct Args {
    /// Path to model directory or HuggingFace model ID.
    /// If omitted, the CLI auto-detects a local model from common directories and HF cache.
    #[arg(long)]
    model_path: Option<String>,

    /// Maximum agent turns (generate-execute cycles) per query
    #[arg(long, default_value_t = 10)]
    max_turns: usize,

    /// Maximum tokens to generate per turn
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Disable CUDA graph (useful for debugging)
    #[arg(long, default_value_t = false)]
    no_cuda_graph: bool,

    /// Max KV cache tokens on GPU. Excess offloads to CPU.
    /// Use a small value (e.g. 512) to test KV offload behavior.
    #[arg(long)]
    max_gpu_kv: Option<usize>,
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
    Exit,
    Unknown(String),
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_repl(
    engine: &mut dyn AgentEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return run_interactive_repl(engine, backend_name, max_turns, max_tokens, temperature);
    }

    run_piped_repl(engine, backend_name, max_turns, max_tokens, temperature)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_repl_banner(
    engine: &dyn AgentEngine,
    backend_name: &str,
    tools: &[Tool],
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) {
    println!();
    println!("=== agent-infer REPL ===");
    println!("Model: {}", engine.model_id());
    println!("Backend: {}", backend_name);
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
    println!("Type your query and press Enter. Type '/help' for commands.");
    println!("Type 'quit' or 'exit' to stop.");
    println!();
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn history_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".agent-infer-history"))
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_interactive_repl(
    engine: &mut dyn AgentEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools();
    let mut session = AgentSession::new();
    let mut editor = DefaultEditor::new()?;
    let history = history_path();

    if let Some(path) = history.as_ref() {
        if let Err(err) = editor.load_history(path) {
            log::debug!("Skipping history load from {}: {err}", path.display());
        }
    }

    print_repl_banner(
        engine,
        backend_name,
        &tools,
        max_turns,
        max_tokens,
        temperature,
    );

    loop {
        match editor.readline("\x1b[1;32m> \x1b[0m") {
            Ok(line) => {
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
                    input,
                    max_turns,
                    max_tokens,
                    temperature,
                )? {
                    break;
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
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

    if let Some(path) = history.as_ref() {
        if let Err(err) = editor.save_history(path) {
            log::debug!("Skipping history save to {}: {err}", path.display());
        }
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_piped_repl(
    engine: &mut dyn AgentEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools();
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut session = AgentSession::new();

    print_repl_banner(
        engine,
        backend_name,
        &tools,
        max_turns,
        max_tokens,
        temperature,
    );

    loop {
        print!("\x1b[1;32m> \x1b[0m");
        stdout.flush()?;

        let mut input = String::new();
        let bytes = stdin.lock().read_line(&mut input)?;
        if bytes == 0 {
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if !handle_repl_input(
            engine,
            backend_name,
            &tools,
            &mut session,
            input,
            max_turns,
            max_tokens,
            temperature,
        )? {
            break;
        }
    }

    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn handle_repl_input(
    engine: &mut dyn AgentEngine,
    backend_name: &str,
    tools: &[Tool],
    session: &mut AgentSession,
    input: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
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
            max_turns,
            max_tokens,
            temperature,
        ));
    }

    let start = Instant::now();
    match session.run_turn(
        engine,
        input,
        tools,
        AgentSettings {
            max_turns,
            max_tokens,
            temperature,
        },
    ) {
        Ok(result) => {
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

    Ok(true)
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
        "/quit" | "/exit" => Some(ReplCommand::Exit),
        _ => Some(ReplCommand::Unknown(command.to_string())),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn execute_repl_command(
    command: ReplCommand,
    engine: &mut dyn AgentEngine,
    backend_name: &str,
    tools: &[Tool],
    session: &mut AgentSession,
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
        ReplCommand::Stats => {
            print_session_stats(
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
    println!("  /reset, /clear   Clear conversation history");
    println!("  /tools           Show available tools");
    println!("  /model           Show active model and backend");
    println!("  /stats           Show session and runtime settings");
    println!("  /save <path>     Save the current session to JSON");
    println!("  /load <path>     Load a saved session JSON");
    println!("  /quit, /exit     Leave the REPL");
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_tools_help(tools: &[Tool]) {
    println!("Tools:");
    for tool in tools {
        println!("  {}: {}", tool.name, tool.description);
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn print_session_stats(
    stats: AgentSessionStats,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
    tool_count: usize,
) {
    println!("Session:");
    println!("  messages: {}", stats.conversation_messages);
    println!("  users: {}", stats.user_messages);
    println!("  assistants: {}", stats.assistant_messages);
    println!("  tool results: {}", stats.tool_messages);
    println!("  tool calls: {}", stats.tool_calls);
    println!("  content chars: {}", stats.content_chars);
    println!("Runtime:");
    println!("  tools enabled: {}", tool_count);
    println!("  max turns: {}", max_turns);
    println!("  max tokens: {}", max_tokens);
    println!("  temperature: {}", temperature);
}

pub fn run() -> Result<()> {
    init_default_logging();

    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), not(feature = "cpu")))]
    {
        anyhow::bail!(
            "agent-infer requires a local inference backend. Rebuild with either \
             the default `cuda` feature, `--no-default-features --features metal,no-cuda`, \
             or `--no-default-features --features cpu,no-cuda`."
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
    {
        let args = Args::parse();
        let model_source = resolve_model_source(args.model_path.as_deref())?;
        log::info!("Loading model from: {}", model_source);
        let load_start = Instant::now();
        let mut engine = LoadedAgentEngine::load(&model_source, !args.no_cuda_graph)?;
        let backend_name = engine.backend_name().to_string();

        if let Some(max_kv) = args.max_gpu_kv {
            #[cfg(feature = "cuda")]
            log::info!(
                "Setting max GPU KV to {} tokens (offload test mode)",
                max_kv
            );
            #[cfg(not(feature = "cuda"))]
            warn!("Ignoring --max-gpu-kv: only supported by the CUDA backend");
            engine.set_max_gpu_kv(max_kv);
        }
        log::info!(
            "Model loaded in {:.1}s (backend={}, model={})",
            load_start.elapsed().as_secs_f64(),
            engine.backend_name(),
            engine.model_id(),
        );
        run_repl(
            &mut engine,
            &backend_name,
            args.max_turns,
            args.max_tokens,
            args.temperature,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ReplCommand, parse_repl_command};

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
    fn parse_repl_command_ignores_normal_chat_input() {
        assert_eq!(parse_repl_command("list files"), None);
    }
}
