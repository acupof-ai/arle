#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::io::{self, BufRead, IsTerminal, Write};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::path::PathBuf;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use std::time::Instant;

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use agent::{
    AgentSession, AgentSessionStats, AgentSettings, AgentTraceEvent, ToolExecutor, ToolPolicy,
};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use chat::{ParsedAssistantResponse, ToolCall, ToolDefinition};
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use infer::server_engine::InferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::DefaultEditor;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use rustyline::error::ReadlineError;
#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use tools::{BuiltinToolPolicyHooks, builtin_tools, execute_tool_call};

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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_repl(
    engine: &mut dyn InferenceEngine,
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
    engine: &dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
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
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools()
        .into_iter()
        .map(|tool| tool.to_definition())
        .collect::<Vec<_>>();
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
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    max_turns: usize,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    let tools = builtin_tools()
        .into_iter()
        .map(|tool| tool.to_definition())
        .collect::<Vec<_>>();
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
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
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
    engine: &mut dyn InferenceEngine,
    backend_name: &str,
    tools: &[ToolDefinition],
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
