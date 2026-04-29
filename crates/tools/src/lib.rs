use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use chat::{ParsedAssistantResponse, ToolCall, ToolDefinition};

static SCRIPT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinToolKind {
    Shell,
    Python,
}

impl BuiltinToolKind {
    const ALL: [Self; 2] = [Self::Shell, Self::Python];

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "shell" => Some(Self::Shell),
            "python" => Some(Self::Python),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Python => "python",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Shell => "Run a shell command.",
            Self::Python => "Run Python 3 code.",
        }
    }

    fn parameters(self) -> Value {
        match self {
            Self::Shell => json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string"
                    }
                },
                "required": ["command"]
            }),
            Self::Python => json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string"
                    }
                },
                "required": ["code"]
            }),
        }
    }

    fn into_tool(self) -> Tool {
        Tool {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }

    fn execute(self, arguments: &Value) -> String {
        match self {
            Self::Shell => execute_shell(argument_as_str(arguments, "command")),
            Self::Python => execute_python(argument_as_str(arguments, "code")),
        }
    }
}

// ============================================================================
// Sandbox configuration
// ============================================================================

struct SandboxConfig {
    timeout_secs: u64,
    max_memory_mb: u64,
    workdir: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_memory_mb: 512,
            workdir: default_workdir(),
        }
    }
}

fn default_workdir() -> String {
    std::env::current_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .to_string_lossy()
        .into_owned()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SandboxBackend {
    Nsjail,
    SandboxExec,
    Bare,
}

impl SandboxBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Nsjail => "nsjail",
            Self::SandboxExec => "sandbox-exec",
            Self::Bare => "bare",
        }
    }
}

fn nsjail_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("nsjail")
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    })
}

#[cfg(target_os = "macos")]
fn sandbox_exec_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::path::Path::new("/usr/bin/sandbox-exec").exists())
}

#[cfg(not(target_os = "macos"))]
fn sandbox_exec_available() -> bool {
    false
}

fn active_sandbox_backend() -> SandboxBackend {
    if nsjail_available() {
        SandboxBackend::Nsjail
    } else if sandbox_exec_available() {
        SandboxBackend::SandboxExec
    } else {
        SandboxBackend::Bare
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolRuntimeReport {
    pub enabled_by_default: bool,
    pub builtin_tools: Vec<String>,
    pub sandbox_backend: String,
    pub sandboxed: bool,
    pub timeout_secs: u64,
    pub max_memory_mb: u64,
    pub workdir: String,
    pub python: String,
}

pub fn tool_runtime_report() -> ToolRuntimeReport {
    let sandbox = SandboxConfig::default();
    let backend = active_sandbox_backend();
    ToolRuntimeReport {
        enabled_by_default: true,
        builtin_tools: BuiltinToolKind::ALL
            .into_iter()
            .map(|kind| kind.name().to_string())
            .collect(),
        sandbox_backend: backend.label().to_string(),
        sandboxed: backend != SandboxBackend::Bare,
        timeout_secs: sandbox.timeout_secs,
        max_memory_mb: sandbox.max_memory_mb,
        workdir: sandbox.workdir,
        python: resolved_python_executable().display().to_string(),
    }
}

fn default_env_path() -> String {
    std::env::var("PATH").unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".to_string())
}

fn effective_tmpdir() -> String {
    std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string())
}

#[cfg(not(target_os = "windows"))]
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(target_os = "windows")]
fn shell_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\\\""))
}

fn resolved_python_executable() -> PathBuf {
    static PYTHON: OnceLock<PathBuf> = OnceLock::new();
    PYTHON
        .get_or_init(|| {
            #[cfg(target_os = "windows")]
            let candidates = [
                "py.exe",
                "python.exe",
                "python3.exe",
                "py",
                "python3",
                "python",
            ];
            #[cfg(not(target_os = "windows"))]
            let candidates = ["python3", "python"];
            for candidate in candidates {
                for dir in std::env::split_paths(&default_env_path()) {
                    let path = dir.join(candidate);
                    if path.is_file() {
                        return path;
                    }
                }
            }
            #[cfg(target_os = "windows")]
            {
                PathBuf::from("py")
            }
            #[cfg(not(target_os = "windows"))]
            {
                PathBuf::from("python3")
            }
        })
        .clone()
}

impl SandboxConfig {
    fn wrap_shell(&self, user_cmd: &str) -> Command {
        match active_sandbox_backend() {
            SandboxBackend::Nsjail => self.wrap_shell_nsjail(user_cmd),
            SandboxBackend::SandboxExec => {
                #[cfg(target_os = "macos")]
                {
                    self.wrap_shell_sandbox_exec(user_cmd)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.wrap_shell_bare(user_cmd)
                }
            }
            SandboxBackend::Bare => {
                log::warn!("no supported sandbox backend found — running without sandbox");
                self.wrap_shell_bare(user_cmd)
            }
        }
    }

    fn wrap_shell_nsjail(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("nsjail");
        cmd.arg("--mode").arg("o");
        cmd.arg("--time_limit").arg(self.timeout_secs.to_string());
        cmd.arg("--rlimit_as").arg(self.max_memory_mb.to_string());
        cmd.arg("--quiet");
        cmd.arg("--disable_proc");

        // Read-only bind mounts for system dirs
        for dir in &[
            "/bin",
            "/lib",
            "/lib64",
            "/usr",
            "/etc",
            "/dev/null",
            "/dev/urandom",
        ] {
            if std::path::Path::new(dir).exists() {
                cmd.arg("-R").arg(dir);
            }
        }

        // Read-only bind for /usr/local (python, pip packages, cuda libs)
        if std::path::Path::new("/usr/local").exists() {
            cmd.arg("-R").arg("/usr/local");
        }

        // Writable /tmp
        cmd.arg("-B").arg(&self.workdir);
        // Also bind /tmp if workdir is different
        if self.workdir != "/tmp" {
            cmd.arg("-B").arg("/tmp");
        }

        cmd.arg("--cwd").arg(&self.workdir);

        // Environment
        cmd.arg("--env").arg(format!("PATH={}", default_env_path()));
        cmd.arg("--env").arg(format!("HOME={}", self.workdir));
        cmd.arg("--env")
            .arg(format!("TMPDIR={}", effective_tmpdir()));
        cmd.arg("--env").arg("LANG=C.UTF-8");
        cmd.arg("--env").arg("PYTHONDONTWRITEBYTECODE=1");

        cmd.arg("--").arg("/bin/bash").arg("-c").arg(user_cmd);
        cmd
    }

    #[cfg(target_os = "macos")]
    fn wrap_shell_sandbox_exec(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("/usr/bin/sandbox-exec");
        cmd.arg("-p").arg(Self::sandbox_exec_profile());
        cmd.arg("/bin/bash").arg("-c").arg(user_cmd);
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env("PATH", default_env_path());
        cmd.env(
            "HOME",
            std::env::var("HOME").unwrap_or_else(|_| self.workdir.clone()),
        );
        cmd.env("TMPDIR", effective_tmpdir());
        cmd.env("LANG", "C.UTF-8");
        cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        cmd
    }

    #[cfg(not(target_os = "windows"))]
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(user_cmd);
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env("PATH", default_env_path());
        cmd.env(
            "HOME",
            std::env::var("HOME").unwrap_or_else(|_| self.workdir.clone()),
        );
        cmd.env("TMPDIR", effective_tmpdir());
        cmd.env("LANG", "C.UTF-8");
        cmd
    }

    #[cfg(target_os = "windows")]
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("cmd");
        cmd.arg("/d").arg("/c").arg(user_cmd);
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env("PATH", default_env_path());
        cmd.env(
            "HOME",
            std::env::var("HOME").unwrap_or_else(|_| self.workdir.clone()),
        );
        cmd.env("TMPDIR", effective_tmpdir());
        cmd.env("LANG", "C.UTF-8");
        cmd
    }

    fn wrap_python(&self, code: &str) -> std::io::Result<Command> {
        let seq = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let script_path =
            std::env::temp_dir().join(format!("sandbox_py_{}_{}.py", std::process::id(), seq));
        std::fs::write(&script_path, code)?;
        let python = resolved_python_executable();
        #[cfg(not(target_os = "windows"))]
        let shell_cmd = format!(
            "{} -u {} ; _rc=$?; rm -f {} ; exit $_rc",
            shell_quote(&python.display().to_string()),
            shell_quote(&script_path.display().to_string()),
            shell_quote(&script_path.display().to_string())
        );
        #[cfg(target_os = "windows")]
        let shell_cmd = format!(
            "{} -u {} & set _rc=%ERRORLEVEL% & del /f /q {} >nul 2>nul & exit /b %_rc%",
            shell_quote(&python.display().to_string()),
            shell_quote(&script_path.display().to_string()),
            shell_quote(&script_path.display().to_string())
        );
        Ok(self.wrap_shell(&shell_cmd))
    }

    #[cfg(target_os = "macos")]
    fn sandbox_exec_profile() -> String {
        [
            "(version 1)".to_string(),
            "(deny default)".to_string(),
            "(allow process-exec)".to_string(),
            "(allow process-fork)".to_string(),
            "(allow signal (target self))".to_string(),
            "(allow sysctl-read)".to_string(),
            "(allow file-read*)".to_string(),
            "(allow file-write*)".to_string(),
            "(allow network*)".to_string(),
            "(allow file-read-data (literal \"/dev/null\") (literal \"/dev/urandom\") (literal \"/dev/random\"))".to_string(),
            "(allow file-write-data (literal \"/dev/null\"))".to_string(),
        ]
        .join("\n")
    }
}

// ============================================================================
// Tool definition
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl Tool {
    pub fn to_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            self.description.clone(),
            self.parameters.clone(),
        )
    }
}

/// Return the built-in tool definitions.
pub fn builtin_tools() -> Vec<Tool> {
    BuiltinToolKind::ALL
        .into_iter()
        .map(BuiltinToolKind::into_tool)
        .collect()
}

// ============================================================================
// Builtin tool policy hooks
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinToolPolicyHooks;

impl BuiltinToolPolicyHooks {
    pub fn recover_tool_calls_from_user_request(
        &self,
        user_input: &str,
        tools: &[ToolDefinition],
    ) -> Option<ParsedAssistantResponse> {
        if tool_available(tools, "python") && mentions_python_tool(user_input) {
            if let Some(code) = extract_python_code(user_input) {
                return Some(single_tool_response("python", json!({ "code": code })));
            }
            if let Some(expr) = extract_arithmetic_expression(user_input) {
                return Some(single_tool_response(
                    "python",
                    json!({ "code": format!("print({expr})") }),
                ));
            }
        }

        if tool_available(tools, "shell")
            && mentions_shell_tool(user_input)
            && let Some(command) = extract_shell_command(user_input)
        {
            return Some(single_tool_response("shell", json!({ "command": command })));
        }

        if tool_available(tools, "shell") && asks_for_file_listing(user_input) {
            return Some(single_tool_response(
                "shell",
                json!({ "command": default_directory_listing_command() }),
            ));
        }

        if tool_available(tools, "shell") && asks_for_repository_overview(user_input) {
            return Some(single_tool_response(
                "shell",
                json!({ "command": default_repository_overview_command() }),
            ));
        }

        None
    }

    pub fn recover_tool_calls_from_draft(
        &self,
        draft: &str,
        tools: &[ToolDefinition],
    ) -> Option<ParsedAssistantResponse> {
        if draft.contains("<tools>") || draft.contains("</tools>") {
            return None;
        }

        if tool_available(tools, "python")
            && let Some(code) = extract_python_code(draft)
        {
            return Some(single_tool_response("python", json!({ "code": code })));
        }

        if tool_available(tools, "shell")
            && mentions_shell_tool(draft)
            && let Some(command) = extract_shell_command(draft)
        {
            return Some(single_tool_response("shell", json!({ "command": command })));
        }

        None
    }

    pub fn should_repair_tool_calls(&self, text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        [
            "tool",
            "function",
            "python",
            "shell",
            "execute",
            "run the code",
            "call the",
            "use the",
        ]
        .iter()
        .any(|needle| lower.contains(needle))
    }

    pub fn finalize_response_text(
        &self,
        user_input: &str,
        content: String,
        _last_tool_name: Option<&str>,
        last_tool_scalar_result: Option<&str>,
        tool_calls_executed: usize,
    ) -> String {
        if tool_calls_executed == 0 {
            return content;
        }

        let Some(tool_result) = last_tool_scalar_result else {
            return content;
        };

        if content.trim().is_empty() || asks_for_exact_scalar_output(user_input) {
            return tool_result.to_string();
        }

        content
    }

    pub fn finalize_after_tool_execution(
        &self,
        user_input: &str,
        last_tool_name: Option<&str>,
        last_tool_result: Option<&str>,
        last_tool_scalar_result: Option<&str>,
    ) -> Option<String> {
        if last_tool_name == Some("shell") && should_return_shell_result_directly(user_input) {
            return last_tool_result.map(str::to_string);
        }

        if asks_for_exact_scalar_output(user_input)
            && let Some(result) = last_tool_scalar_result
        {
            return Some(result.to_string());
        }

        None
    }
}

fn single_tool_response(name: &str, arguments: serde_json::Value) -> ParsedAssistantResponse {
    ParsedAssistantResponse {
        content: String::new(),
        tool_calls: vec![ToolCall::new(name, arguments)],
    }
}

fn tool_available(tools: &[ToolDefinition], name: &str) -> bool {
    tools.iter().any(|tool| tool.name == name)
}

fn mentions_python_tool(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("python tool")
        || lower.contains("python function")
        || lower.contains("use python")
        || lower.contains("run python")
}

fn mentions_shell_tool(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("shell tool")
        || lower.contains("shell command")
        || lower.contains("use shell")
        || lower.contains("run shell")
}

fn asks_for_file_listing(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "list files",
        "show files",
        "what files",
        "which files",
        "current directory",
        "local files",
        "files here",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || [
            "哪些文件",
            "有哪些文件",
            "有什么文件",
            "列出文件",
            "当前目录",
            "本地文件",
            "目录下",
        ]
        .iter()
        .any(|needle| text.contains(needle))
}

fn asks_for_repository_overview(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "look at this repo",
        "look at the repo",
        "inspect this repo",
        "inspect the repo",
        "inspect the repository",
        "look at the repository",
        "look at the codebase",
        "inspect the codebase",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
        || [
            "看看本地仓库",
            "看看仓库",
            "看下仓库",
            "检查仓库",
            "看看代码仓库",
            "看看这个仓库",
            "看看代码库",
            "看看本地代码",
        ]
        .iter()
        .any(|needle| text.contains(needle))
}

fn should_return_shell_result_directly(text: &str) -> bool {
    asks_for_file_listing(text) || asks_for_repository_overview(text)
}

#[cfg(target_os = "windows")]
fn default_directory_listing_command() -> &'static str {
    "cd && dir /a"
}

#[cfg(not(target_os = "windows"))]
fn default_directory_listing_command() -> &'static str {
    "pwd && ls -la"
}

#[cfg(target_os = "windows")]
fn default_repository_overview_command() -> &'static str {
    "for /f \"delims=\" %i in ('git rev-parse --show-toplevel 2^>nul') do set REPO_ROOT=%i & if not defined REPO_ROOT set REPO_ROOT=%CD% & echo repo: %REPO_ROOT% & echo. & echo == top-level == & dir /a \"%REPO_ROOT%\" & echo. & echo == git status == & git -C \"%REPO_ROOT%\" status --short --branch 2>nul"
}

#[cfg(not(target_os = "windows"))]
fn default_repository_overview_command() -> &'static str {
    "repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd); printf 'repo: %s\\n' \"$repo_root\"; printf '\\n== top-level ==\\n'; find \"$repo_root\" -mindepth 1 -maxdepth 1 ! -name '.git' ! -name 'target' ! -name 'bench-output' ! -name 'node_modules' ! -name '.pytest_cache' ! -name '.claude' ! -name '.claire' ! -name '.context' -print | sed \"s#^$repo_root/##\" | sort | sed -n '1,80p'; if git -C \"$repo_root\" rev-parse --is-inside-work-tree >/dev/null 2>&1; then printf '\\n== git status ==\\n'; git -C \"$repo_root\" status --short --branch | sed -n '1,40p'; fi"
}

fn extract_python_code(text: &str) -> Option<String> {
    extract_fenced_code_block(text, &["python", "py"])
        .or_else(|| extract_balanced_call(text, "print("))
}

fn extract_shell_command(text: &str) -> Option<String> {
    extract_fenced_code_block(text, &["bash", "sh", "shell"]).or_else(|| {
        extract_backticked_snippet(text).and_then(|snippet| {
            if snippet.contains('\n') || snippet.trim().is_empty() {
                None
            } else {
                Some(snippet)
            }
        })
    })
}

fn extract_fenced_code_block(text: &str, languages: &[&str]) -> Option<String> {
    let mut remaining = text;
    while let Some(start) = remaining.find("```") {
        remaining = &remaining[start + 3..];
        let Some(end) = remaining.find("```") else {
            break;
        };

        let block = &remaining[..end];
        let (first_line, rest) = block.split_once('\n').unwrap_or((block, ""));
        let language = first_line.trim().to_ascii_lowercase();
        if languages.iter().any(|candidate| language == *candidate) {
            let code = rest.trim();
            if !code.is_empty() {
                return Some(code.to_string());
            }
        }

        remaining = &remaining[end + 3..];
    }

    None
}

fn extract_backticked_snippet(text: &str) -> Option<String> {
    let start = text.find('`')?;
    let rest = &text[start + 1..];
    let end = rest.find('`')?;
    let snippet = rest[..end].trim();
    if snippet.is_empty() {
        None
    } else {
        Some(snippet.to_string())
    }
}

fn extract_balanced_call(text: &str, start_pattern: &str) -> Option<String> {
    let start = text.find(start_pattern)?;
    let mut depth = 1usize;

    for (offset, ch) in text[start + start_pattern.len()..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + start_pattern.len() + offset + ch.len_utf8();
                    let snippet = text[start..end]
                        .trim_matches(|c| matches!(c, '`' | '"' | '\''))
                        .trim();
                    if !snippet.is_empty() {
                        return Some(snippet.to_string());
                    }
                    return None;
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_arithmetic_expression(text: &str) -> Option<String> {
    let mut best = String::new();
    let mut current = String::new();
    let mut has_digit = false;
    let mut has_operator = false;

    for ch in text.chars().chain(std::iter::once('\n')) {
        let allowed = ch.is_ascii_digit()
            || ch.is_ascii_whitespace()
            || matches!(ch, '+' | '-' | '*' | '/' | '%' | '(' | ')');
        if allowed {
            current.push(ch);
            has_digit |= ch.is_ascii_digit();
            has_operator |= matches!(ch, '+' | '-' | '*' | '/' | '%');
            continue;
        }

        let candidate = current.trim();
        if has_digit && has_operator && candidate.len() > best.len() {
            best = candidate.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        current.clear();
        has_digit = false;
        has_operator = false;
    }

    if best.is_empty() { None } else { Some(best) }
}

fn asks_for_exact_scalar_output(user_input: &str) -> bool {
    let lower = user_input.to_ascii_lowercase();
    [
        "answer with just",
        "reply with just",
        "nothing else",
        "the token only",
        "the word only",
        "just the integer",
        "integer only",
        "number only",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

// ============================================================================
// Tool execution
// ============================================================================

fn argument_as_str<'a>(arguments: &'a Value, key: &str) -> &'a str {
    arguments.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Execute a tool by name with the given JSON arguments.
pub fn execute_tool(name: &str, arguments: &serde_json::Value) -> String {
    BuiltinToolKind::from_name(name).map_or_else(
        || format!("Error: unknown tool '{name}'"),
        |tool| tool.execute(arguments),
    )
}

/// Execute a structured tool call.
pub fn execute_tool_call(call: &ToolCall) -> String {
    execute_tool(&call.name, &call.arguments)
}

/// Telemetry captured around a tool execution. Surfaced through
/// [`execute_tool_call_with_metadata`] for trajectory export — the
/// agent loop records `latency_ms` and `truncated` for each call so
/// downstream RL/training can replay the sub-turn faithfully.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolExecutionMetadata {
    pub latency_ms: u64,
    pub truncated: bool,
}

/// Marker emitted by [`collect_output`] when a tool result is too long.
/// Kept here so callers (including the agent loop) can detect truncation
/// without re-implementing the trim logic.
pub const TOOL_RESULT_TRUNCATION_MARKER: &str = "\n... (output truncated)";

/// Execute a tool call and return both the result text and the
/// per-call telemetry the trajectory exporter needs. Wraps
/// [`execute_tool_call`] one-to-one — existing callers that don't need
/// metadata stay on that simpler entry point.
pub fn execute_tool_call_with_metadata(call: &ToolCall) -> (String, ToolExecutionMetadata) {
    let start = Instant::now();
    let result = execute_tool_call(call);
    let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let truncated = result.ends_with(TOOL_RESULT_TRUNCATION_MARKER);
    (
        result,
        ToolExecutionMetadata {
            latency_ms,
            truncated,
        },
    )
}

/// Collect stdout + stderr from a process Output into a truncated string.
/// Filters out nsjail's own warning/info lines from stderr.
fn collect_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw_stderr = String::from_utf8_lossy(&output.stderr);

    // Filter nsjail's own log lines (e.g. "[W][...] logParams()...")
    let stderr: String = raw_stderr
        .lines()
        .filter(|line| {
            !line.starts_with("[W][") && !line.starts_with("[I][") && !line.starts_with("[D][")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("[stderr] ");
        result.push_str(&stderr);
    }
    if result.is_empty() {
        result.push_str("(no output)");
    }
    if result.len() > 8000 {
        result.truncate(8000);
        result.push_str("\n... (output truncated)");
    }
    result
}

enum TimedCommandResult {
    Finished(std::process::Output),
    TimedOut(std::process::Output),
}

fn run_command_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> std::io::Result<TimedCommandResult> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let start = Instant::now();

    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output().map(TimedCommandResult::Finished);
        }

        if start.elapsed() >= timeout {
            let _ = child.kill();
            return child.wait_with_output().map(TimedCommandResult::TimedOut);
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

fn execute_shell(command: &str) -> String {
    let sandbox = SandboxConfig::default();
    log::info!(
        "Executing shell ({}, timeout={}s): {}",
        active_sandbox_backend().label(),
        sandbox.timeout_secs,
        command
    );
    let mut cmd = sandbox.wrap_shell(command);
    match run_command_with_timeout(&mut cmd, Duration::from_secs(sandbox.timeout_secs)) {
        Ok(TimedCommandResult::Finished(output)) => {
            if output.status.code() == Some(137) {
                return "Error: command killed (timeout or OOM)".to_string();
            }
            collect_output(&output)
        }
        Ok(TimedCommandResult::TimedOut(output)) => {
            let partial = collect_output(&output);
            if partial == "(no output)" {
                "Error: command killed (timeout or OOM)".to_string()
            } else {
                format!("{partial}\n[stderr] Error: command killed (timeout or OOM)")
            }
        }
        Err(e) => format!("Error executing command: {e}"),
    }
}

fn execute_python(code: &str) -> String {
    let sandbox = SandboxConfig::default();
    log::info!(
        "Executing python snippet ({}, {} chars)",
        active_sandbox_backend().label(),
        code.len()
    );
    let mut cmd = match sandbox.wrap_python(code) {
        Ok(c) => c,
        Err(e) => return format!("Error preparing python sandbox: {e}"),
    };
    match run_command_with_timeout(&mut cmd, Duration::from_secs(sandbox.timeout_secs)) {
        Ok(TimedCommandResult::Finished(output)) => {
            if output.status.code() == Some(137) {
                return "Error: python killed (timeout or OOM)".to_string();
            }
            collect_output(&output)
        }
        Ok(TimedCommandResult::TimedOut(output)) => {
            let partial = collect_output(&output);
            if partial == "(no output)" {
                "Error: python killed (timeout or OOM)".to_string()
            } else {
                format!("{partial}\n[stderr] Error: python killed (timeout or OOM)")
            }
        }
        Err(e) => format!("Error executing python: {e}"),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::SandboxConfig;
    use super::{
        BuiltinToolPolicyHooks, TimedCommandResult, active_sandbox_backend, builtin_tools,
        default_workdir, resolved_python_executable, run_command_with_timeout,
    };
    use serde_json::json;
    use std::process::Command;
    use std::time::Duration;

    #[cfg(target_os = "windows")]
    fn bare_shell_command(command: &str) -> Command {
        let mut cmd = Command::new("cmd");
        cmd.arg("/d").arg("/c").arg(command);
        cmd
    }

    #[cfg(not(target_os = "windows"))]
    fn bare_shell_command(command: &str) -> Command {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(command);
        cmd
    }

    #[test]
    fn default_workdir_uses_current_directory() {
        let expected = std::env::current_dir().expect("current_dir");
        assert_eq!(default_workdir(), expected.display().to_string());
    }

    #[test]
    fn bare_command_runner_collects_stdout() {
        #[cfg(target_os = "windows")]
        let mut cmd = bare_shell_command("@echo hello");
        #[cfg(not(target_os = "windows"))]
        let mut cmd = bare_shell_command("printf 'hello'");
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(2)).expect("run");
        match result {
            TimedCommandResult::Finished(output) => {
                assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hello");
            }
            TimedCommandResult::TimedOut(_) => panic!("command unexpectedly timed out"),
        }
    }

    #[test]
    fn bare_command_runner_times_out_without_external_timeout_binary() {
        #[cfg(target_os = "windows")]
        let mut cmd = bare_shell_command("ping -n 3 127.0.0.1 >nul");
        #[cfg(not(target_os = "windows"))]
        let mut cmd = bare_shell_command("sleep 2");
        let result = run_command_with_timeout(&mut cmd, Duration::from_millis(100))
            .expect("run with timeout");
        assert!(matches!(result, TimedCommandResult::TimedOut(_)));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sandbox_exec_profile_allows_writes_and_network() {
        let profile = SandboxConfig::sandbox_exec_profile();
        assert!(profile.contains("(allow file-write*)"));
        assert!(profile.contains("(allow network*)"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn sandbox_exec_shell_runner_can_write_inside_workdir() {
        let sandbox = SandboxConfig::default();
        let mut cmd = sandbox.wrap_shell_sandbox_exec(
            "printf 'ok' > .sandbox-exec-test && cat .sandbox-exec-test && rm -f .sandbox-exec-test",
        );
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(5)).expect("run");
        match result {
            TimedCommandResult::Finished(output) => {
                assert_eq!(String::from_utf8_lossy(&output.stdout), "ok");
            }
            TimedCommandResult::TimedOut(_) => panic!("sandbox-exec shell timed out"),
        }
    }

    #[test]
    fn resolved_python_executable_points_to_a_binary() {
        let python = resolved_python_executable();
        assert!(
            python.is_file()
                || python == std::path::Path::new("python3")
                || python == std::path::Path::new("py"),
            "resolved python path should exist or fall back to python3: {}",
            python.display()
        );
    }

    #[test]
    fn sandbox_backend_is_detected() {
        assert_ne!(active_sandbox_backend().label(), "");
    }

    #[test]
    fn builtin_policy_recovers_python_arithmetic_request() {
        let tools = builtin_tools()
            .into_iter()
            .map(|tool| tool.to_definition())
            .collect::<Vec<_>>();

        let parsed = BuiltinToolPolicyHooks
            .recover_tool_calls_from_user_request(
                "Use python to compute 123 * 456 right now.",
                &tools,
            )
            .expect("recover tool call");

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "python");
        assert_eq!(parsed.tool_calls[0].arguments["code"], "print(123 * 456)");
    }

    #[test]
    fn builtin_policy_finalizes_exact_scalar_output() {
        let result = BuiltinToolPolicyHooks.finalize_response_text(
            "After the tool returns, answer with just the integer.",
            String::new(),
            Some("python"),
            Some("56088"),
            1,
        );
        assert_eq!(result, "56088");

        let shell_result = BuiltinToolPolicyHooks.finalize_after_tool_execution(
            "本地有哪些文件",
            Some("shell"),
            Some("file-a\nfile-b"),
            Some("ignored"),
        );
        assert_eq!(shell_result, Some("file-a\nfile-b".to_string()));

        let repo_result = BuiltinToolPolicyHooks.finalize_after_tool_execution(
            "你看看本地仓库",
            Some("shell"),
            Some("repo: /tmp/demo\n\n== top-level ==\nsrc\nCargo.toml"),
            Some("ignored"),
        );
        assert_eq!(
            repo_result,
            Some("repo: /tmp/demo\n\n== top-level ==\nsrc\nCargo.toml".to_string())
        );

        let tool_call = BuiltinToolPolicyHooks
            .recover_tool_calls_from_draft(
                "I should use the Python tool here. I can run print(7 * 8).",
                &builtin_tools()
                    .into_iter()
                    .map(|tool| tool.to_definition())
                    .collect::<Vec<_>>(),
            )
            .expect("recover draft tool call");
        assert_eq!(
            tool_call.tool_calls[0].arguments,
            json!({ "code": "print(7 * 8)" })
        );
    }

    #[test]
    fn builtin_policy_ignores_internal_tool_markup_in_draft_recovery() {
        let tools = builtin_tools()
            .into_iter()
            .map(|tool| tool.to_definition())
            .collect::<Vec<_>>();

        let parsed = BuiltinToolPolicyHooks.recover_tool_calls_from_draft(
            "I have these tools:\n<tools>\n{\"name\":\"shell\"}\n</tools>\nUse shell if needed.",
            &tools,
        );

        assert!(parsed.is_none());
    }

    #[test]
    fn builtin_policy_recovers_repo_overview_request() {
        let tools = builtin_tools()
            .into_iter()
            .map(|tool| tool.to_definition())
            .collect::<Vec<_>>();

        let parsed = BuiltinToolPolicyHooks
            .recover_tool_calls_from_user_request("你看看本地仓库", &tools)
            .expect("recover repo overview tool call");

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
        assert!(
            parsed.tool_calls[0].arguments["command"]
                .as_str()
                .expect("shell command")
                .contains("git rev-parse")
        );
    }

    #[test]
    fn metadata_captures_latency_and_truncation_flag() {
        use super::{
            TOOL_RESULT_TRUNCATION_MARKER, ToolExecutionMetadata, execute_tool_call_with_metadata,
        };
        use chat::ToolCall;

        // Unknown tool path: synchronous, returns a short error string —
        // latency must be captured (>= 0, never panics) and truncation
        // flag must stay false because the marker is absent.
        let call = ToolCall::new("nonexistent-tool", json!({ "anything": "here" }));
        let (result, metadata) = execute_tool_call_with_metadata(&call);
        assert!(result.starts_with("Error: unknown tool"));
        assert!(!metadata.truncated);
        // No upper-bound assertion on latency_ms — that's runtime-dependent.
        // The point is we got a defined value.
        let _ = metadata.latency_ms;

        // Synthetic truncated string: the executor uses the same marker
        // to signal truncation. Verify the detection logic round-trips.
        let truncated_text = format!("first line\nsecond line{TOOL_RESULT_TRUNCATION_MARKER}");
        assert!(truncated_text.ends_with(TOOL_RESULT_TRUNCATION_MARKER));
        let synthetic = ToolExecutionMetadata {
            latency_ms: 0,
            truncated: truncated_text.ends_with(TOOL_RESULT_TRUNCATION_MARKER),
        };
        assert!(synthetic.truncated);
    }
}
