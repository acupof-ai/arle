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
    Bash,
    Python,
    Read,
    Write,
    Replace,
    Ocr,
}

impl BuiltinToolKind {
    const ALL: [Self; 6] = [
        Self::Bash,
        Self::Python,
        Self::Read,
        Self::Write,
        Self::Replace,
        Self::Ocr,
    ];

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "bash" => Some(Self::Bash),
            "python" => Some(Self::Python),
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "replace" => Some(Self::Replace),
            "ocr" => Some(Self::Ocr),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::Python => "python",
            Self::Read => "read",
            Self::Write => "write",
            Self::Replace => "replace",
            Self::Ocr => "ocr",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Bash => "Run a bash command.",
            Self::Python => "Run Python 3 code.",
            Self::Read => {
                "Read a file and return numbered lines, optionally sliced to a 1-based line range."
            }
            Self::Write => {
                "Write content to a file, creating parent directories and overwriting any existing file."
            }
            Self::Replace => "Replace a single unique occurrence of a string in a file.",
            Self::Ocr => {
                "Extract text from an image file (PNG/JPG/…) or image URL using the local \
                 DeepSeek-OCR model. The model downloads automatically on first use \
                 (Apple Silicon / Metal only). Pass mode=markdown for documents/tables."
            }
        }
    }

    fn parameters(self) -> Value {
        match self {
            Self::Bash => json!({
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
            Self::Read => json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "start": {
                        "type": "integer"
                    },
                    "end": {
                        "type": "integer"
                    }
                },
                "required": ["path"]
            }),
            Self::Write => json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    }
                },
                "required": ["path", "content"]
            }),
            Self::Replace => json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "old": {
                        "type": "string"
                    },
                    "new": {
                        "type": "string"
                    }
                },
                "required": ["path", "old", "new"]
            }),
            Self::Ocr => json!({
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Path to a local image file or an http(s) image URL."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["free", "grounding", "markdown"],
                        "description": "free = plain text (default); markdown = document/table layout; grounding = text + bounding boxes."
                    }
                },
                "required": ["image"]
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
            Self::Bash => execute_bash(argument_as_str(arguments, "command")),
            Self::Python => execute_python(argument_as_str(arguments, "code")),
            Self::Read => execute_read(
                argument_as_str(arguments, "path"),
                argument_as_i64(arguments, "start"),
                argument_as_i64(arguments, "end"),
            ),
            Self::Write => execute_write(
                argument_as_str(arguments, "path"),
                argument_as_str(arguments, "content"),
            ),
            Self::Replace => execute_replace(
                argument_as_str(arguments, "path"),
                argument_as_str(arguments, "old"),
                argument_as_str(arguments, "new"),
            ),
            Self::Ocr => execute_ocr(
                argument_as_str(arguments, "image"),
                argument_as_str(arguments, "mode"),
            ),
        }
    }
}

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
    fn configure_native_command(&self, cmd: &mut Command) {
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env("PATH", default_env_path());
        cmd.env(
            "HOME",
            std::env::var("HOME").unwrap_or_else(|_| self.workdir.clone()),
        );
        cmd.env("TMPDIR", effective_tmpdir());
        cmd.env("LANG", "C.UTF-8");
    }

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

        // python, pip packages, cuda libs live under /usr/local
        if std::path::Path::new("/usr/local").exists() {
            cmd.arg("-R").arg("/usr/local");
        }

        cmd.arg("-B").arg(&self.workdir);
        if self.workdir != "/tmp" {
            cmd.arg("-B").arg("/tmp");
        }

        cmd.arg("--cwd").arg(&self.workdir);

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
        self.configure_native_command(&mut cmd);
        cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        cmd
    }

    #[cfg(not(target_os = "windows"))]
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(user_cmd);
        self.configure_native_command(&mut cmd);
        cmd
    }

    #[cfg(target_os = "windows")]
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("cmd");
        cmd.arg("/d").arg("/c").arg(user_cmd);
        self.configure_native_command(&mut cmd);
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

pub fn builtin_tools() -> Vec<Tool> {
    BuiltinToolKind::ALL
        .into_iter()
        .map(BuiltinToolKind::into_tool)
        .collect()
}

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

        if tool_available(tools, "bash")
            && mentions_shell_tool(user_input)
            && let Some(command) = extract_shell_command(user_input)
        {
            return Some(single_tool_response("bash", json!({ "command": command })));
        }

        if tool_available(tools, "bash") && asks_for_file_listing(user_input) {
            return Some(single_tool_response(
                "bash",
                json!({ "command": default_directory_listing_command() }),
            ));
        }

        if tool_available(tools, "bash") && asks_for_repository_overview(user_input) {
            return Some(single_tool_response(
                "bash",
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

        if tool_available(tools, "bash")
            && mentions_shell_tool(draft)
            && let Some(command) = extract_shell_command(draft)
        {
            return Some(single_tool_response("bash", json!({ "command": command })));
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
        if last_tool_name == Some("bash") && should_return_shell_result_directly(user_input) {
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

    (!best.is_empty()).then_some(best)
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

fn argument_as_str<'a>(arguments: &'a Value, key: &str) -> &'a str {
    arguments.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Optional integer argument (absent / non-integer -> `None`).
fn argument_as_i64(arguments: &Value, key: &str) -> Option<i64> {
    arguments.get(key).and_then(Value::as_i64)
}

/// Resolve a tool-supplied path against the sandbox workdir, mirroring how
/// [`execute_bash`] / [`execute_python`] run with `cwd = SandboxConfig.workdir`.
/// Absolute paths are used as-is.
fn resolve_sandbox_path(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        PathBuf::from(SandboxConfig::default().workdir).join(p)
    }
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
    let truncated = result.contains(TOOL_RESULT_TRUNCATION_MARKER);
    (
        result,
        ToolExecutionMetadata {
            latency_ms,
            truncated,
        },
    )
}

/// Truncate `s` to ~`max_chars`, keeping the head AND tail with the middle
/// elided. Command/test output puts the important summary (pytest failures,
/// tracebacks) at the END, so a head-only cut drops exactly what the agent
/// needs. The elision embeds [`TOOL_RESULT_TRUNCATION_MARKER`] so callers
/// detect truncation via `contains`.
fn clip_middle(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let head_n = max_chars / 2;
    let tail_n = max_chars - head_n;
    let head: String = s.chars().take(head_n).collect();
    let tail: String = s.chars().skip(total - tail_n).collect();
    let omitted = total - head_n - tail_n;
    format!("{head}{TOOL_RESULT_TRUNCATION_MARKER}: {omitted} chars omitted ...\n{tail}")
}

/// Filters nsjail's own warning/info lines from stderr and appends the exit
/// code so the agent can tell success from failure.
fn collect_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let raw_stderr = String::from_utf8_lossy(&output.stderr);

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
    let mut result = clip_middle(&result, 8000);
    if let Some(code) = output.status.code() {
        result.push_str(&format!("\n[exit {code}]"));
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

fn execute_bash(command: &str) -> String {
    let sandbox = SandboxConfig::default();
    log::info!(
        "Executing bash ({}, timeout={}s): {}",
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

/// Pure formatting half of [`execute_read`]: given in-memory file `contents`,
/// produce the numbered-line view (1-based `{n}\t{line}`) with the same
/// windowing, footer, per-line clip and `clip_middle` backstop. Does NO
/// filesystem IO — `path` is used only for the empty-file note. Shared by the
/// local-fs reader and the container-sandbox executor so both hardened views
/// stay byte-for-byte identical.
///
/// `start`/`end` are inclusive 1-based bounds; either may be omitted. Without
/// `end`, at most `READ_WINDOW` lines from `start` are returned and a
/// `[lines s-e of total]` footer tells the agent more remains. Over-long lines
/// are clipped so a minified file can't blow the context.
pub fn format_read(contents: &str, path: &str, start: Option<i64>, end: Option<i64>) -> String {
    const READ_WINDOW: i64 = 400;
    const MAX_LINE_CHARS: usize = 2000;

    let lines: Vec<&str> = contents.lines().collect();
    let total = lines.len() as i64;
    if total == 0 {
        return format!("(file {path} is empty)");
    }

    let start = start.unwrap_or(1).clamp(1, total);
    let end = end.unwrap_or(start + READ_WINDOW - 1).clamp(start, total);

    let joined = (start..=end)
        .map(|n| {
            let line = lines[(n - 1) as usize];
            if line.chars().count() > MAX_LINE_CHARS {
                let clipped: String = line.chars().take(MAX_LINE_CHARS).collect();
                format!("{n}\t{clipped}… (line truncated)")
            } else {
                format!("{n}\t{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = clip_middle(&joined, 20_000);
    if start > 1 || end < total {
        out.push_str(&format!("\n[lines {start}-{end} of {total}]"));
    }
    out
}

fn execute_read(path: &str, start: Option<i64>, end: Option<i64>) -> String {
    let resolved = resolve_sandbox_path(path);
    if resolved.is_dir() {
        return format!("ERROR: {path} is a directory (use bash `ls` to list it)");
    }
    let contents = match std::fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(_) => return format!("ERROR: no such file: {path}"),
    };

    format_read(&contents, path, start, end)
}

fn execute_write(path: &str, content: &str) -> String {
    let resolved = resolve_sandbox_path(path);
    if let Some(parent) = resolved.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("ERROR: failed to create parent dirs for {path}: {e}");
    }
    match std::fs::write(&resolved, content) {
        Ok(()) => format!("wrote {} bytes to {path}", content.len()),
        Err(e) => format!("ERROR: failed to write {path}: {e}"),
    }
}

/// Pure match-counting + guard half of [`execute_replace`]: given in-memory
/// `contents`, validate the `old`/`new` pair and produce the post-replace
/// string. Does NO filesystem IO. On any guard failure (empty `old`,
/// `old == new`, 0 matches, >1 matches) returns `Err(message)` with the exact
/// same error strings the tool surfaces; on a unique match returns
/// `Ok(new_file_contents)`. `path` is used only in the error messages. Shared
/// by the local-fs replacer and the container-sandbox executor.
pub fn apply_replace(contents: &str, path: &str, old: &str, new: &str) -> Result<String, String> {
    if old.is_empty() {
        return Err("ERROR: 'old' is empty; provide the exact text to replace".to_string());
    }
    if old == new {
        return Err("ERROR: 'old' and 'new' are identical; nothing to change".to_string());
    }

    let count = contents.matches(old).count();
    match count {
        0 => Err(format!(
            "ERROR: 'old' not found in {path}; copy it EXACTLY incl. whitespace/indentation (read the file to check)"
        )),
        1 => Ok(contents.replacen(old, new, 1)),
        n => Err(format!(
            "ERROR: 'old' occurs {n} times in {path}; add surrounding lines to make the match unique"
        )),
    }
}

fn execute_replace(path: &str, old: &str, new: &str) -> String {
    let resolved = resolve_sandbox_path(path);
    let contents = match std::fs::read_to_string(&resolved) {
        Ok(c) => c,
        Err(_) => return format!("ERROR: no such file: {path}"),
    };

    match apply_replace(&contents, path, old, new) {
        Err(msg) => msg,
        Ok(replaced) => match std::fs::write(&resolved, replaced) {
            Ok(()) => format!("OK: replaced 1 occurrence in {path}"),
            Err(e) => format!("ERROR: failed to write {path}: {e}"),
        },
    }
}

/// Canonical DeepSeek-OCR model id (mirrors `cli::model_catalog::DEEPSEEK_OCR_MODEL_ID`).
const OCR_MODEL_ID: &str = "sahilchachra/unlimited-ocr-mxfp8-mlx";

/// Run OCR on an image by shelling out to `arle ocr <image> --json --mode <mode>`.
///
/// Self-invokes the running binary (`current_exe`) so the agent reuses the exact
/// same OCR path as the CLI: backend isolation (no GPU deps in this crate), no
/// second big model held in-process, and auto-download on first use. Reports
/// whether the model is already cached locally so the agent can set expectations
/// (first call may take a minute to download ~3.6 GB).
fn execute_ocr(image: &str, mode: &str) -> String {
    let image = image.trim();
    if image.is_empty() {
        return "ERROR: 'image' is required (a local path or http(s) URL)".to_string();
    }
    // Resolve a local path against the sandbox workdir so relative paths match
    // the other tools; URLs and absolute paths pass through unchanged.
    let image_arg = if image.starts_with("http://") || image.starts_with("https://") {
        image.to_string()
    } else {
        resolve_sandbox_path(image).to_string_lossy().into_owned()
    };

    let mode = match mode.trim() {
        "" | "free" => "free",
        "grounding" => "grounding",
        "markdown" => "markdown",
        other => return format!("ERROR: unknown ocr mode '{other}' (free|grounding|markdown)"),
    };

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return format!("ERROR: cannot locate the arle binary for OCR: {e}"),
    };

    let model_cached = ocr_model_is_cached();
    if !model_cached {
        log::info!(
            "ocr tool: DeepSeek-OCR model not cached; `arle ocr` will download it on first use"
        );
    }

    let mut cmd = Command::new(&exe);
    cmd.arg("ocr")
        .arg(&image_arg)
        .arg("--mode")
        .arg(mode)
        .arg("--json");
    // OCR (load + optional download + inference) can exceed the shell sandbox's
    // 30s budget; give it a dedicated, generous timeout.
    let timeout = if model_cached {
        Duration::from_secs(180)
    } else {
        Duration::from_secs(900)
    };
    match run_command_with_timeout(&mut cmd, timeout) {
        Ok(TimedCommandResult::Finished(output)) => {
            if output.status.success() {
                let text = ocr_extract_text(&output.stdout);
                if text.trim().is_empty() {
                    "(no text detected in the image)".to_string()
                } else {
                    text
                }
            } else {
                let err = String::from_utf8_lossy(&output.stderr);
                format!(
                    "ERROR: OCR failed: {}",
                    err.lines().last().unwrap_or("unknown error")
                )
            }
        }
        Ok(TimedCommandResult::TimedOut(_)) => {
            "ERROR: OCR timed out (model download or inference took too long)".to_string()
        }
        Err(e) => format!("ERROR: failed to run OCR: {e}"),
    }
}

/// Parse the `{ "text": … }` document `arle ocr --json` prints; fall back to the
/// raw stdout if it isn't the expected shape.
fn ocr_extract_text(stdout: &[u8]) -> String {
    let raw = String::from_utf8_lossy(stdout);
    let line = raw.lines().rev().find(|l| l.trim_start().starts_with('{'));
    if let Some(line) = line
        && let Ok(v) = serde_json::from_str::<Value>(line)
        && let Some(text) = v.get("text").and_then(Value::as_str)
    {
        return text.to_string();
    }
    raw.trim().to_string()
}

/// Best-effort check for whether the DeepSeek-OCR model is already in the HF
/// cache (`$HF_HOME/hub` / `~/.cache/huggingface/hub/models--<org>--<repo>` with
/// a non-empty `snapshots/`). Kept dependency-light so the `tools` crate stays
/// host-only (no infer-util / infer-metal dep).
fn ocr_model_is_cached() -> bool {
    let hub_root = if let Some(v) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(v)
    } else if let Some(v) = std::env::var_os("HF_HOME") {
        PathBuf::from(v).join("hub")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache/huggingface/hub")
    } else {
        return false;
    };
    let dir_name = format!("models--{}", OCR_MODEL_ID.replace('/', "--"));
    let snapshots = hub_root.join(dir_name).join("snapshots");
    std::fs::read_dir(&snapshots)
        .map(|mut entries| entries.any(|e| e.is_ok()))
        .unwrap_or(false)
}
