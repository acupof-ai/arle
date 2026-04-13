use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use infer_chat::{ProtocolToolCall, ProtocolToolDefinition};

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
    *AVAILABLE.get_or_init(|| Path::new("/usr/bin/sandbox-exec").exists())
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
    pub fn to_definition(&self) -> ProtocolToolDefinition {
        ProtocolToolDefinition::new(
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
// Tool execution
// ============================================================================

fn argument_as_str<'a>(arguments: &'a Value, key: &str) -> &'a str {
    arguments.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Execute a tool by name with the given JSON arguments.
pub fn execute_tool(name: &str, arguments: &serde_json::Value) -> String {
    BuiltinToolKind::from_name(name)
        .map(|tool| tool.execute(arguments))
        .unwrap_or_else(|| format!("Error: unknown tool '{name}'"))
}

/// Execute a structured tool call.
pub fn execute_tool_call(call: &ProtocolToolCall) -> String {
    execute_tool(&call.name, &call.arguments)
}

/// Collect stdout + stderr from a process Output into a truncated string.
/// Filters out nsjail's own warning/info lines from stderr.
fn collect_output(output: std::process::Output) -> String {
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
            collect_output(output)
        }
        Ok(TimedCommandResult::TimedOut(output)) => {
            let partial = collect_output(output);
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
            collect_output(output)
        }
        Ok(TimedCommandResult::TimedOut(output)) => {
            let partial = collect_output(output);
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
    use super::{
        SandboxConfig, TimedCommandResult, active_sandbox_backend, default_workdir,
        resolved_python_executable, run_command_with_timeout,
    };
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
                || python == std::path::PathBuf::from("python3")
                || python == std::path::PathBuf::from("py"),
            "resolved python path should exist or fall back to python3: {}",
            python.display()
        );
    }

    #[test]
    fn sandbox_backend_is_detected() {
        assert_ne!(active_sandbox_backend().label(), "");
    }
}
