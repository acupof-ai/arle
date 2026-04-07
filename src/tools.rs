use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use infer::chat_protocol::{ToolCall as ProtocolToolCall, ToolDefinition};

static SCRIPT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinToolKind {
    Shell,
    Python,
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
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
            Self::Shell => {
                "Execute a shell command and return its stdout and stderr. Use this for file operations, system inspection, running programs, etc."
            }
            Self::Python => {
                "Execute a Python 3 code snippet and return its stdout and stderr. Use this for calculations, data processing, or any task best done in Python."
            }
        }
    }

    fn parameters(self) -> Value {
        match self {
            Self::Shell => json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }),
            Self::Python => json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python 3 code to execute"
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
// Sandbox configuration — nsjail backend
// ============================================================================

/// Process sandbox using nsjail.
///
/// Provides:
///   - Mount namespace: rootfs is read-only bind mounts, only /tmp is writable
///   - PID namespace: isolated PID tree
///   - Network isolation: no network access by default
///   - Time limit: SIGKILL after `timeout_secs`
///   - Memory limit: RLIMIT_AS capped at `max_memory_mb`
///   - Minimal environment: only PATH, HOME, TMPDIR, LANG
///
/// Falls back to bare execution if nsjail is not found.
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

/// Check once whether nsjail is available.
fn nsjail_available() -> bool {
    use std::sync::OnceLock;
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

impl SandboxConfig {
    /// Build a Command that runs `user_cmd` inside nsjail.
    fn wrap_shell(&self, user_cmd: &str) -> Command {
        if !nsjail_available() {
            log::warn!("nsjail not found — running without sandbox");
            return self.wrap_shell_bare(user_cmd);
        }

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
        cmd.arg("--env").arg("PATH=/usr/local/bin:/usr/bin:/bin");
        cmd.arg("--env").arg(format!("HOME={}", self.workdir));
        cmd.arg("--env").arg(format!("TMPDIR={}", self.workdir));
        cmd.arg("--env").arg("LANG=C.UTF-8");
        cmd.arg("--env").arg("PYTHONDONTWRITEBYTECODE=1");

        cmd.arg("--").arg("/bin/bash").arg("-c").arg(user_cmd);
        cmd
    }

    /// Bare fallback when nsjail is unavailable.
    fn wrap_shell_bare(&self, user_cmd: &str) -> Command {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(user_cmd);
        cmd.current_dir(&self.workdir);
        cmd.env_clear();
        cmd.env(
            "PATH",
            std::env::var("PATH").unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".to_string()),
        );
        cmd.env(
            "HOME",
            std::env::var("HOME").unwrap_or_else(|_| self.workdir.clone()),
        );
        cmd.env(
            "TMPDIR",
            std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string()),
        );
        cmd.env("LANG", "C.UTF-8");
        cmd
    }

    /// Build a sandboxed Command for running a Python snippet.
    fn wrap_python(&self, code: &str) -> std::io::Result<Command> {
        let seq = SCRIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
        let script_path = format!(
            "{}/sandbox_py_{}_{}.py",
            self.workdir,
            std::process::id(),
            seq
        );
        std::fs::write(&script_path, code)?;
        let shell_cmd = format!(
            "python3 -u {} ; _rc=$?; rm -f {} ; exit $_rc",
            script_path, script_path
        );
        Ok(self.wrap_shell(&shell_cmd))
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
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
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
    log::info!("Executing shell (nsjail): {}", command);
    let sandbox = SandboxConfig::default();
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
    log::info!("Executing python snippet (nsjail, {} chars)", code.len());
    let sandbox = SandboxConfig::default();
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
    use super::{default_workdir, run_command_with_timeout};
    use std::process::Command;
    use std::time::Duration;

    #[test]
    fn default_workdir_uses_current_directory() {
        let expected = std::env::current_dir().expect("current_dir");
        assert_eq!(default_workdir(), expected.display().to_string());
    }

    #[test]
    fn bare_command_runner_collects_stdout() {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg("printf 'hello'");
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(2)).expect("run");
        match result {
            super::TimedCommandResult::Finished(output) => {
                assert_eq!(String::from_utf8_lossy(&output.stdout), "hello");
            }
            super::TimedCommandResult::TimedOut(_) => panic!("command unexpectedly timed out"),
        }
    }

    #[test]
    fn bare_command_runner_times_out_without_external_timeout_binary() {
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg("sleep 2");
        let result = run_command_with_timeout(&mut cmd, Duration::from_millis(100))
            .expect("run with timeout");
        assert!(matches!(result, super::TimedCommandResult::TimedOut(_)));
    }
}
