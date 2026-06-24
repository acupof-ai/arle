//! Per-task filesystem sandbox for agent-based OPD training.
//!
//! Backs an agentic OPD loop where a student LLM edits a real repo checked out
//! at a base commit and we run hidden tests as the reward signal. The
//! [`SandboxToolExecutor`] runs the coding agent's read/write/replace/bash tools
//! against a per-task workdir on the local filesystem (no container — the
//! caller is responsible for isolation if it wants any). The free functions
//! [`boot_workdir`] / [`diff_workdir`] / [`score_workdir`] / [`reset_workdir`]
//! manage the workdir lifecycle: stage a tree, diff the agent's candidate patch,
//! apply the hidden `test_patch` + run `fail_to_pass` tests, and reset between
//! samples.
//!
//! Path resolution is rooted at the per-task `workdir` and rejects `..` escapes
//! so a misbehaving agent can't read or clobber files outside its sandbox.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use agent::ToolExecutor;
use anyhow::{Context, Result, anyhow, bail};
use chat::ToolCall;
use tools::{apply_replace, format_read};

/// Maximum characters of combined stdout/stderr surfaced from a bash tool
/// result before head+tail clipping kicks in.
const BASH_OUTPUT_CLIP: usize = 8000;

/// How often `run_bash` / `score_workdir` poll the child for completion while
/// enforcing the Rust-side timeout. Small enough that the wall-clock timeout
/// stays tight, large enough not to spin.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Run `command` under a wall-clock `timeout`, isolating it in its own process
/// group and capturing combined stdout+stderr to a temp file instead of a pipe.
///
/// This fixes a real hang: a tool command that backgrounds a process (`foo &`)
/// or spawns a child that inherits and holds the stdout pipe makes
/// `wait_with_output()` block reading that pipe until the grandchild closes it —
/// i.e. forever for a daemon, leaving the parent stuck in `wait4()`. Capturing
/// to a file removes the pipe entirely; running in a fresh process group and
/// killing the whole group on exit/timeout reaps any lingering grandchildren so
/// they can't linger or hold the parent.
///
/// Returns `(combined_output, exit_code, killed_by_timeout)`.
fn run_captured(
    mut command: Command,
    timeout: Duration,
) -> std::io::Result<(Vec<u8>, Option<i32>, bool)> {
    use std::os::unix::process::CommandExt;

    let tmp = tempfile::NamedTempFile::new()?;
    let stdout_handle = tmp.reopen()?;
    let stderr_handle = stdout_handle.try_clone()?; // dup'd fd shares the offset → interleaved
    command
        .stdin(Stdio::null())
        .stdout(stdout_handle)
        .stderr(stderr_handle)
        .process_group(0); // new group; pgid == child pid, so we can signal the whole tree

    let mut child = command.spawn()?;
    let pgid = child.id() as i32;

    let deadline = Instant::now() + timeout;
    let mut killed = false;
    let code = loop {
        match child.try_wait()? {
            Some(status) => break status.code(),
            None => {
                if Instant::now() >= deadline {
                    killed = true;
                    kill_group(pgid);
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    };
    // Reap grandchildren that outlived the leader (e.g. a backgrounded process),
    // even on a clean exit — otherwise they linger and can block the parent.
    kill_group(pgid);

    let output = fs::read(tmp.path()).unwrap_or_default();
    Ok((output, code, killed))
}

/// `kill -KILL -<pgid>` — signal the whole process group (negative pid). Uses the
/// `kill` binary to avoid a libc dependency; failure is ignored (the group may be
/// empty already, which is the common case).
fn kill_group(pgid: i32) {
    let _ = Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pgid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Runs the coding agent's read/write/replace/bash tools against a per-task
/// repo directory on the local filesystem.
///
/// `workdir` is both the cwd for bash and the base for path resolution. Tool
/// `path` arguments are joined onto it and validated to stay inside it.
pub struct SandboxToolExecutor {
    /// Per-task repo root (cwd for bash; base for path resolution).
    workdir: PathBuf,
    /// Wall-clock budget for a single `bash` tool invocation. Enforced in Rust
    /// (not the coreutil `timeout`, which is absent on macOS).
    bash_timeout_secs: u64,
    /// Prepended to `PYTHONPATH` for bash (e.g. `"lib:test"`). Any inherited
    /// `PYTHONPATH` is preserved by appending it after this prefix.
    pythonpath: Option<String>,
}

impl SandboxToolExecutor {
    pub fn new(workdir: PathBuf, bash_timeout_secs: u64, pythonpath: Option<String>) -> Self {
        Self {
            workdir,
            bash_timeout_secs,
            pythonpath,
        }
    }

    /// Join `workdir` + `path`, rejecting any `..` escape so the agent cannot
    /// read or write outside its sandbox. Returns `Err(message)` with a
    /// tool-surfaced error string on rejection.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(format!("ERROR: absolute paths are not allowed: {path}"));
        }
        for component in candidate.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(format!("ERROR: path escapes the sandbox ('..'): {path}"));
            }
        }
        Ok(self.workdir.join(candidate))
    }

    /// Spawn `bash -lc <cmd>` in the workdir with a Rust-enforced timeout and
    /// collect stdout+stderr into a single tool result string.
    fn run_bash(&self, cmd: &str) -> String {
        let mut command = Command::new("bash");
        command.arg("-lc").arg(cmd).current_dir(&self.workdir);

        if let Some(prefix) = &self.pythonpath {
            let combined = match std::env::var("PYTHONPATH") {
                Ok(existing) if !existing.is_empty() => format!("{prefix}:{existing}"),
                _ => prefix.clone(),
            };
            command.env("PYTHONPATH", combined);
        }

        match run_captured(command, Duration::from_secs(self.bash_timeout_secs)) {
            Ok((output, code, killed)) => format_bash_output(&output, &[], code, killed),
            Err(e) => format!("ERROR: failed to run bash: {e}"),
        }
    }
}

impl ToolExecutor for SandboxToolExecutor {
    fn execute(&self, call: &ToolCall) -> String {
        match call.name.as_str() {
            "read" => {
                let path = call
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let start = call.arguments.get("start").and_then(|v| v.as_i64());
                let end = call.arguments.get("end").and_then(|v| v.as_i64());
                let resolved = match self.resolve(path) {
                    Ok(p) => p,
                    Err(msg) => return msg,
                };
                if resolved.is_dir() {
                    return format!("ERROR: {path} is a directory (use bash `ls` to list it)");
                }
                let contents = match fs::read_to_string(&resolved) {
                    Ok(c) => c,
                    Err(_) => return format!("ERROR: no such file: {path}"),
                };
                format_read(&contents, path, start, end)
            }
            "write" => {
                let path = call
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let content = call
                    .arguments
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let resolved = match self.resolve(path) {
                    Ok(p) => p,
                    Err(msg) => return msg,
                };
                if let Some(parent) = resolved.parent()
                    && let Err(e) = fs::create_dir_all(parent)
                {
                    return format!("ERROR: failed to create parent dirs for {path}: {e}");
                }
                match fs::write(&resolved, content) {
                    Ok(()) => format!("wrote {} bytes to {path}", content.len()),
                    Err(e) => format!("ERROR: failed to write {path}: {e}"),
                }
            }
            "replace" => {
                let path = call
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let old = call
                    .arguments
                    .get("old")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let new = call
                    .arguments
                    .get("new")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let resolved = match self.resolve(path) {
                    Ok(p) => p,
                    Err(msg) => return msg,
                };
                let contents = match fs::read_to_string(&resolved) {
                    Ok(c) => c,
                    Err(_) => return format!("ERROR: no such file: {path}"),
                };
                match apply_replace(&contents, path, old, new) {
                    Err(msg) => msg,
                    Ok(replaced) => match fs::write(&resolved, replaced) {
                        Ok(()) => format!("OK: replaced 1 occurrence in {path}"),
                        Err(e) => format!("ERROR: failed to write {path}: {e}"),
                    },
                }
            }
            "bash" => {
                let cmd = call
                    .arguments
                    .get("cmd")
                    .or_else(|| call.arguments.get("command"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.run_bash(cmd)
            }
            other => format!("Error: unknown tool '{other}'"),
        }
    }
}

/// Clip a string to `max_chars` keeping HEAD + TAIL. Tests / tracebacks live at
/// the tail, so both ends are preserved. (Local copy — `tools::clip_middle` is
/// private.)
fn clip_middle(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let head_n = max_chars / 2;
    let tail_n = max_chars - head_n;
    let head: String = s.chars().take(head_n).collect();
    let tail: String = s.chars().skip(total - tail_n).collect();
    format!("{head}\n... (output truncated) ...\n{tail}")
}

/// Combine stdout then stderr into a single bash tool result. stderr is
/// prefixed with `[stderr] ` only when non-empty; `(no output)` when both are
/// empty. Appends `[exit {code}]` when the process exited, or
/// `[killed: timeout]` when the Rust timeout fired. Clipped to ~8000 chars.
fn format_bash_output(stdout: &[u8], stderr: &[u8], code: Option<i32>, killed: bool) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);

    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str("[stderr] ");
        combined.push_str(&stderr);
    }
    if combined.is_empty() {
        combined.push_str("(no output)");
    }

    if killed {
        combined.push_str("\n[killed: timeout]");
    } else if let Some(code) = code {
        combined.push_str(&format!("\n[exit {code}]"));
    }

    clip_middle(&combined, BASH_OUTPUT_CLIP)
}

/// Stage a repo tree into a fresh per-task workdir and git-init it so edits can
/// be diffed.
///
/// `staged_tree` is a dir holding the repo already checked out at `base_commit`
/// (cloning/checkout is the caller's job, out of scope). Copies it into
/// `work_root/instance_id`, optionally runs `setup_cmd` (e.g. `pip install`) in
/// the workdir, then `git init` + `add` + commit `"base"` under a deterministic
/// identity. Returns the path to the workdir.
pub fn boot_workdir(
    work_root: &Path,
    instance_id: &str,
    staged_tree: &Path,
    setup_cmd: Option<&str>,
) -> Result<PathBuf> {
    let workdir = work_root.join(instance_id);
    if workdir.exists() {
        fs::remove_dir_all(&workdir)
            .with_context(|| format!("failed to clear existing workdir {}", workdir.display()))?;
    }
    fs::create_dir_all(&workdir)
        .with_context(|| format!("failed to create workdir {}", workdir.display()))?;

    // Copy the staged tree's CONTENTS into the workdir. `<src>/.` copies the
    // directory contents (incl. dotfiles) rather than nesting the dir itself.
    let src_contents = format!("{}/.", staged_tree.display());
    run_checked(
        Command::new("cp")
            .arg("-a")
            .arg(&src_contents)
            .arg(&workdir),
        "cp -a staged tree",
    )?;

    if let Some(setup_cmd) = setup_cmd {
        run_checked(
            Command::new("bash")
                .arg("-lc")
                .arg(setup_cmd)
                .current_dir(&workdir),
            "setup_cmd",
        )?;
    }

    run_checked(
        Command::new("git")
            .arg("init")
            .arg("-q")
            .current_dir(&workdir),
        "git init",
    )?;
    run_checked(
        Command::new("git")
            .arg("add")
            .arg("-A")
            .current_dir(&workdir),
        "git add",
    )?;
    run_checked(
        Command::new("git")
            .args([
                "-c",
                "user.email=a@b",
                "-c",
                "user.name=a",
                "commit",
                "-q",
                "-m",
                "base",
                "--allow-empty",
            ])
            .current_dir(&workdir),
        "git commit base",
    )?;

    Ok(workdir)
}

/// `git -C workdir diff` — the agent's candidate patch (unstaged edits vs the
/// committed base).
pub fn diff_workdir(workdir: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workdir)
        .arg("diff")
        .output()
        .with_context(|| format!("failed to run git diff in {}", workdir.display()))?;
    if !output.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Apply the hidden `test_patch` (skip if empty), then run the `fail_to_pass`
/// tests. Returns `(passed, log_tail)`; `passed` is true iff pytest exits 0.
/// The test command runs under a Rust-enforced timeout (like `run_bash`).
pub fn score_workdir(
    workdir: &Path,
    test_patch: &str,
    fail_to_pass: &[String],
    pythonpath: Option<&str>,
    timeout_secs: u64,
) -> Result<(bool, String)> {
    if !test_patch.trim().is_empty() {
        let patch_file = workdir.join(".arle_test_patch.diff");
        fs::write(&patch_file, test_patch)
            .with_context(|| format!("failed to write test patch {}", patch_file.display()))?;
        let apply = Command::new("git")
            .arg("-C")
            .arg(workdir)
            .arg("apply")
            .arg(&patch_file)
            .output()
            .with_context(|| format!("failed to run git apply in {}", workdir.display()))?;
        let _ = fs::remove_file(&patch_file);
        if !apply.status.success() {
            bail!(
                "git apply of test_patch failed: {}",
                String::from_utf8_lossy(&apply.stderr)
            );
        }
    }

    // `python3 -m pytest <tests... each quoted> -q -p no:cacheprovider`.
    let quoted: Vec<String> = fail_to_pass
        .iter()
        .map(|t| format!("'{}'", t.replace('\'', r"'\''")))
        .collect();
    let cmd = format!(
        "python3 -m pytest {} -q -p no:cacheprovider",
        quoted.join(" ")
    );

    let mut command = Command::new("bash");
    command.arg("-lc").arg(&cmd).current_dir(workdir);
    if let Some(pythonpath) = pythonpath {
        let combined = match std::env::var("PYTHONPATH") {
            Ok(existing) if !existing.is_empty() => format!("{pythonpath}:{existing}"),
            _ => pythonpath.to_string(),
        };
        command.env("PYTHONPATH", combined);
    }

    let (output, code, killed) = run_captured(command, Duration::from_secs(timeout_secs))
        .with_context(|| "failed to run pytest".to_string())?;
    let passed = !killed && code == Some(0);
    let log_tail = format_bash_output(&output, &[], code, killed);
    Ok((passed, log_tail))
}

/// Discard the agent's edits + any applied `test_patch`, restoring the
/// committed base for the next sample: `git checkout -- .` then
/// `git clean -fdq`.
pub fn reset_workdir(workdir: &Path) -> Result<()> {
    run_checked(
        Command::new("git")
            .arg("-C")
            .arg(workdir)
            .args(["checkout", "--", "."]),
        "git checkout",
    )?;
    run_checked(
        Command::new("git")
            .arg("-C")
            .arg(workdir)
            .args(["clean", "-fdq"]),
        "git clean",
    )?;
    Ok(())
}

/// Run a `Command` and return an error including stderr if it does not exit 0.
fn run_checked(command: &mut Command, label: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("failed to spawn `{label}`"))?;
    if !output.status.success() {
        return Err(anyhow!(
            "`{label}` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::process::Command;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall::new(name, args)
    }

    #[test]
    fn bash_does_not_hang_on_backgrounded_child() {
        // Regression: a tool command that backgrounds a process inheriting the
        // stdout pipe used to block `wait_with_output()` until that grandchild
        // exited (forever for a daemon), stranding the parent in `wait4()`.
        // `run_captured` writes to a temp file (no pipe to hold) and kills the
        // whole process group, so `execute` must return promptly.
        let tmp = tempfile::tempdir().unwrap();
        let exec = SandboxToolExecutor::new(tmp.path().to_path_buf(), 30, None);
        let start = std::time::Instant::now();
        let out = exec.execute(&call("bash", json!({"command": "echo ready; sleep 30 &"})));
        let elapsed = start.elapsed();
        assert!(out.contains("ready"), "bash output: {out}");
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "run_bash blocked {}s on a backgrounded child — the pipe-hang regressed",
            elapsed.as_secs()
        );
    }

    #[test]
    fn executor_read_write_replace_bash() {
        let tmp = tempfile::tempdir().unwrap();
        let exec = SandboxToolExecutor::new(tmp.path().to_path_buf(), 30, None);

        let wrote = exec.execute(&call(
            "write",
            json!({"path": "hello.txt", "content": "alpha UNIQUE_TOKEN omega\n"}),
        ));
        assert!(wrote.contains("wrote"), "write result: {wrote}");

        let read = exec.execute(&call("read", json!({"path": "hello.txt"})));
        assert!(
            read.contains("UNIQUE_TOKEN") && read.contains("1\t"),
            "read result: {read}"
        );

        let replaced = exec.execute(&call(
            "replace",
            json!({"path": "hello.txt", "old": "UNIQUE_TOKEN", "new": "REPLACED"}),
        ));
        assert!(
            replaced.contains("OK: replaced"),
            "replace result: {replaced}"
        );

        let bashed = exec.execute(&call("bash", json!({"cmd": "echo hi"})));
        assert!(
            bashed.contains("hi") && bashed.contains("[exit 0]"),
            "bash result: {bashed}"
        );

        // `..` escape is rejected.
        let escaped = exec.execute(&call("read", json!({"path": "../secret"})));
        assert!(escaped.contains("escapes the sandbox"), "escape: {escaped}");
    }

    #[test]
    fn boot_diff_reset_roundtrip() {
        let work_root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        fs::write(staged.path().join("a.txt"), "x\n").unwrap();

        let workdir = boot_workdir(work_root.path(), "inst1", staged.path(), None).unwrap();
        assert!(workdir.join("a.txt").exists());

        let exec = SandboxToolExecutor::new(workdir.clone(), 30, None);
        let wrote = exec.execute(&call("write", json!({"path": "a.txt", "content": "y\n"})));
        assert!(wrote.contains("wrote"), "write: {wrote}");

        let diff = diff_workdir(&workdir).unwrap();
        assert!(diff.contains("a.txt"), "diff should mention a.txt: {diff}");

        reset_workdir(&workdir).unwrap();
        let diff_after = diff_workdir(&workdir).unwrap();
        assert!(
            diff_after.trim().is_empty(),
            "diff after reset should be empty: {diff_after}"
        );
    }

    #[test]
    fn score_detects_failing_test() {
        // Gate on pytest availability — skip if python3/pytest is absent.
        let probe = Command::new("python3")
            .args(["-m", "pytest", "--version"])
            .output();
        match probe {
            Ok(out) if out.status.success() => {}
            _ => return,
        }

        let work_root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        fs::write(
            staged.path().join("test_x.py"),
            "def test_y():\n    assert False\n",
        )
        .unwrap();

        let workdir = boot_workdir(work_root.path(), "inst_score", staged.path(), None).unwrap();
        let (passed, log) =
            score_workdir(&workdir, "", &["test_x.py::test_y".into()], None, 60).unwrap();
        assert!(!passed, "failing test must not pass; log: {log}");
    }
}
