//! Per-task filesystem sandbox for agent-based OPD training.
//!
//! Backs the cc-harness agentic OPD loop ([`crate::cc_harness`]): the agent
//! edits a real repo checked out at a base commit and hidden tests are the
//! reward signal. No container — the caller is responsible for isolation if it
//! wants any. [`run_captured`] is the fork-safe subprocess primitive both
//! scoring and the cc spawn use.
//!
//! Without a container the one isolation this module does enforce is import
//! resolution: [`workdir_pythonpath`] puts the task's own tree ahead of
//! site-packages for both the agent and the scorer, and the harness refuses
//! the agent's bare `pip install`. Everything else the agent's Bash can reach
//! is still shared.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};

use crate::spawner::{SpawnClient, SpawnRequest, SpawnResponse};

fn request_from_command(
    command: &Command,
    combined_timeout: bool,
    timeout: Duration,
) -> SpawnRequest {
    SpawnRequest {
        program: command.get_program().to_string_lossy().into_owned(),
        args: command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect(),
        cwd: command.get_current_dir().map(|d| d.to_owned()),
        env: command
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect(),
        combined_timeout,
        timeout_s: timeout.as_secs(),
    }
}

fn spawn_via_helper(
    client: &SpawnClient,
    command: &Command,
    combined_timeout: bool,
    timeout: Duration,
) -> std::io::Result<SpawnResponse> {
    client.run(&request_from_command(command, combined_timeout, timeout))
}

struct PlainOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    success: bool,
    status: String,
}

fn plain_output(command: &mut Command, label: &str) -> Result<PlainOutput> {
    if let Some(client) = SpawnClient::from_env() {
        let resp = spawn_via_helper(&client, command, false, Duration::ZERO)
            .with_context(|| format!("failed to spawn `{label}` via sandbox-spawner"))?;
        let success = resp.exit_code == Some(0);
        return Ok(PlainOutput {
            stdout: resp.stdout,
            stderr: resp.stderr,
            success,
            status: match resp.exit_code {
                Some(c) => format!("exit code: {c}"),
                None => "terminated by signal".to_string(),
            },
        });
    }
    let output = command
        .output()
        .with_context(|| format!("failed to spawn `{label}`"))?;
    Ok(PlainOutput {
        status: output.status.to_string(),
        success: output.status.success(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

const BASH_OUTPUT_CLIP: usize = 8000;

const POLL_INTERVAL: Duration = Duration::from_millis(50);

pub(crate) fn run_captured(
    command: Command,
    timeout: Duration,
) -> std::io::Result<(Vec<u8>, Option<i32>, bool)> {
    // Route through the pre-CUDA spawner helper when set (dodges ELKEID);
    // otherwise spawn directly (byte-identical default).
    if let Some(client) = SpawnClient::from_env() {
        let resp = spawn_via_helper(&client, &command, true, timeout)?;
        return Ok((resp.stdout, resp.exit_code, resp.timed_out));
    }

    let tmp = tempfile::NamedTempFile::new()?;
    let stdout_handle = tmp.reopen()?;
    let stderr_handle = stdout_handle.try_clone()?; // dup'd fd shares the offset → interleaved

    // Use the standalone `setsid` binary, NOT `Command::process_group(0)`.
    // The latter runs `setpgid` in a pre_exec hook between fork() and exec();
    // in a multi-threaded CUDA-resident parent, fork() snapshots a libc/CUDA
    // lock held by another thread and the child's setpgid deadlocks before
    // exec. `setsid` creates the new session AFTER exec, so no in-child hook.
    // macOS has no `setsid` binary: spawn directly there.
    let mut command = if cfg!(target_os = "linux") {
        let prog = command.get_program().to_owned();
        let args: Vec<_> = command.get_args().map(|a| a.to_owned()).collect();
        let envs: Vec<_> = command
            .get_envs()
            .map(|(k, v)| (k.to_owned(), v.map(|v| v.to_owned())))
            .collect();
        let cwd = command.get_current_dir().map(|d| d.to_owned());

        let mut wrapped = Command::new("setsid");
        wrapped.arg(prog).args(args);
        if let Some(dir) = cwd {
            wrapped.current_dir(dir);
        }
        for (key, val) in envs {
            match val {
                Some(v) => {
                    wrapped.env(key, v);
                }
                None => {
                    wrapped.env_remove(key);
                }
            }
        }
        wrapped
    } else {
        command
    };
    command
        .stdin(Stdio::null())
        .stdout(stdout_handle)
        .stderr(stderr_handle);

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
                    let _ = child.kill(); // direct child (the whole tree on Linux, via the group kill below)
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

fn kill_group(pgid: i32) {
    // Never signal our own group: a mis-derived pgid must not SIGKILL the
    // caller's session (observed under `cargo test -p train --features cuda`).
    // SAFETY: getpgrp takes no arguments and cannot fail.
    if pgid <= 1 || pgid == unsafe { libc::getpgrp() } {
        return;
    }
    let _ = Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pgid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

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

pub fn boot_workdir(
    work_root: &Path,
    instance_id: &str,
    staged_tree: &Path,
    setup_cmd: Option<&str>,
) -> Result<PathBuf> {
    // An ancestor CLAUDE.md costs ~31K tokens/request of CC preamble.
    static ANCESTOR_CLAUDE_MD_WARN: std::sync::Once = std::sync::Once::new();
    if let Some(dir) = work_root.ancestors().find(|a| a.join("CLAUDE.md").exists()) {
        ANCESTOR_CLAUDE_MD_WARN.call_once(|| {
            eprintln!(
                "[sandbox] WARN: work root {} sits under {} which carries CLAUDE.md — CC ingests \
                 it as per-request preamble; stage --work-root outside any repo",
                work_root.display(),
                dir.display()
            );
        });
    }

    let workdir = work_root.join(instance_id);
    if workdir.exists() {
        fs::remove_dir_all(&workdir)
            .with_context(|| format!("failed to clear existing workdir {}", workdir.display()))?;
    }
    fs::create_dir_all(&workdir)
        .with_context(|| format!("failed to create workdir {}", workdir.display()))?;

    // `<src>/.` copies contents (incl. dotfiles) rather than nesting the dir.
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

pub fn diff_workdir(workdir: &Path) -> Result<String> {
    let mut command = Command::new("git");
    command.arg("-C").arg(workdir).arg("diff");
    let output = plain_output(&mut command, "git diff")?;
    if !output.success {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `PYTHONPATH` that resolves the task's OWN tree first.
///
/// Scoring an import against anything else is not scoring the edit. A stale
/// editable install is enough to break this: an agent that ran `pip install -e .`
/// inside one rollout leaves a `.pth` in the shared site-packages pointing at
/// that rollout's directory, and from then on every task's `import <pkg>` on the
/// box resolves there. That happened on the H20 pod (2026-08-23) and silently
/// scored six flake8 tasks against one unrelated leftover.
///
/// `src` is included when present so a src-layout repo resolves without the
/// operator having to know its layout. `extra` is the caller's
/// `--pythonpath`, sandbox-relative, and still comes after.
pub(crate) fn workdir_pythonpath(workdir: &Path, extra: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    let src = workdir.join("src");
    if src.is_dir() {
        parts.push(src.to_string_lossy().into_owned());
    }
    parts.push(workdir.to_string_lossy().into_owned());
    if let Some(extra) = extra.filter(|e| !e.is_empty()) {
        parts.push(extra.to_owned());
    }
    if let Ok(existing) = std::env::var("PYTHONPATH")
        && !existing.is_empty()
    {
        parts.push(existing);
    }
    parts.join(":")
}

pub fn score_workdir(
    workdir: &Path,
    test_patch: &str,
    fail_to_pass: &[String],
    pythonpath: Option<&str>,
    timeout_secs: u64,
) -> Result<(f32, String)> {
    if !test_patch.trim().is_empty() {
        // Reset the patch's test paths to base before applying, so `git apply`
        // lands cleanly even if the rollout dirtied a test file. The student's
        // source fix in other files is preserved.
        // A `+++ b/<path>` is a real header only when it follows a `--- ` line.
        let mut after_minus_header = false;
        for line in test_patch.lines() {
            if after_minus_header && let Some(path) = line.strip_prefix("+++ b/") {
                let path = path.trim_end();
                if !path.is_empty() && path != "/dev/null" {
                    let mut checkout = Command::new("git");
                    checkout
                        .arg("-C")
                        .arg(workdir)
                        .args(["checkout", "--", path]);
                    // Ignore failure: a path the patch CREATES has no base to reset to.
                    let _ = plain_output(&mut checkout, "git checkout test path");
                }
            }
            after_minus_header = line.starts_with("--- ");
        }

        // Hand `git apply` the bare filename: `git -C workdir` resolves it
        // relative to workdir, so a workdir-relative path would double the prefix.
        let patch_name = ".arle_test_patch.diff";
        let patch_file = workdir.join(patch_name);
        fs::write(&patch_file, test_patch)
            .with_context(|| format!("failed to write test patch {}", patch_file.display()))?;
        let mut apply_cmd = Command::new("git");
        apply_cmd
            .arg("-C")
            .arg(workdir)
            .arg("apply")
            .arg(patch_name);
        let apply = plain_output(&mut apply_cmd, "git apply")?;
        let _ = fs::remove_file(&patch_file);
        if !apply.success {
            bail!(
                "git apply of test_patch failed: {}",
                String::from_utf8_lossy(&apply.stderr)
            );
        }
    }

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
    // No bytecode: scoring must execute edited source, not cached pyc.
    command.env("PYTHONDONTWRITEBYTECODE", "1");
    command.env("PYTHONPATH", workdir_pythonpath(workdir, pythonpath));

    let (output, code, killed) = run_captured(command, Duration::from_secs(timeout_secs))
        .with_context(|| "failed to run pytest".to_string())?;
    let log_tail = format_bash_output(&output, &[], code, killed);

    // Fall back to binary (exit-0 → 1.0) when killed or summary unparseable,
    // so a timed-out/crashed run never mints spurious credit.
    let reward = match parse_pytest_counts(&String::from_utf8_lossy(&output)) {
        Some((passed, _)) if !killed => {
            (passed as f32 / fail_to_pass.len().max(1) as f32).clamp(0.0, 1.0)
        }
        _ => {
            if code == Some(0) {
                1.0
            } else {
                0.0
            }
        }
    };
    Ok((reward, log_tail))
}

fn parse_pytest_counts(text: &str) -> Option<(usize, usize)> {
    let count_of = |line: &str, kind: &str| -> Option<usize> {
        line.split(", ").find_map(|seg| {
            let seg = seg.strip_prefix("= ").unwrap_or(seg);
            let (n, rest) = seg.trim().split_once(' ')?;
            // First word of `rest` is the status ("passed", "error", "in", ...).
            let word = rest.split_whitespace().next()?;
            (word == kind).then(|| n.parse().ok()).flatten()
        })
    };
    for line in text.lines().rev() {
        let passed = count_of(line, "passed");
        let failed = count_of(line, "failed");
        let errored = count_of(line, "error").or_else(|| count_of(line, "errors"));
        if passed.is_some() || failed.is_some() || errored.is_some() {
            return Some((
                passed.unwrap_or(0),
                failed.unwrap_or(0) + errored.unwrap_or(0),
            ));
        }
    }
    None
}

fn run_checked(command: &mut Command, label: &str) -> Result<()> {
    let output = plain_output(command, label)?;
    if !output.success {
        return Err(anyhow!(
            "`{label}` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}
