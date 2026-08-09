//! Per-task filesystem sandbox for agent-based OPD training.
//!
//! Backs the cc-harness agentic OPD loop ([`crate::cc_harness`]): the agent
//! edits a real repo checked out at a base commit and hidden tests are the
//! reward signal. [`boot_workdir`] / [`diff_workdir`] / [`score_workdir`]
//! manage the workdir lifecycle (no container — the caller is responsible for
//! isolation if it wants any): stage a tree, diff the agent's candidate patch,
//! apply the hidden `test_patch` + run `fail_to_pass` tests. [`run_captured`]
//! is the fork-safe subprocess primitive both scoring and the cc spawn use.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};

use crate::spawner::{SpawnClient, SpawnRequest, SpawnResponse};

/// Build a [`SpawnRequest`] from a `Command` (program/args/cwd/env) for the helper.
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

/// Run a `Command` via the pre-CUDA spawner helper. `combined_timeout=true`
/// mirrors `run_captured` (setsid + process-group timeout, combined output);
/// `false` mirrors `Command::output()` (separate stdout/stderr, no timeout).
fn spawn_via_helper(
    client: &SpawnClient,
    command: &Command,
    combined_timeout: bool,
    timeout: Duration,
) -> std::io::Result<SpawnResponse> {
    client.run(&request_from_command(command, combined_timeout, timeout))
}

/// Captured result of a plain (no-timeout, separate streams) spawn — the subset
/// of `std::process::Output` the cp/git call sites use.
struct PlainOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    success: bool,
    /// Human-readable exit status for error messages.
    status: String,
}

/// `Command::output()` equivalent, routed through the spawner helper when its env
/// is set (agent-OPD rollout) and run directly otherwise (byte-identical default).
/// This is the `combined_timeout=false` path: stdout/stderr stay separate, no
/// timeout. Covers cp/git (`run_checked`), `git diff`, and `git apply`.
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
/// Returns `(combined_output, exit_code, killed_by_timeout)`. Also the cc
/// harness's `claude -p` spawn path (`crate::cc_harness`).
pub(crate) fn run_captured(
    command: Command,
    timeout: Duration,
) -> std::io::Result<(Vec<u8>, Option<i32>, bool)> {
    // Route through the pre-CUDA spawner helper when agent-OPD set its socket env;
    // the helper does the setsid fork on a non-CUDA process (dodges ELKEID). When
    // unset (normal serve/CLI) we spawn directly below — byte-identical default.
    if let Some(client) = SpawnClient::from_env() {
        let resp = spawn_via_helper(&client, &command, true, timeout)?;
        return Ok((resp.stdout, resp.exit_code, resp.timed_out));
    }

    let tmp = tempfile::NamedTempFile::new()?;
    let stdout_handle = tmp.reopen()?;
    let stderr_handle = stdout_handle.try_clone()?; // dup'd fd shares the offset → interleaved

    // Re-parent the tool command under the standalone `setsid` binary so it runs
    // in its OWN session+group (leader pid == the spawned child's pid). On a
    // timeout we `kill -KILL -<pgid>` the whole subtree and reap any backgrounded
    // grandchildren.
    //
    // We DO NOT use `Command::process_group(0)`. That routes the group change
    // through an in-child `pre_exec` hook between `fork()` and `exec()`; in the
    // agent-OPD rollout the parent is a MULTI-THREADED, CUDA-resident process, so
    // `fork()` snapshots a libc/CUDA lock held by another (now-absent) thread and
    // the child's `setpgid` deadlocks before `exec` — the round-0 repo-overview
    // `bash -lc` never runs, the rollout hangs the full `bash_timeout_secs`, and
    // the group-kill then SIGKILLs the whole student tree (the load deadlock is
    // fixed by stream_synchronize, this is the NEXT blocker on the same loop). The
    // standalone `setsid` binary creates the new session AFTER `exec`, so there is
    // no in-child hook and the spawn stays fork-safe.
    // macOS has no standalone `setsid` binary: spawn directly there (the timeout
    // still kills the direct child; only backgrounded grandchildren can outlive
    // it). Linux — the production lane — keeps the full session re-parent + kill.
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

/// `kill -KILL -<pgid>` — signal the whole process group (negative pid). Uses the
/// `kill` binary to avoid a libc dependency; failure is ignored (the group may be
/// empty already, which is the common case).
fn kill_group(pgid: i32) {
    // Never signal our own group: a mis-derived pgid must not SIGKILL the
    // caller's session (observed 3× deterministic under `cargo test -p train
    // --features cuda` on the pod, 2026-07-24; exact mis-derivation hop
    // unpinned — this guard closes the class).
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
    // An ancestor CLAUDE.md cost ~31K tokens/request of CC preamble
    // (wins/2026-07-24-agent-opd-gpu-busy-frac-measured-go.md). Per-run
    // invariant of work_root, so warn once, not per boot.
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

/// Apply the hidden `test_patch` (skip if empty), then run the `fail_to_pass`
/// tests. Returns `(reward, log_tail)` where `reward` is the partial-credit
/// fraction of `fail_to_pass` tests that passed, in `[0.0, 1.0]`; `1.0` iff all
/// `fail_to_pass` pass (⇔ pytest exit-0). Falls back to binary (exit-0 → 1.0)
/// when the run is killed or the pytest summary can't be parsed.
/// The test command runs under a Rust-enforced timeout (like `run_bash`).
pub fn score_workdir(
    workdir: &Path,
    test_patch: &str,
    fail_to_pass: &[String],
    pythonpath: Option<&str>,
    timeout_secs: u64,
) -> Result<(f32, String)> {
    if !test_patch.trim().is_empty() {
        // The hidden test_patch is ground truth; the student must not be able to
        // block or skew scoring by editing the (hidden) test files it touches.
        // Reset just those paths to the committed base before applying, so
        // `git apply` lands cleanly even if the rollout dirtied a test file — the
        // student's SOURCE fix in other files is preserved. (Without this, a
        // plain `git apply` against a student-dirtied test file fails "patch does
        // not apply" and a real fix is mis-scored as failing.)
        // A `+++ b/<path>` is a real file header only when it directly follows a
        // `--- ` line — gating on that avoids mistaking a patch CONTENT line that
        // happens to start with `+++ b/` for a path to reset.
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

        // Write via the full (possibly-relative) path from the process cwd, but
        // hand `git apply` the bare filename: `git -C workdir` resolves a
        // file-arg relative to workdir, so a workdir-relative path would double
        // the prefix ("can't open patch") whenever work_root is relative.
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
    // Mirror run_bash: no bytecode, and ignore any stale pyc a rollout left
    // behind (scoring must execute the edited source, not cached bytecode).
    command.env("PYTHONDONTWRITEBYTECODE", "1");
    if let Some(pythonpath) = pythonpath {
        let combined = match std::env::var("PYTHONPATH") {
            Ok(existing) if !existing.is_empty() => format!("{pythonpath}:{existing}"),
            _ => pythonpath.to_string(),
        };
        command.env("PYTHONPATH", combined);
    }

    let (output, code, killed) = run_captured(command, Duration::from_secs(timeout_secs))
        .with_context(|| "failed to run pytest".to_string())?;
    let log_tail = format_bash_output(&output, &[], code, killed);

    // Partial-credit reward = passed / requested. Fall back to binary (exit-0 →
    // 1.0) when killed or the summary is unparseable (collection crash / no tail
    // line), so a timed-out or crashed run never mints spurious credit.
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

/// Parse the LAST pytest `-q` summary line (pytest prints it last) into
/// `(passed, failed + errored)`. Robust to any subset of
/// {passed, failed, error(s), skipped, xfailed}: e.g. `"3 passed, 2 failed in
/// 0.4s"` → `(3, 2)`, `"2 passed, 1 error in 0.3s"` → `(2, 1)`. Returns `None`
/// when no summary line is present (collection crash / empty output).
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

/// Run a `Command` and return an error including stderr if it does not exit 0.
/// Routes through the sandbox-spawner when its env is set (agent-OPD rollout).
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// `plain_output`/`run_captured` both gate on the process-global
    /// `ARLE_SPAWNER_SOCKET` (`SpawnClient::from_env`) — any test that spawns a
    /// subprocess must hold this lock so `spawner_routing_matches_direct` /
    /// `spawner::tests::server_client_echo_roundtrip` (which set that env var
    /// mid-test) can never race a concurrent, unrelated subprocess spawn onto
    /// their helper socket.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::spawner::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn parse_pytest_counts_reads_last_summary() {
        assert_eq!(
            parse_pytest_counts("3 passed, 2 failed in 0.4s"),
            Some((3, 2))
        );
        assert_eq!(parse_pytest_counts("5 passed in 0.2s"), Some((5, 0)));
        assert_eq!(parse_pytest_counts("1 failed in 0.1s"), Some((0, 1)));
        assert_eq!(
            parse_pytest_counts("2 passed, 1 error in 0.3s"),
            Some((2, 1))
        );
        assert_eq!(parse_pytest_counts("collection crashed, no summary"), None);
    }

    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "ELKEID kills CUDA-linked fork+exec on the pod (reference_h20_pod_elkeid_kills_cuda_forks); logic covered on cpu lanes"
    )]
    fn run_captured_does_not_hang_on_backgrounded_child() {
        let _guard = env_guard();
        // Regression: a command that backgrounds a process inheriting the stdout
        // pipe used to block `wait_with_output()` until that grandchild exited
        // (forever for a daemon), stranding the parent in `wait4()`.
        // `run_captured` writes to a temp file (no pipe to hold) and kills the
        // whole process group, so it must return promptly.
        let tmp = tempfile::tempdir().unwrap();
        let mut cmd = Command::new("bash");
        cmd.args(["-lc", "echo ready; sleep 30 &"])
            .current_dir(tmp.path());
        let start = std::time::Instant::now();
        let (out, code, killed) = run_captured(cmd, Duration::from_secs(30)).unwrap();
        let elapsed = start.elapsed();
        assert!(
            String::from_utf8_lossy(&out).contains("ready"),
            "output: {out:?}"
        );
        assert_eq!(code, Some(0));
        assert!(!killed);
        assert!(
            elapsed < Duration::from_secs(5),
            "run_captured blocked {}s on a backgrounded child — the pipe-hang regressed",
            elapsed.as_secs()
        );
    }

    #[test]
    fn boot_diff_roundtrip() {
        let _guard = env_guard();
        let work_root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        fs::write(staged.path().join("a.txt"), "x\n").unwrap();

        let workdir = boot_workdir(work_root.path(), "inst1", staged.path(), None).unwrap();
        assert!(workdir.join("a.txt").exists());

        fs::write(workdir.join("a.txt"), "y\n").unwrap();
        let diff = diff_workdir(&workdir).unwrap();
        assert!(diff.contains("a.txt"), "diff should mention a.txt: {diff}");
    }

    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "ELKEID kills CUDA-linked fork+exec on the pod (reference_h20_pod_elkeid_kills_cuda_forks); logic covered on cpu lanes"
    )]
    fn score_detects_failing_test() {
        let _guard = env_guard();
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
        let (reward, log) =
            score_workdir(&workdir, "", &["test_x.py::test_y".into()], None, 60).unwrap();
        let passed = reward >= 1.0;
        assert!(!passed, "failing test must not pass; log: {log}");
    }

    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "ELKEID kills CUDA-linked fork+exec on the pod (reference_h20_pod_elkeid_kills_cuda_forks); logic covered on cpu lanes"
    )]
    fn score_resets_student_dirtied_test_file_before_applying_patch() {
        let _guard = env_guard();
        // Regression: a rollout that edits a (hidden) test file used to make the
        // plain `git apply` of the base-context test_patch fail "patch does not
        // apply", mis-scoring a real fix as failing. score_workdir now resets the
        // patch's paths to base first, so apply lands cleanly.
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
            "def test_y():\n    assert True\n",
        )
        .unwrap();
        let workdir = boot_workdir(work_root.path(), "inst_dirty", staged.path(), None).unwrap();

        fs::write(workdir.join("test_x.py"), "STUDENT GARBAGE\n").unwrap();

        let test_patch = concat!(
            "diff --git a/test_x.py b/test_x.py\n",
            "--- a/test_x.py\n",
            "+++ b/test_x.py\n",
            "@@ -1,2 +1,2 @@\n",
            " def test_y():\n",
            "-    assert True\n",
            "+    assert False\n",
        );

        // Must NOT bail on apply (the reset makes it land), and the patched test
        // (assert False) then fails → passed=false.
        let (reward, log) = score_workdir(
            &workdir,
            test_patch,
            &["test_x.py::test_y".into()],
            None,
            60,
        )
        .unwrap();
        let passed = reward >= 1.0;
        assert!(
            !passed,
            "patched test asserts False → must fail; log: {log}"
        );
    }

    /// Routing through the spawner helper must produce byte-identical sandbox
    /// behavior to the default direct path. Stand up `serve_loop` on a temp
    /// socket, then run the cp/git (`boot_workdir`), bash, and `git diff` paths
    /// once routed (`ARLE_SPAWNER_SOCKET` set) and once direct (unset), asserting
    /// equivalence. Serialized on a global lock because the spawner env is
    /// process-global and Rust runs tests in parallel.
    #[test]
    #[cfg_attr(
        feature = "cuda",
        ignore = "ELKEID kills CUDA-linked fork+exec on the pod (reference_h20_pod_elkeid_kills_cuda_forks); logic covered on cpu lanes"
    )]
    fn spawner_routing_matches_direct() {
        use crate::spawner::{LISTEN_ENV, SOCKET_ENV, serve_loop};
        // One lock shared with spawner.rs's server test — cargo runs both in
        // parallel and they race on the same process-global spawner env.
        let _guard = crate::spawner::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let sock_dir = tempfile::tempdir().unwrap();
        let sock = sock_dir.path().join("spawn.sock");
        // SAFETY: serialized by ENV_LOCK; no rollout threads in the test.
        unsafe {
            std::env::set_var(LISTEN_ENV, &sock);
        }
        let server = std::thread::spawn(serve_loop);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !sock.exists() {
            assert!(std::time::Instant::now() < deadline, "helper never bound");
            std::thread::sleep(Duration::from_millis(10));
        }
        // The helper inherits LISTEN_ENV; clear it for the test process so the
        // client path is selected only when SOCKET_ENV is set below.
        // SAFETY: serialized by ENV_LOCK.
        unsafe {
            std::env::remove_var(LISTEN_ENV);
        }

        // Build a staged tree once; run boot/bash/diff direct, then routed.
        let run_once = |routed: bool| -> (String, String) {
            // SAFETY: serialized by ENV_LOCK.
            unsafe {
                if routed {
                    std::env::set_var(SOCKET_ENV, &sock);
                } else {
                    std::env::remove_var(SOCKET_ENV);
                }
            }
            assert_eq!(
                crate::spawner::SpawnClient::from_env().is_some(),
                routed,
                "from_env() gating must match the env state"
            );
            let work_root = tempfile::tempdir().unwrap();
            let staged = tempfile::tempdir().unwrap();
            fs::write(staged.path().join("a.txt"), "x\n").unwrap();
            // boot_workdir exercises cp -a + git init/add/commit via run_checked.
            let workdir =
                boot_workdir(work_root.path(), "inst_route", staged.path(), None).unwrap();
            // bash via run_captured (combined_timeout path).
            let mut cmd = Command::new("bash");
            cmd.args(["-lc", "echo ROUTE_MARKER"]).current_dir(&workdir);
            let (out, code, killed) = run_captured(cmd, Duration::from_secs(30)).unwrap();
            let bashed = format!("{}|{code:?}|{killed}", String::from_utf8_lossy(&out));
            // edit + git diff via diff_workdir (plain_output path).
            fs::write(workdir.join("a.txt"), "y\n").unwrap();
            let diff = diff_workdir(&workdir).unwrap();
            // Normalize the workdir-specific temp path out of outputs.
            (
                bashed,
                diff.replace(&workdir.to_string_lossy().to_string(), "WD"),
            )
        };

        let (bash_direct, diff_direct) = run_once(false);
        let (bash_routed, diff_routed) = run_once(true);

        assert!(
            bash_routed.contains("ROUTE_MARKER") && bash_routed.contains("Some(0)|false"),
            "routed bash: {bash_routed}"
        );
        assert_eq!(bash_direct, bash_routed, "bash output must match direct");
        assert!(
            diff_routed.contains("a.txt"),
            "routed diff must mention a.txt: {diff_routed}"
        );
        assert_eq!(
            diff_direct, diff_routed,
            "git diff output must match direct"
        );

        // SAFETY: serialized by ENV_LOCK; teardown.
        unsafe {
            std::env::remove_var(SOCKET_ENV);
        }
        drop(server); // detached; process exit reaps it
        let _ = std::fs::remove_file(&sock);
    }
}
