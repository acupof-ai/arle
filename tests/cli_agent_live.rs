#![cfg(all(feature = "cli", any(feature = "cuda", feature = "metal")))]

use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

fn live_model_available() -> bool {
    std::env::var("ARLE_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
        || std::env::var("AGENT_INFER_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .is_some()
        || infer::hf_hub::discover_local_model().is_some()
}

fn live_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn run_cli(args: &[&str], lines: &[&str], timeout: Duration) -> Output {
    let exe = env!("CARGO_BIN_EXE_arle");
    let reads_stdin_prompt = args.iter().any(|arg| *arg == "--stdin");
    let mut child = Command::new(exe)
        .arg("--max-turns")
        .arg("4")
        .arg("--max-tokens")
        .arg("96")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn arle");

    {
        let mut stdin = child.stdin.take().expect("stdin");
        for line in lines {
            writeln!(stdin, "{line}").expect("write prompt");
        }
        if !reads_stdin_prompt {
            writeln!(stdin, "exit").expect("write exit");
        }
    }

    let start = Instant::now();
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            return child.wait_with_output().expect("wait_with_output");
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            panic!("ARLE CLI timed out after {:?}", timeout);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
#[ignore = "requires a local model auto-detected by the CLI"]
fn cli_run_prompt_executes_tool_for_local_file_query() {
    let _guard = live_test_guard();

    if !live_model_available() {
        eprintln!("Skipping live CLI test: no local model available");
        return;
    }

    let output = run_cli(
        &["run", "--prompt", "本地有哪些文件", "--json"],
        &[],
        Duration::from_secs(120),
    );

    assert!(
        output.status.success(),
        "CLI exited with failure\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("one-shot json output");
    assert!(json["tool_calls_executed"].as_u64().unwrap_or(0) >= 1);
    assert!(
        json["text"]
            .as_str()
            .expect("one-shot text")
            .contains("/path/to/code/agent-infer"),
        "CLI output did not contain the expected working directory\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires a local model auto-detected by the CLI"]
fn cli_repl_handles_multiple_turns_and_reset() {
    let _guard = live_test_guard();

    if !live_model_available() {
        eprintln!("Skipping live CLI test: no local model available");
        return;
    }

    let output = run_cli(
        &["run"],
        &[
            "Use the python tool to compute 2 + 2. After the tool returns, answer with just the integer.",
            "/reset",
            "Use the python tool to compute 3 + 3. After the tool returns, answer with just the integer.",
        ],
        Duration::from_secs(120),
    );

    assert!(
        output.status.success(),
        "CLI exited with failure\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("(conversation reset)"),
        "CLI did not acknowledge reset\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("\u{1b}[1;34m4\u{1b}[0m") && stdout.contains("\u{1b}[1;34m6\u{1b}[0m"),
        "CLI did not produce the expected final answers for both turns\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires a local model auto-detected by the CLI"]
fn cli_run_stdin_uses_shell_for_file_listing_queries() {
    let _guard = live_test_guard();

    if !live_model_available() {
        eprintln!("Skipping live CLI test: no local model available");
        return;
    }

    let output = run_cli(
        &["run", "--stdin", "--json"],
        &["本地有哪些文件"],
        Duration::from_secs(120),
    );

    assert!(
        output.status.success(),
        "CLI exited with failure\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("one-shot json output");
    assert!(
        json["tool_calls_executed"].as_u64().unwrap_or(0) >= 1,
        "CLI did not execute the shell tool from stdin input\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        json["text"]
            .as_str()
            .expect("one-shot text")
            .contains("/path/to/code/agent-infer"),
        "CLI shell output did not include the expected working directory\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
}
