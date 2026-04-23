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
        .expect("live test lock")
}

fn run_cli_session(lines: &[&str], timeout: Duration) -> Output {
    let exe = env!("CARGO_BIN_EXE_arle");
    let mut child = Command::new(exe)
        .arg("--max-turns")
        .arg("4")
        .arg("--max-tokens")
        .arg("96")
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
        writeln!(stdin, "exit").expect("write exit");
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
fn cli_auto_detects_local_model_and_executes_python_tool() {
    let _guard = live_test_guard();

    if !live_model_available() {
        eprintln!("Skipping live CLI test: no local model available");
        return;
    }

    let output = run_cli_session(
        &[
            "Use the python tool to compute 123 * 456. Do not do the math mentally. After the tool returns, answer with just the integer.",
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
        stdout.contains("[tool: python]"),
        "CLI did not execute the python tool\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("56088"),
        "CLI output did not contain expected computed value\nstdout:\n{}\nstderr:\n{}",
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

    let output = run_cli_session(
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
        stdout.matches("[tool: python]").count() >= 2,
        "CLI did not execute the python tool twice\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
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
fn cli_uses_shell_for_file_listing_queries() {
    let _guard = live_test_guard();

    if !live_model_available() {
        eprintln!("Skipping live CLI test: no local model available");
        return;
    }

    let output = run_cli_session(&["本地有哪些文件"], Duration::from_secs(120));

    assert!(
        output.status.success(),
        "CLI exited with failure\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("[tool: shell]"),
        "CLI did not execute the shell tool\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("/path/to/code/agent-infer"),
        "CLI shell output did not include the expected working directory\nstdout:\n{}\nstderr:\n{}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
}
