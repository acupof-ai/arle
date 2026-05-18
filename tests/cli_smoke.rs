#![cfg(feature = "cli")]

#[path = "cli_test_support.rs"]
mod cli_test_support;

use cli_test_support::{run_arle, stderr, stdout};

#[test]
fn root_help_mentions_explicit_run_entrypoint() {
    let output = run_arle(&["--help"]);
    assert!(
        output.status.success(),
        "arle --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("run"));
    assert!(help.contains("serve"));
    assert!(help.contains("Start the interactive agent REPL."));
    assert!(help.contains("Start the OpenAI-compatible server."));
    assert!(help.contains("Explicit alias for the interactive agent REPL."));
    assert!(help.contains("arle --doctor"));
    assert!(help.contains("arle train test"));
}

#[test]
fn run_help_exposes_one_shot_inputs() {
    let output = run_arle(&["run", "--help"]);
    assert!(
        output.status.success(),
        "arle run --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("--prompt"));
    assert!(help.contains("--stdin"));
    assert!(help.contains("--json"));
    assert!(help.contains("--no-tools"));
    assert!(help.contains("tool-call stats"));
}

#[test]
fn serve_help_exposes_unified_server_frontdoor() {
    let output = run_arle(&["serve", "--help"]);
    assert!(
        output.status.success(),
        "arle serve --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("--backend"));
    assert!(help.contains("--model-path"));
    assert!(help.contains("OpenAI-compatible serving"));
    assert!(help.contains("arle serve --backend metal"));
}

#[test]
fn train_help_lists_primary_workflows() {
    let output = run_arle(&["train", "--help"]);
    assert!(
        output.status.success(),
        "arle train --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("arle train env"));
    assert!(help.contains("arle train test"));
    assert!(help.contains("arle train estimate-memory"));
    assert!(help.contains("opd"));
}

#[test]
fn doctor_json_reports_schema_and_compiled_backend() {
    let output = run_arle(&["--doctor", "--json"]);
    assert!(
        output.status.success(),
        "arle --doctor --json failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let value: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("doctor output is valid json");
    assert_eq!(value["schema_version"], 3);
    assert_eq!(value["mode"], "doctor");
    assert!(value.get("compiled_backend").is_some());
    assert!(value.get("gpu").is_some());
    assert!(value.get("tools").is_some());
    assert!(value.get("checks").is_some());
}

#[test]
fn train_test_reports_opd_pending_stub() {
    let output = run_arle(&["train", "test"]);
    assert!(
        output.status.success(),
        "arle train test stub should exit 0\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );
    let combined = format!("{}{}", stdout(&output), stderr(&output));
    assert!(combined.contains("OPD"));
}
