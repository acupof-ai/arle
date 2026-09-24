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
    assert!(help.contains("Serve over the Anthropic and OpenAI APIs."));
    assert!(help.contains("Explicit alias for the interactive agent REPL."));
    assert!(help.contains("arle --doctor"));
    assert!(!help.contains("arle train"));
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
    assert!(help.contains("Anthropic Messages and OpenAI APIs"));
    assert!(help.contains("arle serve --backend metal"));
}

#[test]
fn train_subcommand_is_gone() {
    let output = run_arle(&["train", "--help"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unrecognized subcommand 'train'"));
}
