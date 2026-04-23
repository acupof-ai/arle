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
    assert!(help.contains("Start the interactive agent REPL."));
    assert!(help.contains("Explicit alias for the interactive agent REPL."));
    assert!(help.contains("arle --doctor"));
    assert!(help.contains("arle train test --backend metal --json"));
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
    assert!(help.contains("tool-call stats"));
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
    assert!(help.contains("arle train test --backend metal --json"));
    assert!(help.contains(
        "arle train pretrain --corpus corpus.txt --tokenizer tokenizer.json --preset small-25m"
    ));
}

#[test]
fn train_pretrain_help_no_longer_panics() {
    let output = run_arle(&["train", "pretrain", "--help"]);
    assert!(
        output.status.success(),
        "arle train pretrain --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("Total optimizer steps"));
    assert!(help.contains("Gradient accumulation steps before each optimizer update"));
    assert!(help.contains("Resume from an existing checkpoint directory"));
    assert!(help.contains("Advanced pretrain flags still work after `--`"));
}

#[test]
fn data_help_lists_conversion_examples() {
    let output = run_arle(&["data", "--help"]);
    assert!(
        output.status.success(),
        "arle data --help failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let help = stdout(&output);
    assert!(help.contains("arle data download --repo tatsu-lab/alpaca --file alpaca_data.json"));
    assert!(help.contains("arle data convert --input alpaca.jsonl --format alpaca"));
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
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["mode"], "doctor");
    assert!(value.get("compiled_backend").is_some());
    assert!(value.get("gpu").is_some());
    assert!(value.get("checks").is_some());
}

#[cfg(feature = "cpu")]
#[test]
fn train_test_cpu_json_smoke_is_machine_readable() {
    let output = run_arle(&["train", "test", "--backend", "cpu", "--json"]);
    assert!(
        output.status.success(),
        "arle train test failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );

    let value: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("train test output is valid json");
    assert_eq!(value["backend"], "cpu");
    assert!(value["servable_model_dir"].is_string());
    assert_eq!(value["steps"][0]["name"], "convert");
    assert_eq!(value["steps"][0]["status"], "ok");
    assert_eq!(value["steps"][3]["name"], "eval");
    assert_eq!(value["steps"][3]["status"], "ok");
    assert!(value["eval_summary"].is_object());
}
