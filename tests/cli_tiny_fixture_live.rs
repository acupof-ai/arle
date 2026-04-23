#![cfg(all(feature = "cli", any(feature = "cuda", feature = "metal")))]

use std::fs;

#[path = "cli_test_support.rs"]
mod cli_test_support;

use cli_test_support::{run_arle, stderr, stdout, unique_temp_dir};

fn expected_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "metal"
    }
}

#[test]
fn train_test_fixture_checkpoint_runs_through_real_backend_cli() {
    let backend = expected_backend();
    let root_dir = unique_temp_dir("arle-cli-fixture");
    fs::create_dir_all(&root_dir).expect("create fixture dir");

    let root_dir_arg = root_dir.display().to_string();
    let train_args = vec![
        "train",
        "test",
        "--backend",
        backend,
        "--json",
        "--out-dir",
        root_dir_arg.as_str(),
    ];
    let train_output = run_arle(&train_args);
    assert!(
        train_output.status.success(),
        "arle train test failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&train_output),
        stderr(&train_output)
    );

    let train_json: serde_json::Value =
        serde_json::from_str(&stdout(&train_output)).expect("train test output is valid json");
    assert_eq!(train_json["backend"], backend);

    let model_dir = train_json["servable_model_dir"]
        .as_str()
        .expect("train test exposes servable model dir")
        .to_string();
    assert!(
        model_dir.ends_with("/sft/latest"),
        "unexpected fixture model dir: {model_dir}"
    );

    let run_args = vec![
        "--model-path",
        model_dir.as_str(),
        "run",
        "--prompt",
        "Say hello in one word.",
        "--json",
    ];
    let run_output = run_arle(&run_args);
    assert!(
        run_output.status.success(),
        "arle run failed\nstdout:\n{}\nstderr:\n{}",
        stdout(&run_output),
        stderr(&run_output)
    );

    let run_json: serde_json::Value =
        serde_json::from_str(&stdout(&run_output)).expect("run output is valid json");
    assert_eq!(run_json["backend"], backend);
    assert!(
        run_json["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "prompt token count missing\nstdout:\n{}\nstderr:\n{}",
        stdout(&run_output),
        stderr(&run_output)
    );
    assert!(
        run_json["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "completion token count missing\nstdout:\n{}\nstderr:\n{}",
        stdout(&run_output),
        stderr(&run_output)
    );
    assert_eq!(run_json["tool_calls_executed"], 0);

    fs::remove_dir_all(&root_dir).expect("remove fixture dir");
}
