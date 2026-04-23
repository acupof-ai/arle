# Train CLI Core Cleanup

## Context

- Goal: tighten the core `crates/cli/src/train_cli.rs` flow after the earlier
  behavior fixes so the unified train/data surface stays clean as more commands
  get added.
- Review focus:
  - avoid repeating the same resolve-run-error pattern across train commands
  - keep `train test` on one canonical internal report shape instead of
    maintaining separate human and JSON step state
  - preserve the already-fixed CLI behavior while simplifying the code path

## What Worked

- Collapsed `train pretrain`, `train sft`, and `train eval` onto one shared
  `run_resolved_train_invocation(...)` helper. The resolve phase, dry-run path,
  and error reporting now share one canonical flow.
- Replaced the duplicated `convert_status/pretrain_status/sft_status/eval_status`
  fields in `TrainTestReport` with a single `steps` list. Human output and
  `--json` output now read from the same state instead of rebuilding the step
  table in a second struct.
- Kept `TrainTestReport` as a pure result object by removing the embedded
  `json` render toggle. `run_train_test(...)` now decides how to print from the
  CLI args, while temp-dir cleanup runs through one shared post-run path instead
  of duplicated success/failure branches.
- Kept the external behavior stable while simplifying internals:
  - `agent-infer train test --backend metal --json`
  - `agent-infer train pretrain --corpus README.md --tokenizer models/Qwen3-0.6B --dry-run --json`
- Verification stayed clean:
  - `cargo test -p cli --release --no-default-features --features no-cuda`
  - `cargo clippy -p cli --release --no-default-features --features no-cuda --no-deps -- -D warnings`
  - `cargo build -p agent-infer --release --no-default-features --features cli,metal,no-cuda`
  - `cargo fmt --manifest-path infer/Cargo.toml --all -- --check`

## Rule

- When several CLI commands differ only by "resolve args, maybe dry-run, then
  dispatch", keep one shared control path for that pattern. Duplicating the
  wrapper logic makes later DX fixes drift across commands.
- For machine-readable CLI output, keep one canonical internal data model and
  render from it; avoid a second near-duplicate report struct just for JSON.
- CLI / operator-DX only; no runtime hot path changed, so no performance
  benchmark entry was required.
