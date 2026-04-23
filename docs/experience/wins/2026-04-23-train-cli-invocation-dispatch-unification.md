# Train CLI Invocation Dispatch Unification

## Context

- Goal: keep the unified `agent-infer train ...` / `agent-infer data ...`
  surface from drifting internally after the earlier CLI fixes landed.
- Review focus:
  - remove the remaining split between train-command wrappers and data-command
    wrappers around dry-run rendering and real dispatch
  - preserve the already-fixed `data` exit-code contract while still reducing
    wrapper duplication
  - keep the CLI core simple enough that adding a new subcommand follows one
    obvious pattern

## What Worked

- Unified all `train` commands onto one `run_train_command(...)` path. The
  resolve-error-render-dispatch flow is now shared across `pretrain`, `sft`,
  `eval`, `grpo`, and `multi-turn`.
- Unified `data convert` and `data download` onto one
  `run_passthrough_invocation(...)` path. They now share dry-run rendering with
  the rest of the CLI without wrapping or rewriting the child exit code.
- Pulled the shared dry-run decision into `dry_run_exit(...)`, so there is only
  one place deciding whether to print a `ResolvedInvocation` versus execute it.
- Replaced the duplicated command list in `train env` with one
  `TRAIN_ENV_COMMANDS` constant.
- Added targeted unit tests for the shared wrapper contracts:
  - train-command dispatch executes the resolved argv for non-dry-run calls
  - passthrough invocation preserves the child exit code exactly

## Verification

- `cargo test -p cli --release --no-default-features --features no-cuda`
- `cargo clippy -p cli --release --no-default-features --features no-cuda -- -D warnings`
- `cargo build -p agent-infer --release --no-default-features --features cli,no-cuda`
- `cargo build -p agent-infer --release --no-default-features --features cli,metal,no-cuda`
- `target/release/agent-infer data convert` -> exit `2`
- `target/release/agent-infer data convert --input /tmp/does-not-exist.jsonl --format chat --output /tmp/out.chat.jsonl` -> exit `1`
- `target/release/agent-infer train pretrain --corpus README.md --tokenizer models/Qwen3-0.6B --dry-run --json`
- `cargo run -p agent-infer --release --no-default-features --features cli,metal,no-cuda -- train test --backend metal --json`

## Rule

- In this CLI, dry-run rendering is a cross-cutting concern, not a per-command
  special case. Keep exactly one shared helper that decides whether to print
  the resolved invocation or execute it.
- Preserve child exit-code semantics for passthrough subcommands by sharing the
  wrapper flow without wrapping the child result into a different error model.
- CLI / operator-DX only; no runtime hot path changed, so no performance
  benchmark entry was required.
