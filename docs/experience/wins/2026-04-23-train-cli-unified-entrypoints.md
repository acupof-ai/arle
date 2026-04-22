# Unified train/data entrypoints under `agent-infer`

## Context

- Goal: stop making users memorize separate `crates/train` binaries and expose
  one coherent CLI surface from the existing `agent-infer` package.
- Constraint: keep the current train binaries as the source of truth instead of
  reimplementing training argument parsing or training loops inside `crates/cli`.
- Scope: CLI / DX only. No runtime hot-path changes and no benchmark required.

## What Worked

- Added a single top-level command tree:
  - `agent-infer train pretrain|sft|grpo|multi-turn|eval`
  - `agent-infer data download|convert`
- Kept the train-side implementation canonical by adding thin
  `dispatch_from_args(...)` adapters to the existing train binaries instead of
  creating a second parser/runtime layer.
- Used passthrough argument capture in `crates/cli` so new train flags keep
  working without touching the top-level CLI grammar every time.
- Verified the command tree with:
  - `cargo test -p cli --release --no-default-features --features no-cuda`
  - `cargo clippy -p cli --release --no-default-features --features no-cuda -- -D warnings`
  - `cargo build -p agent-infer --release --no-default-features --features cli,no-cuda`
  - `cargo build -p agent-infer --release --no-default-features --features cli,metal,no-cuda`
- Added real CLI smoke checks:
  - `agent-infer data convert ...` successfully converted Dolly JSONL into
    canonical chat JSONL.
  - `agent-infer train pretrain --bogus` surfaced the underlying train parser's
    `unknown flag --bogus`, proving the unified CLI dispatch reaches the real
    entrypoint instead of only parsing top-level syntax.
- Removed dead live-refresh code from `crates/cli/src/tps.rs` after the REPL
  simplification so `clippy -D warnings` stays clean.

## Rule

- For CLI normalization work, centralize the user-facing command tree in
  `crates/cli`, but keep training job semantics owned by the existing train
  entrypoints. Add thin reusable argv adapters instead of duplicating flag
  parsing or trainer logic.
