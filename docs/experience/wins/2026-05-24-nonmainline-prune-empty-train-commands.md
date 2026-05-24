# T3 Non-Mainline Prune — Empty Train Command Namespace

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T3 and
`docs/projects/2026-05-18-opd-only-pivot.md`.

## Context

- T3 requires one deletion cluster per commit with zero-usage evidence.
- The 2026-05-18 OPD-only pivot deleted legacy train binaries and command
  modules. After that deletion, `crates/train/src/commands.rs` only contained
  a comment saying the old command namespace was gone.
- `arle train ...` dispatch now lives in `crates/cli/src/train_cli.rs`, so
  keeping `train::commands` as a public empty module preserved a dead API
  shape from the pre-pivot train surface.

## What Worked

- Deleted `crates/train/src/commands.rs`.
- Removed `pub mod commands` from `crates/train/src/lib.rs`.
- Updated `crates/train/Cargo.toml` and `docs/codebase-map.md` so the active
  truth surfaces point at the CLI front door instead of the retired command
  namespace.

## Zero-Usage Evidence

```bash
rg -n "train::commands|crate::commands|commands::|pub mod commands|commands.rs|mod commands|src/commands|dispatch_from_args" \
  crates src infer README.md docs/codebase-map.md docs/support-matrix.md \
  docs/projects/2026-05-24-opd-mainline-task-backlog.md
```

- Exit code 1; no active code or active truth-surface hits remain.
- Historical research/review docs still mention old command files as past
  evidence and were intentionally not rewritten.

## Verification

```bash
cargo fmt --check -p train
cargo check -p train --no-default-features
cargo clippy -p train --no-default-features --no-deps -- -D warnings
cargo test -p train --no-default-features --lib
```

- All commands exited 0.
- `cargo test -p train --no-default-features --lib` ran 81 tests.
- This was CPU-only and did not touch P5 PID 28950, GPU runtime,
  `infer/src/kv_tier/`, or `infer/src/scheduler/`.

## Rule

- After deleting a command family, delete the empty public namespace too.
  A comment-only module still advertises an API shape and keeps stale docs
  alive.
