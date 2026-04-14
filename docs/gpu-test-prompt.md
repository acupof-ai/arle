# Route A GPU Validation Prompt

Use this checklist when validating the 2026-04-15 Route-A refactor on a CUDA
host after the local Darwin lane has finished the structural rollback.

## Goal

Confirm that folding the abandoned runtime split back into `infer` changed only
package/module structure and did not change build behavior or runtime semantics.

## Commands

Run these from the repo root, in order:

```bash
cargo fmt --all -- --check
cargo check --workspace --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features metal
cargo test --workspace --release --no-default-features --features metal
cargo clippy --workspace --no-default-features --features metal -- -D warnings
```

## What to inspect

- `infer/src/lib.rs` should have no legacy `#[path]` redirects into the
  removed split-tree modules.
- `infer/src/agent_engine.rs`, `infer/src/types.rs`,
  `infer/src/events.rs`, and `infer/src/scheduler/policy.rs` should exist.
- `Cargo.toml` workspace members should no longer list the four reverted crates.
- `crates/` should contain only `infer-agent`, `infer-chat`, `infer-cli`,
  `infer-tools`, `mlx-sys`, and `README.md`.

## Final sweep

Run the Route-A legacy-reference grep used in the acceptance checklist.

Expected results:

- zero `.rs` hits
- zero `.toml` hits
- `.md` hits only in `docs/archives/cuda-crate-extraction.md` and
  `docs/experience/wins/2026-04-14-kv-quant-audit.md`
