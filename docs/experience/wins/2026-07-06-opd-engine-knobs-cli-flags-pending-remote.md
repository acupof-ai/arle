# OPD rollout arm: `--rollout-engine {infer,train}` replaces the env toggle

`bench-exempt` — CLI-surface refactor. `--rollout-engine` selects the OPD rollout
arm (default `infer`); the CUDA rollout paths are unchanged, so a perf A/B would
only re-measure the already-benched P4 infer-rollout default. CUDA build is
`pending-remote` (no nvcc on this Mac); non-CUDA build + full train test suite
green locally.

## Context

`AGENTS.md` / the user's standing preference: run-shaping knobs should be CLI
flags, not magic env. The rollout arm was env-only (`ARLE_OPD_INFER_ROLLOUT`).
Per the user's "只保留 `--rollout-engine`, 其他全部删除" — this lands the one
flag as the sole entry and deletes the env fallback entirely (deletion-style
refactor, no migration layer, no half-state).

## What Worked

- **`--rollout-engine {infer,train}`** on `TrainOpdArgs` (`crates/cli/src/args.rs`);
  `apply_opd_rollout_engine` installs it into a set-once `OnceLock` that
  `infer_rollout_flag_enabled()` reads (`crates/train/src/opd.rs`). Unset →
  `infer` (the fast default); `train` → the train-crate O(n²) A/B baseline arm.
- **Deleted**: the `ARLE_OPD_INFER_ROLLOUT` env branch, the never-landed
  `--engine-offload` flag + its `OpdEngineOffloadArg` enum + the
  `set_engine_offload_override` machinery. `engine_offload_mode()` reverts to its
  original env-only form (`ARLE_OPD_ENGINE_OFFLOAD`) — untouched pre-existing
  behavior, not part of this flag surface.
- Non-CUDA build accepts the flag but it is inert (`#[cfg(not(cuda))]` stub); the
  CPU path has no infer engine.

Verification: `cargo check -p cli --features metal,no-cuda` green; `cargo test -p
train --features no-cuda` 163 lib + integration green; clippy clean. CUDA build =
pending-remote (cudarc needs nvcc).

## Rule

When a flag replaces an env toggle, delete the env path rather than keeping a
"flag > env > default" fallback — one entry, one default, no migration layer.
The set-once `OnceLock` read by the existing resolver avoids churning the deep
CUDA-gated call sites that consume the value.
