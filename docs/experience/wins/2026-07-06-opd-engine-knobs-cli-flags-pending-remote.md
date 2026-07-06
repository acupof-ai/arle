# OPD rollout-engine + engine-offload magic env → CLI flags

`bench-exempt` — a flag-surface refactor: `--rollout-engine` / `--engine-offload`
now resolve the same values the `ARLE_OPD_*` env vars did (env kept as fallback),
so behavior is byte-identical for existing runs. The CUDA offload/rollout paths
themselves are unchanged; a perf A/B would only re-measure the already-benched
P4 infer-rollout default and the 05-30 offload win. CUDA typecheck is
`pending-remote` (no nvcc on this Mac); non-CUDA build + full train test suite
green locally.

## Context

`AGENTS.md` / the user's standing preference is "env 就删除 … 默认用模型派生"
— magic env vars should become CLI flags with model-derived defaults. Two
*functional* OPD knobs were env-only: `ARLE_OPD_INFER_ROLLOUT` (rollout engine
arm) and `ARLE_OPD_ENGINE_OFFLOAD` (VRAM time-share mode). Pure diagnostic env
(`ARLE_OPD_STEP_TRACE`, `_VRAM_TRACE`, `_BACKWARD_PROFILE`, `_LOG_GRAD_NORM`, …)
are left as-is — they are trace toggles, not run-shaping switches.

## What Worked

- **`--rollout-engine {infer,train}`** and **`--engine-offload
  {off,student,teacher,all}`** added to `TrainOpdArgs` (`crates/cli/src/args.rs`),
  wired via `apply_opd_engine_overrides` at the top of `run_opd_from_dirs`.
- **Resolution order = flag > env > default**, via set-once `OnceLock`
  overrides consulted by the existing `infer_rollout_flag_enabled()` /
  `engine_offload_mode()` resolvers (`crates/train/src/opd.rs`). No call-site
  signature churn (the resolvers are read from deep CUDA-gated call sites), no
  parallel old+new path, legacy env still honored for one migration cycle.
- The `all` offload flag doc names its known step-2 illegal-address on the W4A8
  Marlin teacher reload and points to `teacher` as the safe choice.
- Non-CUDA build accepts the flags but they are inert (`#[cfg(not(cuda))]` stub);
  the CPU path has no infer engine and no VRAM to offload.

Verification: `cargo check -p cli --features metal,no-cuda` green; `cargo test -p
train --features no-cuda` 163 + all integration tests green. CUDA build =
pending-remote (cudarc needs nvcc).

## Rule

Promote run-shaping env vars to CLI flags but keep the env as a set-once
fallback consulted by the *same* resolver — this avoids touching deep gated
call-site signatures and leaves a clean migration cycle, rather than forking a
parallel flag-only path (a half-state).
