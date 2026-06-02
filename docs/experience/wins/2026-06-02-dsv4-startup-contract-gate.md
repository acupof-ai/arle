# DSv4 startup contract gate for SGLang-path claims

## Context

DSv4 optimization was drifting into local operator tuning while the executable
path still differed from SGLang's best-practice path. A launch could silently
serve on the replicated-token debug lane and still look like a performance run.

## What Worked

- Added `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` and `ARLE_DSV4_HIGH_PERF=1` as
  explicit high-performance declarations, with the old
  `ARLE_DSV4_SGLANG_PATH=1` kept as an alias.
- Added a generic `ModelForward::validate_scheduler_contract` hook so models
  can validate scheduler-owned knobs before slot state allocation.
- DSv4 now logs the startup contract: profile, fallback lane, topology, KV pool
  format, CUDA graph support/reason, MoE backend, expert backend, FlashMLA
  gates, shared KV pool, and incremental KV.
- In high-performance profile, DSv4 fails fast if the SGLang best-practice
  contract is not active. The current binary still reports token-owned DP/EP
  sharding and graph-captured DSv4 metadata as missing, so this is a guardrail,
  not a performance claim.

## Verification

- `cargo fmt --check`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features no-cuda`

Remote DSv4 build/startup-log validation is pending.

## Rule

High-performance DSv4 launches must fail before serving if they are actually on
the debug/fallback path. Performance numbers are not comparable unless the
startup contract proves the intended topology, KV layout, graph mode, DeepEP,
and DeepGEMM path are active.
