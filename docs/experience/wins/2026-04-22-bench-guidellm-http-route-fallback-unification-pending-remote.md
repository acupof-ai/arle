# HTTP route fallback unification — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that structured `404/405` cleanup for the top-level HTTP router and session sub-router does not move canonical TTFT / ITL / throughput outside noise while making the API contract more uniform and diagnosable.

## Hypothesis

- The change is boundary-only:
  - top-level unknown routes now return structured OpenAI-style `404`
  - top-level wrong-method requests now return structured OpenAI-style `405`
  - session-route unknown paths and wrong methods now return structured JSON bodies instead of framework-default text
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh http-route-fallback-unification
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/http_server*.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP route fallback tranche
- Feature set:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Non-default flags / env vars: none

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- Local verification was limited to focused no-CUDA tests and `clippy -D warnings`.

## Learnings

- Route-level `404/405` responses are part of the API contract; leaving them as framework-default text weakens client diagnostics even when the happy path is already structured.
- Session sub-routers need the same fallback discipline as the top-level OpenAI-compatible routes, otherwise nested APIs drift in UX.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
