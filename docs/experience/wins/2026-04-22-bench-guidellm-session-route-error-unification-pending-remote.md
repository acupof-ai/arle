# Session route error unification — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove the `/v1/sessions/*` HTTP-boundary cleanup does not move canonical serving TTFT / ITL / throughput outside noise while making the API contract more uniform and diagnosable.

## Hypothesis

- The change is boundary-only:
  - malformed JSON, missing `Content-Type`, and oversized bodies now return structured JSON errors
  - the session sub-router now uses one shared rejection path instead of mixed framework-default responses
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh session-route-error-unification
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/http_server/sessions.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the session-route DX tranche
- Feature set:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server::sessions:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Non-default flags / env vars: none

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- Local verification was limited to focused no-CUDA tests and clippy.

## Learnings

- Session-specific sub-routers should not fall back to framework-default text errors when the top-level API already promises structured JSON.
- DX cleanups on HTTP boundaries still need explicit benchmark bookkeeping even when they are expected to be perf-neutral.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server::sessions:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
