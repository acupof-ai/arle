# HTTP session Allow header — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that restoring standard `Allow` headers on session-route `405` responses does not move canonical TTFT / ITL / throughput outside noise while improving API usability.

## Hypothesis

- The change is boundary-only:
  - `/v1/sessions/*` keeps its existing structured JSON error body
  - session-route `405` responses now also emit a standard `Allow` header for `save`, `load`, `manifest`, and delete routes
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh http-session-allow-header
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/http_server/sessions.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP session Allow-header tranche
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

- Nested route groups should preserve the same HTTP recovery semantics as the top-level API; otherwise clients see inconsistent `405` behavior within one service.
- Header correctness matters on internal/session APIs too, not just on OpenAI-compatible routes.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
