# HTTP auth challenge header — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that adding standard Bearer auth challenge headers to HTTP `401` responses does not move canonical TTFT / ITL / throughput outside noise while improving client interoperability.

## Hypothesis

- The change is boundary-only:
  - unauthorized responses now include `WWW-Authenticate: Bearer realm="agent-infer"`
  - JSON error bodies and runtime request handling stay unchanged
- Expected performance impact is within noise because scheduler, batching, and backend kernels are untouched.

## Command

```bash
scripts/bench_guidellm.sh http-auth-challenge-header
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/error.rs` / `infer/src/http_server.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP auth-header tranche
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

- Standard auth challenge headers are part of HTTP usability, not optional polish; many clients use them to recognize authentication failures cleanly.
- The API can stay OpenAI-style in body shape while still behaving like a correct Bearer-protected HTTP service at the header layer.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
