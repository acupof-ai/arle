# HTTP JSON rejection parity — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that top-level JSON rejection cleanup does not move canonical TTFT / ITL / throughput outside noise while making malformed-body diagnostics match the session sub-router.

## Hypothesis

- The change is boundary-only:
  - top-level OpenAI routes now distinguish missing `Content-Type`, malformed JSON, and oversized request bodies more precisely
  - `BytesRejection` now maps to structured `payload_too_large` / `invalid_body` errors instead of falling through the generic `invalid_json` bucket
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh http-json-rejection-parity
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/http_server.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP JSON rejection parity tranche
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

- Boundary diagnostics should classify transport/body failures separately from JSON schema failures; otherwise clients cannot tell whether to shrink a request, fix headers, or repair payload structure.
- Keeping top-level OpenAI routes and nested session routes on the same rejection taxonomy reduces surprise for SDK and CLI wrappers.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
