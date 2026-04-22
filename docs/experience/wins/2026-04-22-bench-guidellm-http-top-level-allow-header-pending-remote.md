# HTTP top-level Allow header — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that restoring standard `Allow` headers on top-level HTTP `405` responses does not move canonical TTFT / ITL / throughput outside noise while improving client usability.

## Hypothesis

- The change is boundary-only:
  - top-level `405` responses keep the structured OpenAI-style JSON body
  - known routes now also emit a standard `Allow` header describing the accepted method set
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh http-top-level-allow-header
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/error.rs` / `infer/src/http_server.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP top-level Allow-header tranche
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

- A structured JSON `405` body is only half the contract; standard `Allow` headers are what clients and proxies use to recover automatically.
- Replacing framework fallbacks should preserve core HTTP semantics, not just payload shape.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
