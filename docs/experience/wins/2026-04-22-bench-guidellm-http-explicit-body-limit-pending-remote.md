# HTTP explicit body limit — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove that making the top-level HTTP body limit explicit and aligning it with the session routes does not move canonical TTFT / ITL / throughput outside noise while improving large-prompt usability.

## Hypothesis

- The change is boundary-only:
  - top-level OpenAI routes now use an explicit `16 MiB` request body limit
  - session routes reuse the same constant instead of an inlined limit
  - requests below that size continue to work, and oversized requests still fail with the structured `payload_too_large` contract
- Expected performance impact is within noise because request scheduling, batching, and backend kernels are unchanged.

## Command

```bash
scripts/bench_guidellm.sh http-explicit-body-limit
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file is the required `pending-remote` stub for an `infer/src/http_server*.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP explicit body-limit tranche
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

- Relying on framework-default body limits is a DX trap for long-prompt APIs; the limit should be explicit, intentional, and shared across related route groups.
- A deliberate larger limit can improve API usability without changing hot-path runtime behavior.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
