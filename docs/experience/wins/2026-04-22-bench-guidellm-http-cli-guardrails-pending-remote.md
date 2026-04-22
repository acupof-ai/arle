# HTTP / CLI guardrails — guidellm pending-remote, 2026-04-22

**Status:** `pending-remote`  
**Commissioned by:** [`docs/plans/train-eval-infer-dx-v1.md`](../../plans/train-eval-infer-dx-v1.md)

## Goal

- Regression check: prove the HTTP request-guardrail cleanup does not move canonical `guidellm` TTFT / ITL / throughput outside noise while improving API/CLI stability and DX.

## Hypothesis

- The HTTP changes are entry-boundary only:
  - malformed JSON is mapped into OpenAI-style error bodies
  - invalid sampling knobs fail before scheduler submission
  - non-finite logprobs are dropped instead of risking serialization failures
- The CLI changes are parse-time only:
  - invalid `--max-turns`, `--max-tokens`, and `--temperature` values fail in clap before runtime
- Expected perf impact is within noise because no scheduler, kernel, batching, or backend implementation changed.

## Command

```bash
scripts/bench_guidellm.sh http-cli-guardrails
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file exists as the required `pending-remote` stub for an `infer/src/http_server.rs` diff.

## Environment

- Backend: HTTP serving surface
- Hardware: pending remote / canonical bench host for the affected backend
- Commit: pending local commit for the HTTP / CLI guardrail tranche
- Feature set:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo test -p cli --release --no-default-features --features no-cuda --lib -- --nocapture`
- Non-default flags / env vars: none

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- Local verification was limited to focused no-CUDA tests and `clippy -D warnings`.

## Learnings

- HTTP API guardrails still need an explicit benchmark stub even when the change is expected to be perf-neutral.
- Serialization-safety and request-validation cleanups belong at the boundary so runtime regressions fail as structured client errors instead of deeper in the serving stack.

## Δ vs baseline

- Pending remote execution against the most recent relevant HTTP-serving baseline.

## Artefacts

- Local verification only:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo test -p cli --release --no-default-features --features no-cuda --lib -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
  - `cargo clippy -p cli --release --lib --no-default-features --features no-cuda -- -D warnings`
- Remote canonical artefacts: pending.
