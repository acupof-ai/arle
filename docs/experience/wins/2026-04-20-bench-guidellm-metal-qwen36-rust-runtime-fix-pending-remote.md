# metal qwen3.6 rust runtime fix — guidellm pending-remote, 2026-04-20

**Status:** `pending-remote`
**Commissioned by:** [`docs/plans/2026-04-15-metal-backend-acceptance-plan.md`](../../plans/2026-04-15-metal-backend-acceptance-plan.md)

## Goal

- Regression check: prove the Qwen3.6/Qwen3.5-MoE Metal runtime fix restores HTTP serving without moving canonical TTFT / ITL / throughput outside expected noise.

## Hypothesis

- The fix is runtime-path correctness, not a kernel change:
  - `Qwen35StepDriver::ensure_cpp_session_drained()` now no-ops in Rust mode instead of unconditionally requiring `cpp_model`.
  - Qwen3.6/Qwen3.5-MoE no longer advertises live prefix replay that depends on the compiled Qwen3.5 step path.
- Expected perf impact is neutral to slightly positive on Qwen3.6 because the failing scheduler/runtime path now stays on the intended Rust MoE execution route.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen36-rust-runtime-fix
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file exists to satisfy the required `pending-remote` stub flow for a runtime-affecting Metal diff.

## Environment

- Backend: Metal
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Hardware: pending remote / dedicated Apple Silicon bench host
- Commit: `80b25b2`
- Feature set: `cargo build --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda`
- Non-default flags / env vars: none

## Results — sweep headline table

- Pending remote execution.

## Problems

- No canonical `guidellm` run was executed locally in this turn.
- `cargo clippy --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda -- -D warnings` still fails on pre-existing unrelated warnings elsewhere in the Metal tree; this fix did not attempt a repo-wide lint cleanup.

## Learnings

- Shared Qwen3.5-family helpers must branch on `Qwen35StepMode` before touching compiled-model state; Qwen3.6/MoE reuses the family request state but intentionally lacks `cpp_model`.
- Runtime-only fixes still need a performance stub even when the code change is correctness-first.

## Δ vs baseline

- Pending remote execution against the most recent relevant Metal baseline.

## Artefacts

- Local verification only:
  - `cargo check --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda`
  - `cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda loads_qwen36_config_with_nested_moe_block -- --nocapture`
  - `cargo run --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_bench -- --model '/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46' --prompt-tokens 1024 --generation-tokens 128 --warmup 1 --runs 1`
  - `cargo run --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_serve -- --model-path '/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46' --port 18080 --bind 127.0.0.1 --warmup 0`
  - `curl -sS -X POST http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' --data @/tmp/qwen36_http.json`
- Local smoke observations:
  - `metal_bench`: prompt `62.4 tok/s`, generation `81.7 tok/s`, TTFT `16400 ms` at `prompt=1024`, `gen=128`
  - HTTP `/v1/chat/completions`: request completed successfully after the runtime fix; the prior `Qwen3.5 C++ step path missing compiled model` failure did not reproduce
- Remote canonical artefacts: pending.
