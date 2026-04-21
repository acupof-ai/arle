# Metal mixed-contract cleanup and Qwen3.5/Qwen3.6 session guard — guidellm regression check, metal, 2026-04-21

## Goal

- Regression check for the Phase-1 Metal mixed-batch contract cleanup, plus validation that the Qwen3.5/Qwen3.6 compiled-session regression is fixed.

## Hypothesis

- Renaming the Metal mixed-batch path into a model-agnostic runtime contract should keep the Qwen3 sync baseline roughly flat.
- Guarding Qwen3.5/Qwen3.6 prefix export against nested compiled-session replay, and fixing `qwen35_session_begin/end` error-path state handling, should eliminate the REPL/session breakage where a failed prefix publish poisoned the next decode step.

## Command

```bash
cargo build --release --no-default-features --features metal,no-cuda,cli
cargo check -p infer --release --no-default-features --features metal,no-cuda
cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings
cargo test -p infer --release --no-default-features --features metal,no-cuda backend::metal::scheduler::tests:: -- --nocapture

target/release/metal_serve \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --port 8020 \
  --warmup 1

scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-mixed-contract-regression-512 \
  --target http://127.0.0.1:8020 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --profile synchronous \
  --max-seconds 30

printf '你好\n/quit\n' | \
  target/release/agent-infer \
    --model-path mlx-community/Qwen3.6-35B-A3B-4bit
```

Invoked via: `scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-mixed-contract-regression-512 ...`

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3-0.6B-4bit` for the regression bench; `mlx-community/Qwen3.6-35B-A3B-4bit` for the REPL smoke
- **Hardware:** Apple M4 Pro / 48 GB / macOS 26.3.1
- **Commit base during run:** `00ab66e`
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda,cli`
- **Non-default flags / env vars:** none
- **Server launch:** `target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8020 --warmup 1`

## Results — synchronous headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| sync | 965.3 | 1176.2 | 5.92 | 5.96 | 108.05 | 0.30 |

## Problems

- The REPL smoke still showed model-emitted `<think>...</think>` content in the terminal output. That is separate from the compiled-session bug fixed here.
- This run was a minimum regression check (`synchronous`, 30s), not a full sweep.

## Learnings

- The Metal runtime can be simplified at the contract boundary without introducing a meaningful sync-path collapse, as long as the scheduler budget stays on the committed `max_batch_tokens=512` line.
- Qwen3.5/Qwen3.6 compiled sessions are globally owned by the compiled model, not by individual Rust drivers. Replay-based prefix export must not attempt a nested `session_begin` while a live request still owns that model session.
- `qwen35_session_begin/end` precondition failures must not clear the live compiled session as a side effect.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-phase-metrics-unified-prefill-policy.md](./2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-phase-metrics-unified-prefill-policy.md)

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ synchronous | 899.5 ms | 965.3 ms | +7.3% |
| out tok/s @ synchronous | 109.57 | 108.05 | -1.4% |

## Artefacts

- Raw: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-mixed-contract-regression-512/benchmarks.json`
- CSV: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-mixed-contract-regression-512/benchmarks.csv`
- HTML: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-mixed-contract-regression-512/benchmarks.html`

## Notes

- What changed in the code since baseline:
  - Metal runtime mixed path now goes through one generic `try_mixed_batch` contract instead of a Qwen3-named runtime branch.
  - Qwen3.5/Qwen3.6 replay-based prefix export skips while the live compiled session is active.
  - `qwen35_session_begin/end` no longer clear the current compiled session on precondition failure.
- Qwen3.6 REPL smoke no longer reproduced:
  - `Metal live prefix publish failed ... qwen35_session_begin requires an inactive session`
  - `Metal decode step failed ... qwen35_compiled_step_session requires an active session`
- Follow-ups:
  - investigate `<think>` leakage in the REPL output path separately
