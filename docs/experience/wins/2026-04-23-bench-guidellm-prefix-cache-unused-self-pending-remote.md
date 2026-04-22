# Prefix-cache unused-self lint fix pending remote verification

## Goal

- Regression-check the `infer/src/prefix_cache.rs` lint-only cleanup after
  dropping an unused `&self` parameter from the block-selection pin helper.

## Hypothesis

- The change should be performance-neutral because it only turns a pure helper
  into an associated function; no runtime data flow, branching, or allocation
  behavior changes.

## Command

```bash
target/release/metal_serve \
  --model-path models/Qwen3-0.6B \
  --port 8011 \
  --bind 127.0.0.1 \
  --warmup 0

scripts/bench_guidellm.sh metal-prefix-cache-unused-self \
  --target http://127.0.0.1:8011 \
  --model Qwen/Qwen3-0.6B \
  --processor models/Qwen3-0.6B
```

Invoked via the canonical wrapper: `scripts/bench_guidellm.sh`.

## Environment

- **Backend:** `metal`
- **Model:** `models/Qwen3-0.6B`
- **Hardware:** Apple Silicon local dev host
- **Commit:** local working tree after `8387e34`
- **Feature set:** `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve`
- **Non-default flags / env vars:** `--warmup 0`, `--target http://127.0.0.1:8011`, `--model Qwen/Qwen3-0.6B`, `--processor models/Qwen3-0.6B`
- **Server launch:** `target/release/metal_serve --model-path models/Qwen3-0.6B --port 8011 --bind 127.0.0.1 --warmup 0`

## Results

- Status: `pending-remote`
- Local code verification completed:
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda prefix_cache:: -- --nocapture`
- Local canonical bench attempt seeded partial artefacts only:
  - `bench-output/2026-04-23-metal-prefix-cache-unused-self/service_stats_before.txt`
  - `bench-output/2026-04-23-metal-prefix-cache-unused-self/service_stats_trace.jsonl`
  - `bench-output/2026-04-23-metal-prefix-cache-unused-self/guidellm.log`
- `benchmarks.json`, `benchmarks.csv`, and `benchmarks.html` were not produced
  before the local run was terminated.

## Problems

- The local canonical `guidellm` sweep against `models/Qwen3-0.6B` kept a
  single request active and continued past the expected local verification
  window, so it did not yield a complete benchmark artefact set.
- Because the sweep did not finish locally, there is no trustworthy Δ table for
  this runtime touch yet.

## Learnings

- Even code-motion-only changes under `infer/src/` still need an explicit bench
  paper trail; when the canonical sweep does not finish, capture the exact
  local attempt and mark the run `pending-remote` instead of silently skipping
  the benchmark gate.
- `service_stats_trace.jsonl` is still useful when `guidellm` stalls: it tells
  us the server kept serving requests and narrows the failure surface to the
  benchmark harness / workload pair rather than the immediate code change.

## Δ vs baseline

- **Baseline reference:** [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md](./2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md)
- Delta table: `pending-remote` because the local sweep did not complete and
  the local model/weights differ from that canonical 4-bit baseline.

## Artefacts

- Partial raw trace: `bench-output/2026-04-23-metal-prefix-cache-unused-self/service_stats_trace.jsonl`
- Partial log: `bench-output/2026-04-23-metal-prefix-cache-unused-self/guidellm.log`
- Service trace (before): `bench-output/2026-04-23-metal-prefix-cache-unused-self/service_stats_before.txt`
