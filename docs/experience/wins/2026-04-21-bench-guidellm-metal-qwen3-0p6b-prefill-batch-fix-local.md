# Metal Qwen3-0.6B scheduler prefill batching fix — local quick regression, 2026-04-21

## Goal

- Regression-check the Metal Qwen3 scheduler/runtime changes that remove scalarized prompt prefill and contain runtime panics, while measuring the local serving delta on `Qwen3-0.6B-4bit`.

## Hypothesis

- Collapsing each prompt chunk into one batched prefill graph should cut TTFT sharply and recover most of the lost HTTP throughput.
- Guarding runtime panics per request/batch should stop the allocator/resource-limit crash from taking down the whole `metal_serve` loop.

## Command

```bash
scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-local-opt1 \
  --quick \
  --target http://127.0.0.1:8019 \
  --model Qwen3-0.6B-4bit \
  --processor models/Qwen3-0.6B
```

Server launch:

```bash
target/release/metal_serve \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --port 8019 \
  --warmup 1
```

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3-0.6B-4bit`
- **Hardware:** Apple M4 Pro, 48 GB unified memory, macOS 26.3.1
- **Commit:** `457ea1f` + local uncommitted diff
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Server launch:** `target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8019 --warmup 1`

## Canonical params

- This turn used the local quick regression shape, not the full canonical `4096 / 256` sweep.
- Profile: `concurrent`
- Data: `prompt_tokens=512,output_tokens=128`
- Max seconds: `60`
- Warmup: `5`
- Concurrency set: `1,2,4,8`

## Results — quick headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc1 | 103.8 | 809.2 | 4.08 | 7.42 | 190.88 | 1.491 |
| conc2 | 147.1 | 195.0 | 8.73 | 11.61 | 191.72 | 1.509 |
| conc4 | 188.8 | 199.5 | 17.64 | 18.23 | 202.76 | 1.655 |
| conc8 | 2682.0 | 3015.9 | 18.05 | 18.46 | 198.62 | 1.636 |

## Problems

- This is a local quick regression-check, not the full canonical `4096 / 256` sweep.
- `conc8` still shows queueing / tail-latency pressure (`TTFT p50 2682 ms`) even though throughput stays near the `conc4` ceiling.
- The pre-change local run had already shown allocator/resource-limit failures at `conc2+`; this patch fixes the acute crash, but it does not yet solve high-concurrency scheduling fairness.

## Learnings

- For Metal Qwen3, scalarized prompt prefill was the primary serving killer. Batching prompt chunks at the request-state layer moved `conc1` from multi-second TTFT into low-hundreds-ms territory.
- The earlier MLX allocator/resource-limit panic was survivability-critical. Catching batch/request panics in the runtime loop turns a server-wide failure into per-request cleanup, which makes the quick profile usable again.
- After the prefill fix, the next real limiter is queueing/active-slot behavior at higher concurrency, not raw decode token rate.

## Δ vs baseline

- **Baseline:** local quick snapshot in `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local/`

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ conc1 | 2094.1 ms | 103.8 ms | -95.0% |
| out tok/s @ conc1 | 44.25 | 190.88 | +331.4% |
| out tok/s @ conc2 | 57.89 | 191.72 | +231.2% |

- Availability delta:
  - baseline `conc4`: `0 out tok/s`, `0 req/s actual`
  - now `conc4`: `202.76 out tok/s`, `1.655 req/s actual`
  - baseline `conc8`: `0 out tok/s`, `0 req/s actual`
  - now `conc8`: `198.62 out tok/s`, `1.636 req/s actual`

## Artefacts

- Raw: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local-opt1/benchmarks.json`
- CSV: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local-opt1/benchmarks.csv`
- HTML: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local-opt1/benchmarks.html`
- Prior local quick baseline: `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local/benchmarks.json`

## Notes

- What changed in the code since baseline:
  - [`infer/src/backend/metal/request_state.rs`](../../../infer/src/backend/metal/request_state.rs) now batches Qwen3 prompt chunks through `run_tokens(...)` instead of scalarizing prefill into one-token graphs.
  - [`infer/src/backend/metal/runtime.rs`](../../../infer/src/backend/metal/runtime.rs) now wraps prefill/decode execution in panic guards and aborts only the affected requests on runtime failure.
  - [`infer/src/backend/metal/qwen35.rs`](../../../infer/src/backend/metal/qwen35.rs) gained a mixed-length packed-decode equivalence test to pin the already-enabled Qwen3.5 varlen path.
  - [`infer/src/backend/metal/AGENTS.md`](../../../infer/src/backend/metal/AGENTS.md) now documents the current Metal-vs-CUDA execution model and corrects the stale claim that Qwen3.5 varlen decode was still scaffold-only.
- Suspected cause of remaining `conc8` TTFT inflation: the runtime no longer dies, so the visible bottleneck is now queue buildup / active-slot residency rather than prompt-side graph materialization.
- Follow-ups:
  - Run the full canonical `4096 / 256` Metal sweep once this runtime patch is committed.
  - Investigate `conc8` queueing: active-slot caps, request admission fairness, and decode-first scheduling pressure.
