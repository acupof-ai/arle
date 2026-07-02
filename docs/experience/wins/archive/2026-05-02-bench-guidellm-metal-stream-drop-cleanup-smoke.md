# Metal stream-drop cleanup smoke — guidellm, metal-m4pro, 2026-05-02

## Goal

- Regression-check a Metal runtime cleanup that treats dropped streaming clients
  as cancellation instead of service failure, and removes stale Qwen3.5 DFlash
  request-state fields.

## Hypothesis

- The cleanup should not break Qwen3.5 MLX 4bit serving, packed decode, or the
  Metal runtime metrics path.

## Command

```bash
./scripts/start_metal_serve.sh models/Qwen3.5-0.8B-MLX-4bit 8019 -- --warmup 0

./scripts/bench_guidellm.sh metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke \
  --target http://127.0.0.1:8019 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor models/Qwen3.5-0.8B-MLX-4bit \
  --smoke \
  --trace-interval-ms 1000
```

## Environment

- **Backend:** Metal
- **Model:** Qwen3.5-0.8B MLX 4bit, `models/Qwen3.5-0.8B-MLX-4bit`
- **Hardware:** Apple M4 Pro
- **OS / MLX:** macOS 26.3.1, MLX 0.31.1
- **Commit:** `8c5dd20` plus uncommitted Metal cleanup
- **Feature set:** `cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve`
- **Non-default flags / env vars:** `--warmup 0`

## Results

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 112 | 10.2 | 109.8 | 164 | 11.04 | 4.31 | 0.35 | 4.24 | 4.83 | 5.6 | 5.6 | 0.18 | 0.24 | 1 | 93.43 | 3088.88 | 3034.18 | 14364 | 448 | 5.6 |

## Service Trace

| metric | value |
|---|---:|
| samples | 44 ok / 0 failed |
| peak waiting | 1 |
| peak active | 1 |
| peak running_batch | 1 |
| peak prefill_queue | 0 |
| prefix_hit_rate peak | 0.0% |
| prefix_skip_rate peak | 0.0% |
| KV fetch/store wait samples | 0 |

## Problems

- The initial canonical sweep
  `metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup` was interrupted and
  produced no `benchmarks.json`; it is not a valid benchmark result.
- This smoke run is intentionally a short regression check, not a published
  saturation claim.

## Learnings

- The typed stream-drop path exercised during the smoke as
  `Metal batched decode client dropped`; it is now logged as cancellation, not
  as a post-process service error.
- Removing the stale request-state GDR tape/snapshot fields does not affect the
  active Qwen3.5 DFlash tests; rollback state lives in `dflash.rs`.

## Delta vs baseline

- First run for this exact GuideLLM smoke profile:
  `Qwen3.5-0.8B-MLX-4bit`, HTTP serving, `conc1`, smoke prompt/output shape.
- Nearest same-model local helper baseline is
  [`2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`](2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md),
  but it used `metal_bench` 1024/256 rather than GuideLLM HTTP smoke, so no
  percentage delta is reported.

## Artefacts

- Raw: `bench-output/2026-05-02-metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke/benchmarks.json`
- CSV: `bench-output/2026-05-02-metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke/benchmarks.csv`
- HTML: `bench-output/2026-05-02-metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke/benchmarks.html`
- Service trace: `bench-output/2026-05-02-metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke/service_stats_trace.jsonl`
- Service trace summary: `bench-output/2026-05-02-metal-m4pro-qwen35-0p8b-mlx4bit-stream-drop-cleanup-smoke/service_stats_trace_summary.md`
