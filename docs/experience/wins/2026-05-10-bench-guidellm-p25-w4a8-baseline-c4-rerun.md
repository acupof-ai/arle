# P2.5 W4A8 Marlin baseline c=4 repeat - guidellm fixed-c, CUDA, 2026-05-10

## Goal

Repeat the legacy explicit W4A8 Marlin c=4 arm after the first treatment pair
showed a c=4 TTFT p99 tail outlier.

## Hypothesis

If the first c=4 p99 regression was run noise, a fresh matched baseline/treatment
c=4 pair should return to the normal 1.7-1.9s TTFT p99 band with neutral ITL and
throughput.

## Params

| Field | Value |
|---|---|
| Profile | `concurrent` |
| Concurrency | `4` |
| Prompt / output | `4096 / 256` tokens |
| Duration | `--max-seconds 120`, `--warmup 10` |
| Seed | `20260416` |
| Target | `http://localhost:8000` |
| Wrapper | `scripts/bench_guidellm.sh p25-w4a8-baseline-c4-rerun` |

## Env

| Field | Value |
|---|---|
| Backend | CUDA |
| Hardware | NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78) |
| Workspace commit | `6e704a7` |
| Target substrate commit | `8e9901e perf(cuda): add opt-in W4A8 Marlin auto config selector` |
| Model | `infer/models/Qwen3-4B-GPTQ-W4A8-zpfix` |
| Model id | `Qwen3-4B-GPTQ-W4A8-zpfix` |
| Processor | `infer/models/Qwen3-4B` |
| Feature set | `cargo build --release -p infer --features cuda` |
| Non-default env | none for selector arm |
| Server launch | `./target/release/infer --model-path infer/models/Qwen3-4B-GPTQ-W4A8-zpfix --port 8000 --num-slots 8 --max-seq-len 5120` |

## Command

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh p25-w4a8-baseline-c4-rerun \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --processor infer/models/Qwen3-4B \
  --concurrencies 4 --max-seconds 120 --warmup 10
```

## Results

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | E2E mean | E2E p99 | out tok/s | total tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | 1707.7 | 118.4 | 1629.0 | 1889.7 | 23.94 | 17.33 | 0.06 | 17.32 | 17.47 | 17.48 | 6.13 | 6.31 | 169.38 | 2880.11 | 311372 | 19456 | 0.655 |

## Results - service-side metrics

| metric | value |
|---|---:|
| samples | 137 ok / 0 failed |
| peak active | 4 |
| peak waiting | 0 |
| peak running_batch | 4 |
| peak prefill_queue | 3 |
| peak kv_util | 79.1% |
| prefix hit rate | 0.0% |
| prefix skip rate | 0.0% |
| kv fetch waiters | 0/137 |
| kv store queue samples | 0/0 |

## Results - request accounting

| metric | value |
|---|---:|
| completed input tokens | 311372 |
| completed output tokens | 19456 |
| incomplete input tokens | 0 |
| incomplete output tokens | 0 |

## Problems

- Fixed-concurrency `--concurrencies` puts the wrapper in exploration mode, so
  no wins entry was auto-seeded. This snapshot was filled manually from the raw
  artefacts.
- Server warmup hit the known W4A8 prefill warmup retry at `B=8` and
  `2048` tokens/row, then completed at `1024` tokens/row. The benchmark itself
  reported 0 request errors and no queue growth.

## Learnings

The repeated baseline returned to the same c=4 band as the original baseline:
TTFT mean 1707.7 ms, TTFT p99 1889.7 ms, ITL p99 17.48 ms.

## Delta vs baseline

Repeat control arm for the c=4 auto-iteration pair.

## Artefacts

- Raw: `bench-output/2026-05-10-p25-w4a8-baseline-c4-rerun/benchmarks.json`
- CSV: `bench-output/2026-05-10-p25-w4a8-baseline-c4-rerun/benchmarks.csv`
- HTML: `bench-output/2026-05-10-p25-w4a8-baseline-c4-rerun/benchmarks.html`
- Headline: `bench-output/2026-05-10-p25-w4a8-baseline-c4-rerun/headline_table.md`
- Service trace: `bench-output/2026-05-10-p25-w4a8-baseline-c4-rerun/service_stats_trace_summary.md`
- Server log: `bench-output/server-logs/2026-05-10-p25-w4a8-baseline-c4-rerun-server.log`
