# P2.5 W4A8 Marlin auto-config c=4 repeat - guidellm fixed-c, CUDA, 2026-05-10

## Goal

Repeat the `INFER_MARLIN_W4A8_AUTOCONFIG=1` c=4 arm after the first treatment
pair showed a c=4 TTFT p99 tail outlier.

## Hypothesis

If the first c=4 p99 regression was run noise, the matched repeat should be
within +/-5% of the repeated legacy baseline on TTFT, ITL, and throughput.

## Params

| Field | Value |
|---|---|
| Profile | `concurrent` |
| Concurrency | `4` |
| Prompt / output | `4096 / 256` tokens |
| Duration | `--max-seconds 120`, `--warmup 10` |
| Seed | `20260416` |
| Target | `http://localhost:8000` |
| Wrapper | `scripts/bench_guidellm.sh p25-w4a8-autoconfig-c4-rerun` |

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
| Non-default env | `INFER_MARLIN_W4A8_AUTOCONFIG=1` |
| Server launch | `INFER_MARLIN_W4A8_AUTOCONFIG=1 ./target/release/infer --model-path infer/models/Qwen3-4B-GPTQ-W4A8-zpfix --port 8000 --num-slots 8 --max-seq-len 5120` |

## Command

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
INFER_MARLIN_W4A8_AUTOCONFIG=1 \
scripts/bench_guidellm.sh p25-w4a8-autoconfig-c4-rerun \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --processor infer/models/Qwen3-4B \
  --concurrencies 4 --max-seconds 120 --warmup 10
```

## Results

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | E2E mean | E2E p99 | out tok/s | total tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc4 | 1706.6 | 117.4 | 1628.1 | 1883.9 | 23.92 | 17.33 | 0.06 | 17.31 | 17.47 | 17.48 | 6.13 | 6.29 | 169.48 | 2881.84 | 311372 | 19456 | 0.655 |

## Results - service-side metrics

| metric | value |
|---|---:|
| samples | 133 ok / 0 failed |
| peak active | 4 |
| peak waiting | 0 |
| peak running_batch | 4 |
| peak prefill_queue | 3 |
| peak kv_util | 79.0% |
| prefix hit rate | 0.0% |
| prefix skip rate | 0.0% |
| kv fetch waiters | 0/133 |
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

The c=4 treatment repeat was neutral versus the repeated legacy baseline:
TTFT p99 improved by 0.31%, ITL p99 was unchanged, and total tok/s improved by
0.06%. This kills the hypothesis that the first c=4 p99 spike was a stable
auto-config regression.

## Delta vs baseline

Baseline: [`2026-05-10-bench-guidellm-p25-w4a8-baseline-c4-rerun.md`](2026-05-10-bench-guidellm-p25-w4a8-baseline-c4-rerun.md)

| metric | baseline repeat | auto-config repeat | Delta |
|---|---:|---:|---:|
| c=4 TTFT mean | 1707.7 ms | 1706.6 ms | -0.06% |
| c=4 TTFT p99 | 1889.7 ms | 1883.9 ms | -0.31% |
| c=4 ITL mean | 17.33 ms | 17.33 ms | +0.00% |
| c=4 ITL p99 | 17.48 ms | 17.48 ms | +0.00% |
| c=4 out tok/s | 169.38 | 169.48 | +0.06% |
| c=4 total tok/s | 2880.11 | 2881.84 | +0.06% |

## Artefacts

- Raw: `bench-output/2026-05-10-p25-w4a8-autoconfig-c4-rerun/benchmarks.json`
- CSV: `bench-output/2026-05-10-p25-w4a8-autoconfig-c4-rerun/benchmarks.csv`
- HTML: `bench-output/2026-05-10-p25-w4a8-autoconfig-c4-rerun/benchmarks.html`
- Headline: `bench-output/2026-05-10-p25-w4a8-autoconfig-c4-rerun/headline_table.md`
- Service trace: `bench-output/2026-05-10-p25-w4a8-autoconfig-c4-rerun/service_stats_trace_summary.md`
- Server log: `bench-output/server-logs/2026-05-10-p25-w4a8-autoconfig-c4-rerun-server.log`
