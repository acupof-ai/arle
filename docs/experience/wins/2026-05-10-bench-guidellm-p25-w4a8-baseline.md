# P2.5 W4A8 Marlin baseline - guidellm fixed-c, CUDA, 2026-05-10

## Goal

Measure the legacy explicit W4A8 Marlin config arm for the P2.5 auto-config
A/B closeout.

## Hypothesis

The env-unset legacy path should reproduce the existing W4A8 fixed-concurrency
baseline at c=1 and c=4, with no request errors and no queue growth.

## Params

| Field | Value |
|---|---|
| Profile | `concurrent` |
| Concurrency | `1,4` |
| Prompt / output | `4096 / 256` tokens |
| Duration | `--max-seconds 120`, `--warmup 10` |
| Seed | `20260416` |
| Target | `http://localhost:8000` |
| Wrapper | `scripts/bench_guidellm.sh p25-w4a8-baseline` |

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
scripts/bench_guidellm.sh p25-w4a8-baseline \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1,4 --max-seconds 120 --warmup 10
```

## Results

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | E2E mean | E2E p99 | out tok/s | total tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 415.4 | 62.8 | 404.9 | 747.2 | 15.14 | 13.57 | 0.01 | 13.57 | 13.59 | 13.59 | 3.88 | 4.21 | 66.29 | 1127.12 | 118813 | 7424 | 0.255 |
| conc4 | 1738.0 | 116.9 | 1712.9 | 1864.6 | 24.03 | 17.30 | 0.05 | 17.29 | 17.45 | 17.46 | 6.15 | 6.27 | 168.75 | 2869.46 | 311372 | 19456 | 0.655 |

## Results - service-side metrics

| metric | value |
|---|---:|
| samples | 264 ok / 0 failed |
| peak active | 4 |
| peak waiting | 0 |
| peak running_batch | 4 |
| peak prefill_queue | 3 |
| peak kv_util | 78.4% |
| prefix hit rate | 0.0% |
| prefix skip rate | 0.0% |
| kv fetch waiters | 0/264 |
| kv store queue samples | 0/0 |

## Results - request accounting

| metric | value |
|---|---:|
| completed input tokens | 430185 |
| completed output tokens | 26880 |
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

Legacy explicit config is a stable matched-control arm for c=1 and c=4; it is
the correct baseline for licensing the opt-in selector.

## Delta vs baseline

First arm in this matched A/B pair.

## Artefacts

- Raw: `bench-output/2026-05-10-p25-w4a8-baseline/benchmarks.json`
- CSV: `bench-output/2026-05-10-p25-w4a8-baseline/benchmarks.csv`
- HTML: `bench-output/2026-05-10-p25-w4a8-baseline/benchmarks.html`
- Headline: `bench-output/2026-05-10-p25-w4a8-baseline/headline_table.md`
- Service trace: `bench-output/2026-05-10-p25-w4a8-baseline/service_stats_trace_summary.md`
- Server log: `bench-output/server-logs/2026-05-10-p25-w4a8-baseline-server.log`
