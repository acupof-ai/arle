# P2.5 W4A8 Marlin auto-config - guidellm fixed-c, CUDA, 2026-05-10

## Goal

Measure `INFER_MARLIN_W4A8_AUTOCONFIG=1` against the legacy explicit W4A8
Marlin config arm for the P2.5 auto-config A/B closeout.

## Hypothesis

The QQQ-style auto selector should be neutral-or-better versus the explicit
legacy config at c=1 and c=4. Gate: all shapes within +/-5%; soft win only at
>=2% latency improvement.

## Params

| Field | Value |
|---|---|
| Profile | `concurrent` |
| Concurrency | `1,4` |
| Prompt / output | `4096 / 256` tokens |
| Duration | `--max-seconds 120`, `--warmup 10` |
| Seed | `20260416` |
| Target | `http://localhost:8000` |
| Wrapper | `scripts/bench_guidellm.sh p25-w4a8-autoconfig` |

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
scripts/bench_guidellm.sh p25-w4a8-autoconfig \
  --model Qwen3-4B-GPTQ-W4A8-zpfix \
  --processor infer/models/Qwen3-4B \
  --concurrencies 1,4 --max-seconds 120 --warmup 10
```

## Results

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | E2E mean | E2E p99 | out tok/s | total tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 416.1 | 64.5 | 405.1 | 756.8 | 15.14 | 13.56 | 0.01 | 13.56 | 13.58 | 13.58 | 3.88 | 4.22 | 66.29 | 1127.18 | 118813 | 7424 | 0.255 |
| conc4 | 1760.6 | 137.7 | 1865.9 | 2034.3 | 24.14 | 17.33 | 0.05 | 17.32 | 17.47 | 17.48 | 6.18 | 6.44 | 168.62 | 2867.21 | 311372 | 19456 | 0.655 |

## Results - service-side metrics

| metric | value |
|---|---:|
| samples | 255 ok / 0 failed |
| peak active | 4 |
| peak waiting | 0 |
| peak running_batch | 4 |
| peak prefill_queue | 3 |
| peak kv_util | 86.7% |
| prefix hit rate | 0.0% |
| prefix skip rate | 0.0% |
| kv fetch waiters | 0/255 |
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
- c=4 TTFT p99 regressed by +9.10% in this first pair. Per bench auto-iteration
  rules this triggered a matched c=4 repeat rather than a one-shot license
  decision.

## Learnings

c=1 was neutral. c=4 mean latency and throughput were neutral, but the first
c=4 tail sample was not sufficient for a license decision; matched repeat was
required.

## Delta vs baseline

Baseline: [`2026-05-10-bench-guidellm-p25-w4a8-baseline.md`](2026-05-10-bench-guidellm-p25-w4a8-baseline.md)

| metric | baseline | auto-config | Delta |
|---|---:|---:|---:|
| c=1 TTFT mean | 415.4 ms | 416.1 ms | +0.17% |
| c=1 TTFT p99 | 747.2 ms | 756.8 ms | +1.28% |
| c=1 ITL mean | 13.57 ms | 13.56 ms | -0.07% |
| c=1 ITL p99 | 13.59 ms | 13.58 ms | -0.07% |
| c=1 out tok/s | 66.29 | 66.29 | +0.00% |
| c=1 total tok/s | 1127.12 | 1127.18 | +0.01% |
| c=4 TTFT mean | 1738.0 ms | 1760.6 ms | +1.30% |
| c=4 TTFT p99 | 1864.6 ms | 2034.3 ms | +9.10% |
| c=4 ITL mean | 17.30 ms | 17.33 ms | +0.17% |
| c=4 ITL p99 | 17.46 ms | 17.48 ms | +0.11% |
| c=4 out tok/s | 168.75 | 168.62 | -0.08% |
| c=4 total tok/s | 2869.46 | 2867.21 | -0.08% |

## Artefacts

- Raw: `bench-output/2026-05-10-p25-w4a8-autoconfig/benchmarks.json`
- CSV: `bench-output/2026-05-10-p25-w4a8-autoconfig/benchmarks.csv`
- HTML: `bench-output/2026-05-10-p25-w4a8-autoconfig/benchmarks.html`
- Headline: `bench-output/2026-05-10-p25-w4a8-autoconfig/headline_table.md`
- Service trace: `bench-output/2026-05-10-p25-w4a8-autoconfig/service_stats_trace_summary.md`
- Server log: `bench-output/server-logs/2026-05-10-p25-w4a8-autoconfig-server.log`
