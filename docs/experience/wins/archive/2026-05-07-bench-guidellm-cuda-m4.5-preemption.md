# M4.5 — CUDA GuideLLM Canonical Drain Gate

## Goal

- **Regression gate / diagnosis:** confirm the M4.5 disconnected-client slot cleanup unblocks the canonical GuideLLM sweep and drains the server after KV pressure.

## Hypothesis

- With `763d6bb`, the sweep can still drive `kv_util` to 100%, but cancelled clients no longer leave active slots resident; the run should produce `benchmarks.json` and finish with `active=0 waiting=0`.

## Command

```bash
RUST_LOG=info RUST_BACKTRACE=full \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run --release -p infer --no-default-features --features cuda -- \
  --model-path /home/ckl/projects/arle/infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 5120

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-m4.5-preemption \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B
```

Invoked via: `scripts/bench_guidellm.sh cuda-m4.5-preemption --target http://localhost:8000 --model Qwen/Qwen3-4B --processor /home/ckl/projects/arle/infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM, driver 595.71.05, CUDA 13.2.78
- **Commit:** 763d6bb
- **Feature set:** `cargo run --release -p infer --no-default-features --features cuda`
- **Non-default flags / env vars:** `--max-seq-len 5120`, `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python`, `TORCH_CUDA_ARCH_LIST=8.9`
- **Server launch:** see command above.
- **Scheduling envelope:** `max_num_batched_tokens=16384 | 16384, chunked_prefill_size=2048 | 2048, max_prefill_tokens=16384 | 16384, mem_fraction_static=0.85 | 0.85, max_slots=14 | (n/a - SGLang has no fixed cap)`

## Canonical params (resolved by wrapper)

- `--profile sweep`
- `--data prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Workload: `default`
- Wrapper: `scripts/bench_guidellm.sh cuda-m4.5-preemption`

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sync | 539.1 | 74.2 | 517.3 | 803.2 | 17.71 | 15.67 | 0.04 | 15.68 | 15.73 | 15.73 | 15.73 | 4.54 | 4.78 | 1 | 57.17 | 972.03 | 977.4 | 57358 | 3584 | 0.217 |
| throughput | 13566.3 | 8008.8 | 6425.4 | 24562.8 | 92.31 | 39.47 | 22 | 31.3 | 38.54 | 138.98 | 138.98 | 23.67 | 41.86 | 11 | 149.63 | 2544.33 | 3241 | 90134 | 5632 | 0.367 |
| 0.23541666666666666r/s | 2534.2 | 520.5 | 2689.9 | 2703.5 | 47.18 | 37.43 | 3.47 | 38.58 | 38.81 | 38.81 | 38.81 | 12.08 | 12.59 | 3 | 52.53 | 893.29 | 1011.27 | 49164 | 3072 | 0.2 |
| 0.25416666666666665r/s | 2537 | 540.8 | 2701.3 | 2711.7 | 57.31 | 47.59 | 6.47 | 49.77 | 49.87 | 49.87 | 49.87 | 14.67 | 15.43 | 3 | 52.98 | 900.9 | 1086.15 | 49164 | 3072 | 0.2 |
| 0.2729166666666667r/s | 2538.4 | 540.1 | 2702.7 | 2717.7 | 68.49 | 58.81 | 6.76 | 61.19 | 61.42 | 61.42 | 61.42 | 17.54 | 18.37 | 4 | 53.02 | 901.55 | 1162.88 | 49164 | 3072 | 0.2 |
| 0.2916666666666667r/s | 2517.2 | 581.6 | 2700.1 | 2734.6 | 90.11 | 80.6 | 16.28 | 82.83 | 94.45 | 94.45 | 94.45 | 23.07 | 26.82 | 4 | 44.97 | 764.72 | 1247.67 | 40970 | 2560 | 0.167 |
| 0.3104166666666667r/s | 2496.8 | 613.7 | 2704.7 | 2746.3 | 116.31 | 106.97 | 22.16 | 115.34 | 120.79 | 120.79 | 120.79 | 29.78 | 33.55 | 4 | 39.33 | 668.75 | 1327.24 | 36873 | 2304 | 0.15 |
| 0.3291666666666667r/s | 2489.7 | 621 | 2706.7 | 2728.8 | 123.28 | 114 | 16.25 | 119.76 | 121.27 | 121.27 | 121.27 | 31.56 | 33.65 | 5 | 40.25 | 684.46 | 1401.31 | 36873 | 2304 | 0.15 |
| 0.34791666666666665r/s | 2527.5 | 572.9 | 2713 | 2743.8 | 129.41 | 120.01 | 2.32 | 120.69 | 121.54 | 121.54 | 121.54 | 33.13 | 33.74 | 6 | 43.52 | 740.07 | 1472.34 | 40970 | 2560 | 0.167 |
| 0.3666666666666667r/s | 2523.9 | 590.7 | 2716.7 | 2754.9 | 130 | 120.62 | 1.9 | 121.24 | 121.56 | 121.56 | 121.56 | 33.28 | 33.75 | 6 | 44.48 | 756.36 | 1542.97 | 40970 | 2560 | 0.167 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `673` (ok: `673`, failed: `0`)
- Peak waiting: `501`
- Peak active: `12`
- Peak running_batch: `12`
- Peak prefill_queue: `10`
- Plan labels: `idle=24`, `decode=10252`, `prefill=73`, `split=0`, `mixed=446`
- Peak kv_util: `100.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/673`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 500 | 501 |
| kv_util | 63.7% | 90.5% | 98.8% | 99.9% | 100.0% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 1 | 3 | 7 | 12 | 12 |
| prefill_tokens | 0 | 2048 | 2048 | 14339 | 16384 |
| tokens_out | 10174 | 19067 | 27443 | 36094 | 36742 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 12 |
| peak waiting | 501 |
| peak prefill_queue | 10 |
| peak kv_util | 100.0% |
| final active / waiting | 0 / 0 |
| plan labels | idle=24, decode=10252, prefill=73, split=0, mixed=446 |
| `prefix_hit_rate` | 0.0% |
| `prefix_skip_rate` | 0.0% |
| `kv_fetch_q` | 0/16 |
| `kv_fetch_waiters` | 0/673 samples >0 |
| `kv_store_q` | 0/16 |
| `kv_store` | sub:0, done:0, fail:0, rej:0 |
| `kv_bp` | fetch:0, store:0 |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | 491640 |
| incomplete input tokens | 2363392 |
| completed output tokens | 30720 |
| incomplete output tokens | 5749 |
| final service `tokens_out` | 36742 |

## Problems

- First attempt before `763d6bb` reproduced the stall: `active=12 waiting=513 scheduled=0 decode_rows=0 prefill_rows=0`, with artifacts at `bench-output/2026-05-07-cuda-m4.5-preemption/`.
- Throughput mode still shows high tail latency (`ITL p99/p50 = 4.44`) and many incomplete input tokens under GuideLLM saturation. This is a perf/stability issue, not the M4.5 drain bug: after run2 the service reported `active=0 waiting=0`.

## Learnings

- Treat closed client channels as scheduler pressure, not as harmless frontend cancellation: active slots with closed `delta_tx` must be released even when no decode plan is runnable.
- Waiting queues must drop closed requests before building admission plans, otherwise a cancelled load generator can keep the scheduler hot with unservable work.
- The useful M4.5 signal was not "no KV pressure"; run2 hit `kv_util=100.0%` and `waiting=501`, but continued issuing decode/mixed plans and drained.

## Δ vs baseline

- **Baseline:** [2026-05-07-m4-guidellm-canonical-stuck.md](../errors/2026-05-07-m4-guidellm-canonical-stuck.md)
- **Delta:** first publishable M4.5 canonical run. The baseline produced no `benchmarks.json` because the server stuck at `active=12 waiting=560 scheduled=0 decode_rows=0 prefill_rows=0`.

## Artefacts

- Raw: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/benchmarks.json`
- CSV:  `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/benchmarks.csv`
- HTML: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/benchmarks.html`
- Service trace (before): `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_before.txt`
- Service trace (during): `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_trace.jsonl`
- Service trace (after):  `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_after.txt`
- Service trace (summary): `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: `763d6bb fix(scheduler): drain cancelled CUDA slots under pressure`.
- Suspected cause of remaining tail latency: saturation sweep queues many 4097-token prompts against 14 slots on a 16 GB card; this is expected pressure for the canonical sweep and should be handled in M6 perf analysis.
- Follow-ups: M6 can use this run as the first non-stuck CUDA GuideLLM baseline; no M4.5 rollback needed.

## Service Trace

- Poll interval: `1000ms`
- Before: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_before.txt`
- During: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_trace.jsonl`
- After: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_after.txt`
- Summary: `/home/ckl/projects/arle/bench-output/2026-05-07-cuda-m4.5-preemption-run2/service_stats_trace_summary.md`
