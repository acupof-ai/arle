# guidellm sweep longctx-32k-phase1-baseline-r2-0464fb3e — guidellm sweep, longctx-32k-phase1-baseline-r2-0464fb3e, 2026-04-30

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Second run of the required three-run pre-patch Phase 1 P1.1 longctx-32k
  baseline anchor at commit `0464fb3e`.

## Hypothesis

- This should keep the same pre-patch failure shape as run 1: c=1 near 10
  output tok/s, c=4 unstable and below the SGLang 16.27 output tok/s anchor,
  with `Mixed > 0`, `Split = 0`, and near-full KV pressure.

## Command

```bash
WORKLOAD=longctx-32k scripts/bench_guidellm.sh \
  longctx-32k-phase1-baseline-r2-0464fb3e \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-phase1-baseline-r2-0464fb3e --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA host from
  the NVIDIA driver stack, nvcc `/usr/local/cuda/bin/nvcc` reports 12.8
- **Commit:** 0464fb3e
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `TORCH_CUDA_ARCH_LIST=8.9`, `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_TARGET_DIR=/tmp/arle-target`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs`

## Canonical params (resolved by wrapper)

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 300`
- `--random-seed 20260416`
- `--rate 1,4`
- `--outputs json --outputs csv --outputs html`
- Workload: `longctx-32k`
- Wrapper: `scripts/bench_guidellm.sh <backend-label> --workload longctx-32k`

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conc1 | 12507.2 | 53.9 | 12497.5 | 12631.5 | 106.53 | 57.9 | 0.47 | 58.06 | 58.19 | 58.19 | 58.19 | 27.27 | 27.38 | 1 | 9.8 | 1264.12 | 1322.48 | 360459 | 2816 | 0.033 |
| conc4 | 79344.3 | 30431.7 | 100862.3 | 100863.3 | 505.99 | 196.82 | 108.64 | 120.01 | 350.46 | 350.46 | 350.46 | 129.54 | 131.47 | 0 | 8.07 | 1041.18 | 1522.84 | 98307 | 768 | 0.01 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `670` (ok: `670`, failed: `0`)
- Peak waiting: `1`
- Peak active: `3`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Plan labels: `idle=5272618`, `decode=6119`, `prefill=442`, `split=0`, `mixed=28`
- Peak kv_util: `98.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/670`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 1 | 1 | 1 |
| kv_util | 83.8% | 89.7% | 94.8% | 98.0% | 98.0% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 1 | 1 | 3 | 3 |
| prefill_tokens | 0 | 2048 | 4096 | 6144 | 6144 |
| tokens_out | 4882 | 6418 | 7186 | 7186 | 7186 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 3 |
| peak waiting | 1 |
| peak prefill_queue | 1 |
| peak kv_util | 98.0% |
| `plan_label.idle` | 5272618 |
| `plan_label.decode` | 6119 |
| `plan_label.prefill` | 442 |
| `plan_label.split` | 0 |
| `plan_label.mixed` | 28 |
| `prefix_hit_rate` | peak 0.0%, q75 0.0% |
| `prefix_skip_rate` | peak 0.0% |
| `kv_fetch_q` | 0/16 |
| `kv_fetch_waiters` | 0 |
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
| completed input tokens | 458766 |
| incomplete input tokens | 131072 |
| completed output tokens | 3584 |
| incomplete output tokens | 10 |

## Problems

- c=4 was highly unstable across the same pinned commit and workload:
  `out tok/s=8.07`, but c=4 completed input was only `98307` tokens and
  c=4 TTFT p50 rose to `100862.3 ms`.
- The run still remained below the pinned SGLang c=4 baseline
  (`8.07 / 16.27 = 0.496x`) and far below the Phase 1 entrance target.
- The trace confirms the same qualitative scheduler gap as run 1:
  `Mixed=28`, `Split=0`, `prefill_queue=1`, and peak `kv_util=98.0%`.

## Learnings

- Throughput-only c=4 numbers are noisy at this failure edge; the anchor needs
  the full three-run mean plus TTFT/tail and request-accounting context.
- Run 2 did more c=4 output than run 1, but paid for it with 100s TTFT and
  near-full KV pressure. That is still the same admission/eviction problem,
  not proof that the scheduler is healthy.

## Δ vs baseline

- **Baseline:** `2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-r1-0464fb3e.md`
- **Delta table:** run 2 vs run 1 at the same `0464fb3e` commit.

| metric | baseline | now | Δ% |
|---|---|---|---|
| out tok/s @ c=1 | 10.00 | 9.80 | -2.0% |
| out tok/s @ c=4 | 0.90 | 8.07 | +796.7% |
| TTFT p50 @ c=4 | 39264.8 ms | 100862.3 ms | +156.9% |
| completed output tokens | 3072 | 3584 | +16.7% |
| incomplete output tokens | 511 | 10 | -98.0% |
| plan_label.mixed | 16 | 28 | +75.0% |
| plan_label.split | 0 | 0 | 0.0% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: none; this run was executed from
  detached `0464fb3e` after reverting the invalid headroom patch on `main`.
- Suspected cause of regression vs SGLang: admission/eviction policy at the
  near-full KV edge, not kernel TFLOPs.
- Follow-ups: run baseline r3, aggregate mean/stddev across r1-r3, then only
  design a patch from the documented SGLang policy gaps.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r2-0464fb3e/service_stats_trace_summary.md`
