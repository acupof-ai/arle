# guidellm sweep longctx-32k-phase1-baseline-r1-0464fb3e — guidellm sweep, longctx-32k-phase1-baseline-r1-0464fb3e, 2026-04-30

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- First run of the required three-run pre-patch Phase 1 P1.1 longctx-32k
  baseline anchor at commit `0464fb3e`.

## Hypothesis

- This should reproduce the pre-patch near-full-KV behavior observed in the
  plan-label rerun: c=1 near 10 output tok/s, c=4 far below the SGLang
  16.27 output tok/s baseline, with `Mixed > 0` and `Split = 0`.

## Command

```bash
WORKLOAD=longctx-32k scripts/bench_guidellm.sh \
  longctx-32k-phase1-baseline-r1-0464fb3e \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-phase1-baseline-r1-0464fb3e --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 13.0 host,
  nvcc from `/usr/local/cuda/bin/nvcc`
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
| conc1 | 12488.7 | 47 | 12491 | 12587.7 | 104.44 | 55.87 | 0.09 | 55.87 | 56.1 | 56.1 | 56.1 | 26.74 | 26.9 | 1 | 10 | 1289.49 | 1347.65 | 360459 | 2816 | 0.037 |
| conc4 | 39264.8 | 0 | 39264.8 | 39264.8 | 1258.5 | 1109.45 | 0 | 1109.45 | 1109.45 | 1109.45 | 1109.45 | 322.18 | 322.18 | 1 | 0.9 | 116.73 | 0 | 32769 | 256 | 0 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `654` (ok: `654`, failed: `0`)
- Peak waiting: `1`
- Peak active: `4`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Plan labels: `idle=5271863`, `decode=3053`, `prefill=222`, `split=0`, `mixed=16`
- Peak kv_util: `99.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/654`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 1 | 1 |
| kv_util | 80.7% | 92.5% | 95.7% | 98.8% | 99.0% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 1 | 3 | 3 | 3 |
| prefill_tokens | 0 | 2048 | 2048 | 6144 | 6144 |
| tokens_out | 1288 | 2824 | 2826 | 2826 | 3594 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 4 |
| peak waiting | 1 |
| peak prefill_queue | 1 |
| peak kv_util | 99.0% |
| `plan_label.idle` | 5271863 |
| `plan_label.decode` | 3053 |
| `plan_label.prefill` | 222 |
| `plan_label.split` | 0 |
| `plan_label.mixed` | 16 |
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
| completed input tokens | 393228 |
| incomplete input tokens | 131072 |
| completed output tokens | 3072 |
| incomplete output tokens | 511 |

## Problems

- c=4 completed only one request in the 300s window:
  `out tok/s=0.90`, `c4 incomplete output tokens=510`
  (`511` total across c=1+c=4).
- c=4 output throughput is only `0.055x` of the pinned SGLang c=4 baseline
  (`0.90 / 16.27`), far below the Phase 1 entrance target.
- The trace confirms the known failure shape: `Mixed=16`, `Split=0`,
  `prefill_queue=1`, and peak `kv_util=99.0%`.

## Learnings

- This run should be treated as run 1 of the three-run pre-patch anchor, not
  as an optimization result.
- The pre-patch failure is reproducible at `0464fb3e`: the scheduler enters
  mixed mode but loses c=4 progress at the near-full KV edge.

## Δ vs baseline

- **Baseline:** first run of the new three-run pre-patch baseline anchor.
- **Reference:** `2026-04-30-bench-guidellm-longctx-32k-phase1-s5-plan-label.md`
  at the same commit family reported `c1=9.98`, `c4=0.90`, `Mixed=16`.

| metric | baseline | now | Δ% |
|---|---|---|---|
| out tok/s @ c=1 | n/a | 10.00 | n/a |
| out tok/s @ c=4 | n/a | 0.90 | n/a |
| TTFT p50 @ c=4 | n/a | 39264.8 ms | n/a |
| plan_label.mixed | n/a | 16 | n/a |
| plan_label.split | n/a | 0 | n/a |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: none; this run was executed from
  detached `0464fb3e` after reverting the invalid headroom patch on `main`.
- Suspected cause of regression vs SGLang: admission/eviction policy at the
  near-full KV edge, not kernel TFLOPs.
- Follow-ups: run baseline r2/r3, aggregate mean/stddev, then only design a
  patch from the documented SGLang policy gaps.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r1-0464fb3e/service_stats_trace_summary.md`
