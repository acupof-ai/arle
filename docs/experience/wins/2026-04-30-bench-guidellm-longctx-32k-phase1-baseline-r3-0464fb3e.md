# guidellm sweep longctx-32k-phase1-baseline-r3-0464fb3e — guidellm sweep, longctx-32k-phase1-baseline-r3-0464fb3e, 2026-04-30

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Third run of the required three-run pre-patch Phase 1 P1.1 longctx-32k
  baseline anchor at commit `0464fb3e`.

## Hypothesis

- This should keep the same pre-patch failure shape as runs 1 and 2: c=1 near
  10 output tok/s, c=4 below the SGLang 16.27 output tok/s anchor, with mixed
  decode+prefill steps and no split-prefill path.

## Command

```bash
WORKLOAD=longctx-32k scripts/bench_guidellm.sh \
  longctx-32k-phase1-baseline-r3-0464fb3e \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-phase1-baseline-r3-0464fb3e --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

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

GuideLLM generated raw JSON/CSV/HTML, then returned exit code 4 because c=4
had no successful requests. The table below records the emitted summary.

| rate | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 | TPOT p95 | conc mean | out tok/s | total tok/s | in tok/s | completed in | incomplete in | completed out | incomplete out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 12500.4 ms | 12611.0 ms | 56.8 ms | 57.2 ms | 105.4 ms | 105.9 ms | 1.0 | 9.5 | 1333.9 | 1324.4 | 360459 | 32768 | 2816 | 1 |
| conc4 | n/a | n/a | n/a | n/a | n/a | n/a | 4.0 | 0.1 | 282.9 | 282.8 | 0 | 131072 | 0 | 38 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `809` (ok: `809`, failed: `0`)
- Peak waiting: `2`
- Peak active: `3`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Plan labels observed during trace: `idle=10613280`, `decode=8933`,
  `prefill=658`, `split=0`, `mixed=52`
- Plan-label delta from before to after: `decode +2807`, `prefill +215`,
  `split +0`, `mixed +24`
- Peak kv_util: `91.4%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/809`
- KV store queue samples >0: `0/0`

## Service Trace Distribution

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 1 | 1 | 2 | 2 |
| kv_util | 73.3% | 83.8% | 91.2% | 91.4% | 91.4% |

## Service Token Counters

| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 1 | 2 | 2 | 2 |
| prefill_tokens | 0 | 2048 | 2048 | 4096 | 4096 |
| tokens_out | 8743 | 10025 | 10025 | 10025 | 10025 |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 3 |
| peak waiting | 2 |
| peak prefill_queue | 1 |
| peak kv_util | 91.4% |
| `plan_label.decode_delta` | 2807 |
| `plan_label.prefill_delta` | 215 |
| `plan_label.split_delta` | 0 |
| `plan_label.mixed_delta` | 24 |
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
| completed input tokens | 360459 |
| incomplete input tokens | 163840 |
| completed output tokens | 2816 |
| incomplete output tokens | 39 |

## Problems

- GuideLLM marked the result invalid because c=4 had zero successful requests.
- The c=4 300s window produced only `38` incomplete output tokens and no
  completed request, while leaving 3 active/running requests in the server
  after snapshot.
- The service trace still showed mixed decode+prefill activity
  (`mixed_delta=24`) and no split path (`split_delta=0`), matching the same
  qualitative scheduler failure seen in runs 1 and 2.

## Learnings

- The pre-patch c=4 failure mode is severe enough to produce invalid benchmark
  result sets, not just low throughput.
- For same-server sequential runs, plan-label counters are cumulative. This
  entry therefore records before/after deltas for the scheduler counters.

## Δ vs baseline

- **Baseline:** `2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-r2-0464fb3e.md`
- **Delta table:** run 3 vs run 2 at the same `0464fb3e` commit.

| metric | baseline | now | Δ% |
|---|---|---|---|
| out tok/s @ c=1 | 9.80 | 9.50 | -3.1% |
| out tok/s @ c=4 | 8.07 | 0.10 | -98.8% |
| c=4 successful requests | 3 | 0 | -100.0% |
| c=4 completed output tokens | 768 | 0 | -100.0% |
| c=4 incomplete output tokens | 10 | 38 | +280.0% |
| plan_label.mixed | 28 observed | 24 delta | n/a |
| plan_label.split | 0 observed | 0 delta | 0.0% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: none; this run was executed from
  detached `0464fb3e` after reverting the invalid headroom patch on `main`.
- Suspected cause of regression vs SGLang: admission/eviction policy at the
  near-full KV edge, not kernel TFLOPs.
- Follow-ups: aggregate r1-r3 mean/stddev, correct earlier service-counter
  wording where needed, then only design a patch from the documented SGLang
  policy gaps.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-baseline-r3-0464fb3e/service_stats_trace_summary.md`
