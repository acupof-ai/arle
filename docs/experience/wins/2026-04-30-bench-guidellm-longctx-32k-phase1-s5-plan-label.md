# guidellm sweep longctx-32k-phase1-s5-plan-label — guidellm sweep, longctx-32k-phase1-s5-plan-label, 2026-04-30

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Rerun Phase 1 S5 after adding scheduler plan-label counters, so the
  longctx-32k acceptance gate can prove `Mixed > 0` and `Split = 0` from
  `/v1/stats`.

## Hypothesis

- The new counters should not change request behavior, and the S5 trace should
  show monotonic plan-label totals for decode, prefill, split, and mixed.

## Command

```bash
scripts/bench_guidellm.sh longctx-32k-phase1-s5-plan-label \
  --workload longctx-32k \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `scripts/bench_guidellm.sh longctx-32k-phase1-s5-plan-label --workload longctx-32k --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 13.0 host,
  nvcc from `/usr/local/cuda/bin/nvcc`
- **Commit:** 0464fb3e
- **Feature set:** `cargo build -p infer --release --no-default-features --features cuda --bin infer`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `PATH=/usr/local/cuda/bin:$PATH`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_HOME=/tmp/arle-cargo-home`, `CARGO_TARGET_DIR=/tmp/arle-target`,
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
| conc1 | 12530.1 | 64.5 | 12521.5 | 12662.3 | 104.62 | 55.9 | 0.11 | 55.88 | 56.16 | 56.16 | 56.16 | 26.78 | 26.94 | 1 | 9.98 | 1287.01 | 1344.93 | 360459 | 2816 | 0.037 |
| conc4 | 39275.6 | 0 | 39275.6 | 39275.6 | 1258.86 | 1109.78 | 0 | 1109.78 | 1109.78 | 1109.78 | 1109.78 | 322.27 | 322.27 | 1 | 0.9 | 116.7 | 0 | 32769 | 256 | 0 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `656` (ok: `656`, failed: `0`)
- Peak waiting: `1`
- Peak active: `4`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Plan labels: `idle=5293289`, `decode=3053`, `prefill=222`, `split=0`, `mixed=16`
- Peak kv_util: `99.0%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/656`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 0 | 0 | 1 | 1 |
| kv_util | 80.7% | 91.2% | 95.7% | 98.8% | 99.0% |


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
| `plan_label.idle` | 5293289 |
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

- c=4 is worse than the earlier S5 run: only one c=4 request completed inside
  the 300s window (`out tok/s=0.9`, incomplete output tokens `510`), so this
  run is not a performance baseline replacement.
- The plan-label gate is now machine-checkable and passes (`Mixed=16`,
  `Split=0`), but the performance gate still fails: c=4 throughput is far
  below SGLang `16.27 out tok/s`.
- `idle` is noisy because the scheduler can spin through Idle selections while
  a request is active but no GPU work is launchable; use `decode/prefill/split/mixed`
  for Phase 1 gate decisions.

## Learnings

- The earlier uncertainty is resolved: Qwen3-4B FP8 longctx does enter Mixed
  on this build, and Split stays at zero.
- Phase 1 remains blocked by long-prompt throughput and c=4 progress, not by
  Mixed/Split observability.
- The c=4 instability makes P1.1 profiling more urgent: we need to explain why
  a retry at the same envelope completed only one c=4 request while the prior
  S5 run completed six c=4 requests.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-s5-arle.md`
- **Delta table:** this run is primarily an observability rerun; c=4 is kept
  as a regression signal, not as a replacement baseline.

| metric | baseline | now | Δ% |
|---|---|---|---|
| out tok/s @ c=1 | 9.99 | 9.98 | -0.1% |
| out tok/s @ c=4 | 9.96 | 0.90 | -91.0% |
| TTFT p50 @ c=4 | 39535.2 ms | 39275.6 ms | -0.7% |
| plan_label.mixed | n/a | 16 | n/a |
| plan_label.split | n/a | 0 | n/a |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: `0464fb3e` adds plan-label counters
  and trace-summary parsing; intended to be observability-only.
- Suspected cause of c=4 regression: not explained by this diff yet. The trace
  shows peak KV utilization `99.0%`, high idle count, and only one completed c=4
  request, so the next pass should profile admission/progress under near-full
  KV pressure.
- Follow-ups: P1.1 c=4 long-prefill profile with plan labels, then either fix
  admission/overlap or open the FP8 prefill tensor-core kernel project.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-plan-label/service_stats_trace_summary.md`
