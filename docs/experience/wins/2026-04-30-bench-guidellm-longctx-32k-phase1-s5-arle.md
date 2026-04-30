# guidellm sweep longctx-32k-phase1-s5-arle — guidellm sweep, longctx-32k-phase1-s5-arle, 2026-04-30

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. `scripts/bench_throughput.py` is legacy only.
> Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Goal

- Phase 1 S5 ARLE-side canonical longctx-32k run for Qwen3-4B FP8 KV on
  the local L4 host, before the pinned SGLang baseline.

## Hypothesis

- The S1/S2 mixed FP8 path and S4 harness should complete the
  prompt=32768/output=256 workload at c=1 and c=4 without request errors,
  while keeping Qwen3-4B on the mixed longctx path.

## Command

```bash
scripts/bench_guidellm.sh longctx-32k-phase1-s5-arle \
  --workload longctx-32k \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Invoked via: `scripts/bench_guidellm.sh longctx-32k-phase1-s5-arle --workload longctx-32k --target http://127.0.0.1:8000 --model Qwen3-4B --processor infer/models/Qwen3-4B`

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 13.0 host,
  nvcc from `/usr/local/cuda/bin/nvcc`
- **Commit:** f4816d65
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
| conc1 | 12486.2 | 81.8 | 12491.2 | 12648.8 | 104.47 | 55.92 | 0.09 | 55.91 | 56.15 | 56.15 | 56.15 | 26.75 | 26.97 | 1 | 9.99 | 1288.5 | 1346.58 | 360459 | 2816 | 0.037 |
| conc4 | 69576.1 | 30105.8 | 39535.2 | 99837.9 | 415.74 | 144.53 | 89.5 | 96.82 | 343.62 | 343.62 | 343.62 | 106.44 | 187.36 | 3 | 9.96 | 1285.3 | 1580.97 | 196614 | 1536 | 0.02 |

## Service Trace Peaks

- Poll interval: `1000ms`
- Samples: `736` (ok: `736`, failed: `0`)
- Peak waiting: `1`
- Peak active: `4`
- Peak running_batch: `3`
- Peak prefill_queue: `1`
- Peak kv_util: `98.9%`
- Prefix hit rate: peak `0.0%`, q75 `0.0%`
- Prefix skip rate peak: `0.0%`
- Peak mem: `n/a` (delta vs before: `n/a`)
- Server ttft_p99 peak: `n/a`
- KV fetch queue samples >0: `0/0`
- KV fetch waiter samples >0: `0/736`
- KV store queue samples >0: `0/0`
- Tier wait peaks: fetch `n/a`, store `n/a`

## Service Trace Distribution


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| waiting | 0 | 1 | 1 | 1 | 1 |
| kv_util | 83.4% | 92.7% | 94.8% | 98.9% | 98.9% |


## Service Token Counters


| metric | q25 | q50 | q75 | q99 | peak |
|---|---:|---:|---:|---:|---:|
| decode_tokens | 0 | 1 | 1 | 3 | 3 |
| prefill_tokens | 0 | 2048 | 4096 | 6144 | 6144 |
| tokens_out | 1760 | 3040 | 4578 | 4578 | 4578 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 4 |
| peak waiting | 1 |
| peak prefill_queue | 1 |
| peak kv_util | 98.9% |
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
| completed input tokens | 557073 |
| incomplete input tokens | 163840 |
| completed output tokens | 4352 |
| incomplete output tokens | 11 |

## Problems

- The c=4 leg did not scale over c=1 on output throughput:
  `conc1=9.99 tok/s`, `conc4=9.96 tok/s`.
- GuideLLM emitted a stability warning for c=4:
  `ITL p99/p50 = 3.55` (`p50=96.82 ms`, `p99=343.62 ms`).
- c=4 TTFT is high and variable: p50 `39535.2 ms`, p99 `99837.9 ms`.
- SGLang baseline is not captured yet, so ARLE/SGLang Phase 1 pass/fail is
  still pending.

## Learnings

- The canonical FP8 envelope starts and runs on one L4 with
  `TokenKVPool=136976` tokens, but W1 throughput is effectively saturated by
  long-prompt prefill at c=4 rather than decode capacity.
- The S5 acceptance gate needs the pinned SGLang row before deciding whether
  this is within the Phase 1 `>=0.95x` target or a prefill-kernel gap.

## Δ vs baseline

- **Baseline:** first local ARLE S5 run; pinned SGLang baseline pending.
- **Delta table:** first run.

| metric | baseline | now | Δ% |
|---|---|---|---|
| out tok/s @ c=1 | n/a | 9.99 | n/a |
| out tok/s @ c=4 | n/a | 9.96 | n/a |
| TTFT p50 @ c=4 | n/a | 39535.2 ms | n/a |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: no code change during this run;
  environment bring-up and S3 smoke are recorded in
  `2026-04-30-longctx-l4-local-bringup.md` and
  `2026-04-30-longctx-s3-local-longprompt-smoke.md`.
- Suspected cause of any regression: n/a until SGLang baseline lands.
- Follow-ups: run `scripts/bench_sglang_longctx.sh longctx-32k-phase1-s5`
  after stopping the ARLE service to free the L4.

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-32k-phase1-s5-arle/service_stats_trace_summary.md`
