# phase15-evictable-c1-guard - guidellm longctx c1 guard, CUDA, 2026-05-01

## Goal

- Supplement the Phase 1 close evidence with the required S5 c=1/360s
  longctx guard for the evictable-prefix admission patch.

## Hypothesis

- The evictable-prefix admission patch should preserve c=1 behavior and,
  ideally, keep the c=1 longctx row within the SGLang parity gate required by
  Phase 1 S5.

## Command

```bash
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs

LONGCTX_SECONDARY_C1_ONLY=1 WORKLOAD=longctx-32k LONGCTX_MAX_SECONDS=360 \
  scripts/bench_guidellm.sh phase15-evictable-c1-guard \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 24 GB class VRAM
- **Runtime patch commit:** `051b1081`
- **Repo head during run:** `9e002fa9`
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Non-default flags / env vars:** FP8 KV, `--num-slots 16`,
  `--max-seq-len 131072`, `--mem-fraction-static 0.85`,
  `--max-num-batched-tokens 16384`, `--max-prefill-tokens 16384`,
  `--schedule-policy fcfs`
- **Server launch:** shown above

## Canonical params

- `--profile concurrent`
- `--data prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256`
- `--max-seconds 360`
- `--random-seed 20260416`
- `--rate 1`
- Workload: `longctx-32k`
- Wrapper: `scripts/bench_guidellm.sh`

## Results - headline table

| rate | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s | total tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc1 | 12540.6 ms | 12645.9 ms | 56.84 ms | 56.93 ms | 9.83 | 1267.65 | 425997 | 3328 | 0.036 |

Successful-only recompute:

| metric | value |
|---|---:|
| successful requests | 13 |
| total output tokens | 3328 |
| effective out tok/s (`total_output_tokens / 360`) | 9.244 |
| TTFT p50 from raw timings | 12540.6 ms |
| ITL p50 from raw request stats | 56.84 ms |

## Results - service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | 1 |
| peak waiting | 0 |
| peak running_batch | 1 |
| peak prefill_queue | 0 |
| peak kv_util | 95.9% |
| plan labels | `idle=4468,decode=3323,prefill=239,split=0,mixed=0` |
| `prefix_hit_rate` | peak `0.0%`, q75 `0.0%` |
| `prefix_skip_rate` | peak `0.0%` |
| `kv_fetch_waiters` | `0/380` |

## Results - request accounting

| metric | value |
|---|---:|
| completed input tokens | 425997 |
| incomplete input tokens | 32768 |
| completed output tokens | 3328 |
| incomplete output tokens | 1 |

## Problems

- The guard does not meet Phase 1 S5 SGLang parity:
  - SGLang c=1 primary: `11.67 out tok/s`
  - SGLang c=1 secondary: `11.57 out tok/s`
  - ARLE GuideLLM c=1: `9.83 out tok/s`
  - ARLE effective c=1: `9.244 out tok/s`
- One request was incomplete at the 360s boundary.

## Learnings

- The evictable-prefix admission patch fixed the c=4 KV-pool edge but did not
  close the single-concurrency long-prompt prefill gap.
- Full Phase 1 S5 close remains blocked until c=1 reaches the parity gate.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **SGLang target:** `docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`

| metric | baseline / target | now | delta |
|---|---:|---:|---:|
| ARLE pre-patch c1 mean | 9.77 out tok/s | 9.83 out tok/s | +0.6% |
| SGLang c1 secondary | 11.57 out tok/s | 9.83 out tok/s | -15.0% |
| SGLang c1 secondary | 11.57 out tok/s | 9.244 effective out tok/s | -20.1% |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-05-01-phase15-evictable-c1-guard/benchmarks.json`
- CSV: `/content/workspace/agent-infer/bench-output/2026-05-01-phase15-evictable-c1-guard/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-05-01-phase15-evictable-c1-guard/benchmarks.html`
- Service trace: `/content/workspace/agent-infer/bench-output/2026-05-01-phase15-evictable-c1-guard/service_stats_trace_summary.md`

## Notes

- This is a blocking guard entry, not a promotion entry.
- Follow-up: profile c=1 long-prompt prefill before starting Phase 2
  implementation.
