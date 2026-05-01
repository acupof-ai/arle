# guidellm invalid result phase15-evictable-c4-r2 — 2026-05-01

This entry records the fresh-server rerun for the Phase 1.5 evictable-prefix
admission patch. It reproduced the r1 measurement failure: GuideLLM completed
requests and wrote artifacts, but the wrapper rejected the run because TTFT p50
and ITL p50 were both `0.0` with non-zero output tokens. This is not a valid
sample for the required three-run c=4 mean.

## Goal

- Check whether `phase15-evictable-c4-r1` was a one-off GuideLLM timing
  artifact by rerunning the same c=4 longctx workload from a fresh server.

## Hypothesis

- If r1 was process-state contamination, a fresh server should produce a valid
  GuideLLM timing table while preserving the non-deadlocked completion behavior.

## Command

```bash
WORKLOAD=longctx-32k LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 \
  scripts/bench_guidellm.sh phase15-evictable-c4-r2 \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 12.8 toolchain
- **Commit:** `051b1081`
- **Feature set:** `cargo build -p infer --release --features cuda`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs`
- **KV pool:** `136976` max tokens, `8561` pages, `11.0 GB`, FP8E4M3

## Results

GuideLLM headline table, invalid:

| metric | value |
|---|---:|
| completed input tokens | 917,508 |
| completed output tokens | 7,168 |
| median concurrency | 4.0 |
| request/s mean | 0.1 |
| output tok/s mean | 27.7 |
| total tok/s mean | 3568.0 |
| request latency p50 | 33.9s |
| request latency p95 | 111.5s |
| TTFT p50 | 0.0 ms invalid |
| ITL p50 | 0.0 ms invalid |
| TPOT p50 | 132.4 ms |

Service trace:

| metric | value |
|---|---:|
| trace samples | 329 |
| peak active | 4 |
| peak running_batch | 4 |
| peak prefill_queue | 2 |
| peak kv_util | 100.0% |
| `plan_label.decode` | 262 |
| `plan_label.prefill` | 106 |
| `plan_label.mixed` | 2 |
| final active | 0 |
| final tokens_out | 1032 |

Request accounting from `benchmarks.json`:

| metric | value |
|---|---:|
| created requests | 40 |
| successful requests | 28 |
| cancelled requests | 12 |
| errored requests | 0 |

## Problems

- The invalid TTFT/ITL-zero shape reproduced from a fresh server. This is now a
  blocking measurement issue, not a one-off contaminated run.
- The patch still appears to avoid the baseline deadlock mode: the service
  drained to `active=0`, with 28 successful GuideLLM requests.
- `kv_util` still peaked at `100.0%`; this patch may improve liveness without
  yet providing a valid throughput-gate measurement.

## Learnings

- Stop the c=4 three-run throughput mean until the GuideLLM timing path is
  fixed or explained. Continuing r3 would produce another invalid sample and
  waste the GPU window.
- Next technical step should inspect `/v1/completions` streaming chunk timing
  and GuideLLM's `first_request_iteration` handling. The raw JSON contains
  non-zero first/last iteration timestamps for some requests but aggregate
  TTFT/ITL p50 collapses to zero.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`
- **Mission target:** `docs/experience/wins/2026-04-30-bench-sglang-longctx-longctx-32k-phase1-s5.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| c4 valid output tok/s | 2.99 mean | invalid | n/a |
| c4 deadlock/zero mode | 1/3 baseline runs | 0/2 observed, both invalid | qualitative improvement |
| valid entrance-gate sample | 2/3 baseline runs | 0/2 | regression vs measurement gate |

## Artefacts

- Raw: `bench-output/2026-05-01-phase15-evictable-c4-r2/benchmarks.json`
- CSV: `bench-output/2026-05-01-phase15-evictable-c4-r2/benchmarks.csv`
- HTML: `bench-output/2026-05-01-phase15-evictable-c4-r2/benchmarks.html`
- Command: `bench-output/2026-05-01-phase15-evictable-c4-r2/command.txt`
- Log: `bench-output/2026-05-01-phase15-evictable-c4-r2/guidellm.log`
- Service trace summary: `bench-output/2026-05-01-phase15-evictable-c4-r2/service_stats_trace_summary.md`

## Notes

- Code delta since baseline: `051b1081` adds cascade-aware radix evictable block
  enumeration, counts only GPU-resident `block_to_pages` capacity in
  `effective_pool_free_pages`, uses that in admission, and aligns mixed decode
  retraction with the same effective-free budget.
- Follow-up: debug timing validity before running r3 or any mixed-mode/agent-loop
  publication runs.
