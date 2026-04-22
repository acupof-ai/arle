# Qwen3-4B L4 c16 regression after tier readmission fixes

## Goal

- Confirm that the T1 byte-drain and slower-tier recall heuristic fixes do not regress the canonical `Qwen3-4B` L4 `c16` throughput lane.

## Hypothesis

- Canonical random long prompts should remain effectively tier-cold (`prefix_hit_rate ~= 0`, no staged fetches), so the throughput delta should stay within noise even after enabling real T2/T3 recall.

## Command

```bash
scripts/bench_guidellm.sh qwen3-4b-l4-c16-tier-recall-b70c03b \
  --target http://127.0.0.1:8058 \
  --model 1cfa9a7208912126459214e8b04321603b3df60c \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen3-4B`
- **Hardware:** NVIDIA L4
- **Commit base:** `b70c03b` plus local tier-readmission fixes in this tranche
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Server flags:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384`

## Results

| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|
| `16` | `7779.0` | `30395.2` | `59.9` | `119.14` | `0.364` |

Service-side trace summary:

| metric | value |
|---|---:|
| peak active | `10` |
| peak waiting | `7` |
| peak prefill_queue | `6` |
| peak kv_util | `98.2%` |
| `kv_store` | `sub:52,done:52,fail:0,rej:0` |
| `prefix_hit_rate` | `0.0%` |
| `prefix_skip_rate` | `0.0%` |
| `kv_fetch_q` | `0/16` |

## Problems

- The benchmark is still tier-cold: there are no staged fetches and no prompt-token skip. The slower-tier fixes therefore do not get exercised by this workload.
- The run is exploration mode (`--concurrencies 16`), so this entry is written manually rather than auto-seeded by the wrapper.

## Learnings

- These tier fixes are effectively neutral on the canonical `c16` lane because the workload does not hit reusable prefixes.
- The remaining `Qwen3 c16` throughput gap versus `sglang` is still elsewhere; the tiered-KV work closed correctness gaps, not the mainline decode/prefill bottleneck.

## Δ vs baseline

Baseline: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-080ee05-observability-trace-run3/benchmarks.json`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| out tok/s | `120.87` | `119.14` | `-1.4%` |
| TTFT p50 (ms) | `7232.1` | `7779.0` | `+7.6%` |
| ITL p50 (ms) | `59.73` | `59.90` | `+0.3%` |

## Artefacts

- Raw: `bench-output/2026-04-22-qwen3-4b-l4-c16-tier-recall-b70c03b-run2/benchmarks.json`
- CSV: `bench-output/2026-04-22-qwen3-4b-l4-c16-tier-recall-b70c03b-run2/benchmarks.csv`
- HTML: `bench-output/2026-04-22-qwen3-4b-l4-c16-tier-recall-b70c03b-run2/benchmarks.html`
- Service trace: `bench-output/2026-04-22-qwen3-4b-l4-c16-tier-recall-b70c03b-run2/service_stats_trace_summary.md`
- Server log: local PTY session during the bench run
