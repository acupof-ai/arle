# cuda-l4 canonical sweep, FP8 KV — full perf-fix stack landed

> Headline canonical sweep after today's full fix chain
> (`83e67ff2`, `47bad713`, `c4109b29`, `8f6965c3`, `4089fde2`,
> `d53c6d8d`). All pre-2026-04-29 wins entries used either `--fast`
> 30s or unaligned defaults — this is the first canonical sweep at
> the SGLang-aligned envelope (`max_prefill_tokens=16384`,
> `mem_fraction_static=0.85`).

## Goal

Establish the canonical FP8-KV baseline at SGLang-aligned defaults so
future regressions can diff against a real number, not a single-run
`--fast` outlier. Goal type: **publication baseline**.

## Hypothesis

- TTFT p50 at `sync` rate ≈ 700 ms (single-request prefill).
- ITL p50 at `sync` ≈ 35 ms (decode floor).
- Throughput-mode out tok/s ≈ 75-80 (admission saturated).
- Sweep peak (just below saturation) ≈ 55-60 tok/s, TTFT 900-1000 ms.

## Command

```bash
target/release/infer \
    --model-path models/Qwen3-4B --port 8000 \
    --num-slots 16 --max-seq-len 4608
    # All other params at SGLang-aligned defaults
    # (max_prefill_tokens=16384, mem_fraction_static=0.85, kv-cache-dtype=fp8)

bash scripts/bench_guidellm.sh canonsweep-v3-fp8
```

## Environment

- **Backend:** cuda
- **Model:** Qwen3-4B (bf16 weights, FP8E4M3 paged KV)
- **Hardware:** NVIDIA L4 sm_89, 22 GB, CUDA 12.8
- **Commit:** d53c6d8d
- **Feature set:** `cargo build --release --features cuda` (implies `tilelang-attn`)
- **Non-default flags / env vars:** none — full-default canonical run
- **Server launch:** `target/release/infer --model-path models/Qwen3-4B --num-slots 16`
- **Pool size:** 148,256 tokens / 11.5 GB at fraction=0.85
- **Scheduling envelope at boot:**
  ```
  max_num_batched_tokens=16384 | 16384 (SGLang)
  chunked_prefill_size=2048 | 2048
  max_prefill_tokens=16384 | 16384
  mem_fraction_static=0.85 | 0.85
  max_slots=16 | (n/a — SGLang has no fixed cap)
  ```

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 725.2 | 737.8 | 36.3 | 36.38 | 25.85 | 0.1 |
| throughput | 13417 | 23301.6 | 74.77 | 108.84 | 76.04 | 0.267 |
| 0.12083333333333332r/s | 903.9 | 925.2 | 40.45 | 40.52 | 29.68 | 0.1 |
| 0.14166666666666666r/s | 914.3 | 926 | 41.48 | 44.08 | 33.91 | 0.117 |
| 0.16249999999999998r/s | 917.4 | 941.2 | 42.22 | 44.99 | 38.07 | 0.133 |
| 0.18333333333333335r/s | 937.4 | 958.9 | 46.16 | 48.68 | 41.32 | 0.15 |
| 0.20416666666666666r/s | 926.5 | 947.4 | 46.73 | 49.31 | 45.94 | 0.167 |
| 0.22499999999999998r/s | 916.7 | 954.5 | 47.19 | 52.25 | 49.49 | 0.183 |
| 0.24583333333333335r/s | 966.3 | 977.4 | 51.72 | 54.32 | 52.64 | 0.2 |
| 0.26666666666666666r/s | 933.6 | 972.3 | 52.47 | 54.9 | 56 | 0.217 |


## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | ... |
| peak waiting | ... |
| peak prefill_queue | ... |
| peak kv_util | ... |
| `prefix_hit_rate` | ... |
| `prefix_skip_rate` | ... |
| `kv_fetch_q` | ... |
| `kv_fetch_waiters` | ... |
| `kv_store_q` | ... |
| `kv_store` | ... |
| `kv_bp` | ... |
| `tier_recall` | ... / n/a |
| `tier_src` | ... / n/a |
| `tier_promoted` | ... / n/a |
| `tier_fallback` | ... / n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | ... |
| incomplete input tokens | ... |
| completed output tokens | ... |
| incomplete output tokens | ... |

## Problems

- <anything that degraded, crashed, or deviated from the watch-list>

## Learnings

- <generalizable rule or tuning takeaway>

## Δ vs baseline

- **Baseline:** <link to prior `2026-04-29-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/benchmarks.html`
- Service trace (before): `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_before.txt`
- Service trace (during): `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_trace.jsonl`
- Service trace (after):  `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_after.txt`
- Service trace (summary): `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_trace_summary.md`

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">

## Service Trace

- Poll interval: `1000ms`
- Before: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_before.txt`
- During: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_trace.jsonl`
- After: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_after.txt`
- Summary: `/content/workspace/agent-infer/bench-output/2026-04-29-canonsweep-v3-fp8/service_stats_trace_summary.md`
