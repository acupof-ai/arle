# guidellm sweep qwen3-4b l4 c16 prefill-overlap-prefix-aware-7f8d9c8 — guidellm sweep, qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8, 2026-04-22

## Goal

- **Type:** regression
- Measure the net `c16` effect of two scheduler changes on the canonical L4
  `Qwen3-4B` workload: batched prefill completion and prefix-aware deferred
  waiting ordering.

## Hypothesis

- Batched first-token completion after prefill should shave a small amount of
  per-wave scheduler tail cost.
- Prefix-aware deferral ordering should be throughput-neutral on the canonical
  `guidellm` workload because this workload has effectively no prefix reuse.

## Command

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8041 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8 \
  --target http://127.0.0.1:8041 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B` bf16
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, runtime CUDA `13.0`
- **Code state:** `7f8d9c8` plus local scheduler edits in `infer/src/scheduler/cuda/prefill.rs` and `infer/src/scheduler/cuda/runtime.rs`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Weights path:** `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

## Results

### Bench headline

| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok | incomplete output tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c16` | `6956.2` | `49250.7` | `49253.7` | `62.74` | `71.26` | `76.25` | `106.72` | `0.364` | `6630` | `40960` | `510` |

### Service trace headline

| metric | value |
|---|---:|
| samples | `420` |
| peak active | `10` |
| peak waiting | `8` |
| peak running_batch | `10` |
| peak prefill_queue | `9` |
| peak kv_util | `98.2%` |
| after prefix_hit_rate | `0.0%` |

## Problems

- The canonical `guidellm` `4096-in / 256-out` run still shows `prefix_hit_rate=0.0%`, so the new deferred waiting ordering is not exercised by this workload in any meaningful way.
- Throughput and latency are effectively flat against the latest `c16` trace baseline, so batched prefill completion alone does not move the dominant refill-wave cost enough to show up in end-to-end `c16`.

## Learnings

- On this workload, **prefix-aware waiting is latent infrastructure**, not an immediately visible throughput lever; a prefix-reuse-heavy workload is required to measure it.
- Batched prefill completion keeps the prefill tail cleaner, but **it is not the missing SGLang-scale gain**. The remaining `c16` gap still sits in the long-prompt prefill/refill wave itself.
- The changes are safe as a scheduler cleanup step because the canonical `c16` regression check stays neutral while targeted unit coverage expands.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-unified-budget-bottleneck.md`

| metric | baseline | current | Δ |
|---|---:|---:|---:|
| `c16` out tok/s | `106.81` | `106.72` | `-0.1%` |
| `c16` TTFT p50 (ms) | `6939.7` | `6956.2` | `+0.2%` |
| `c16` ITL p50 (ms) | `62.73` | `62.74` | `+0.0%` |

## Artefacts

- Raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/benchmarks.json`
- CSV: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/benchmarks.csv`
- HTML: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/benchmarks.html`
- GuideLLM log: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/guidellm.log`
- Service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/service_stats_trace_summary.md`
- Service trace samples: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8/service_stats_trace.jsonl`
- Server trace shards: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8-server/traces`
