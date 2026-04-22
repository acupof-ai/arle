# Qwen3-4B CUDA c16 unified-budget bottleneck trace on `f35a861`

## Goal

- Diagnosis: confirm the remaining `c16` bottleneck after CUDA scheduler budget unification on the latest local `main`.

## Hypothesis

- Unifying budget arithmetic should not materially change the steady-state ceiling by itself.
- The remaining loss should still be refill-wave cost under near-full KV utilization, not admission drift or tier-queue backpressure.

## Bench anchor

- Matched bench anchor on the same commit and workload:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`

## Capture params

- Trace type: `guidellm` concurrent `c16` bench + `/v1/stats` service trace at `200ms` + server-side Chrome trace via `--trace-output-path`.
- Capture window: full `60s` bench window plus natural drain.
- Not a GPU profiler run: no `nsys` / `ncu`, so kernel table / roofline stay out of scope.

## Command

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

 target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8033 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget \
  --target http://127.0.0.1:8033 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, runtime CUDA `13.0`
- **Code state:** `f35a861`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Weights path:** `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

## Results

### Bench headline

| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p95 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok | incomplete output tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c16` | `6939.7` | `49204.1` | `49205.9` | `62.73` | `71.25` | `76.18` | `106.81` | `0.364` | `6630` | `40960` | `510` |

### Service trace headline

| metric | value |
|---|---:|
| samples | `419` |
| peak active | `10` |
| peak waiting | `8` |
| peak running_batch | `10` |
| peak prefill_queue | `9` |
| peak kv_util | `98.2%` |
| weighted active | `8.47` |

### Loaded-window split

- Loaded window length: `64s`
- `active=10`: `35s`
- `active=8`: `14s`
- `prefill_rows>0`: `16s`
- `decode_only`: `48s`
- `waiting>0 && running_batch=0`: `0s`

### Step-cost split

| phase | wall time | step p50 (ms) | step p95 (ms) |
|---|---:|---:|---:|
| `decode_only` | `48s` | `59.5` | `60.5` |
| `prefill_active` | `16s` | `2071.4` | `2167.4` |
| `decode+prefill` | `8s` | `2071.4` | `2117.3` |

### Queue / tier pressure

| signal | p50 | p95 | max |
|---|---:|---:|---:|
| `prefill_queue` | `0` | `4` | `9` |
| `running_batch` | `4` | `10` | `10` |
| `kv_util %` | `93.4` | `97.7` | `98.2` |
| `kv_store_q` | `0` | `0` | `0` |
| `tier_store_wait ms` | `0` | `0` | `0` |

## Findings

- The unified budget refactor does **not** reintroduce the old admission-collapse pattern. The loaded window now spends `35s` at `active=10` and never enters a `waiting>0 && running_batch=0` hole.
- The remaining tax is **prefill refill cost**, not decode cost. Decode-only steps sit around `60ms`, while any step with prefill work jumps to about `2.1s`.
- Roughly `16s / 64s = 25%` of loaded wall time is spent with `prefill_rows>0`, which directly explains why throughput stalls near `106 tok/s` even though steady-state active occupancy is mostly `8–10` rows.
- This trace does **not** show tier-queue saturation: `kv_store_q` stays at `0` and `tier_store_wait` stays at `0ms`. The next lever is therefore reducing long-prompt refill-wave cost under `~98%` KV utilization, not more coordinator queue math.

## Problems

- This run started the server directly from the shell, so server stdout was not persisted to a repo-local `infer.log`; the diagnosis here therefore relies on `service_stats_trace.jsonl` and Chrome trace artefacts.
- `--trace-output-path` emitted Chrome trace shards only; there is still no cheaper request-spine JSONL artefact in this run directory.

## Learnings

- After budget unification, more budget-policy churn is unlikely to buy much at `c16`; the scheduler already holds an `8–10` row active set without idle admission gaps.
- The next optimization target is the cost of each `4096`-token refill wave. If that `~2.1s` step cost drops, throughput should move; if it does not, more page-budget cleanup work will mostly reshuffle the same ceiling.

## Δ vs baseline

- **Prior diagnosis baseline:** `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`
- **Same-commit bench anchor:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`

| metric | prior diagnosis | now | Δ% |
|---|---:|---:|---:|
| `c16` out tok/s | `96.16` | `106.81` | `+11.1%` |
| `c16` TTFT p50 (ms) | `7079.0` | `6939.7` | `-2.0%` |
| `c16` incomplete input tok | `45056` | `40960` | `-9.1%` |
| loaded-window `active=0 && waiting>=11` | `7s` | `0s` | `-100%` |

## Artefacts

- Raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget/benchmarks.json`
- Service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget/service_stats_trace_summary.md`
- Service trace samples: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget/service_stats_trace.jsonl`
- Chrome trace shards: `bench-output/infer-qwen3-4b-l4-c16-trace-f35a861-unified-budget-server/traces`
