# guidellm sweep qwen3-4b l4 c16 async-prefill-overlap-18c116d — guidellm sweep, qwen3-4b-l4-c16-18c116d-async-prefill-overlap, 2026-04-22

## Goal

- **Type:** optimization
- Validate the delete-style async prefill rewrite on the canonical L4 `Qwen3-4B`
  `c16` long-prompt workload and measure how much of the remaining `sglang`
  gap it closes.

## Hypothesis

- Splitting batched prefill into launch + next-turn completion should let the
  scheduler overlap CPU admission work with the heaviest refill wave, raise the
  time spent at the physical `active=10` ceiling, and materially increase `c16`
  output throughput.
- The rewrite should not change the physical ceiling itself (`peak active`
  remains `10` on this L4 / KV budget shape), so any gain should come from
  better refill pacing and lower backlog, not a larger live set.

## Command

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# run 1
 target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8041 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap \
  --target http://127.0.0.1:8041 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200

# run 2
 target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8042 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun \
  --target http://127.0.0.1:8042 \
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
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`
- **Commit:** `18c116d`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Weights path:** `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`, `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`, `--trace-output-path ...`

## Results

### Bench headline

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed req | incomplete req | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run1` | `7231.1` | `29146.0` | `59.73` | `59.92` | `123.76` | `0.364` | `28` | `8` | `7140` | `32768` |
| `run2` | `7390.9` | `29485.7` | `59.78` | `59.96` | `123.00` | `0.364` | `28` | `8` | `7140` | `32768` |
| **mean** | **`7311.0`** | **`29315.9`** | **`59.76`** | **`59.94`** | **`123.38`** | **`0.364`** | **`28`** | **`8`** | **`7140`** | **`32768`** |

### Variance

| metric | mean | stdev | variance % |
|---|---:|---:|---:|
| `c16` out tok/s | `123.38` | `0.38` | `0.31%` |
| `c16` TTFT p50 (ms) | `7311.0` | `79.9` | `1.09%` |
| `c16` ITL p50 (ms) | `59.76` | `0.03` | `0.04%` |

### Service trace headline

| run | peak active | peak waiting | peak running_batch | peak prefill_queue | peak kv_util |
|---|---:|---:|---:|---:|---:|
| `run1` | `10` | `8` | `10` | `6` | `98.2%` |
| `run2` | `10` | `8` | `10` | `6` | `98.2%` |

### Trace-derived occupancy (`run1`)

| metric | baseline `7f8d9c8` | `18c116d` | Δ |
|---|---:|---:|---:|
| loaded window (s) | `64` | `64` | `+0` |
| `active=10` time (s) | `35` | `43` | `+8` |
| `prefill_rows>0` time (s) | `16` | `19` | `+3` |
| `prefill_rows=4 && active=10` time (s) | `2` | `11` | `+9` |
| `waiting>0 && decode_rows=0 && prefill_rows=0` (s) | `0` | `0` | `+0` |

## Problems

- The benchmark still ends with `8` incomplete requests (`32768` prompt tokens), so the rewrite improves refill pacing but does not fully eliminate long-context tail backlog.
- The request-spine JSONL artefact still did not land under `--trace-output-path`; this run relies on `/v1/stats`, server logs, and Chrome trace shards rather than `request_events.jsonl`.
- The remaining gap to the earlier `sglang` reference is still about `10%` on output throughput.

## Learnings

- The async prefill rewrite is a **real throughput lever**. The gain does not come from a higher physical ceiling — `peak active` stays `10` — but from keeping the scheduler at that ceiling longer and pushing larger refill waves while already full.
- The rewrite removes the old synchronous prefill bubble cleanly enough that `c16` output throughput rises by about `15.6%` with low run-to-run variance.
- The remaining gap is no longer admission collapse. It is the refill tail that is still visible through repeated prefix-cache demotion / fallback under backlog; this is where the next `sglang`-parity work belongs.

## Δ vs baseline

- **Primary local baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-prefill-overlap-prefix-aware-7f8d9c8.md`
- **Reference `sglang` baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | baseline | current | Δ |
|---|---:|---:|---:|
| `c16` out tok/s vs prior infer | `106.72` | `123.38` | `+15.6%` |
| `c16` TTFT p50 (ms) vs prior infer | `6956.2` | `7311.0` | `+5.1%` |
| `c16` ITL p50 (ms) vs prior infer | `62.74` | `59.76` | `-4.8%` |
| `c16` incomplete input tok vs prior infer | `40960` | `32768` | `-20.0%` |
| `c16` out tok/s vs `sglang` | `137.07` | `123.38` | `-10.0%` |

## Artefacts

- `run1` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/benchmarks.json`
- `run1` CSV: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/benchmarks.csv`
- `run1` HTML: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/benchmarks.html`
- `run1` GuideLLM log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/guidellm.log`
- `run1` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/service_stats_trace_summary.md`
- `run1` service trace samples: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap/service_stats_trace.jsonl`
- `run1` server trace shards: `bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-server/traces`
- `run1` server log: `bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-server/infer.log`
- `run2` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun/benchmarks.json`
- `run2` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun/service_stats_trace_summary.md`
- `run2` server trace shards: `bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun-server/traces`
- `run2` server log: `bench-output/infer-qwen3-4b-l4-c16-18c116d-async-prefill-overlap-rerun-server/infer.log`

## Notes

- The strongest visible trace change is not a lower peak wait queue but a longer residency at `active=10` and much more time spent doing `4`-row prefill waves while already full.
- Mid-run logs still show repeated prefix-cache demotion fallback (`core.rs:1716` / `core.rs:1728`), so host-tier / demotion pressure remains the next bottleneck to attack if we want the last `~10%` against `sglang`.
