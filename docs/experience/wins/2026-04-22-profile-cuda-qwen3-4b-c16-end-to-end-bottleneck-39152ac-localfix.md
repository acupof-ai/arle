# Qwen3-4B CUDA c16 end-to-end bottleneck trace on `39152ac + localfix`

## Goal

- Diagnosis: use end-to-end request/service trace on the current local `39152ac + local compile fix` tree to explain why `c16` still trails `sglang`.

## Hypothesis

- The remaining `c16` gap is not single-step decode latency.
- The bottleneck should be active-set underfill at steady state: once the first long-prompt wave completes, the scheduler should fail to refill back to `10/16` because GPU KV pages remain pinned in the prefix cache.

## Bench anchor

- Matched bench baseline on the same local tree and workload:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c4-c8-c16-39152ac-localfix.md`

## Capture params

- Trace type: `guidellm` concurrent `c16` bench + `/v1/stats` service trace at `200ms` + server-side Chrome trace via `--trace-output-path`.
- Capture window: full `60s` bench window plus natural stream drain.
- Not a GPU profiler run: no `nsys` / `ncu`, so kernel table / roofline are intentionally out of scope for this trace.

## Command

```bash
target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8026 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-trace-39152ac-localfix-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-trace-39152ac-localfix \
  --target http://127.0.0.1:8026 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, `nvidia-smi` runtime CUDA `13.0`
- **Code state:** `39152ac` plus local compile fix in `infer/src/scheduler/cuda/core.rs`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Weights path:** `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

## Results

### Bench headline

| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c16` | `7079.0` | `41787.6` | `46.1` | `68.1` | `96.16` | `0.364` | `6375` | `45056` |

### Service trace headline

| metric | value |
|---|---:|
| samples | `478` |
| peak active | `10` |
| peak waiting | `11` |
| peak running_batch | `10` |
| peak prefill_queue | `9` |
| peak kv_util | `99.1%` |

### Loaded-window time split

- Loaded window length: `78s`
- `decode_rows=10`: `15s`
- `decode_rows=5`: `35s`
- `active=10 && waiting>=6`: `19s`
- `active=5 && waiting>=11`: `38s`
- `active=0 && waiting>=11`: `7s`
- `prefill_rows>0`: `8s`

### KV page arithmetic

- Token KV pool: `44304` tokens = `2769` pages @ `page_size=16`
- One long request at this workload needs about `4097 + 256 = 4353` tokens = `273` pages
- Raw pool ceiling: `2769 / 273 = 10.14` request-equivalents
- After each decode wave, prefix-cache retention falls only to `1384` pages = `5.07` request-equivalents
- Remaining free list: `1385` pages = `5.07` request-equivalents

That matches the observed steady-state collapse from `10 active` to `5 active`.

### Prefill refill cost

- First refill wave after the initial `10` completions:
  - cleanup / demotion step: `491ms`
  - `4x4096` prefill chunk batch: `2745ms`
  - trailing `1x4096` prefill chunk batch: `798ms`
- Repeated steady-state 5-request cycles:
  - `req11-15`: `15.280s`
  - `req16-20`: `15.335s`
  - `req21-25`: `15.438s`

So each steady-state wave spends roughly `3.5s / 15.3s ≈ 23%` of wall time refilling the next `5` requests before decode can continue.

### Decode is not the slow part

Derived from `/v1/stats` samples:

| decode rows | mean rows/s | p50 rows/s |
|---|---:|---:|
| `10` | `166.42` | `167.22` |
| `5` | `109.00` | `108.93` |
| `1` | `28.29` | `28.25` |

The per-step decode path stays stable; throughput drops because the active set collapses from `10` rows to `5`, not because a `5`-row step becomes intrinsically broken.

## Problems

- This run no longer writes `request_events.jsonl`; only Chrome trace JSON landed under `--trace-output-path`. The bottleneck analysis therefore relies on `/v1/stats` plus scheduler logs rather than the cheaper request-spine artefact.
- The tree is not a clean upstream commit because `39152ac` required a local compile fix in `infer/src/scheduler/cuda/core.rs`.

## Findings

1. **The hard capacity ceiling is still `~10` live requests.** The L4 paged KV pool is only `2769` pages, and this workload consumes `273` pages per request, so the raw live-set ceiling is `10.14` requests.
2. **The steady-state bottleneck is prefix-cache page retention.** After the first `10` requests finish, prefix-cache demotion leaves `1384` pages retained, which is almost exactly half the pool. That leaves room for only `~5` new requests, and the service trace immediately settles at `active=5`, `waiting=11`.
3. **The throughput loss is active-set underfill, not decode kernel slowdown.** During the loaded window, the system spends `35s` in `decode_rows=5` versus only `15s` in `decode_rows=10`. ITL remains around `46ms`, but total output throughput falls because decode runs with half the rows.
4. **Refill stalls are material but secondary.** Each steady-state 5-request wave burns about `3.5s` in cleanup + chunked prefill before decode resumes, about `23%` of each `15.3s` cycle.

## Learnings

- On long-prompt `c16`, first compute the page budget in request-equivalents before tuning scheduler heuristics; the live-set ceiling is visible directly from `TokenKVPool` startup logs.
- When `peak active` is `10` but steady-state settles at `5`, inspect prefix-cache retention first; retained GPU pages can halve effective concurrency even after OOM is gone.
- For this workload, improving `c16` now means reducing GPU prefix-cache retention / demotion pressure, not chasing single-token decode latency.

## Δ vs baseline

- Versus the matched local bench anchor (`39152ac + localfix`), this trace run is directionally the same on throughput (`96.16 tok/s` vs `95.02 tok/s`) and adds the explanation for the remaining gap.
- Versus `sglang` `c16` baseline (`137.07 tok/s`), the remaining deficit is `-29.8%`.

## Artefacts

- Bench raw output: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-39152ac-localfix/benchmarks.json`
- Bench log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-39152ac-localfix/guidellm.log`
- Service trace: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-39152ac-localfix/service_stats_trace.jsonl`
- Service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-39152ac-localfix/service_stats_trace_summary.md`
- Server log: `bench-output/infer-qwen3-4b-l4-c16-trace-39152ac-localfix-server/infer.log`
- Chrome trace directory: `bench-output/infer-qwen3-4b-l4-c16-trace-39152ac-localfix-server/traces`
