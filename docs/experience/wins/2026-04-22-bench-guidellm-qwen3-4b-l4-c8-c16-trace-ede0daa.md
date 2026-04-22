# Qwen3-4B CUDA c8/c16 trace rerun on `ede0daa`

## Goal

- Diagnosis + regression check: confirm whether latest pulled CUDA scheduler still reproduces the old `c8/c16` zero-token / prefill-OOM failure, and verify that end-to-end request trace artefacts now land on disk.

## Hypothesis

- The latest pulled scheduler should no longer fail high-concurrency prefills with `CUDA_ERROR_OUT_OF_MEMORY`.
- `c8` and `c16` should produce valid throughput numbers again.
- Any remaining zero-token requests near the benchmark tail should be benchmark/client cutoff artefacts, not internal OOM.

## Command

```bash
# smoke: trace artefact bring-up
/tmp/agent-infer-target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8016 \
  --num-slots 4 \
  --max-seq-len 1024 \
  --trace-output-path bench-output/2026-04-22-trace-smoke-ede0daa-run6/traces

# c8
/tmp/agent-infer-target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8017 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-ede0daa/traces
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c8-trace-ede0daa \
  --target http://127.0.0.1:8017 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 8 \
  --max-seconds 60 \
  --warmup 5

# c16
/tmp/agent-infer-target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8018 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa/traces
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-trace-ede0daa \
  --target http://127.0.0.1:8018 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5

# short c16 cause check after explicit client-cutoff cause wiring
/tmp/agent-infer-target/release/infer \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --port 8019 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa-causecheck/traces
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-trace-ede0daa-causecheck \
  --target http://127.0.0.1:8019 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 16 \
  --max-seconds 30 \
  --warmup 5
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, `nvidia-smi` runtime CUDA `13.0`
- **Commit:** `ede0daa`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`, `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`, `--trace-output-path ...`
- **Server launch:** direct `infer` binary launch, one server per concurrency leg

## Results

### Trace smoke

- `bench-output/2026-04-22-trace-smoke-ede0daa-run6/traces/request_events.jsonl` now lands correctly.
- Chrome trace JSON also lands: `bench-output/2026-04-22-trace-smoke-ede0daa-run6/traces/1776841288621_db7aa37d3832c59f524222ee53f5172e.json`.

### Throughput / latency headline

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c8` | `5652.3` | `29640.9` | `46.22` | `65.83` | `85.73` | `0.273` | `5120` | `12288` |
| `c16` | `20144.9` | `42887.8` | `46.27` | `77.05` | `94.99` | `0.364` | `6400` | `45056` |
| `c16` cause-check (`30s`) | `7482.7` | `27133.1` | `60.25` | `76.93` | `107.22` | `0.400` | `3840` | `45056` |

### Request-trace summary

- `c8` request trace produced `23` finish events, `2` zero-token finishes.
- `c16` request trace produced `37` finish events, `9` zero-token finishes.
- The short final `c16` cause-check run produced `7` zero-token finishes, all tagged `client_delta_closed_while_prefill_queued`.
- No `prefill_batch_failed: CUDA_ERROR_OUT_OF_MEMORY` reproduced on the latest pulled code.

## Problems

- `c8/c16` still leave a tail of in-flight requests unfinished when the benchmark window ends; this shows up as zero-token server-side finishes in the request trace.
- Before the final cause patch, these tail finishes had no explicit terminal cause in `request_events.jsonl`.
- This is a trace attribution gap, not the earlier catastrophic OOM regression: the latest runs still produce valid completed-request throughput/latency numbers.

## Learnings

- The old high-concurrency failure mode changed class: on the latest scheduler, `c8/c16` no longer die on first-chunk prefill OOM; the remaining zero-token tail is dominated by client cutoff while the request is still queued for prefill.
- End-to-end request trace needs two outputs: Chrome trace JSON for timeline inspection and a cheap `request_events.jsonl` spine for per-request aggregation.
- The most useful request-level join keys are now stable on disk: `request_id`, `scheduler_iteration`, `slot_idx`, and `terminal_cause`.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`
- **Prior regression diagnosis:** `docs/experience/errors/2026-04-22-cuda-l4-zero-token-trace-root-cause.md`

| metric | prior infer | now | Δ% |
|---|---:|---:|---:|
| `c8` out tok/s | `57.71` | `85.73` | `+48.6%` |
| `c16` out tok/s | `45.08` | `94.99` | `+110.7%` |
| `c8` vs `sglang` out tok/s | `107.79` | `85.73` | `-20.5%` |
| `c16` vs `sglang` out tok/s | `137.07` | `94.99` | `-30.7%` |

## Artefacts

- Trace smoke request events: `bench-output/2026-04-22-trace-smoke-ede0daa-run6/traces/request_events.jsonl`
- Trace smoke Chrome JSON: `bench-output/2026-04-22-trace-smoke-ede0daa-run6/traces/1776841288621_db7aa37d3832c59f524222ee53f5172e.json`
- `c8` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-ede0daa-run2/benchmarks.json`
- `c8` request events: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-ede0daa/traces/request_events.jsonl`
- `c8` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-trace-ede0daa-run2/service_stats_trace_summary.md`
- `c16` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa-run2/benchmarks.json`
- `c16` request events: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa/traces/request_events.jsonl`
- `c16` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa-run2/service_stats_trace_summary.md`
- `c16` cause-check raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa-causecheck-run2/benchmarks.json`
- `c16` cause-check request events: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-trace-ede0daa-causecheck/traces/request_events.jsonl`

## Notes

- The key fix in this patch stack is observability: request finish / zero-token / client-cutoff causes now survive to disk, and the trace output path finally emits real artefacts again.
- The catastrophic regression documented earlier (`prefill batch failed: CUDA_ERROR_OUT_OF_MEMORY`) is no longer the limiting failure mode on the latest pulled code.
- Remaining follow-up is performance, not correctness: `c8/c16` are now valid but still behind the earlier `sglang` reference at the same workload.
