# CUDA c16 serial trace exposes active-set underfill

## Context

Local CUDA scheduler iteration on 2026-04-21 combined two changes:

- scheduler-side prompt token reuse + adaptive decode headroom reservation
- `scripts/bench_guidellm.sh` serial execution + automatic `/v1/stats` trace capture

Environment and commands:

- Host: NVIDIA L4, driver 580.82.07
- Model: `Qwen/Qwen3-4B`
- Server:
  `./target/release/infer --model-path Qwen/Qwen3-4B --port 8000 --num-slots 16 --max-seq-len 4608 --enable-mixed-chunk=true --chunked-prefill-size 4096 --max-prefill-tokens 16384`
- Regression bench:
  `./scripts/bench_guidellm.sh cuda-l4-c16-mixed-reserve --fast --model Qwen/Qwen3-4B --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- Serial trace smoke:
  `./scripts/bench_guidellm.sh cuda-l4-trace-serial-smoke --target http://127.0.0.1:8000 --model Qwen/Qwen3-4B --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --fast --trace-interval-ms 500`

Observed outputs:

- `bench-output/2026-04-21-cuda-l4-c16-mixed-reserve/`
  - TTFT p50 `4017.8 ms`
  - TTFT p99 `7354.9 ms`
  - ITL p50 `87.35 ms`
  - ITL p99 `135.63 ms`
  - out tok/s `55.97`
  - req/s `0.267`
- `bench-output/2026-04-21-cuda-l4-trace-serial-smoke/service_stats_trace_summary.md`
  - peak waiting `15`
  - peak active `1`
  - peak kv_util `93.0%`

The new serial trace path worked as intended: every bench run now produces
before/during/after `/v1/stats` artefacts in the same output directory, and
the wrapper rejects concurrent bench runs via a global lock.

## Root Cause

The bad c16 result is not a GuideLLM accounting bug. The service trace shows
that the server is failing to keep a wide active set under load:

1. client concurrency reached 16 (`peak waiting=15` proves demand exists)
2. the scheduler only sustained `active=1` at the trace level
3. KV utilization still climbed into the 90% range, so the system is not idle;
   it is filling memory without converting that pressure into parallel decode

That narrows the current bottleneck to scheduler/runtime behavior under long
prompt pressure: active admission + mixed/decode progression are underfilling
the live batch, so throughput collapses even though requests are queued.

## Fix

- Keep the tokenization cache and adaptive decode reservation changes.
- Keep the serial bench + trace wrapper changes; they turned this from a vague
  "slow c16" complaint into a reproducible scheduler diagnosis.
- Next optimization pass should target active-set width directly:
  - trace `running_batch` and `prefill_queue` occupancy per tick
  - record why decode-active ticks fail to materialize more than one live slot
  - correlate retract/requeue and host-demotion pressure with active-slot drops

## Rule

When a high-concurrency GuideLLM run looks implausibly slow, require a
same-run service trace before touching bench parameters. If the trace shows
`waiting >> active`, treat the issue as scheduler underfill first, not as a
benchmark-tool problem.
