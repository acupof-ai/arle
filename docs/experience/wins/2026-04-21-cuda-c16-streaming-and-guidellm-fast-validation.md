# CUDA c16 streaming + GuideLLM fast validation, 2026-04-21

## Context

- Goal: validate that the current unified mixed path no longer shows the prior invalid benchmark signature (`TTFT/ITL = 0`, impossible output throughput), and that `c16` long-prompt streaming no longer trips `CUDA_ERROR_ILLEGAL_ADDRESS` / early finish at `16 slots / 4608 seq / 4096-in`.
- Scope for this iteration included one scheduler test fix plus runtime validation:
  - remove stale unit-test reference in `infer/src/scheduler/cuda/runtime.rs` so scheduler-focused test runs are green again;
  - rerun direct streaming probes and GuideLLM fast profile on `/v1/completions`.

## What Worked

- Scheduler-focused library tests run successfully after deleting the stale runtime test import/path:
  - `CUDA_HOME=/usr/local/cuda ZIG=/tmp/zig-tool/zig-x86_64-linux-0.15.2/zig cargo test -p infer --release --lib scheduler:: -- --nocapture`
  - Result: `83 passed; 0 failed`.
- CUDA/no-cuda type-check gate remains green:
  - `CUDA_HOME=/usr/local/cuda ZIG=/tmp/zig-tool/zig-x86_64-linux-0.15.2/zig cargo check -p infer --no-default-features --features cuda,no-cuda`
- Runtime server config used for validation:
  - `target/release/infer --model-path Qwen/Qwen3-4B --port 8017 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`
  - Hardware: NVIDIA L4 23GB, driver 580.82.07, CUDA 13.0.
- Direct streaming probes (`/v1/completions`, `stream=true`, `ignore_eos=true`) succeeded:
  - `c4`, `4096-in/64-out`: all requests completed with 64 tokens.
  - `c16`, `4096-in/64-out`: all 16 completed with 64 tokens.
  - `c16`, `4096-in/256-out`: all 16 completed with 256 tokens.
- GuideLLM fast run on completions path produced non-zero latency metrics (no short-circuit signature):
  - Command:
    - `./scripts/bench_guidellm.sh cuda-l4-c16-recheck --fast --target http://127.0.0.1:8017 --model Qwen/Qwen3-4B --processor Qwen/Qwen3-4B`
  - Headline:
    - `TTFT p50=3204.8ms`, `TTFT p99=24234.9ms`
    - `ITL p50=74.58ms`, `ITL p99=82.6ms`
    - `out tok/s=55.41`, `req/s actual=0.233`

## Rule

- Keep `/v1/completions` streaming as the canonical latency source for GuideLLM validation in this repo; if a run reports implausible throughput with `TTFT/ITL` at zero, treat it as a pipeline/signature bug and verify with direct streaming probes before trusting benchmark aggregates.

## Artefacts

- Fast benchmark raw output:
  - `bench-output/2026-04-21-cuda-l4-c16-recheck/benchmarks.json`
  - `bench-output/2026-04-21-cuda-l4-c16-recheck/benchmarks.csv`
  - `bench-output/2026-04-21-cuda-l4-c16-recheck/benchmarks.html`
