# Qwen3-4B CUDA c4/c8/c16 rerun on `e9eac60`

## Goal

- Regression-check the latest pulled CUDA scheduler on the long-prompt high-concurrency workload after the admission-headroom alignment and postlaunch emit overlap series.

## Hypothesis

- `c4` should improve materially because `assign_slots()` no longer over-admits requests whose future KV headroom is already claimed by active slots.
- `c8/c16` should at least hold the latest valid `ede0daa` numbers instead of regressing back toward the older invalid/underfilled runs.

## Command

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo build --release -p infer --bin infer

MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# c4
target/release/infer \
  --model-path "$MODEL" \
  --port 8024 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c4-e9eac60 \
  --target http://127.0.0.1:8024 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL" \
  --concurrencies 4 \
  --max-seconds 60 \
  --warmup 5

# c8
target/release/infer \
  --model-path "$MODEL" \
  --port 8025 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c8-e9eac60 \
  --target http://127.0.0.1:8025 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL" \
  --concurrencies 8 \
  --max-seconds 60 \
  --warmup 5

# c16
target/release/infer \
  --model-path "$MODEL" \
  --port 8026 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c16-e9eac60 \
  --target http://127.0.0.1:8026 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5
```

## Environment

- **Backend:** CUDA
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, `nvidia-smi` runtime CUDA `13.0`
- **Commit:** `e9eac60`
- **Feature set:** `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo build --release -p infer --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`, `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`
- **Server launch:** direct `target/release/infer` launch, one server per concurrency leg

## Results

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c4` | `2837.9` | `14601.4` | `43.40` | `43.57` | `71.13` | `0.255` | `4608` | `0` |
| `c8` | `5828.2` | `29706.9` | `46.09` | `66.31` | `85.46` | `0.273` | `5120` | `12288` |
| `c16` | `7580.6` | `42769.9` | `46.11` | `77.63` | `95.36` | `0.364` | `6400` | `45056` |

## Problems

- `c8/c16` still end the benchmark window with unfinished requests (`incomplete input tok` stays non-zero), so high-concurrency tail latency remains the main open issue.
- This run did not include the earlier local request-level trace patch stack; it is throughput/latency validation only.

## Learnings

- Admission headroom alignment is enough to recover the old `c4` collapse: once later admissions stop borrowing pages already promised to active long prompts, `c4` TTFT and throughput move back into the same range as `sglang`.
- `c8/c16` are no longer admission-catastrophic, but they are still bounded by tail backlog rather than per-token decode speed.

## Δ vs baseline

- **Primary baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c8-c16-trace-ede0daa.md`
- **Older paired table for `c4` and `sglang` reference:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| `c4` out tok/s vs older infer baseline | `36.70` | `71.13` | `+93.8%` |
| `c4` out tok/s vs `sglang` | `74.05` | `71.13` | `-3.9%` |
| `c8` out tok/s vs `ede0daa` | `85.73` | `85.46` | `-0.3%` |
| `c8` out tok/s vs `sglang` | `107.79` | `85.46` | `-20.7%` |
| `c16` out tok/s vs `ede0daa` | `94.99` | `95.36` | `+0.4%` |
| `c16` out tok/s vs `sglang` | `137.07` | `95.36` | `-30.4%` |

## Artefacts

- `c4` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-e9eac60/benchmarks.json`
- `c4` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-e9eac60/service_stats_trace_summary.md`
- `c4` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c4-e9eac60-server/infer.log`
- `c8` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-e9eac60/benchmarks.json`
- `c8` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-e9eac60/service_stats_trace_summary.md`
- `c8` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-e9eac60-server/infer.log`
- `c16` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-e9eac60/benchmarks.json`
- `c16` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-e9eac60/service_stats_trace_summary.md`
- `c16` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-e9eac60-server/infer.log`

## Notes

- This rerun validates the latest pulled scheduler series on the exact long-prompt workload that previously exposed the admission-vs-execution mismatch.
- The new code closes most of the `c4` gap immediately, but does not yet move the `c8/c16` ceiling meaningfully beyond the prior valid `ede0daa` snapshot.
