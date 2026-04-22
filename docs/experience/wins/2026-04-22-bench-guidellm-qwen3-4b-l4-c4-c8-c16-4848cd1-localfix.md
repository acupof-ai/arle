# Qwen3-4B CUDA c4/c8/c16 rerun on `4848cd1` + local metrics compile fix

## Goal

- Regression-check the latest pulled CUDA scheduler after the async emit worker series.

## Hypothesis

- `c8/c16` should improve modestly if moving stopless emit work off the scheduler thread removes a meaningful CPU-side bubble.

## Command

```bash
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo build --release -p infer --bin infer

MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# c4
target/release/infer \
  --model-path "$MODEL" \
  --port 8034 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c4-4848cd1-localfix \
  --target http://127.0.0.1:8034 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL" \
  --concurrencies 4 \
  --max-seconds 60 \
  --warmup 5

# c8
target/release/infer \
  --model-path "$MODEL" \
  --port 8035 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c8-4848cd1-localfix \
  --target http://127.0.0.1:8035 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL" \
  --concurrencies 8 \
  --max-seconds 60 \
  --warmup 5

# c16
target/release/infer \
  --model-path "$MODEL" \
  --port 8036 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  2026-04-22-infer-qwen3-4b-l4-c16-4848cd1-localfix \
  --target http://127.0.0.1:8036 \
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
- **Base commit:** `4848cd1`
- **Local runtime delta:** `infer/src/metrics.rs` adds the missing `ServerMetrics` methods required by the pulled scheduler code to compile
- **Feature set:** `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig cargo build --release -p infer --bin infer`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`, `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`
- **Server launch:** direct `target/release/infer` launch, one server per concurrency leg

## Results

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c4` | `2838.4` | `14625.5` | `43.71` | `43.85` | `70.77` | `0.255` | `4608` | `0` |
| `c8` | `5923.9` | `31220.0` | `46.09` | `63.57` | `86.51` | `0.273` | `5120` | `12288` |
| `c16` | `7562.9` | `42756.2` | `46.11` | `77.57` | `95.39` | `0.364` | `6400` | `45056` |

## Problems

- The pulled scheduler series did not compile as-is on this host because `infer/src/metrics.rs` lacked the new `ServerMetrics` methods referenced from scheduler code.
- `c8/c16` still finish the benchmark window with non-zero incomplete input tokens, so tail backlog remains the dominant open issue.

## Learnings

- The async emit worker series does not move the throughput ceiling materially on this workload by itself; the new numbers are effectively flat versus the immediately prior valid snapshot.
- The latest pull included an upstream compile drift between scheduler and metrics surfaces, so fresh runtime pulls still need a build gate before any benchmark interpretation.

## Δ vs baseline

- **Primary baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c4-c8-c16-e9eac60.md`
- **Older `sglang` reference:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| `c4` out tok/s vs `e9eac60` | `71.13` | `70.77` | `-0.5%` |
| `c4` out tok/s vs `sglang` | `74.05` | `70.77` | `-4.4%` |
| `c8` out tok/s vs `e9eac60` | `85.46` | `86.51` | `+1.2%` |
| `c8` out tok/s vs `sglang` | `107.79` | `86.51` | `-19.7%` |
| `c16` out tok/s vs `e9eac60` | `95.36` | `95.39` | `+0.0%` |
| `c16` out tok/s vs `sglang` | `137.07` | `95.39` | `-30.4%` |

## Artefacts

- `c4` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-4848cd1-localfix/benchmarks.json`
- `c4` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-4848cd1-localfix/service_stats_trace_summary.md`
- `c4` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c4-4848cd1-localfix-server/infer.log`
- `c8` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-4848cd1-localfix/benchmarks.json`
- `c8` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-4848cd1-localfix/service_stats_trace_summary.md`
- `c8` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-4848cd1-localfix-server/infer.log`
- `c16` raw bench: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-4848cd1-localfix/benchmarks.json`
- `c16` service trace summary: `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-4848cd1-localfix/service_stats_trace_summary.md`
- `c16` server log: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-4848cd1-localfix-server/infer.log`

## Notes

- This run should be read as “latest scheduler pull plus local metrics compile fix”, not as a pure upstream `4848cd1` snapshot.
- On this workload, the async emit worker series is measurable only at `c8` and only modestly; `c16` is effectively unchanged.
