# Scheduler Overlap Gap Instrumentation

## Context

The overlap follow-up needed data before moving CPU work into additional
threads. Existing step-phase stats exposed admission, prefill, decode, emit,
and step total, but not the cleanup phase that releases completed slots,
prefix blocks, and KV pages.

## What Worked

Added scheduler-loop EMAs for cleanup and full loop total:

- `infer_scheduler_step_cleanup_microseconds`
- `infer_scheduler_loop_total_microseconds`
- `/v1/stats` suffix: `cleanup:<us>,loop_total:<us>`

The c=16 Qwen3-4B micro run used the canonical wrapper in exploration mode:

```bash
scripts/bench_guidellm.sh arle-cuda-l4-qwen3-a2-overlap-micro \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --trace-interval-ms 250
```

Headline:

| shape | TTFT p50 | ITL p50 | out tok/s | trace samples |
|---|---:|---:|---:|---:|
| c=16, 4096/256, 30s | 12865.2 ms | 73.24 ms | 109.66 | 231 |

Phase trace at steady decode:

| phase | representative EMA |
|---|---:|
| admission | 6-7 us |
| prefill | 0 us |
| decode | 73.1-73.4 ms |
| emit | 12-13 us |
| cleanup | 8 us |
| loop_total | 73.1-73.4 ms |

The completion burst had one visible cleanup spike in the server log:
`cleanup=8445us` when 16 requests finished together. Its EMA settled around
`cleanup:569us`, still far below the decode step. KV coordinator queues stayed
empty (`kv_fetch_waiters=0`, `kv_store_q=0/16`) and waiting stayed at 0.

## Decision

No scheduler-thread split was made in this tranche. The largest observed gap
is the GPU decode path, not admission, detokenize/emit, sampling postprocess,
or KV release. Adding a new thread/channel at this point would add queueing and
lifetime complexity without moving the measured bottleneck.

## Verification

```bash
CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo test -p infer --features cuda server_metrics --lib

CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig CUDA_HOME=/usr/local/cuda \
  cargo build --release -p infer --features cuda
```

Raw artefacts:

- `bench-output/2026-04-29-arle-cuda-l4-qwen3-a2-overlap-micro/benchmarks.json`
- `bench-output/2026-04-29-arle-cuda-l4-qwen3-a2-overlap-micro/service_stats_trace.jsonl`
- `bench-output/2026-04-29-arle-cuda-l4-qwen3-a2-overlap-micro/service_stats_trace_summary.md`

## Rule

Only split scheduler CPU work after the trace shows a CPU-owned phase is
material at the target workload. Cleanup spikes should be tracked, but a
sub-millisecond EMA does not justify a new cross-thread ownership boundary.
