# P2.B.6 MagicDec Sparse-KV Bench Did Not Enter Spec Path

## Context

P2.B.6 was intended to validate the end-to-end MagicDec sparse-KV path after
P2.B.4 (`59e379e1`) with self-spec draft mode and sparse KV enabled.

During setup, the scheduler config already had sparse-spec fields, but the
server CLI did not expose them. This tranche added the missing flags:

- `--spec-sparse-kv-enabled`
- `--spec-sparse-recent-tokens`
- `--spec-sparse-top-k-pages`

## Commands

Build:

```bash
ZIG=/tmp/zig14/zig CUDA_HOME=/usr/local/cuda \
  cargo build --release -p infer --features cuda
```

Server:

```bash
RUST_LOG=info CUDA_HOME=/usr/local/cuda ./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --num-slots 16 \
  --max-seq-len 131072 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 16384 \
  --max-prefill-tokens 16384 \
  --schedule-policy fcfs \
  --spec-enabled \
  --spec-draft-k 5 \
  --spec-acceptance-threshold 0.3 \
  --spec-draft-model self \
  --spec-sparse-kv-enabled
```

Bench:

```bash
LONGCTX_CONCURRENCIES=4 WORKLOAD=longctx-32k GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  scripts/bench_guidellm.sh phase2b-magicdec-sparse-c4-bench \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Results

Environment:

- GPU: NVIDIA L4
- Model: Qwen3-4B
- KV cache: FP8 E4M3
- KV pool: 136,976 tokens, 8,561 pages, page size 16
- Duration: 300 s

| Metric | Value | Status |
| --- | ---: | --- |
| Phase 1 baseline effective throughput | 26.169 tok/s | reference |
| GuideLLM headline output throughput | 28.0 tok/s | invalid result set |
| Successful-only throughput | 23.893 tok/s | FAIL, -8.7% vs Phase 1 |
| Successful requests | 28 | recorded |
| Completed output tokens | 7,168 | recorded |
| Spec draft / verified / accepted tokens | 0 / 0 / 0 | FAIL |
| Spec acceptance rate | 0.0% | FAIL |
| Spec step latency count | 0 | FAIL |
| Plan labels | idle=3, decode=517, prefill=122, split=0, mixed=2 | recorded |
| Peak KV utilization | 100.0% | recorded |

GuideLLM marked the run invalid because TTFT p50 and ITL p50 were both zero
despite successful requests with non-zero output tokens. The successful-only
token count is still useful as a conservative throughput check, but this run
must not be claimed as a lift.

Acceptance criteria:

- Throughput `>= 26.169 tok/s`: **FAILED** (`23.893 tok/s`, -8.7%).
- Acceptance rate `>= 0.3`: **FAILED** (`0.0%`).
- Ideal MagicDec reproducibility rate `>= 0.5`: **FAILED**.

## Root Cause

The sparse draft path did not run. Runtime stats after the bench reported
`spec=draft:0,verified:0,accepted:0,accept_rate:0.0%,step_latency_count:0`.

The current sparse view builder selects pages from the prefix cache:

- `infer/src/scheduler/cuda/spec_path.rs`: `build_sparse_draft_views` calls
  `prefix_cache.select_sparse_pages_for_draft_tokens_with_attached(...)`.

However, active long-prompt requests are published to the prefix cache only
during request cleanup:

- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs`: finished requests call
  `publish_to_prefix_cache(...)` before freeing the slot.

For the active 32k prompts under decode, the request KV pages are in the paged
KV pool and slot state, but they are not yet represented as selectable prefix
cache nodes. The sparse selector therefore returns an empty view and the spec
path falls back to normal decode without recording draft or verifier metrics.

## Fix

The P2.B.7 fix should source sparse draft pages from active slot-owned KV pages,
not only from completed prefix-cache entries. The selector needs a live-slot
view over materialized prompt/decode pages while preserving the invariant that
the verifier pass uses full KV.

Add a fallback metric such as `spec_sparse_view_empty_total` before the next
bench so this no-op condition cannot silently pass through the scheduler.

## Artifacts

- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench/server.log`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/benchmarks.json`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/benchmarks.csv`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/benchmark_report.html`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/guidellm.log`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/service_stats_before.txt`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/service_stats_after.txt`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/service_stats_manual_after.txt`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/service_stats_trace.jsonl`
- `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-bench-run2/service_stats_trace_summary.md`
