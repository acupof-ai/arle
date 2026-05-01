# P2.B.7 MagicDec Sparse-KV Rerun Entered Spec Path But Regressed Throughput

## Context

P2.B.6 previously failed because sparse self-spec never entered the verifier
path: active long-prompt pages were still private to the paged KV pool and not
published into the radix prefix cache, so sparse view construction returned
empty views and spec metrics stayed `0/0/0`.

P2.B.7 (`beff488b`) added active-slot sparse page fallback and
`spec_sparse_view_empty_total`. This rerun tested the same longctx-32k c=4
envelope after the fix.

## Hypothesis

P2.B.7 should make sparse self-spec run for active long prompts:

- `spec=draft/verified/accepted` should be non-zero.
- `empty_sparse_views` should stay zero.
- acceptance should be at least `0.3`.
- throughput should not regress below the Phase 1 FP8 baseline
  `26.169 tok/s`.

## Command

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
LONGCTX_CONCURRENCIES=4 LONGCTX_MAX_SECONDS=300 WORKLOAD=longctx-32k \
GUIDELLM__MP_CONTEXT_TYPE=forkserver \
  scripts/bench_guidellm.sh phase2b-magicdec-sparse-c4-rerun-post-p2b7 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **GPU:** NVIDIA L4, 23,034 MiB
- **Driver:** 580.82.07
- **CUDA target:** sm_89
- **Model:** Qwen3-4B, `infer/models/Qwen3-4B`
- **Commit:** `65786543`
- **KV cache:** FP8 E4M3 paged pool
- **KV pool:** 136,976 tokens, 8,561 pages, page size 16
- **Feature set:** `cargo build --release -p infer --features cuda`

## Results

P2.B.7 fixed the original no-spec bug:

| Metric | Value | Status |
| --- | ---: | --- |
| Spec draft / verified / accepted | 941 / 941 / 702 | PASS, path entered |
| Spec acceptance rate | 74.6% | PASS |
| Empty sparse views | 0 | PASS |
| Spec step latency samples | 49 | PASS |

Throughput regressed severely:

| Metric | Value | Status |
| --- | ---: | --- |
| Phase 1 FP8 baseline effective throughput | 26.169 tok/s | reference |
| GuideLLM headline output throughput | 3.19 tok/s | FAIL, -87.8% |
| Completed-output/300s effective throughput | 4.27 tok/s | FAIL, -83.7% |
| Completed output tokens | 1,280 | recorded |
| Completed requests | 5 | recorded |
| Request latency p50 / p95 | 450.8 s / 451.0 s | FAIL |
| TTFT p50 / p99 | 81.374 s / 81.624 s | FAIL |
| ITL p50 / p95 | 1,319 ms / 1,319 ms | FAIL |
| Plan labels | decode=49, prefill=18, mixed=18, split=0 | recorded |
| Peak KV utilization | 98.5% | recorded |

Acceptance criteria:

- Throughput `>= 26.169 tok/s`: **FAILED**.
- Acceptance rate `>= 0.3`: **PASSED**.
- Sparse view should not be empty: **PASSED**.

## Root Cause

P2.B.7 proved sparse page selection now reaches active long-prompt KV pages, but
the current MagicDec implementation is still not a cheap draft path.

Observed service stats:

```text
spec=draft:941,verified:941,accepted:702,empty_sparse_views:0,accept_rate:74.6%,step_latency_count:49
step_last=33026.3ms step_p50=5000.0ms
step_phase_us=adm:2696,prefill:6342,decode:17541844,emit:12,total:17550800,cleanup:48019
kv_util=98.5% peak_mem=20727.1MB tokens_out=941
```

The likely contributors are:

1. Sparse self-spec currently drafts by repeatedly running sparse decode steps
   on the target model. Even with sparse pages, that is still a target-model
   forward loop before verifier work.
2. The verifier then runs full-KV target verification, so each spec step pays
   draft plus verify overhead.
3. KV utilization reaches 98.5%, so every long-prompt c=4 step runs at the
   same memory-pressure edge that Phase 1 had to fix for normal decode.
4. Sparse decode forces eager/single-row behavior, which destroys the normal
   batched decode cadence. The service trace shows only 49 decode steps over a
   very long wall-clock run.

The high acceptance rate is real but not useful yet: acceptance is no longer
the limiting factor. Draft/verifier step cost is.

## Fix

Do not claim Phase 2.B lift from this path yet.

Next engineering direction:

1. Profile sparse self-spec with Nsight Systems over one steady decode window;
   measure launches per accepted token and per-step kernel time.
2. Reduce sparse draft cost before more acceptance tuning:
   - batch K sparse draft positions instead of K sequential target forwards, or
   - make sparse draft use a cheaper kernel/layout, or
   - defer Phase 2.B until sparse-KV attention can run materially below full
     decode cost.
3. Keep P2.B.7 because it fixed the no-spec selector bug and exposed the real
   bottleneck.
4. Continue to FP8 KV audit before any further sparse-KV optimization, because
   Phase 1 mission margin depends on the FP8 path.

## Rule

For speculative decoding, acceptance rate is not a win by itself. A spec path
only counts when accepted tokens per second beats the non-spec Phase 1 baseline
under the same KV precision and request envelope.

## Artifacts

- Raw: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/benchmarks.json`
- CSV: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/benchmarks.csv`
- HTML: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/benchmarks.html`
- GuideLLM log: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/guidellm.log`
- Service trace before: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/service_stats_before.txt`
- Service trace during: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/service_stats_trace.jsonl`
- Service trace after: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/service_stats_after.txt`
- Service trace final manual snapshot: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/service_stats_manual_final.txt`
- Service trace summary: `bench-output/2026-05-01-phase2b-magicdec-sparse-c4-rerun-post-p2b7/service_stats_trace_summary.md`

