# Qwen3-4B paged-prefill survives the sweep — root-cause fix (plan hoist)

> The structural fix from `docs/plans/paged-prefill-lifecycle-audit-2026-04-18.md`.
> Five kernel-level fixes weren't enough; the real bug was calling
> FlashInfer's `PrefillPlan` **36× per forward** (once per layer) instead
> of once. Plan writes host-pinned `page_locked_workspace` and enqueues
> `cudaMemcpyAsync` on the compute stream; subsequent plan calls
> overwrite the source buffer before the stream drains, so the
> enqueued copies read the wrong bytes and corrupt FlashInfer's
> `int_workspace`. sglang calls plan once per forward; we now do the
> same via the new `ops::PagedPrefillForward` handle.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B (bf16)
- **Hardware:** NVIDIA L4 24GB, CUDA 13.0, driver 580.82.07, SM 8.9
- **Commit:** pending (structural fix on top of `41e0c59`)
- **Feature set:** `cargo build --release --features cuda -p infer --bin infer`
- **Non-default flags:** `--num-slots 10 --max-seq-len 5120
  --mem-fraction-static 0.88`
- **`prefill_uses_paged_pool()` returns:** `true`
- **Prior snapshots:**
  - [`2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md`](2026-04-17-bench-guidellm-qwen3-4b-infer-l4-p99.md)
    (contig path, Apr-17)
  - [`2026-04-18-bench-guidellm-paged-phase1-qwen3.md`](2026-04-18-bench-guidellm-paged-phase1-qwen3.md)
    (contig path, Apr-18, after the five kernel-level fixes)

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir bench-output/2026-04-18-paged-phase3a-hoisted/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh paged-phase3a-hoisted`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 771.4 | 784.2 | 35.37 | 35.41 | 26.45 | 0.1 |
| throughput | 24403.1 | 48095.7 | 84.33 | 87.93 | 77.05 | 0.333 |
| 0.129r/s | 1206.4 | 1220.5 | 40.63 | 40.73 | 31.25 | 0.117 |
| 0.158r/s | 1206 | 1232.2 | 41.85 | 41.95 | 37.38 | 0.133 |
| 0.188r/s | 1226 | 1247.2 | 46.29 | 46.42 | 42.42 | 0.15 |
| 0.217r/s | 1237.3 | 1252.3 | 47.36 | 47.41 | 47.94 | 0.183 |
| 0.246r/s | 1241.7 | 1270.1 | 52.18 | 52.27 | 52.46 | 0.2 |
| 0.275r/s | 1260.8 | 1295.7 | 57.3 | 57.38 | 56.6 | 0.217 |
| 0.304r/s | 1266.3 | 1298.7 | 59.4 | 59.97 | 61.29 | 0.233 |
| 0.333r/s | 1290.8 | 1326.7 | 64.62 | 65.07 | 64.91 | 0.25 |

## Delta — paged hoisted vs contig (both on Apr-18 HEAD, same hardware)

| metric | contig (83a1239) | **paged hoisted** | Δ |
|---|---:|---:|---:|
| TTFT p50 @ sync | 878.7 ms | **771.4 ms** | **−12.2%** ✓ |
| TTFT p99 @ sync | 897.8 ms | **784.2 ms** | **−12.6%** ✓ |
| TTFT p50 @ 0.246 r/s | 1245 ms | 1241.7 ms | −0.3% (flat) |
| TTFT p99 @ 0.246 r/s | 1267.3 ms | 1270.1 ms | +0.2% (flat) |
| ITL p99 @ sync | 35.44 ms | 35.41 ms | flat |
| ITL p99 @ 0.246 r/s | 52.23 ms | 52.27 ms | flat |
| peak out tok/s (throughput mode) | 97.43 | **77.05** | **−20.9%** 🔴 |
| throughput TTFT p99 | 37757.5 ms | **48095.7 ms** | +27% 🔴 |

## What changed in the code

New struct `ops::PagedPrefillForward` holds the per-forward FlashInfer
plan state + uploaded indptr device buffers. Built **once** before the
per-layer loop via `PagedPrefillForward::new_hd128` or `new_hd256`.
Each layer now only runs (a) the paged-prep kernel (QK norm + RoPE +
paged K/V write) and (b) `plan.run_hd{128,256}` — no plan call, no
indptr re-upload.

- `infer/src/ops/attention.rs`: `PagedPrefillForward` struct +
  constructors; `prefill_attention_paged_batch` /
  `prefill_attention_hd256_paged_batch` signatures changed
  (`&mut BatchPrefillPagedPlan` → `&mut PagedPrefillForward`).
- `infer/src/ops.rs`: export `PagedPrefillForward`.
- `infer/src/model/qwen3/prefill.rs`: build `PagedPrefillForward::new_hd128`
  once in `process_all_layers_batch_paged`, pass `&mut fwd` to each
  of the 36 layer calls.
- `infer/src/model/qwen35/prefill.rs`: same for HD256 (8 full-attn
  layers). Still gated behind `prefill_uses_paged_pool() = false` for
  Qwen3.5 pending the `supports_partial_prefix=false` slot-reuse
  lifecycle fix, but the plan-race class is now fixed for that path
  too when it re-enables.
- `infer/src/model/qwen3/forward.rs`: `prefill_uses_paged_pool() → true`.

## Why paged TTFT is lower

Paged-prefill writes K/V directly into the pool via page-table
indirection — no contiguous→paged migration step after prefill. Saves
a GPU-bound copy per request. The ~100 ms delta on sync TTFT is
consistent with that.

## Why the throughput mode regresses

Unexpected. Hypothesis: under pure-throughput load the paged path may
be hitting the pool's eviction budget more aggressively (since there's
no contiguous scratch absorbing the initial K/V) or the scheduler's
`alloc_pool_tokens_with_retry` is serialising where the contig path
parallelised. Two observations:

- ITL is unchanged at steady rates — it's not a decode regression.
- TTFT p99 in throughput mode jumped from 37.7 s to 48.1 s, and out
  tok/s dropped ~20%. That's a TAIL / admission-thrashing regression
  of the open-rate case, not a per-request kernel perf regression.

Deferred to a follow-up audit — the win here is that the crash is
closed and every steady-rate slot gets the TTFT improvement. The
throughput-mode dip doesn't affect the matched-rate columns the
sglang-parity comparison cares about.

## Rule

When five kernel-level fixes to the same failure mode don't close a
bug, the bug is in the control flow that surrounds those kernels. The
root cause here was **call-site cardinality**: one plan per forward
vs. 36 plans per forward. The kernel-level patches (workspace size,
`total_num_rows` wire, try/catch, page-table trim,
`enable_cuda_graph=false`) are all correct on their own, but they
were chasing symptoms of the race. The actual fix is one commit
changing where `plan.plan_hd128/hd256` is called from.

## Artefacts

- Raw: `bench-output/2026-04-18-paged-phase3a-hoisted/benchmarks.json`
- CSV:  `bench-output/2026-04-18-paged-phase3a-hoisted/benchmarks.csv`
- HTML: `bench-output/2026-04-18-paged-phase3a-hoisted/benchmarks.html`
