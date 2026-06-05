# DSv4 decode 4.365 → 21.615 tok/s (~5×): on-device router async-unblocked by persistent scratch pool

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Model:** DeepSeek-V4-Flash FP8. **Shape:** parity harness, prompt
`671,6102,294,8760,344`, decode 16 tokens, greedy. **Commissioned by:**
[`docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`](../../plans/2026-06-04-dsv4-decode-sglang-class-perf.md)
Steps 2 (#24) + the alloc-kill (#29). **Status:** landed, default-on
(`ARLE_DSV4_GPU_ROUTER=1` + scratch pool), async sync-free, 16/16 correct.

## Context

The host-route baseline profile (committed in the roadmap doc §5) showed decode
was ~76% host overhead: `moe_route` 41.4% (= `cuStreamSynchronize`, per-layer
route D2H+CPU+H2D) and `deepgemm_grouped` 38.8% (= `cuMemAllocAsync`, per-step
grouped-GEMM scratch). #24 moved routing on-GPU (route kernel already 0-diff
licensed) but its async sync-free path produced garbage — bisection
(`SYNC_AFTER_DEEPGEMM` fail / `SYNC_AFTER_COMBINE` pass; post-MoE keepalive,
keepalive-all, same-stream event all FAIL) localized it to a
`cudaMallocAsync`/free/raw-pointer **reuse-boundary** race, not buffer lifetime.

## What Worked

A persistent `Dsv4MoeDecodeScratch` (route buffers, pack/count/slot/route_out,
DeepGEMM grouped workspace, shared-expert workspace) — fixed device addresses,
reused across forwards, no per-step malloc/free. This is the predicted
**double-win**: it kills the alloc cost *and* removes the reuse-boundary race
(fixed addresses → no aliasing), so the async sync-free path is correct with no
sync bridge. Scoped to decode/B=1 (prefill stays on-demand — avoid 512/2048
resident workspace peak, the #27 lesson). Plus a real OOB fix: shared-expert
DeepGEMM built a TMA D descriptor with `m=max_m=128` but allocated only 1 output
row; now allocates `max_m` rows and exposes the `seq_len` view.

**Decode throughput (same harness, same prompt):**

| Config | tok/s | ms/token | vs baseline |
|---|---|---|---|
| host-route baseline | 4.365 | 236.8 | — |
| GPU router + SYNC_AFTER_MOE (bridge) | 6.245 | 160 | +43% |
| **GPU router + scratch pool (async, sync-free)** | **21.615** | **~46** | **+395% (~5×)** |

(nsys-instrumented run: 19.724 tok/s — instrumentation overhead.) Correctness:
`clean_tokens` == oracle, 16/16.

**License gates — both met (after nsys re-profile on clean code):**
- route D2H gone: `cuMemcpyDtoHAsync` 120 calls / 2.16 ms total = **0.018 ms/token**
  (sample only; no per-layer route sync). ✓ #24.
- `cuMemAllocAsync` **87.4 → 2.81 ms/token** (−97%; residual is non-MoE per-layer
  temporaries, a #30 scratch-arena follow-up). ✓ #29.

**New 卡点 (after, per-stage NVTX) — host overhead is gone, kernels dominate:**

| Stage | before % | after % |
|---|---|---|
| `moe_route` | 41.4 | **3.7** |
| `deepgemm_grouped` | 38.8 | **4.3** |
| `mla_attn` | 5.0 | **15.9** (now #1) |
| `lm_head_sample` | 0.3 | **12.7** (~9.4 ms/token) |
| `shared_hc` | 3.9 | 5.4 |

The residual decode time is now real kernel compute (`mla_attn`, `lm_head_sample`)
+ per-kernel launch overhead/gaps. The launch overhead is #25's target (full
decode graph, ~250–400 launches/token → 1); `lm_head_sample` at 9.4 ms/token is a
newly-surfaced kernel-level lever (LM-head GEMM over vocab + sampling).

## Rule

A GPU operator that passes component parity but produces *accumulating* decode
drift only when its host-side sync is removed is a `cudaMallocAsync` reuse-boundary
race, not a buffer-lifetime bug — keepalive (holding `CudaSlice` clones) won't fix
it because the allocator re-hands the same freed address to the next step's
alloc while an async consumer still reads it. The fix is a **persistent scratch
pool** (fixed addresses, no per-step free), which simultaneously kills the
`cuMemAllocAsync` cost. Bisect the sync point to localize, then convert per-step
scratch to a reused arena rather than adding fences.
