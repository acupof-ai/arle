# DSv4 #25 breakable decode graph: launch API −87%, wall flat — decode is GPU-kernel-bound (H20 floor)

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Shape:** parity harness, prompt `671,6102,294,8760,344`. **Commissioned by:**
[`docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`](../../plans/2026-06-04-dsv4-decode-sglang-class-perf.md)
§6 lever 2 (launch overhead). **Status:** landed, **default-off**
(`ARLE_DSV4_DECODE_GRAPH=1`, gated to `seq_len==1 && gpu_router && !deepep`).

## Context

Post-buffer-lever decode was 41.2 ms/token (25.129 tok/s); the after-profile
showed launch API (`cudaLaunchKernel` + `cuLaunchKernelEx`) ≈ 13.36 ms/token =
32% of wall — the largest remaining bucket by the API table. Hypothesis: a full
decode graph collapsing ~250–400 launches/token → ~1 would reclaim it.

## What Worked (and the honest result)

A full 43-layer **breakable** CUDA graph: attn graph segment → eager TP all-reduce
→ MoE graph segment → eager TP all-reduce → tail graph; sampling outside the
graph, DeepEP untouched. Prereqs (both correctness-preserving, verified): the
per-step `start_pos` moved to a **device-scalar ABI** (not baked into the launch
— the cache-as-input capture rule) and `compressed/key_count` bounded to a
**graph-safe fixed capacity**. Mechanism pre-proven on 8 ranks (graph-replay →
fixed buffer → eager NCCL all-reduce on that buffer).

| Metric | before (#29+buffer) | #25 graph | Δ |
|---|---|---|---|
| 16-tok decode | 25.129 tok/s | 25.517 | +1.5% |
| 64-tok guard | — | 25.778 (no bail, 16 exact) | ✓ |
| 128-tok nsys | — | 25.296 (no bail) | ✓ |
| launch API | 13.36 ms/tok | **1.68 ms/tok** | **−87%** |
| wall | 41.2 ms/tok | ~39.5 ms/tok | ~flat |

Correctness: 16/16 == oracle, 64 + 128 no incremental bail.

**The honest finding (§0 framing trap, 3rd this arc):** the 13.36 ms/token
`cudaLaunchKernel` was **CPU-side launch time overlapped with GPU execution** —
not on the wall-clock critical path. The graph removed 87% of it but wall barely
moved because the GPU was the bottleneck all along. `cuStreamSynchronize`
(37.1 ms/token) now waits on **GPU backlog**, not host-route sync. **Decode is
GPU-kernel-bound.** The narrow API-time table (launch = 32%) was the trap; the
wall-clock critical path is the kernels.

## Verdict

This completes the host + launch overhead elimination axis. Cumulative:
**236.8 → ~39.5 ms/token (~6×), 4.365 → 25.5 tok/s.** The remaining ~39.5 ms is
real GPU kernel floor: hybrid MLA attention (~11.8 ms), FP8 GEMV, DeepGEMM
pack/unpad, MHC params. **The 5–6 ms target is an H100/H800 number** (where
SGLang measures it); on H20 (lower FLOPs/bandwidth) the realistic decode floor
for this 671B-class FP8 MoE is ~25–40 ms/token without deep per-kernel rewrites.
The graph lands gated-off — it frees the CPU (value at concurrency / once kernels
shrink) and is the canonical production architecture, but is not a single-stream
B=1 wall win today.

## Rule

`cudaLaunchKernel` aggregate ms/token in an API table is **CPU launch time, much
of which overlaps GPU execution** — it is not wall-clock-critical-path time. A
graph that removes it only speeds wall-clock if launches were actually starving
the GPU. Before committing to graph capture for "launch overhead," confirm the
GPU is *waiting on the launch queue* (gaps between kernels), not the reverse. We
removed 87% of launch API for ~1.5% wall — correct call to verify, kept gated.
