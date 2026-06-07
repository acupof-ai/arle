# DSv4 decode nsys: GAP-dominated (~60% GPU-idle) — the 6ms lever is a WHOLE-STEP graph

## Context

After every per-stage kernel lever A/B'd to a wash (graph/mHC/comm-overlap/DSA-skip/GEMV),
ran nsys on the TP=8 decode (tracer → no NCCL deadlock, unlike ncu) to find the real
critical path. Rank-0 only (gate on INFER_TP_RANK==0).

## What nsys showed (and the load-confound it exposed)

Whole-trace gpukernsum was dominated (67%) by `dsv4_block_scaled_to_fp8_cache_*` (4493
instances, ~65µs). **These are the LOAD-time DeepGEMM weight-cache build** (called only
from `loader.rs::from_dsv4_weight`; moe.rs never calls it) — one-time, NOT the decode.
The whole-trace sum misled until traced to the caller.

**Per-forward DECODE kernels** (the real decode GPU work): mhc_params 35µs, ncclAllReduce
16µs, deep_gemm 20µs, flash_fwd_mla 15µs, get_mla_metadata 16µs, compressor 9µs — summing
to **~6-8ms/forward of GPU**. But the decode wall is ~26ms/token (non-spec). **→ decode is
~60% GPU-IDLE: gap-dominated by per-step host orchestration BETWEEN layers, not the
kernels.**

## Rule / the corrected 6ms lever

- **The decode wall is host-GAP-bound, not kernel-bound.** GPU kernels ≈ 6-8ms; wall ≈
  26ms. This is why every per-kernel optimization was a wash (the kernels aren't the wall)
  and why MTP (+71%) works — it amortizes the whole gappy step over ~1.85 tokens.
- **The earlier decode-graph wash was a PER-LAYER graph** (`forward_tokens_decode_graph`
  does `run_or_capture` per layer) — it can't fill BETWEEN-layer gaps, hence −5%. The
  genuine 6ms lever is a **WHOLE-STEP CUDA graph**: capture the entire 43-layer decode
  forward as one graph → replay with ~0 host gaps → wall collapses toward the ~6-8ms GPU
  floor ≈ 6ms. On-device MoE routing (task #24, done) makes the step graph-capturable (no
  mid-forward host decisions).
- **For decode-wall analysis, trace the kernel CALLER before attributing nsys time** — the
  biggest kernels were the one-time load, not the decode. And separate load vs decode
  windows (NVTX) before reading gpukernsum.
- Next: confirm the decode GPU-busy fraction (decode-window nsys) and build the whole-step
  graph (the substantial lever; the per-layer scaffold exists in forward_tokens_decode_graph
  but must be lifted to one capture for the full forward). DSv4-Flash B=1 decode stands at
  ~15ms (MTP); the whole-step graph is the gate to ~6ms.

## CONFIRMED (sqlite query): decode is ~94% GPU-IDLE — HOST-bound, not GPU-bound

nsys sqlite, post-load window: GPU busy 291ms / wall 4648ms = **6.3% busy, 93.7% IDLE**.
Top decode kernels: sm90_fp8_gemm 59.5ms, mhc_params 49ms, ncclAllReduce 23ms,
deepgemm_pack 13.7ms, flash_mla 10ms — all tiny vs the wall. (Caveat: the window includes
harness inter-run setup, so the tight-loop idle is somewhat <94%; direction unambiguous.)

**This CORRECTS [[feedback_b1_decode_gpu_bound_overhead_removal_wash]]: B=1 decode is
HOST/GAP-bound, NOT GPU-bound.** The earlier "GPU-bound" was inferred from the per-layer
graph wash — but that graph washed because it didn't kill the BETWEEN-layer host, not
because the GPU was saturated. The GPU is ~94% idle; the host per-step orchestration is the
wall. Every per-kernel lever washed because the kernels are ~6% of the wall; MTP works
because it amortizes the host over ~1.85 tokens.

**The 6ms lever is definitively the WHOLE-STEP CUDA graph** — capture the entire 43-layer
decode forward as ONE graph (on-device routing, task #24, makes it capturable) → replay
with ~0 host orchestration → the wall collapses toward the small GPU floor (~6ms or less).
The per-layer scaffold (forward_tokens_decode_graph) must be lifted to a single
whole-forward capture; that is the substantial, now-evidence-licensed next kernel effort.
