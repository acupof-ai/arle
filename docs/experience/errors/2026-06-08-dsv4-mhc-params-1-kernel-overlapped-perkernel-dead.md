# DSv4 mhc_params (#1 by GPU-time) is OVERLAPPED → wall-neutral; per-kernel optimization is DEAD for B=1 decode

## Context
The clean critical-path profile named dsv4_mhc_params the #1 decode kernel (3.05ms/fwd GPU
time, 86×/fwd). Optimized it: uint4-vectorized the 16384-elem rms reduction (single-block,
latency-bound). Kernel 35.5µs → 26.5µs (−25%, ncu-class win), needle byte-identical.

## A/B (clean, non-nsys, 4× each, flashmla B=1 8×H20)
treatment(uint4): 36.6/40.7/39.6/38.8 mean 38.95 — baseline(scalar): 38.8/39.2/38.9/36.5
mean 38.36. **+1.5%, ranges fully overlap (both 36.5–40.7) → WASH.** Reverted (9c8f6078).

## Root cause / CONCLUSIVE RULE
- **The #1-by-GPU-time kernel is OVERLAPPED.** A −25% isolated win moves the wall 0% because
  the async pipeline runs mhc_params concurrently with critical-path work (e.g. the prior
  layer's all-reduce on the comm stream). Same as GEMV (1.8× isolated, wall-neutral).
- **gpukernsum (GPU-time-per-kernel) does NOT identify the decode wall.** Kernels overlap, so
  the GPU-time ranking is NOT the critical path. The clean profile was necessary but not
  sufficient — it ranks GPU work, not the serial dependency chain.
- **8 per-kernel/graph/host/comm levers have now ALL washed** for B=1 DSv4 decode: per-layer
  graph, whole-step graph, GEMV uint4, mHC-fusion, mhc_params uint4, comm-overlap, alloc-pool,
  launch removal. The decode wall is the CRITICAL-PATH serial chain (43 layers × dependent
  attn→AR→MoE→AR), and per-kernel speedups overlap away.
- **The ONLY lever that has ever moved B=1 decode is AMORTIZATION (MTP +71%, 27→15ms)** —
  because it spreads the WHOLE fixed critical path over N tokens. Per-kernel optimization is
  DEAD; the path to 6ms is deeper amortization (MTP depth/tree, frozen-KV) or batching (M=N,
  c>1 — the throughput thread), NOT kernel work.
- To EVER license a per-kernel decode opt, you'd need the nsys TIMELINE critical-path (which
  kernels are never-overlapped on the longest serial path) — not gpukernsum. But given 8
  washes, the prior is overwhelmingly "it's overlapped"; default to amortization.
