# DSv4-Flash decode is LAUNCH-BOUND (nsys) — 30 tok/s is ~5× off the roofline, decode-graph is the lever

Status: nsys-confirmed diagnosis (TP=4, DeepSeek-V4-Flash-FP8, c=1). Settles "is 30 tok/s the floor?"
— NO. It is launch-bound; the decode CUDA graph (gated off for CSA) is the unexploited lever.

## Context
After fixing the c≥2 crash (eager c=1/8/16/32 = 30/62/78/68 tok/s), ckl pushed back on calling 30
tok/s the floor: "硬件 roofline 还差多了". Correct — I had asserted a floor from a memory ("levers
wash"), never computing the roofline (§0: measured ≠ physical floor — compute bytes/(HBM·TP) first).

## Roofline (computed)
DSv4-Flash: hidden 4096, 256 experts top-6 + 1 shared (moe_inter 2048), 43 layers, MLA. Active
params/token ≈ ~10B → at FP8, TP=4 (attn /4, MoE allreduce /4) ≈ 2.6 GB/rank/token (≤8 GB un-sharded).
H20 HBM ~3.4 TB/s → **memory-bound roofline ≈ 0.8–2.5 ms/step = 400–1280 tok/s**. Measured c=1 = 33 ms
(30 tok/s) → **13–42× off**. Not memory-bound.

## nsys (decisive — the SOLID gate)
`nsys profile -t cuda,nvtx --trace-fork-before-exec=true --delay=150 --duration=55` over the 4 workers,
continuous c=1 stream. Stats:
- `cudaLaunchKernel`: **44,174 ms (40%), 8.74M calls** (avg 5.1 µs)
- `cuStreamSynchronize`: 39,464 ms (36%) — the GPU idling, waiting for CPU launches
- GPU kernel compute Σ ≈ **~12 s** — kernels are tiny (2–8 µs each)

Launch + sync (84 s) ≫ actual compute (~12 s). Per step/rank ≈ **1333 kernel launches × 5.1 µs ≈ 7 ms
of pure launch API** plus the idle gaps between them. The 43 layers × ~31 kernels/layer (MLA + 6-expert
MoE GEMMs + indexer/compressor + pack + all-to-all) never amortize at b=1. The all-to-all *comm* is NOT
the bottleneck (the launches/sync dominate) — so EP is the wrong lever; the graph is the right one.

## The lever
The decode CUDA graph (capture the step → 1 replay, eliminating the 1333 launches + idle) is **explicitly
`bail!`'d for any CSA/indexer layer** (attention.rs:8273) — and DSv4-Flash is all-CSA. So the graph
*never ran on the real path*; the memory's "graph washes for DSv4" was measured on a path that bailed.
csa_select_official already has a graph-path (returns None → defers to the batched select, 12821). The
real blocker is the batched select's per-step **H2D of context_lens/positions/block_table** (13334-48) —
a graph bakes the host buffer but the contents change each step. Fix: compute them device-side from
`start_pos_device`, then remove the bail.

Expected: launch overhead 7ms/step → ~0; wall toward the ~12s/1650-step ≈ 7ms compute floor → **~3–5×
(30 → ~100–150 tok/s)**, pending implementation + re-measure.

## Rule
Never call a measured number a "floor" without the computed roofline AND an nsys/launch-time split (§0).
For decode latency, `cudaLaunchKernel` total time vs `cuda_gpu_kern_sum` is the launch-vs-compute verdict;
millions of µs-scale launches = launch-bound = CUDA-graph territory, regardless of any "levers wash" lore
(which may have been measured on a path that bailed out of the graph). nsys the multiproc serve with
`--trace-fork-before-exec=true` to follow the spawned workers.
