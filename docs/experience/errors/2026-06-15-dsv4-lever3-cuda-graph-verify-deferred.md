# DSv4 lever 3 (CUDA-graph the batched verify) — deferred with evidence (verify is compute-bound, not launch-bound)

## Context
After deploying batched MTP (+77% @c=8), lever 3 of the campaign was "CUDA-graph the
multi-row verify to recover launch overhead" (c-sweep #70-class lever). Assessed for the
`都解决掉` (solve-all-remaining) pass.

## Verdict: DEFER (modest ROI × high complexity) — not implemented
### Evidence 1 — the verify is GPU-COMPUTE-bound, not launch-bound
Phase profile (`ARLE_DSV4_MTP_STEP_PROFILE`, batched MTP) — verify ms vs wave size n:
`n=4 → 76ms`, `n=8 → 134ms`, `n=9 → 147ms`. That's ~linear in n (76→134 = 1.76× for 2× n,
slightly sub-linear). A launch-bound phase would be ~FLAT in n (fixed launch count); the
near-linear scaling means the verify is dominated by per-row GPU COMPUTE (the DSv4
60-layer MoE), which CUDA-graph (a launch-overhead lever) does NOT touch. The +77%
batching already removed the per-row launch-gap that made CUDA-graph valuable on the
single-row path (`Dsv4DecodeGraphScratch` is `seq_len==1` only, `dsv4.rs:1548`).

### Evidence 2 — variable-n makes the graph high-complexity
CUDA graphs capture a FIXED kernel topology. The batched verify's topology varies with
the wave size n (the per-slot attention loop has n iterations; n swings 4↔16 as
concurrency changes). A captured graph at n=8 is invalid at n=4 → would need a graph PER
n (capture 4..16) or pad every wave to a fixed n (wasted compute). The existing
single-row decode graph sidesteps this (n always 1). Multi-row variable-n graph capture
is a large, brittle build for the modest launch-overhead recovery above.

### Better alternative if attention-launch is ever the target
The recoverable launch overhead inside the verify is the PER-SLOT attention loop (n×3
small ops/layer). The real fix is **cross-slot batched tree attention** (one FlashMLA
over all N×(depth+1) rows with a block-diagonal tree meta — the "Stage 2" deferred when
the per-slot tree-attn was chosen). That cuts the loop to 1 call/layer (fewer launches +
better occupancy) — strictly better than graphing the many launches. But it's the hard
block-diagonal-FlashMLA kernel, also modest ROI given the verify is compute-bound.

## Disposition
Lever 3 deferred. The decode throughput is verify-compute-bound (DSv4's inherent
60-layer MoE); the +77% batched MTP is near the decode ceiling. Launch-overhead levers
(CUDA-graph, cross-slot attn) are modest single-digit-% for significant complexity.

## Rule
- **Graph/launch levers only pay when the phase is launch-bound (flat-in-batch); a
  phase that scales ~linearly with batch is compute-bound — measure the scaling before
  building a graph.** The single-row path was launch-bound (CUDA-graph helped); the
  batched verify is compute-bound (it won't). [[feedback_measure_batching_before_ceiling]]
