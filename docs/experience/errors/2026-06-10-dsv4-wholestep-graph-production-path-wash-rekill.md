# Whole-step graph on the PRODUCTION path is a WASH (10th wall-neutral lever) — GPU-bound re-confirmed by the cleanest test; nsys launch-gap framing trap formalized

**Date:** 2026-06-10 (final). 8×H20, same binary `571ccbd1`, env-flips, serial
arms, B=1 `dsv4_ab_bench.py`. Output-correctness checked per arm.

## The clean decomposition

| arm | B=1 p50 | output |
|---|---|---|
| masked eager (default) | 39.51 | correct |
| contig eager (`ARLE_DSV4_MOE_CONTIG_DECODE=1`) | 39.38 | correct |
| **whole-step graph + FlashMLA + contig** | **38.92 (−1.5%, WASH)** | **correct, 0 IMA** |
| whole-step graph + pooled tail | 33.41 | correct (pooled GEMM tax ≈ −14%) |
| whole-step graph + masked ALLOC tail | 48.85 | **degenerate** (false speed) |

## What this kills and why it's final

ONE graph replay per token replaced ~2830 host launches on the production
FlashMLA + contig-MoE path, output byte-coherent — and the wall moved −1.5%.
The host launch loop is FULLY hidden behind the GPU chain. This re-confirms
the 2026-06-08 CONCLUSIVE (B=1 decode is GPU-execution-bound) on today's
stack with a strictly stronger experiment.

- **The nsys "launch drizzle 7.5 ms/token" was NOT recoverable host time**:
  inherent inter-kernel dispatch latency + CUPTI per-kernel observer tax. The
  skew-anatomy lever board #1 (graph, predicted −5~7 ms) is REFUTED.
- **"Host launch burst spans 27.7/30.8 ms" ≠ host-pacer**: the burst is
  API-call wall time OVERLAPPED with GPU execution (queue-paced), not a
  serial prefix. Launch-span framing joins NVTX-sync absorption in the
  framing-trap list.
- **+24% with degenerate output is the most dangerous failure shape**: the
  masked alloc-tail "win" was the MoE partially no-oping on aliased VAs.
  Speed deltas are meaningless until the correctness gate passes.

## Rules

- **Captured bodies allocate NOTHING.** Stream-ordered allocs + clone_htod'd
  constants inside capture become graph nodes whose Rust-side frees land
  after capture → replays touch aliased/freed VAs → silent corruption or
  false speedups. Every buffer in a captured body is a persistent scratch
  address; constants are filled at init.
- **A graph wash on a GPU-bound chain is expected, not surprising**: launch
  removal only pays when the host is the pacer. Test host-pacer with a
  whole-step graph BEFORE building per-path capture-safety.
- nsys gap decompositions must subtract inherent dispatch + observer tax
  before claiming recoverable host time — A/B the graph, don't trust the gap.
- **Sampling/RNG never enters a capture with by-value state** (audited vs the
  SGLang Omni frozen-(seed,offset) catastrophe, flashinfer sampling baked a
  by-value RNG handle into the graph → frozen u → AR self-conditioning locks
  into repetition attractors, cross-boot drift makes it look flaky). ARLE is
  structurally immune today: BOTH decode graphs (DSv4 `tail_graph`, Qwen
  `GraphBucket`) end at the logits buffer with sampling outside the capture,
  and the RNG is stateless counter-based `splitmix64(seed, position)` — the
  philox-contract design; there is no mutable offset to freeze. If sampling
  is ever moved in-graph for the lm_head-tail ~0.6 ms, position must live in
  a device buffer advanced by a captured kernel (by-reference, the PyTorch
  graphsafe pattern) — and note our greedy (temperature=0) graph gates are
  BLIND to this bug class; a same-seed cross-boot 8/8 token-id probe is the
  matching gate.

## What the campaign keeps (durable instrumentation/infra wins)

Universal warm-before-capture; rearm_warm request-boundary mechanism;
3 capture-hazard classes fixed across the production path (constants→init
also −43 sched-meta calls/token in eager: 38.99 → 39.51 across the two
binaries); -1 sentinels devicified; contig-MoE rehabilitated (≈ masked at
B=1, old −24% kill was the non-contig pooled variant). The graph is correct
and env-gated for future stacks where the host becomes the wall.

## The lever board after this kill

1. **MTP** (÷~1.85) — the only proven B=1 multiplier; serve integration is
   THE next tranche.
2. GPU-work reduction on the serial chain (fusion: 65 kernels/layer; mHC,
   AR+rmsnorm) — license each by component A/B.
3. Lockstep start-offset (~2-4 ms skew) — real on the GPU timeline, but with
   a GPU-bound chain its recoverable share needs a direct experiment
   (rotate/parallelize TickAdmissions sends), not inference from gaps.

## Refs

- Ladder + capture fixes: [`wins/2026-06-10-dsv4-flashmla-graph-ima-fixed-ladder.md`](../wins/2026-06-10-dsv4-flashmla-graph-ima-fixed-ladder.md)
- Skew anatomy (its lever #1 hereby corrected): [`wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md`](../wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md)
- 6-08 CONCLUSIVE: re-confirmed.
