# DSv4 whole-step decode graph: WORKS + byte-identical, but WALL-NEUTRAL → decode is GPU-bound (CONCLUSIVE)

## Context

Implemented the whole-step decode CUDA graph (the hypothesized decode-6ms lever): ONE
capture over the entire per-token forward (all 43 layers + 86 all-reduces + tail),
replayed with ~0 host orchestration. Gated ARLE_DSV4_WHOLE_STEP_GRAPH (commit 81c64546).
This is the CLEANEST possible test of host-bound vs GPU-bound: it removes ALL host
orchestration between kernels.

## A/B (8×H20 TP=8, needle B=1, DECODE_GRAPH=1 + GPU_ROUTER=1 + FLASHMLA=0 so the gate engages)

| config | tok/s | output |
|---|---:|---|
| eager (no graph) | 30.38 | [223,30793,…,8308] |
| per-portion graph | 30.13 | byte-identical |
| **whole-step graph** | **30.32** | **byte-identical** |

- **The whole-step capture SUCCEEDS**: 86 NCCL all-reduces + ~860 kernels in ONE CUDA
  graph captures + replays correctly (byte-identical). (86-AR single capture was the
  feared blocker — it works.)
- **WALL-NEUTRAL**: 30.3 vs 30.4 (flat). Removing ALL host orchestration moves the wall 0%.

## CONCLUSIVE RULE (resolves the whole investigation)

- **B=1 decode is GPU-EXECUTION-bound, NOT host-bound.** The whole-step graph is the clean
  test (zero host between kernels) and it's wall-neutral — so the host orchestration is NOT
  the wall. The earlier nsys "94% GPU-idle" was a harness artifact (window included
  inter-run idle), as flagged. My memory's ORIGINAL "GPU-bound" was right; the "host-gap"
  correction was wrong.
- **Every host/graph/overhead lever is DEFINITIVELY a wash** (now proven by the cleanest
  possible test): decode graph (per-layer + whole-step), comm-overlap, mHC fuse,
  alloc-pool, launch removal. The GPU kernel execution IS the wall.
- **6ms requires LESS GPU WORK** — fewer/faster critical-path kernels + all-reduces, or
  less compute. Not host/graph/launch optimization. The GPU work is mostly vendored
  (FlashMLA, DeepGEMM) + the serial dependency chain (43 layers × attn→AR→MoE→AR), which
  is the hard, vendored/architectural frontier. MTP (+71%, the landed lever) works because
  it amortizes the GPU work over ~1.85 tokens — the only mechanism that moved the GPU-bound
  wall.
- The whole-step graph is kept gated (validated-correct, capturable) as infrastructure +
  the definitive diagnostic; it is wall-neutral on this model so NOT default-on, and it's
  locked to the !flashmla decode-graph base (can't help the faster FlashMLA path).
