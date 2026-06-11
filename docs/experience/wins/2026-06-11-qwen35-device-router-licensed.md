# Qwen3.5/3.6 device-side MoE routing LICENSED — and the formula miss that re-ranks the board

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.6-35B-A3B, H20.
**Commit:** `874f8cfb`. **A/B:** same binary, `ARLE_QWEN35_GPU_ROUTER` env
flip, n=3 per cell, idle box.

## Result

| case | gen128 tok/s (n=3) | needle 3k+64 |
|---|---|---|
| TP=1 device route | **40.84** (40.65/41.05/40.83) | 9.07 s PASS |
| TP=1 host route | 39.61 (39.41/39.88/39.55) | 9.71 s PASS |
| TP=2 device route | **60.61** (60.55/60.63/60.65) | 5.20 s PASS |
| TP=2 host route | 56.93 (57.29/56.93/56.57) | 6.00 s PASS |

Δ decode: TP=1 **+3.1%**, TP=2 **+6.5%**; needle wall −6.6% / −13.3%.
Smoke strings identical across all four cells; needle PASS ×4.
**Verdict: default-ON licensed** — strictly positive at every cell, σ ≤ 0.4
tok/s, zero numerics regression, and the tradeoff column is clean (fewer
host ops, one extra zero-bias buffer).

## The honest part: the formula missed by an order of magnitude

The operator review predicted 2.3–3.5× from removing 40 per-token
host-routing roundtrips + unblocking launch-queue pipelining. Actual: 1.03–
1.07×. The prediction over-weighted the sync cost on the post-workspace
base: with per-call allocations already gone (`1e0f05e1`), each `ctx.sync`
drained a shallow queue, and launch issue itself (~1,074 launches/token)
plus the remaining kernel work dominates. Decode is still at ~5.4% of the
4 TB/s roofline (40.8 vs ~750 tok/s) — **the binding constraint moved and
must be re-measured (nsys per-token timeline) before the next decode-side
lever is picked.** Prefill-side levers (#2 DeepGEMM grouped, #3 chunked
GDR, #4 paged attention) are unaffected by this miss — their formulas are
byte/FLOP-based, not sync-count-based.

## Rule

- A sync-removal formula must price the QUEUE DEPTH behind the sync at the
  time of the fix, not at audit time — landing another orchestration fix
  first (workspace reuse) deflated the sync cost this formula assumed.
- License thresholds are for default flips, not for keeping strictly-positive
  zero-tradeoff wins: +3–6% with σ<1% and no regression keeps default-ON
  even though it misses the 10% re-tune bar.
