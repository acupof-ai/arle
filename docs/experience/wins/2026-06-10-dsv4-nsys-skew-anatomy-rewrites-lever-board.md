# nsys skew anatomy: B=1 decode wall = launch-gap drizzle (5-7ms) + lockstep start-offset (2-4ms) + compute (9.5ms) — lever board rewritten

**Date:** 2026-06-10 (late). **Captures:** `nsys_skew_{oneshot,ncclcomm}.{nsys-rep,sqlite}`
(B=1, 256 tok, `ARLE_DSV4_NVTX=1`, both comm arms, same binary `a0fd3a12`).
**Analyzer:** `scripts/`… pod `skew_analysis.py` (kernel-level GPU timestamps,
cross-rank instance matching by per-device ordinal — host NVTX is enqueue-time
and was NOT used for GPU claims).

## What the per-kernel GPU timeline proves (one-shot arm; 22016 AR + 11006 AG instances)

1. **Protocol is fully dead**: exec-after-last-arrival = 5.0 µs (AR) / 5.6 µs
   (AG) — bench speed, on-chain, in production. First-arriver duration 47 µs
   (spin) vs last-arriver 4.9 µs confirms spin semantics. The comm wash is
   mechanically closed: nccl arm exec was 15.0/12.9 µs, the −10 µs/op gain
   minus ~5 µs/op staging ≈ noise. Staging copies measured 1.2 µs p50.
2. **Arrival spread δ: p50 42 µs (AR) / 71 µs (AG), p90 160/278, heavy tail** —
   and it is SYSTEMATIC: rank 7 is persistently last (its spin total 0.9 s vs
   2.5–2.9 s on every other rank).
3. **MoE imbalance KILLED as the skew source**: per-rank grouped-GEMM time is
   dead even (918–928 ms), FlashMLA/GEMV even. All compute legs equal.
4. **The lag is host-side**: rank 7's stream shows ~34 k gaps >20 µs
   (≈ one per collective, ~37 µs avg) right after barrier releases — its host
   feeds the stream a beat late. Mechanism hypothesis (strong): the lockstep
   `TickAdmissions` broadcast serializes 7 TCP writes in rank order; rank 7
   starts every step last, and with hosts pacing near GPU speed the initial
   offset persists through all 43 layers. (Cross-check: skew exists identically
   under the nccl arm — comm-implementation-independent.)
5. **The BIGGEST idle block is not skew**: gap-size decomposition per token —
   <20 µs launch gaps × ~2000 kernels = **4.7–7.5 ms/token on every rank**
   (2800 kernels/step = 65/layer); 20–100 µs collective-lag gaps = 1.8 ms
   (typical rank) – 4.6 ms (rank 7); **zero** >2 ms blocks (no per-step tail).

## Wall composition @ 25.8 ms/token (rank 7, the pacer)

| block | ms/token | share |
|---|---|---|
| compute kernels | ~9.5 | 37% |
| per-kernel launch gaps (<20 µs × ~2000) | ~7.5 | 29% |
| per-collective host lag (20–100 µs × ~133) | ~4.6 | 18% |
| collective spin | ~3.5 | 14% |
| misc | ~0.7 | 3% |

## Lever board (rewritten)

1. **Whole-step CUDA graph — RE-LICENSE on today's stack.** The 5–7.5 ms
   launch-gap block is exactly what a graph removes. The 2026-06-08
   "wall-neutral CONCLUSIVE" verdict predates official-kernel adoption,
   lockstep, and this measurement — and that experiment proved the graph
   WORKS byte-identical. Critically: **one-shot CAR is graph-capturable while
   NCCL-under-TP is not** — today's "wall-neutral" comm work is the enabler
   that makes the multi-rank decode step graphable at all. Predicted: −5~7 ms
   (25.8 → ~19).
2. **Cheapen the lockstep step-start** (−2~4 ms): pre-serialized buffer +
   non-serialized sends, rotate send order per tick, or a shared-memory
   doorbell replacing per-worker TCP. Also fixes the rank-7 systematic bias.
3. **MTP** (÷~1.85) — unchanged as the only multiplier; stacks on 1+2:
   ~19 → ~16 → **~8–9 ms/token ≈ 110–125 tok/s**. The 6 ms target is visible
   for the first time.
4. Kernel-count reduction (65 kernels/layer is enormous) — same target as #1
   via fusion instead of graphing; graph is cheaper to try first.

## Rules

- **Kernel-level GPU timestamps, not host NVTX, for cross-rank claims** —
  host ranges measure the enqueue loop.
- **A "GPU-bound" verdict expires when the stack is rebuilt.** Re-measure
  idle composition after major kernel/scheduler changes; the 6-08 graph
  conclusion did not survive official kernels + lockstep.
- Per-collective δ is a SPREAD, not additive wall cost — wall accounting
  needs (max-arrival − prev-release) + exec, never Σδ (framing trap).

## Refs

- Comm wash + per-op numbers: `errors/2026-06-10-dsv4-oneshot-comm-wall-neutral-skew-bound.md`
- Graph works byte-identical: `wins/2026-06-08-dsv4-wholestep-graph-wall-neutral-gpu-bound-CONCLUSIVE.md` (its WALL verdict superseded by this entry)
- Pod: `nsys_skew_*` reps/sqlites, `skew_analysis.py`
