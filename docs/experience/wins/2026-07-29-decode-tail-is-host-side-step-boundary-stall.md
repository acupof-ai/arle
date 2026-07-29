# The decode tail is a host-side step-boundary stall, and it is the biggest lever left

## Context

At c=16 the dense arm loses 55.4% of its decode wall to spikes (17.3% of gaps
above 3× p50, p90 426 ms vs p50 66.6 ms); MoE has the same shape at 6.1% /
21.7%. Two hypotheses died cheaply first: it is not the prefill chunk
(`chunked_prefill_size` 512/1024/2048 moves p90 by nothing —
[entry](../errors/2026-07-29-dense-decode-tail-is-not-the-prefill-chunk.md)),
and it is not preemption (zero `parking the request` warnings in any run log).

## What Worked

nsys over a steady c=16 window, measuring GPU-idle rather than kernel time.

| | dense 27B | MoE 35B-A3B |
|---|---:|---:|
| window | 40.1 s | 30.1 s |
| GPU busy | **59.4%** | **81.1%** |
| GPU idle | 40.7% | 18.9% |
| idle gaps > 50 ms | 47, totalling 14.7 s | 26, totalling 4.3 s |
| that as share of window | **36.7%** | **14.2%** |
| largest gaps | 704, 681, 675, 673, 670, 655, 653 ms | 264, 233, 230, 226, 217 ms |

Every large gap is bounded by the same pair: `argmax_batch_kernel` (the end of a
decode step — sampling) before it, `embedding_batched_native_kernel` (the start
of the next step) after it. **The stall is entirely host-side, between two
steps, with nothing on the GPU.** That is why chunk size did nothing: no prefill
is running during the spike.

Both models have it; dense is ~2.6× worse on both count and size. 47 gaps against
~64 requests in flight is about one per request completion.

## Why this reframes the work

The MoE expert-kernel tranche just bought 1.22× on the steady-state step — a
1.22× on the 81% the GPU is actually busy. Meanwhile 19% (MoE) to 41% (dense) of
the wall is the GPU doing nothing. **At c=16 the host loop is now a larger lever
than any remaining kernel.**

## Still open

What the engine does between `argmax` and the next `embedding` has not been
attributed. It is per-step host work that scales with the model, so the
candidates are the per-finish path (recurrent sidecar save, radix
`publish_prefix_blocks`) and per-step planning. 123,928 DtoH copies in a 40 s
window is the other thing worth explaining — their transfer time is only 404 ms,
but each may carry a synchronization.

Attribution needs host-side profiling (nsys `--trace osrt,nvtx` or the existing
`qwen35_profile` hooks widened to the step boundary), not more inference.

## Rule

**Measure GPU idle, not kernel time.** Three rounds of this investigation
profiled what the GPU was running and found nothing wrong with it, because the
problem was the GPU running nothing at all. A kernel summary cannot show you an
empty timeline — you have to ask for the gaps.
