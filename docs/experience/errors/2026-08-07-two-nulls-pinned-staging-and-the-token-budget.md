# Two nulls: pinned readback staging, and the per-tick token budget — CUDA, 2026-08-07

## Context

After the day's two accepted wins (verify linear core batched, rollback replay
batched), two further changes were measured and both came back null. Recording
them because each was predicted from an analysis that looked sound, and each
analysis was wrong in a way worth naming.

## Null 1 — pinned readback staging is a wash

**Predicted:** an nsys API ledger over a 30.75 s pure-decode window put
**1.64 s (15%)** in `cuMemHostAlloc` + `cuMemFreeHost`, 3552 calls paired 1:1
with 3487 D2H readbacks. `clone_dtoh` lands in a pageable `Vec`, so the driver
page-locks a staging buffer, copies, and frees it — ~615 µs to allocate, ~298 µs
to free, to move a handful of i32 tokens.

**Fixed in `7b8a66603`:** added `PinnedSlot<T>` (the pinned twin of `SliceSlot`'s
exact-length reuse) and routed the three readbacks that run every tick through
it — the batched draft draw, the single-token autoregressive draft step, and
`argmax_rows`.

**Measured** (counterbalanced, short prompts + 512 generation so the window sits
in the phase the fix acts on):

| c | C TPOT | D TPOT | Δ | C p90 | D p90 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.87 | 8.88 | +0.1% | 26.5 | 26.5 | +0.1% |
| 8 | 31.48 | 31.32 | −0.5% | 44.7 | 44.8 | +0.1% |
| 16 | 57.83 | 57.99 | +0.3% | 78.8 | 78.4 | −0.6% |

Within-arm spread at c=16 was C 58.15/57.52 and D 57.12/58.86 — the arms
overlap, so the −13.9% the c=16 p99 shows is inside the noise.

Repeated on the long-agent anchor (the other phase, the other dataset), also
counterbalanced:

| c | C TPOT | D TPOT | Δ | Δ total tok/s |
|---:|---:|---:|---:|---:|
| 1 | 8.609 | 8.521 | −1.0% | +0.5% |
| 2 | 18.618 | 18.444 | −0.9% | −0.1% |
| 4 | 34.126 | 33.036 | −3.2% | −1.6% |
| 8 | 60.782 | 61.769 | **+1.6%** | +1.1% |
| 16 | 115.444 | 112.579 | −2.5% | −0.0% |

The sign flips at c=8 and the within-arm spread at c=16 is 3.2% (C
117.31/113.58), so this is a null on both workloads, not a small win on one.

**Root cause of the wrong prediction: a profiler-measured host-side cost
fraction does not extrapolate to an unprofiled run.** nsys slows the host far
more than the GPU — the profiled run did 13687 total tok/s against 32954
unprofiled, a 2.4× dilation. The 1.64 s of `cuMemHostAlloc` is real wall time
*under nsys*; its **share** of the tick is an artifact of everything around it
having been slowed differently. A 15% share measured under a profiler is not a
15% share in production.

**Kept, not reverted.** It is strictly less driver work, it cannot regress, and
the codebase already documents the same fix for its snapshot path. But it is
recorded as a wash, and the ledger it came from is not evidence for the next
one.

## Null 2 — the per-tick token budget does not bind

**Predicted:** `max_num_batched_tokens` was a hardcoded 16384 with no CLI path.
The roofline ridge `M* = FLOPS / (2·BW)` fits at ~65 tokens on this box (the
profile's `dense_ffn` curve gives a 0.173 ms/layer memory floor and a
0.00268 ms/token compute slope). 16384 is ~250× that, a 16384-token step carries
~890 ms of compute by the same fit, and `itl_p99` is 758.8 ms against a
110.52 ms mean — so the tail looked like a decode row waiting behind a prefill
step, and shrinking the budget looked like it would cut it.

**Exposed in `ed92c6d8c`** as `--max-num-batched-tokens`, then swept on one
binary with 16384 run twice as its own noise control:

| budget | TPOT | p90 | p99 | total tok/s |
|---:|---:|---:|---:|---:|
| 16384 (1st) | 115.34 | 549.6 | 766.3 | 34091.0 |
| 2048 | 111.98 | 537.1 | 774.7 | 33484.4 |
| 512 | **126.88** | 586.6 | 828.0 | 33011.0 |
| 16384 (control) | 108.70 | 518.1 | 755.9 | 33815.6 |

Same-config repeat spread is **5.7%** — larger than the effect being looked for.
Against the two 16384 runs' mean of 112.02: **2048 is 0.0%**, and **512 is 13.3%
worse**.

**Root cause of the wrong prediction: I used the roofline ridge as an upper
bound. It is a lower bound.** Below the ridge, batching is free throughput
because the weight read is amortized over more tokens. Above it, more tokens
cost time proportional to the work they add — which is not waste, it is the
work. Lowering the budget does not remove work; it splits the same work across
more forwards, each re-paying the weight read. That is exactly the 13.3% the 512
arm lost.

The only cost of sitting above the ridge is **latency exposure** — a decode row
sharing the tick waits a whole step — and the measurement says that exposure is
not material here: with prefix caching each agent turn appends only ~864 tokens,
so a handful of concurrent prefill rows never approach 16384 and the budget
almost never binds. The arithmetic said so before the sweep did and I did not
check it: 16384 tokens at the measured 160 TFLOP/s is ~5.5 s of compute, but
TTFT p90 at c=16 is 2.90 s, so a step that large was never happening.

**Verdict: 16384 stays.** The knob stays too — the per-tick budget being
untestable was itself a defect — but there is no evidence for a different
default, and the derived-default idea sketched when the flag landed is
withdrawn.

## Rules

**A profiler's share-of-time is not the production share-of-time.** Absolute
kernel times survive profiling; host-side *fractions* do not, because the
profiler taxes the host and the GPU unequally. Size a host-side fix against an
unprofiled A/B, or don't size it.

**Check whether the constraint binds before optimizing it.** Both nulls are the
same omission at different layers: a quantity was analysed correctly in
isolation (pinned alloc really does cost ~0.9 ms; the ridge really is ~65
tokens) and then assumed to be on the critical path without testing that it was.

**Run the control arm twice.** The 5.7% same-config spread on this box is what
made the 2048 arm readable as a null rather than a 3% win. Without it the sweep
would have looked like a small monotone improvement toward smaller budgets,
which is the opposite of what it shows.
