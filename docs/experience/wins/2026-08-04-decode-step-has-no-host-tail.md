# The champion decode step is kernel time end to end — 2026-08-04

> Status: **Measurement, no code change.** Host time in a decode step is
> **0.061 ms** of 19.03, and `Σ kernel` is **19.184 ms** against a 18.97 ms
> wall. There is no host tail and no idle to reclaim; every optimization from
> here has to make a kernel faster. Two earlier headlines on this file —
> "~1.6 ms is host time between steps" and "the lever is launch count" — are
> refuted below by the measurements that replaced them.

## What the nsys ledger said, and why it misled

The two-sided decode-only ledger (ARLE @ T4 vs SGLang 0.5.13, same H20, same
gptq_marlin kernel, same int8 weights) is still correct as far as it goes:

| per decode step | ARLE @ T4 | SGLang | Δ |
|---|---:|---:|---:|
| wall | 20.58 ms | 16.12 ms | +4.46 |
| GPU busy (Σ kernel) | 15.89 | 14.44 | +1.45 |
| **GPU idle** | **4.69** | **1.68** | **+3.01** |
| launches | 1059 | 928 | +131 |
| occupancy | 0.77 | 0.90 | |

A gap analysis localized ARLE's idle further: **98.6% of it is a single
~4.6 ms block per step**, and every one of the 200 sampled blocks sits between
`gemv_handwritten_kernel` (lm_head, last kernel of a step) and
`embedding_batched_native_kernel` (first kernel of the next). Inside those
blocks the CUDA API accounts for only 0.923 ms (`cuGraphLaunch` 0.891, nine
H2D 0.028, sync 0.003, D2H 0.001).

From "a single contiguous block, at the step boundary, with almost no CUDA API
in it" I concluded: host code. **That inference was wrong**, and it was wrong
in the one way a profile cannot self-check — nsys attributes wall to *the GPU
timeline*, so "GPU idle" means the device had no kernel running. It says
nothing about whether a CPU was doing anything.

## What the direct measurement says

`ARLE_STEP_PHASE=1`, champion W8A16, c=1, 32K context, decode-only steps, with
an explicit sync inserted after the graph launch so the GPU wall cannot hide
inside the sampling phase:

```
decode-phase steps=3000  mirror=0.003  meta=0.011  stage=0.002  gpu=18.973  sample=0.040 ms
step-phase   steps=3000  poll=0.000  apply_out=0.005  poll_bg=0.000  admit=0.000  plan=0.000  submit=19.030 ms
```

Identical to three decimals across the 1500/2000/2500/3000-step reports.

**Host total = mirror + meta + stage + sample + apply_out = 0.061 ms of a
19.03 ms step.** Scheduler, admission, planning, and the token/stream tail are
all exactly zero. There is no host tail left to cut — the 1.2 ms resident-page
scan fixed on 2026-08-03 was the last of it.

So the ~4.3 ms of idle is **the GPU sitting between kernels during the graph
replay**:

| | ARLE | SGLang |
|---|---:|---:|
| launches / step | 1059 | 928 |
| intra-GPU idle / step | ~4.3 ms | 1.68 ms |
| **idle per launch** | **~4.0 µs** | **~1.8 µs** |

## What the lever is NOT

The first version of this section proposed **launch count**: ARLE runs 1059
launches to SGLang's 928, so at 4.3 ms ÷ 1059 ≈ 4 µs of dead time per launch,
fusing kernels should return real time.

**That was tested and refuted the same day.** Fusing every residual-add +
RMSNorm pair in all three layer loops removed **192 launches per step** and
moved the GPU wall by 0.00 ms (18.973 → 18.978/19.006). Full account in
[2026-08-04-launch-count-is-not-the-decode-lever](../errors/2026-08-04-launch-count-is-not-the-decode-lever.md);
that tranche is reverted and the conv1d and split fusions queued behind it are
dead on the same evidence.

**4 µs/launch was a quotient, not a rate.** Removing 18% of the launches
removed none of the idle.

## The idle it was named after does not exist

The ~4.3 ms figure came from GPU wall 18.97 minus `Σ kernel ≈ 14.7`, and that
14.7 was the T4 trace's 15.886 reduced by the *expected* gains of two later
tranches, never re-measured. An nsys run on the champion itself puts **Σ kernel
at 19.184 ms** against a 18.97 ms wall, occupancy 0.95: the step is kernel time
end to end. Nothing is idle, which is why removing 192 launches returned
nothing.

Eager launch cost was then measured directly for comparison:
`--qwen35-decode-graph false` runs the same work outside the graph and costs
**+1.55 ms** over ~1000 launches, ~1.6 µs each. Inside a captured graph node
count is free.

The real lever was one kernel:
[FA3 decode splits](2026-08-04-fa3-decode-splits-fill-the-sms.md).

## Rule

**"GPU idle" in a profile is a statement about the device timeline, never
about the host.** The two are only the same when the host is what the device
is waiting for, and that is exactly what a whole-step CUDA graph removes.
Before attributing device idle to host code, put a timer *in the host code* —
it cost one build here and it inverted the conclusion.

Corollary, learned the hard way twice in one day: an in-process phase timer
must be gated on the phase it claims to measure. The first cut of this
instrument averaged over all steps and reported `submit=109 ms` (one 32K
prefill costs ~25 s); the second gated on the plan being *submitted* rather
than the plan `apply_output` was *processing*, and booked a post-prefill radix
seal into the decode bucket as `apply_out=4.264 ms`. The true value is 0.005.
