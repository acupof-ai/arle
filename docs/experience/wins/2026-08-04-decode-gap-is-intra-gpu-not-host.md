# The decode gap vs SGLang is intra-GPU dead time, not host time — 2026-08-04

> Status: **Measurement, no code change.** Supersedes the same-file
> 2026-08-03 version, whose headline ("~1.6 ms is host time between steps")
> the direct in-process timer refutes. **Total host time in a champion decode
> step is 0.061 ms.** The gap is dead time between kernels inside the graph
> replay, and the lever is launch count.

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

## What this makes the lever

**Launch count** — which reverses the priority call in the superseded version.
That version measured a norm+residual fusion opportunity at 0.156 ms of kernel
time and dismissed it as "an order below the actual lever". At ~4 µs of dead
time per launch it is worth its kernel time *plus* its dispatch gap:

| fusion | launches cut | kernel ms | + gap @4 µs | total |
|---|---:|---:|---:|---:|
| residual add into RMSNorm (SGLang ships `FusedAddRMSNormKernel`) | 128 | 0.156 | 0.51 | **~0.67** |
| conv1d prefill+state_update → one kernel | 48 | 0.074 | 0.19 | ~0.26 |
| split2 / split_qkv | 64 | 0.066 | 0.26 | ~0.33 |

Against a measured 1.91 ms residual gap (18.98 vs 17.07), those three are the
whole thing. None is invention: SGLang runs 128 fused norm+add where ARLE runs
256 separate kernels, and one `_causal_conv1d_update` where ARLE runs two.

The 4 µs/launch figure is a derived average (idle ÷ launches), not a measured
per-node cost — it sets the ranking, and each fusion still needs its own
matched A/B before it is licensed.

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
