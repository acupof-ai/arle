# The CP backward is measured for the first time: full attention is at parity to cp=8, GDN carries a constant 7e-3 bias

**Date:** 2026-08-04 · **Commits:** 5d751b56e (grad instrument) + f46ca2fb8 (anchor fix) + e3ff7c368 (degenerate-anchor guard) + 3829d90ce (depth/world-size axes) + 7c5105c83 (8 GDN heads) · **Pod:** 8×H20

## Context

`nd_parallel_parity` ran the full CP step — backward, grad all-reduce, optimizer —
but only ever compared the **loss**. The CP backward had never been checked
against single card. A 27B run then showed matching losses (spread 8.2e-4) with
grad norms 3.744990 (cp=1) vs 1.984009 (cp=2): a 1.89× break the loss-only gate
could not see.

## What worked

Three-way grad-norm comparison (CPU f32 / single-card CUDA / CP) on the same
name-seeded model, plus real axes for the two things a CP bias would have to
scale on: `ARLE_ND_LAYERS` (depth) and `ARLE_ND_CP_SIZE` (world size = ring step
count). Both knobs were inert when first written — depth died on
`layer_types length must equal num_hidden_layers`, and `CP_SIZE` was a `const`
so a 4-GPU run silently reproduced the 2-rank numbers. Fixed before any
conclusion was drawn from them.

**World-size sweep, 8 layers, 7 GDN + 1 full attention, `grad_cp_vs_f32`:**

| cp_size | cp_vs_f32 | single_vs_f32 |
|---|---|---|
| 2 | 7.201e-3 | 7.242e-4 |
| 4 | 7.188e-3 | 7.245e-4 |
| 8 | 7.195e-3 | 7.251e-4 |

Flat to three digits. **The GDN CP bias is a constant ~7e-3 relative, ~10× the
single card's distance from f32, and completely independent of world size.**

**Full attention at cp=8:** 2.259e-3 (scalar ring) / 2.827e-3 (FA3 ring) vs
single-card 2.128e-3 — parity. Depth sweep (2/4/8/16) shows both CP and
single-card drift from f32 growing together as bf16 error accumulates per layer;
the *ratio* does not compound (4.7× → 8.8× → 1.18× → 0.91×, non-monotonic
because each depth is a different random model).

**cp=8 runs end-to-end for both layer families** — the world size 256K training
targets, previously unexercised.

## What this does NOT explain

The 27B's 1.89×. Nothing here compounds: not depth, not world size, not
sequence length (a seq=1024 probe *shrank* the gap 3.3×). The toy bias is three
orders below 89%. The remaining 27B-specific factors are FP8 base weights, MoE,
real seq 32768 with checkpoint offload engaged, and LoRA on attention-qv only.
The discriminating experiment is a per-param grad dump on the 27B at cp=1 vs
cp=2 — a global norm cannot say *which* parameters diverge.

## Rule

- Before reading a sweep, prove each knob moves the thing it names. Two of three
  axes here were inert, and both would have produced a confident wrong answer:
  a flat line reads as "does not scale" whether the mechanism is absent or the
  knob is.
- A constant bias and a compounding one demand different responses. Measuring
  the axis is what tells them apart; without it, 7e-3 at depth 8 could be
  extrapolated to anything at depth 48.
