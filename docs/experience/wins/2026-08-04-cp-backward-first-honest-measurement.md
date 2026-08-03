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

Flat to three digits. **Whatever the GDN CP residual is, it is completely
independent of world size.** The 10× ratio over single-card here is specific to
this config (8 GDN heads); at 2 heads and depth 8 the same measurement gives
1.18×, so the ratio is not a property of the CP path.

**And it is probably not CP-specific at all.** Signed deviations from f32, depth
sweep at 2 GDN heads — CP: −0.25%, −1.85%, +0.58%, +2.18%; single-card: −0.05%,
−0.21%, +0.49%, +2.38%. At depth 8 and 16 the two paths miss f32 in the *same
direction by nearly the same amount*: that residual is shared bf16 error, not
something CP introduces. A per-layer CP bias would show a same-sign, growing
CP−single gap; the gap shrinks (4.70× → 8.75× → 1.18× → 0.92×) until CP is
marginally better than single-card at 15 GDN layers. The depth-2 4.70× that
started this hunt is a small-denominator artifact — single-card is unusually
accurate there (5.286e-4).

**Full attention at cp=8:** 2.259e-3 (scalar ring) / 2.827e-3 (FA3 ring) vs
single-card 2.128e-3 — parity. The FA3 ring's excess over the scalar ring also
fails to survive: worse at depth 2 (6.53e-3 vs 2.49e-3), *better* at depth 8
(3.50e-3 vs 5.84e-3). Sign-flipping with config means it is measurement noise,
not a property — the earlier "FA3 doubles the hybrid gap" reading is withdrawn.

Every delta here sits at 1e-3–1e-2 against a 5e-2 margin, near the floor of
bf16 norm reproducibility, and each depth is a different randomly-initialised
model (f32 norms jump 73.6 / 15.9 / 19.2 / 76.4) rather than one model made
deeper. The series bounds a compounding bias; it does not resolve fine structure.

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
- Compare signed deviations, not magnitudes. Two paths missing the anchor in the
  same direction by the same amount share a cause; a ratio of magnitudes hides
  that and invites attributing shared bf16 error to whichever path is new. The
  first pass of this entry claimed a "constant GDN CP bias" on exactly that
  mistake, from one config's ratio.
