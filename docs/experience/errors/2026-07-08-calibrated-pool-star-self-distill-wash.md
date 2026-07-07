# Calibrated-pool STaR self-distill = wash — smooth-RL build parked

> 2026-07-08 · Driver: ckl · Terminal-Bench agentic-OPD.

## Context

Difficulty-calibrated the generated c3 pool (`filter_inband.py`: 5/6 hard
families land in the 20–60% band), seeded a STaR loop from the +5pp format LoRA
(`INIT_LORA`) to isolate the *capability* gradient, ran round0→round1 on 60 c3
tasks × pass@3.

## Result — flat, not a plateau break

Same-task pass@1 A/B (n=60 common tasks): round0 **75%** (45/60) → round1
**73%** (44/60) = **−2pp**, 11 improved / 12 regressed = noise-level wash. A
mid-run 86% partial was pure ordering bias (easy families evaluated first).

## Root cause

Self-distilling a **75%-already-solved** pool carries no capability gradient —
the passing trajectories are of tasks the seeded model already solves (the
SWE-Pro Δ≈0 lesson, `wins/2026-07-07-terminal-bench-opd-format-distill-lift`).
Per-family the calibration DID produce a 20–60% band, but at **pass@1 per-task**
the format-seeded model clears 75%, so the sweet-spot band FOR THE SEEDED MODEL
is too thin. Calibration fixed the substrate's difficulty *spread*; it did not
change that self-distillation caps once the model already passes the substrate.

## Rule

- **Smooth-RL (in-process actor-learner, WS1-5) is worth building only once the
  loop demonstrably LIFTS.** Both STaR and the co-resident agentic-OPD mode
  currently wash (STaR −2pp here; co-resident's only capability run was on a
  ceilinged held-out set, +1/24 noise). Making a wash faster optimizes nothing.
  WS1 (replay buffer) + WS5 (LoRA TP-shard math) cores are landed and reusable;
  the rest is **parked** until a lifting loop exists.
- **The next lever is a teacher or a lower-recalibrated band, not more
  self-distillation.** Beyond format, lift needs a stronger per-step
  distribution (⑤ GKD: think-on self / DSv4-Flash teacher) OR tasks recalibrated
  to pass@1 40–60% FOR THE SEEDED MODEL (not 75%). Calibrate against the model
  that will actually roll out, not the base.
