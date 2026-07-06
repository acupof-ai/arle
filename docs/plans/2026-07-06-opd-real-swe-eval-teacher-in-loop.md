# Plan — Real SWE-Pro agentic OPD with teacher-in-the-loop rescue

> Status: Active — 2026-07-06 · Driver: ckl (OPD review P5). Continues G6 of
> [2026-07-03-agentic-opd-27b-capability-curve.md](2026-07-03-agentic-opd-27b-capability-curve.md).
> Plan only; the run itself is `pending-remote` on the 8×H20 pod.

## Context / problem

Agentic OPD (`agent-opd`) has **loop stability** evidence (37.3× speedup,
20-round monotonic loss) but **no capability-lift evidence**. The
2026-07-03 kill established why: a synthetic small-repo bug-fix corpus
saturates (27B baseline 8/8 → 24/24 across five difficulty escalations —
any task small enough to stage cheaply is one-shot for a 27B), while real
SWE-bench-Pro instances hit a **0-accept wall** (explore-forever, tool-schema
misuse) — and rejection-sampling RFT has **no gradient at 0 accepts**.

So the capability regime that matters (real-repo scale, spec-driven fixes)
is exactly where plain RFT starves. The licensed path (that kill's G6) is
**teacher-in-the-loop**: a think-on teacher generates rescue trajectories on
the 0-accept tasks, distilled via the existing masked-CE writeback.

## Approach (decomposed)

1. **Stage ≥12 real SWE-Pro instances.** The pod already has 1 train + 3 eval
   ansible instances (269 MB, from the 06-27..30 campaign). Extend to ≥12 with
   measured baseline pass-rates in the 10–40% band (the mid-band gate — probe
   the untrained 27B *first*, never infer difficulty from task design; five
   design-level "harder" judgments were wrong in the kill).
2. **Think-on teacher rescue.** Same 27B Qwen3.6-FP8 weights, thinking enabled
   (zero extra VRAM — precedent: the 2026-06-20 4B BFCL think-on win). On a
   task the base student 0-accepts, the think-on teacher attempts it; a passing
   teacher trajectory becomes a rescue target.
3. **Writeback.** Reuse `masked_writeback_ce_step_dispatch` (response-masked CE
   over the rescue trajectory) — no new writeback path. The `response_mask`
   marks LLM tokens (1) vs tool/env tokens (0), grad scaled by `1/total_targets`.
4. **Curve.** Alternate self-rollout (accepts → self-writeback) with
   teacher-rescue (0-accepts → teacher-writeback); measure held-out pass-rate
   over rounds. Success = a rising curve on the real held-out eval slice.

## Prerequisites / open questions

- Teacher-rescue acceptance rate on real instances — if the think-on teacher
  *also* 0-accepts at real-repo scale, the lane is still starved (decode the
  teacher trajectories at token level before trusting an aggregate — case-as-fact).
- VRAM: think-on teacher shares the student weights but its longer CoT rollout
  + the writeback backward stack against the known agent-OPD tape/forward walls
  (`errors/2026-06-26..29`); the engine-offload knobs (now `--engine-offload`)
  are the mitigation.

## Verification

Pod-only. Baseline probe (untrained 27B on the ≥12 instances) is the gate-zero.
Then the alternating loop with a held-out pass-rate curve; multi-seed if the
lift is <5pp on small-n. Bench/wins entry per §Benchmarks once a curve exists.

## Salvage already banked

The 16-round run on the saturated corpus is a clean **stability** read (loss
falling, pass-rate flat at baseline, zero OOM after the tape-margin fix) — the
synthetic corpus + harness stay as the smoke/regression gate.
