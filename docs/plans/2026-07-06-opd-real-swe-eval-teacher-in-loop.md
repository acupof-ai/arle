# Plan — Real SWE-Pro agentic OPD with teacher-in-the-loop rescue

> Status: Active — 2026-07-06 · Driver: ckl (OPD review P5). Continues G6 of
> [2026-07-03-agentic-opd-27b-capability-curve.md](2026-07-03-agentic-opd-27b-capability-curve.md).
> Pod-only run.

**Verdict:** plain RFT can't show a capability curve — synthetic corpora
saturate (27B 8/8→24/24), real SWE-Pro 0-accepts (no gradient). The licensed
path is teacher-in-the-loop rescue.

## Approach
1. **Stage ≥12 real SWE-Pro instances** with measured baseline in the 10–40%
   band (probe the untrained 27B first — never infer difficulty from design).
   Pod has 1 train + 3 eval ansible instances already.
2. **Think-on teacher rescue** — same 27B weights, thinking on (zero extra VRAM;
   precedent: the 2026-06-20 4B BFCL win). On a 0-accept task, a passing teacher
   trajectory becomes the rescue target.
3. **Writeback** — reuse `masked_writeback_ce_step_dispatch` (response-masked CE).
4. **Curve** — alternate self-rollout (accepts → self) and teacher-rescue
   (0-accepts → teacher); measure held-out pass-rate over rounds.

## Open questions
- Teacher-rescue accept rate at real-repo scale (decode trajectories at token
  level before trusting an aggregate — case-as-fact).
- VRAM: longer teacher CoT + writeback backward vs the known agent-OPD walls
  (`errors/2026-06-26..29`); mitigate with `--rollout-engine train` / offload env.

## Verify
Pod-only. Gate-zero = baseline probe on the ≥12 instances; then the alternating
loop with a held-out curve, multi-seed if lift <5pp.
