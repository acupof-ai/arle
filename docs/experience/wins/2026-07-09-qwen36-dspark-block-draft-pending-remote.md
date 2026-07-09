# Qwen3.6 DSpark block drafter — pending-remote

> Status: pending-remote — code landed, H20 gates + A/B not yet run.

## Context

`--spec-type dspark --mtp-draft-model <dir>`: DSpark/DFlash 5-layer block
drafter as an alternative draft source for the Qwen3.6 CUDA spec-decode path
(plan: [2026-07-09-dspark-dflash-spec-decode-qwen36](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md)).
Verify/rollback substrate reused verbatim; baseline (spec off) byte-identical —
taps cost one `Option` branch per layer.

## Pending gates (H20)

- Correctness: needle x3 + same-config-twice (correct-inference, not
  byte-vs-baseline); acceptance sanity (wrong tap/layout ⇒ acceptance ≈ 0).
- Perf A/B, OPD rollout shape (20–45K ctx, B=1, greedy): no-spec vs MTP-d2 vs
  DSpark backbone-only (z-lab) vs +markov (AEON). Kill: ≤1.15× vs no-spec.
- Known perf debt to re-measure: markov/confidence per-row H2D/D2H syncs,
  80 attn launches per draft block.

Checkpoints staged on pod: `/root/Qwen3.6-27B-DFlash` (backbone),
`/root/dspark-aeon` (+markov), `/root/dspark-fr` (full DSpark, speculators
format — needs conversion).
