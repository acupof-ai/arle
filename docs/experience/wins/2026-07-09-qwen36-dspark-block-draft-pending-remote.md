# Qwen3.6 DSpark block drafter — correctness PASS, perf attribution pending

> Status: pending-remote — correctness gates PASS on H20; net perf is a LOSS
> pending fixed-cost attribution (below).

## Context

`--spec-type dspark --mtp-draft-model <dir>`: DSpark/DFlash 5-layer block
drafter as an alternative draft source for the Qwen3.6 CUDA spec-decode path
(plan: [2026-07-09-dspark-dflash-spec-decode-qwen36](../../plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md)).
Verify/rollback substrate reused verbatim; baseline (spec off) byte-identical —
taps cost one `Option` branch per layer.

## Gate results (H20, GPU 1, binary @ 4b4e1905f, backbone-only z-lab)

- **Correctness PASS**: load `mode=dflash-backbone block=16 taps=[1,16,31,46,61]`;
  same-config-twice byte-identical; needle exact; spec-off output coherent.
- **Acceptance real**: 2.79–5.41 tok/step (code prompts, greedy).
- **Net perf LOSS**: dspark 11.2–18.5 tok/s vs no-spec 41.8–42.8 tok/s.
  Step cost ~**250 ms/step constant** (249/251 ms across prompts) vs
  23.4 ms/step no-spec — a fixed per-block cost, not acceptance-dependent.
  Break-even needs ≤108 ms/step at the observed 4.63 tok/step.

## Next: attribute the 250 ms fixed cost

Hypotheses to measure (env-gated phase log, ARLE_MTP_PHASE-style): draft-block
per-row argmax D2H syncs (~15/step), 80 per-row attn launches/step, verify
16-row eager forward, append_ctx feat path, per-step H2D uploads + `ctx.sync`.
No conclusion until the phase table lands.

Checkpoints on pod: `/root/Qwen3.6-27B-DFlash` (backbone, complete),
`/root/dspark-aeon` (+markov), `/root/dspark-fr` (full DSpark, speculators
format — convert with `scripts/convert_dspark_speculators.py`); fr/aeon
downloads in flight.
