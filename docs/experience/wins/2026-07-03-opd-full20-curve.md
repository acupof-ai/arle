# Full 20-round agentic OPD curve — 438s → 11.75s/round (37.3×)

## Context

End-to-end validation of the two-day optimization series on a full 20-round
agentic-OPD run (up from the prior 10-round spot-checks), same shape as the
day's original baseline: 27B Qwen3.6-FP8, `--share-frozen-base`, 1 toy task,
temperature 0, LoRA attention-qv r16/α32, single H20 GPU. Raw per-round data:
[curve20_data.json](curve20_data.json).

## Results

Steady state (rounds 1–19, round 0 pays first-call kernel warmup):

| metric | value |
|---|---|
| mean wall-clock / round | **11.75s** (vs 438s baseline → **37.3×**) |
| forward | 2.10s |
| backward | 4.21s |
| rollout + tools + sandbox | 5.08s |
| LoRA sync + misc | 0.04s |
| loss, first 5 → last 5 rounds | 0.2651 → 0.0884 |
| round 0 (warmup) | 15.76s |

Every round `passed=1`. Loss descends monotonically apart from two on-policy
variance spikes (round 9: 0.272, round 14: 0.251) — expected trajectory noise
from the just-updated student sampling a new rollout, not a regression; the
trend resumes immediately after each.

Per-round timing is flat within ±0.3s across all 19 steady-state rounds
(forward/backward literally identical to 2 decimal places every round) —
confirms the writeback and rollout paths are shape-stable, not shape-lucky on
one measurement.

## Rule

- A 3-round spot-check licenses a fix; a 20-round run is what proves the fix
  holds under the loop it was built for — do the longer run before calling an
  optimization campaign closed.
