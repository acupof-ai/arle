# DSv4 Route A deletion + device-table dirty-bit contract — pod acceptance 6/6 PASS

## Context

#154's warm-cache needle regression (solo miss up to 98.3% at the pre-fix
HEAD, bisected: origin `0198c3ba7` + amplifier Route A page-sharing series /
`e05a467e6` padding) was fixed by the two-commit series `bbaaea93b` (delete
the entire Route A prefix-reuse machinery, +67/−1553 — pools, tiers,
restore path, `host_to_flashmla` translation, all of it; whole-slot park
kept) + the dirty-bit coherence contract (host tables carry only real
pages; per-(slot,layer) device page tables refresh whenever the host band
changes, prefill AND decode — decode never resynced before). Simplification
sweep `b6d5bd887` (behavior-neutral) followed. Acceptance ran at
`91981b737` (the byte-identical pre-rewrite stack; later commits are the
verified-equal rewrite + a Qwen-lane kv-tier pin fix + the neutral sweep).

## What Worked (TP=4 H20, DeepSeek-V4-Flash-FP8, one boot per lane)

| Lane | Result |
|---|---|
| V1 solo pt=462 ×15, default | 15/15 exact, 14/15 byte-identical to baseline |
| V2 solo ×15, prefix cache off | 15/15 exact, 14/15 byte-identical |
| V3 concurrent n=4 unique ×15 (60 req) | 60/60 exact on the clean boot (attempt 1 died to a foreign cleanup kill; its 26 req included ONE `738391` substitution — non-reproducing, watch-list) |
| V4 concurrent n=2 ×30 (60 req) | 60/60 exact — cleaner than the historical pre-window #150 rate |
| V5 budget | dead pools 105+86+1 MB → 0; pool_total 7187→7380 MB (+193 reclaimed); num_slots 209→209 (clamp binds elsewhere) |
| V6 park (`--kv-oversubscription`) | boots, 2/2 exact — direct reuse intact |

Residuals carried forward, named: ① ~1/15 solo wording-variant (bare
`'738291'`, correct digits, MoE-nondeterminism class, uncorrelated with
prefix cache); ② 1/146 concurrent single-digit substitution in the
SIGTERM-truncated attempt only — same class as pre-window #150, not a #154
regression; re-baseline the `ARLE_DSV4_PROJ_BATCHED_BF16` A/B on this
now-clean baseline before drawing #150 conclusions.

## Rule

- **Delete the mechanism, not just its trigger** — gating reuse off while
  keeping live-path pool writes would have left the content-blind sharing
  running for nothing; full deletion made D2 (stale host→phys translation)
  structurally impossible rather than patched.
- **An acceptance lane must pin binary provenance** (tree commit + binary
  sha256 per run) — this acceptance's verdict transfers to the rewritten
  history only because the rewrite was verified byte-identical to the
  accepted tree.
- Shared-box hazard is real: a foreign agent's pre-launch `pkill` killed a
  mid-lane serve (attempt 1) — rerun on a fresh boot before reading any
  mid-lane anomaly as signal.
