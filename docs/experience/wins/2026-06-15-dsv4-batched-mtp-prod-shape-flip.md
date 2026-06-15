# DSv4 batched MTP decode — default ON (prod-shape licensed, +77% @c=8)

## Context
batched MTP fold WIN (+81% @c=12 short-prompt,
[fold win](2026-06-15-dsv4-batched-mtp-fold-win.md)) was gated OFF
(`ARLE_DSV4_BATCHED_MTP`). ckl: deploy it — verify at the production prompt shape
(~2300-tok) across concurrency, then default-flip. CLAUDE.md: decode default flips need
multi-shape verification.

## Prod-shape A/B (pod 8×H20, same binary, ~2400-tok prompt, --num-slots 16, --spec-type mtp)
| c | batched MTP | per-row MTP | Δ |
|---|---|---|---|
| 4 | 47.94 (avg_active 2.3, noisy) | 41.66 (avg 4.0) | — |
| **8** | **76.68** (avg 7.8) | **43.37** (avg 7.9) | **+77%** |
| 12 | 78.65 (avg 11.7) | 46.11 (avg 7.0, plateaued) | +71% |

The c=8 row is the clean matched comparison (both arms avg_active~7.8): **+77%**. Same
mechanism as short-prompt: **per-row MTP plateaus ~42-46 tok/s (can't sustain >7-8
concurrent — sequential per-slot spec_step); batched scales to ~77** (one amortized wave).

## Correctness at prod shape
Concurrent distinct-word-needle, ~2400-tok context (filler=120), c=8: **7/8 own,
cross-contam=0** (the 1 miss is the shared uncommon-word recall limit; LONGER context did
BETTER than short — 7/8 vs 6/8). No cross-slot contamination at prod context length.

## The flip
`dsv4_batched_mtp_enabled()` default ON (opt-out `ARLE_DSV4_BATCHED_MTP=0`). The executor
gate `spec_on && this && rows >= dsv4_batched_decode_min_rows()` engages batched MTP at
**spec + c>=4** only; c<4 keeps the per-row MTP latency path. The marginal batched-draft
sub-lever (`ARLE_DSV4_BATCHED_MTP_DRAFT`, lever 2a, +2% noise-floor) stays OFF. **The
production `--spec-type mtp` serve now batches decode at c>=4 by default → +77% @c=8.**
Multi-shape licensed: short c=12 +81% + prod ~2400-tok c=8 +77%, both coherent.

**Default-engage confirmed**: served with `ARLE_DSV4_BATCHED_MTP` UNSET (the default) +
`--spec-type mtp` + sustained c=8 → `[dsv4-mtp-batched]` fired **547×** (per-row 334 = the
c<4 ramp/straggler phases). The flip engages by default under sustained concurrency.
(A short-gen needle check `max_tokens=12` showed 0 batched lines — too short to form a
rows>=4 decode wave; the gate correctly stays per-row at c<4. The flip helps SUSTAINED
high-concurrency decode, which is the production serving regime.)

## The full arc (this session)
per-row MTP (gated batched lane disabled) → batched lane wired (Phase A/B FlashMLA,
+58% vs per-row at c=8) → batched MTP (the 理想态): correct-but-regress (sub-mode2 attn
−44%) → tree-attn fix (−32%) → fold commit (THE win, +81%) → correctness verified
(needle, no contam) → principle (verify-compute-bound, per-row plateaus) → lever 2a draft
(marginal, 1-layer MoE) → **prod-shape license → default ON**. DP-attn deferred (prior
doc: 3-4 weeks / ~2.7% decode / scheduler-crux; it's the prefill/scaling track, #3).

## Rule
- **A gated win isn't shipped until multi-shape-licensed + default-flipped.** The +81%
  sat gated; the value is realized only when the production serve takes it by default —
  which requires the prod prompt shape (not just the smoke shape) clearing the bar
  ([[feedback_kv_features_default_on]], CLAUDE.md multi-shape rule).
- **plateau-vs-scale holds across shape.** per-row's ~42-46 ceiling and batched's ~77
  scaling reproduce at ~2400-tok (decode is O(topk) sparse → context-flat); the win is
  the parallelism (one amortized wave vs sequential per-slot), not the prompt length.
