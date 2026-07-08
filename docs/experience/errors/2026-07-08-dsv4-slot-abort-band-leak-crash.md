# DSv4 first-request crash at tight num_slots — per-layer FlashMLA pool sizing/mirroring, NOT a slot-abort leak

> **Correction (2026-07-08, same day):** the title and root-cause hypothesis
> below as originally written were WRONG — filed from a plausible-sounding
> mechanism (admission-reject doesn't release a reservation) without first
> reproducing and decoding the actual failing sequence. Ground-truth repro
> showed the crash fires on the **very first ever request**, no reject/abort
> involved at all. Corrected per CLAUDE.md §0 case-as-fact: decode the actual
> failure before generalizing a mechanism. **FIXED** — see below.

## Context

Discovered 2026-07-08 while verifying the FlashMLA per-layer KV budget fix
(`3ebc763f9`) on the H20 pod via `needle_gate.py` at a deliberately tight
admission boundary (see
`docs/experience/wins/2026-07-08-dsv4-flashmla-budget-needle-gate-pass.md`).
Initially misdiagnosed as an admission-reject-path reservation leak (plausible
from the log line alone, never actually traced). Re-investigated same day
with an actual pod repro + trace, which found the real cause below.

## Root Cause (confirmed via repro, FIXED)

Two bugs in `crates/infer-cuda/src/attention/kv_layout.rs`, both stemming from
the same fact: since the 2026-07-05 per-layer KV-budget fix (`3ebc763f9`),
each DSv4 layer's `flashmla_kv_pool` is sized to **that layer's own**
`flashmla_slot_pages` — layers are NOT uniform (in the production DeepSeek-V4-
Flash-FP8 checkpoint: 3 SlidingWindow layers at 2 pages/slot vs.
CompressedSparse layers up to 104 pages/slot at the tightest tested
`max_total_tokens`). Two call sites hadn't been updated for that
heterogeneity:

1. **`Dsv4KvAdapter::flashmla_total_pages()`** read `self.layers.first()`'s
   own pool capacity to report the host's `total_pages` for admission — but
   `fixed_pages_per_slot` (what `alloc_fixed_band` actually draws) is the
   MAX per-layer requirement across all layers. Layer 0 is typically a cheap
   SlidingWindow layer (2 pages/slot); `total_pages` from it (2) was smaller
   than `fixed_pages_per_slot` (104) on every boot at `num_slots ∈ {1,2}` —
   a **deterministic crash on the very first request**, not anything to do
   with reject/abort.
2. Once (1) is fixed, admission passes but exposes a second bug:
   **`Dsv4KvAdapter::mirror_slot_pages`/`prepare_kv_batch`** sliced the
   host's SHARED page-id list (sized to the largest layer) directly into
   each layer's own, smaller `TokenKVPool::mirror_band` — valid only for the
   largest layer; any smaller layer's own pool (`max_total_pages` as low as
   2) got fed out-of-range host ids (`mirror_band page 103 out of range 2`).

## Fix (landed)

Both in `crates/infer-cuda/src/attention/kv_layout.rs` (~30 lines, 3 hunks):

- `flashmla_total_pages()` now maximizes over `self.layers` by
  `flashmla_slot_pages()` instead of taking `.first()` — tracks the same
  layer `fixed_pages_per_slot` already maximizes over.
- `mirror_slot_pages`/`prepare_kv_batch` now derive each layer's own
  contiguous LOCAL id range (`[slot * layer_slot_pages, +n)`) instead of
  slicing the host's shared id list — safe because DSv4 has no page-radix
  reuse (`reusable_prefix_blocks` was `0` for DSv4 as of this fix; a page id
  carries no cross-layer/cross-request meaning for this model).

## Verification (PASS)

Re-ran the exact repro on the rebuilt binary (TP=4, GPUs 4-7,
`max_total_tokens=26000` → `num_slots=1`): 3 cycles of
[oversized-prompt reject (clean abort, "needs 282 KV pages, pool has 104
free") → normal prompt on the same slot] all succeeded with correct output,
server stayed alive throughout. `needle_gate.py` sanity (500/2000-token
lengths, 3 runs each): exact match 3/3 at both lengths (2000-token run shows
the documented MoE non-determinism floor across runs, not a defect).

## Rule

**Decode the actual failure before generalizing a root-cause mechanism** —
"admission-reject doesn't release a reservation" was a plausible story that
fit the observed crash log, but nobody had traced whether a reject/abort
even occurred before the crash. It hadn't; the crash fired on the first
request, deterministically, from a completely different bug class
(per-layer pool sizing/addressing heterogeneity introduced by an earlier,
correct fix). Same lesson as `2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`'s
case-as-fact rule, now confirmed a second time on the same feature area.

**Secondary rule, still valid**: a per-layer-heterogeneous budget/sizing fix
(like `3ebc763f9`) can expose OTHER code that assumed uniform per-layer sizes
and never got updated — `flashmla_total_pages`/`mirror_slot_pages` both
silently assumed "layer 0's pool = every layer's pool," which was true only
under the OLD uniform-divide budgeting. Any future change that makes
per-layer state genuinely heterogeneous should grep for `.first()`/index-0
shortcuts across the adapter, not just the budget function itself.
