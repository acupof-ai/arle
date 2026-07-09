# DSv4 Route A shared-pool cost collapses TP=4 slot capacity 121→1 at max_total_tokens=16384

## Context

Surfaced 2026-07-09 as a side-finding while benching KV tier L2 vs L2+L3
(`docs/experience/wins/2026-07-09-dsv4-kv-tier-l2-vs-l2l3-guidellm.md`).
Quantified via pod measurement, not guessed.

## Root Cause (confirmed, not a sizing bug)

Route A's two always-GPU-resident shared pools (`Dsv4CompressStatePool`,
`Dsv4SwRingSnapshotPool` — commits `6a78a490d`/`95a2fab94`, no eviction
wired yet, a known deferred gap from step 6) cost **1,538 MB total** at
`max_seq_len=16384` on the production checkpoint (21 `compress_ratio==4`
layers, `sliding_window=128`, `head_dim=512`, `index_head_dim=128`):

| Pool | Bytes | Formula |
|---|---:|---|
| Compress-state (main+indexer) | 840 MB | `21 × (32 MB main + 8 MB indexer)` |
| DSA key-cache | 10 MB | scales with indexer layers |
| SW ring | 688 MB | `43 × 16 MB/layer` |

Every term scales **exactly linearly** with `max_total_tokens` (verified
across 5 points, no multiplier drift) — this is the design's stated
tradeoff (GPU-resident, sized for the full range, no eviction), not a bug.

**The collapse to 1 slot is a separate, pre-existing cliff, only newly
reachable.** `kv_budget_plan()` (`dsv4.rs:2077`) reserves nearly the entire
per-slot-affordable budget for slot state BEFORE sizing the shared FlashMLA
band pool, leaving a thin residual (`pool_budget_total`) computed as the
difference of two large near-equal numbers. Route A's fixed 1,538 MB
term only costs ~9 slots directly (113 affordable vs the historical 121),
but it tips that thin residual from "plenty" to 57 MB — which floor-divides
to exactly 1 slot's FlashMLA band. Between 4608 and 6144 tokens the
residual swings 1,365 MB → 57 MB (24× change over a 1.33× token-count
change): a genuine cliff, not a smooth tradeoff curve.

## Measured breakeven

| max_total_tokens | pool_total residual | affordable slots |
|---:|---:|---:|
| 2048 | 7,187 MB | 209 |
| 4608 | 1,365 MB | **79** |
| 6144 | 57 MB | 2 |
| 8192 | 45 MB | 1 |
| 16384 | 57 MB | 1 |

~4608 is the practical floor for reasonable capacity on this checkpoint at
TP=4; there is no smooth middle ground between 4608 and 6144.

## Fix

Not yet chosen. Three candidates, none evaluated for cost/risk yet:
1. Wire real eviction for the compress-state/ring pools (the already-known
   step-6 gap — would need the deferred `.cu` kernel change for a
   device-side page-table gather, per that step's own finding).
2. Shrink `capacity_blocks` (page the pools instead of sizing for the full
   `max_seq_len` range) — smaller always-resident footprint, same
   eviction-less model.
3. Reserve a floor for `pool_budget_total` before the per-slot reservation
   in `kv_budget_plan()`, so the FlashMLA band pool never gets starved to a
   near-zero residual regardless of how much the per-slot term grows —
   this is a pre-existing ordering issue independent of Route A and would
   also harden future fixed-cost additions from hitting the same cliff.

## Rule

**A fixed shared-pool cost interacts with an existing budget-allocation
ORDER, not just its own absolute size.** 1.5 GB (6.4% of free VRAM) sounds
survivable in isolation, but it collapsed capacity 121→1 because it landed
on the small-residual side of a difference-of-two-large-numbers
computation that was already thin before Route A existed. Before adding any
new fixed-cost term to a budget function, check what it's being subtracted
FROM, not just how big the term itself is.
