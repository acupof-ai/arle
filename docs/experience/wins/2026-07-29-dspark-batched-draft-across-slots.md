# DSpark draft batched across slots — c=16 goes from −22.6% to parity

## Context

DSpark at c=16 was re-measured on the champion binary (KV mirror + batched FA3 +
MoE kernel), because every prior verdict predates all three. It still lost:
125.5 → 97.1 tok/s, TPOT 99.69 → 137.70 ms. The gate (`--spec-max-batch 1`)
was right, but the reason had never been located.

## What Worked

nsys over one per-slot draft, un-instrumented: **2.257 ms, 88.8% GPU busy, 113
kernels**. Not launch-bound — the draft head's MLP GEMMs take 52-54 µs each for
**6 rows**. A GEMM that narrow reads the whole weight matrix regardless of row
count, so B concurrent slots read the same draft weights B times per tick.

One forward over `B * block` rows reads them once. Only the two ring kernels stay
per-slot — they carry each slot's own K/V, which batching cannot remove. No CUDA
change: the GEMMs already take a row count, and the ring kernels take pointers
that offset into the batched buffers.

The argmax comes with it: one batched `argmax_rows_into` over every slot's rows
replaces B blocking D2H reads.

## Measurement

Matched A/B, GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash,
`bench-agent-32k-16x8`, 48 req/point, max_tokens 214, seed 20260416, c=16.

|  | TPOT ms |
| --- | ---: |
| no-spec | 99.69 |
| DSpark block 16 | 137.70 |
| DSpark block 6 | 110.52 |
| **block 6 + batched draft** | **101.86** |
| block 16 + batched draft | 131.23 |

Batching is **−7.8%** TPOT at block 6, −4.7% at block 16. Gate exact=3 DET at 512/4k/16k, 0 errors — the verify is exact, so
output is unchanged by construction.

Block size is now a live lever at concurrency (it was not on the old engine):
16 → 6 is +21%, 6 → 4 collapses to 72.5. The peak sits at 6-8, which is what a
fixed-per-tick overhead plus a row-linear verify predicts.

Inert at the shipped default: `--spec-max-batch 1` never puts two slots in a
tick, so the batched path needs `idx.len() >= 2` and never fires.

## Still open

Parity, not a win. The remaining per-slot serial term is the accept path's
`restore_trunk` + `replay_linear_only` — the same weight-bound redundancy on the
27B *target*, once per slot per tick. It runs whenever `k < depth`, which is
~every chain (mean accepted k = 1.21 against depth 5). In the nsys window
`gated_delta_rule_prefill_recur` is 3,351 ms = **11.1% of the wall**, and 16 of
every 17 calls are those per-slot replays. Batching it needs a varlen
recurrent-state replay across slots.

Accept economics, measured: block 6 commits 2.21 tok per 6 verify rows (0.37
tok/row) against plain decode's 1.0 tok/row. Speculation only pays because verify
cost is strongly sublinear in rows — 242 rows cost 1.54× what 92 rows cost.

## Rule

**A GEMM's cost is its weight traffic until the rows fill the tile.** Six rows
and ninety-six rows cost the same 52 µs, so any per-slot loop around a small
GEMM is doing B× the memory traffic for free. The instrumented split said draft
was 33% of the tick and the kernel timeline said the draft was 88.8% GPU-busy —
the phase counters inflate whatever they measure, because each lap is a sync.
Take the split from the timeline, not from the counters.
