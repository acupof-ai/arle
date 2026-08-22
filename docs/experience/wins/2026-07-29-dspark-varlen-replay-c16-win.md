# DSpark wins at c=16 — varlen partial-accept replay closes the last serial term

## Context

Batching the draft
([entry](2026-07-29-dspark-batched-draft-across-slots.md)) took c=16 from −22.6%
to parity. The remaining per-slot serial term was the rollback: `restore_trunk`
plus a 48-layer conv/GDR replay of the accepted prefix, run once per slot per
tick because mean accepted `k = 1.21` against depth 5 — ~every chain rolls back.

nsys attribution over a c=16 window separated it from real prefill by kernel
duration: 64,526 `gated_delta_rule_prefill_recur` calls of 10-60 µs = **2,019 ms
= 6.7% of the wall**, against 480 calls of ≥150 µs (the 2048-token prefill
chunks). Plus 79,790 short `conv1d_prefill` launches — 0.5% of GPU time but
~1,000 ms of driver time at 12.5 µs per `cudaLaunchKernel`.

## What Worked

Both prefill kernels take an optional per-slot pointer table and row-length
array. Non-null selects a varlen form where `blockIdx.y` is one slot, reading
its own capture and ring for its own `row_len[s]` rows. One launch per layer
replays the whole batch: **1,536 launches per tick → 96.**

`dspark_accept_commit` drops to the pure host accept scan; the executor collects
the partially-accepted chains and runs one `dspark_rollback_batch` after the
commit loop. Nothing between them reads the trunk linear state or `seq_len`, so
the deferral is safe. The opt-in FlashQLA chunked path keeps its per-slot route.

## Measurement

Matched A/B, one binary, one session, GPU 0, ThinkingCap-Qwen3.6-27B-FP8 +
Qwen3.6-27B-DFlash, `bench-agent-32k-16x8`, 48 req/point, max_tokens 214,
seed 20260416, c=16.

|  | TPOT ms |
| --- | ---: |
| no-spec | 102.70 |
| **DSpark block 6** | **98.29** (−4.3%) |
| DSpark block 8 | 98.71 (−3.9%) |

TPOT clears the ±3% drift band. Gate exact=3 DET at 512/4k/16k, 0 errors.

Cumulative TPOT at c=16, from the pre-batching arm: 137.70 → 110.52 (block 6)
→ 101.86 (batched draft) → 98.29 ms (varlen replay).

Block size peaks at 6: 16 → 6 is +21%, 8 is −1.2% from 6, 4 collapses (its run
also OOM'd — spec state at c=16 fills a 96 GB H20).

## Still open

**The default gate stays `--spec-max-batch 1`.** This licenses c=16 only; c=2/4/8
are unmeasured and c=4 was the worst regime in the 2026-07-26 campaign. Flipping
the default needs the full c-sweep plus a fresh accept-rate reading, since a spec
Δ is only comparable within one accept_rate.

## Rule

**Separate a kernel's costs by duration before attributing them.** The same
kernel name carried both the 2048-token prefill chunks and the 2-token replays;
the aggregate said 11% of the wall and hid which half was addressable. One
histogram split it into 4.2% irreducible and 6.7% launch-bound.
