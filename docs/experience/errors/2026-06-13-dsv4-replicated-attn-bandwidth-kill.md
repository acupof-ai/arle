# DSv4 replicated decode attention — correct but −48% B=1; the "B=1 attention is free" premise was false

## Context

Beyond-44 lever: at B=1 decode, DSv4 spends ~86 attention collectives/token
(Q all-gather + output all-reduce, latency-bound). The replicated-attn idea
(`f2601ba6`, opt-in `ARLE_DSV4_REPLICATED_ATTN=1`): each rank computes ALL 64
heads from full-width weights and skips both collectives — premised on "B=1
attention compute is cheap, so replicating it is nearly free and the
collective saving dominates."

It had never run. First serve crashed: `wo_a cols 4096 != local width 32768`.

## Root cause #1 (fixed, `c4328756`): grouped o-projection

DSv4's o-projection is GROUPED — `o_groups=8` (== TP), `wo_a` rows =
`o_groups*o_lora_rank=8192`, `wo_a` cols = `heads_per_group*head_dim=4096`,
Column-sharded so each rank normally owns exactly one group. Sharded works by
coincidence (`sharded wq_b.rows 4096 == wo_a.cols 4096`); replicated needs all
8 groups applied block-diagonally + full-width `wo_b`. The original code
assumed a single full-width `wo_a`. Fix: load `wo_a` as 8 per-group matrices
+ full `wo_b`, loop the proven `dsv4_linear` per group, then full `wo_b`.
**Correctness confirmed**: needle 512 exact-DET / 2048 partial / 6000 exact
(the locked envelope) — the grouped o-proj and global-head-order are right.

## Root cause #2 (the KILL): the premise is false

| arm | B=1 tok/s | TPOT |
|---|---|---|
| baseline (FP8 lane + NUMA) | 44.5 | 22.5ms |
| **replicated-ON** (×3, clean serial) | **23.13 / 23.14 / 23.14** | **43.2ms** |

**−48%.** The replicated compute is NOT free:
- The grouped o-projection is 8 scalar FP8 GEMVs/layer (each streams ~4MB FP8
  weight at ~25% HBM) ≈ **14ms/token** across 43 layers — the same scalar-GEMV
  bandwidth floor that killed the MoE GEMV lane
  (`2026-06-13-dsv4-decode-gemv-lane-bandwidth-kill`).
- Plus every rank now runs the FULL 64-head decode attention (8× the heads)
  and the full-width Q projection.

These ~20ms/token far outweigh the few ms saved by dropping 86 latency-bound
collectives. KILL as default; the lane stays opt-in + correct.

## Rule

- **"Cheap per-token compute → replication is free" must be measured, not
  assumed.** Replicating shifts O(heads) compute onto every rank; at TP=8 that
  is 8× the attention + a grouped o-projection whose scalar-dequant bandwidth
  (25%) dwarfs the collective latency it removes. Same bandwidth lesson, second
  lane.
- **Salvage path (if ever revisited):** a vectorized/uint4 grouped o-projection
  kernel (the FP8-decode-lane trick) would cut the 14ms to ~3.5ms; the
  irreducible 8×-attention cost would still have to clear the collective saving
  on a binding shape before any default flip. Lower priority than d2 spec
  (already +20% → 53.3).
- The grouped-o-projection correctness fix (`c4328756`) is kept: it makes the
  opt-in lane correct (was crashing), and is the substrate for the salvage.
