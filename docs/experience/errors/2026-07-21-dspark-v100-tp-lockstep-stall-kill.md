# DSpark KILL on V100 (sm_70) — TP lockstep stall at world_size=1

## Context

DSpark/DFlash speculative decode was enabled on V100 with
`--spec-type dspark --mtp-draft-model z-lab/Qwen3.6-27B-DFlash`
(DFlash drafter, block=16, taps=[1,16,31,46,61]) to measure the spec-decode
gain on the V100 baseline (Qwen3.6-27B-W4A16, no-spec ITL p50 40.4 ms at c=1).

## Root Cause

DSpark's `tp_lockstep_proposal/accept` mechanism was designed for TP≥2 (H20
pod). On TP=1 V100 (`world_size=1`), the coordinator's lockstep barrier stalls
waiting for cross-rank acks that never arrive:

```
WARN infer_server::coordinator: [coordinator] lockstep stalled:
  tick #2232128 awaiting acks (min_acked=2232124, elapsed=10.000306789s)
```

Each DSpark step (draft 16 tokens + verify) serializes through this stall,
adding ~460 ms/step overhead (ITL 40 → 499 ms). At c=8 the stall compounds
into request timeouts (8/8 errors in 1543 s); at c=16 the client-side
connection retry storm produces 131204 errors in 60 s.

## Fix

Not yet implemented. Two options:
1. **TP=1 fast path**: skip the lockstep coordinator entirely when
   `world_size=1` (no cross-rank sync needed).
2. **Lockstep timeout**: add a self-ack fallback when `min_acked` doesn't
   advance within N ms.

## Rule

DSpark on single-GPU (TP=1) is KILLED until the lockstep stall is fixed.
Multi-GPU TP≥2 DSpark results (H20) are unaffected.

## Resolution — 2026-07-25

Re-bench at HEAD 59b86ee4c (fresh on-box build, symbol-verified): **zero
lockstep-stall WARNs / zero errors over ~2 h of DSpark load** — the
world_size≤1 self-ack bypass is measured effective; stall root cause closed
(#168). DSpark-on-V100 stays KILLED on new grounds: the non-user-facing
8192-page pool floor OOMs 32 GB (#178) and sm_70 greedy output is garbage
(#179) — the −91% was mostly the sm_70 draft+verify path itself.
