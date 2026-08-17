# CP=2 deadlock: divergent tp_sync_min in admission path — CUDA, 2026-08-17

## Context

T3.2b 2D KV ownership sharding (world=4, attn_tp=2, cp=2). The first
needle-ladder run deadlocked at tick #4 (first prefill): all 4 ranks stuck
in NCCL collectives, coordinator tore down after 120s.

GDB traces showed:
- Ranks 0,1 (cp_rank=0, senders): reached the GDN relay, blocked in
  `ncclGroupEnd()` on the attn_cp communicator.
- Ranks 2,3 (cp_rank=1, receivers): never reached the GDN relay — stuck
  upstream in the admission path.

## Root Cause

Under CP sharding (`kv_shard_spec().is_some()`), the prefix lookup is
skipped (`PrefixMatch::empty()`), but `attach_prefix_to_request` was still
called. With an empty match it issues two `tp_sync_min(0)` collectives on
the **global TP communicator** per admitted request (one in
`clamp_prefix_to_backend`, one at the end of attach). The admission
budgeting path (`try_admit_front_waiter`) also called
`clamp_prefix_to_backend` per candidate, issuing a third collective.

The admission loop (`while self.active.len() < running_cap`) can iterate a
different number of times across ranks (throttling, divergent radix
matches). When ranks 0,1 exit the loop and enter the forward pass (GDN
relay on the **attn_cp communicator**), ranks 2,3 are still inside the
loop waiting in `tp_sync_min` on the **global TP communicator**. Neither
side can proceed: ranks 0,1 wait for ranks 2,3 on attn_cp; ranks 2,3 wait
for ranks 0,1 on global TP. Cross-communicator deadlock.

The initial hypothesis (NCCL send/recv on the same stream) was wrong. The
NCCL grouping and send-before-compute changes made during the
investigation are correct improvements and stay, but they were not the
fix.

## Fix

Skip the entire prefix match/clamp/attach path when
`kv_shard_spec().is_some()`. Under CP the ring pass recomputes the whole
prompt — there is no prefix reuse. The empty-match attach was a no-op
that set `prefill_start_pos=0 / Prefilling{0}`, the same state a fresh
request already carries. Three sites changed:

1. `try_admit_front_waiter` budgeting: gate the `reuse_matched_len`
   computation on `kv_shard_spec().is_none()` (skips the per-candidate
   `clamp_prefix_to_backend` collective).
2. `try_admit_front_waiter` attach: `else if kv_shard_spec().is_none()`
   skips `attach_prefix_to_request` (2 collectives).
3. Planner recompute path: same gate on `attach_prefix_to_request`.

The CP-replication branch in `clamp_prefix_to_backend` (the
`kv_shard_spec().is_some()` tp_sync_min) became dead code — deleted.

## Rule

Per-request collectives inside a loop whose iteration count can diverge
across ranks will deadlock cross-communicator. Collective calls must live
at fixed per-step points (unconditional, same count on every rank), not
inside per-request admission/attach paths. Under CP sharding, prefix
reuse does not exist — skip the entire prefix path, don't call it with an
empty match.
