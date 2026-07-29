# CP N=2 NCCL wedge was per-process HashMap param order, not a comm bug

## Context

First real CP N=2 run (synthetic-writeback seq 8192, cp-size 2) hung after
`phase=backward`: no loss printed, both GPUs pinned ~100%, no NCCL WARN, first
all-reduce of the step. Forward/backward collectives (48 all_gather + 48
reduce_scatter) had completed lockstep, so the sharding math was sound. Prior
session filed the hang against `all_reduce_cp_grads` (opd.rs:3304) as the
suspect — a hypothesis, not a root cause.

## Root Cause

`all_reduce_cp_grads` iterates `trainable_params` and issues one NCCL
`all_reduce` per param **by position**. That position order came from
`param_ids`, built by `register_linear` (qwen35.rs) iterating
`LinearWithLora::adapter_name_map()` — a **2-entry `HashMap{lora_a, lora_b}`**.
Rust seeds `HashMap`'s hasher from per-process RNG, and each CP rank is a
separate `current_exe()` re-exec (`train_multiproc.rs`), so every rank drew an
independent lora_A-vs-lora_B order. Two ranks then reduced the **same 64 params
in different orders** — at the first swapped slot one rank issued all_reduce on
`[16,5120]` (lora_A) while the other issued `[12288,16]` (lora_B).

NCCL has no size/shape rendezvous: mismatched element counts into the same
collective make every rank spin on the GPU forever. The **CPU** loops raced
past all 64 params and printed `DONE` on both ranks — because
`all_reduce_sum_device` only *enqueues* async work on the stream. That async
enqueue is exactly why the trace showed both ranks "finishing": the deadlock is
on the device, not the host.

The trace (temporary logging of per-rank `idx→(param_id,shape)`) was decisive:
both ranks logged `total_params=64` and identical param *sets* (sorted-set diff
empty) but the per-idx pairing diverged at 30/64 positions, first at idx=4 — a
pairwise A↔B swap within one LoRA module. That killed the "3304 deadlock loop"
hypothesis (the loop is order-faithful) and localized it upstream to param
registration.

N=1 never hit this: single process (one consistent order) AND the reduce is
gated `cp.is_enabled() && step_optimizer`, so the collective path first executes
only at N≥2.

## Fix

Deterministic order at the source, not a sort at the all-reduce entry.
`LinearWithLora::adapter_ordered()` returns fixed A-then-B `Vec`; the two
param-ordering callers (`register_linear`, `collect_linear_ids`) use it. The
model-level `adapter_name_map()` stays a HashMap — its callers are name-keyed
(checkpoint/sync/EMA), not position-keyed, so their iteration order never
reaches a collective. Commit b8e2ad96b.

## Rule

Any per-rank list handed to a positional collective (CP/TP grad all-reduce,
all_gather, reduce_scatter) MUST have identical order on every rank. A
`HashMap`/`HashSet` anywhere in that list's construction is a latent
cross-process NCCL deadlock, because Rust randomizes hash iteration per process
and each rank is its own process. Symptom is unmistakable and misleading: GPUs
pinned, no NCCL WARN, and the **host** side reports "done" on all ranks (async
enqueue) while the device wedges. When a collective hangs with matching param
*counts*, check the per-rank param *order* before suspecting the comm layer.
