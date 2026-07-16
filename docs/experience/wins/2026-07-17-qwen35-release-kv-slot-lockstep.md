# Qwen3.5/3.6 device KV pool freed eagerly at release_kv_slot — planner never over-commits

> Status: pending-remote — bench + P4 smoke re-run gate on the H20 pod.

## Context

P4 smoke (Qwen3.6-27B FP8, in-process serve, 2 concurrent ~22-32K-token cc
sessions, pool 3437 pages) died engine-fatal:

```
TokenKVPool: out of pages (requested 2048 tokens / 128 new pages, available 6 pages)
```

`execution.rs:270` treats any step error as fatal → engine thread exits →
zombie serve.

## Root cause

Host/device KV pool desync on the Qwen3.5/3.6 arm. The engine frees a slot's
host pages eagerly at `free_slot_pages` (finish/park/requeue,
`infer-core/src/lib.rs:1003`) and notifies the executor via
`release_kv_slot` — but the CUDA dispatch (`infer-cuda/src/executor.rs`) only
implemented it for DSv4. The Qwen35 arm freed its self-allocated
`full_attn_kv` device pool LAZILY at the next occupant's position-0 prefill
(`executor/qwen35.rs` `submit_prefill_row`). Between a long session's finish
and slot reuse, the host admission pool over-reported free pages by the whole
dead slot, so `fit_plan_to_kv_pages` licensed prefill chunks the device pool
could not hold — fatal at submit.

## What Worked

- `Qwen35Executor::release_kv_slot`: return keepalive-parked pages + free the
  `full_attn_kv` slot at engine release time (idempotent with the
  prefill-start free and swap-out/swap-in frees).
- Regression test
  `plan_repair_defers_prefill_continuation_when_pool_nearly_exhausted`
  (infer-core): nearly-exhausted pool + queued chunked-prefill continuation →
  the plan defers the chunk; the un-repaired plan provably over-commits.

## Rule

A backend that self-allocates its device KV must free a slot at
`release_kv_slot`, in the same tick the engine frees the host pages — lazy
free at next occupancy breaks the planner's single-pool accounting.
