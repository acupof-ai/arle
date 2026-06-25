# DSv4 Stage-B dynamic pool — num_slots ceiling LIFTED (boots at 32, was OOM at 16)

Status: budget payoff PROVEN on-device (boot, no OOM); completion correctness pending
TP=8 (a TP=4-only adapter blocker, not the crux). Commit `02e179f4` (coherent budget +
incremental draw, ckl design + my compile-gap finish).

## Context
DSv4 high concurrency was capped by per-slot `num_slots × max_seq` KV reservation
(num_slots=16 OOM, measured). The fix: dynamic shared MLA pool + page-table kernels
([spec](../../plans/2026-06-24-dsv4-stage-b-kernel-spec.md)). Foundation (pack + read
page-table, bit-identity) landed `5eab59ad`/`3020145b`; this is the activation.

## What worked
- **Coherent budget** (`Dsv4KvBudgetPlan`, dsv4.rs): `per_slot` = DSA+state only (MLA
  arena dropped); pool = the REMAINDER `MEM_FRACTION×free − weights − fixed −
  per_slot×num_slots`. So `total = weights + MEM_FRACTION×free` — no over-alloc (unlike
  the reverted greedy tweak that OOM'd).
- **Incremental draw**: `reset_flashmla_slot` no longer reserves the max_seq band;
  `flashmla_alloc_append` draws pages per growth step → slots fragment the shared pool.
- **Read flipped** to `Some(page_table)` + persistent `page_table_batched` (graph-safe).
- **Finish (mine)**: wired `Dsv4LayerKvLayout::new` to size the pool from the coherent
  remainder + relaxed the Stage-A `num_slots×max_seq` contiguous assert (admission-gated).

## Measured (pod, TP=4 on 4×H20, `--max-running-requests 32`)
- **BOOTS at num_slots=32, no OOM.** VRAM ledger: `weights 75289MB + adapter 16029MB +
  Σ 32 slots 5348MB` (= 167MB/slot, the small DSA+state), pool = remainder; used
  95495MB / free 2013MB. **num_slots=16 OOM'd before → ceiling lifted.**
- Completion HTTP 500 at tick #0: **"DSv4 O-LoRA grouped scale shape must be non-empty"**
  — a TP=4 adapter-sharding error (DSv4-Flash O-LoRA is built for TP=8), independent of
  the crux (errors before any KV draw). Completion correctness = **pending TP=8** (needle
  ×3 DET + c跑 when 8 GPUs free).

## Rule
The coherent budget is the load-bearing piece: pool = REMAINDER (not a separate
num_slots×max_seq allocation) keeps total ≤ MEM_FRACTION×free. The reverted tweak
over-allocated by allocating the pool on top of the reserved per-slot — same num_slots
math, opposite VRAM outcome. Always check `total = weights + MEM_FRACTION×free`.
Verify DSv4 end-to-end at TP=8 (the production shape) — TP=4 hits the O-LoRA blocker.
