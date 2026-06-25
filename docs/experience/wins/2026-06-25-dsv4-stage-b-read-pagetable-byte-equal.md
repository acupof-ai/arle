# DSv4 Stage-B Block A — read-side page-table routing, A/B-proven byte-equal to baseline

Status: landed + A/B-verified on pod (TP=4 EP=4, 8×H20). Block A is the
necessary precondition for the dynamic-pool payoff (Blocks B/C); it is byte-equal
to Stage-A baseline by construction (identity table → same physical indices) and
proven so by a same-prompt baseline-vs-treatment A/B.

## Context
DSv4 high-concurrency is capped because the MLA latent KV reserves a full
`max_seq` band per slot (`num_slots × max_seq`); at TP=4 max_seq=16384 the budget
clamps to **21 slots** (measured). The dynamic shared-pool fix needs the read path
to address the pool by a device page table (logical→physical), not by a per-slot
contiguous band base. The pack/write side already routed the table
(`flashmla_device_page_table`); the single-row eager **read** path did not
(`build_indices` passed `None`, decode kernel fed a per-slot `pool_buf.slice(range)`).

## What landed (Block A, `crates/infer-cuda/src/attention.rs`)
- `build_indices` (single-row eager decode) now routes the device page table for
  **all** modes → emits POOL-ABSOLUTE physical indices (was slot-relative `None`).
- The FlashMLA decode kernel is fed the **whole-pool base** (`flashmla_pool_data()`),
  not the per-slot band slice — matching the already-correct batched path
  (`attention.rs` batched ref feeds whole-pool base + table-routed indices).
- Dropped the per-slot `flashmla_pages_byte_range` slice + its `ensure!`.
- V32/GLM also route the table here: only the WRITE/pack side lacks a V32
  device-page-table kernel; the read-side table is mode-neutral and their
  contiguous identity band stays byte-equal.

## Verification (measured, pod TP=4 EP=4, `/host/DeepSeek-V4-Flash`)
- `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
  green (note: this Mac needs `CUDARC_CUDA_VERSION=12060` to skip cudarc's nvcc probe).
- Pod build `--features nccl` green; boots clean at num_slots=16, same budget line
  (per_slot 924MB — Block A does not touch the budget).
- **A/B (the gate)**: same prompts, baseline-binary (HEAD, no Block A) vs Block A
  binary, warm runs → **identical outputs**. Block A == baseline.

## Two false-alarm artifacts caught by case-as-fact (would have wrongly killed A)
1. **needle "738-only"**: with filler, the model reliably emits the needle prefix
   "738" but garbles the tail (`7382/738192/...`). **Baseline does the exact same
   thing** → pre-existing long-context tail-retrieval artifact of this checkpoint,
   NOT a Block A regression. Trivial repeat (`738291`), immediate-state, and
   repeat-3× all return the full 6 digits — the model can emit it; filler degrades
   the tail equally on both binaries.
2. **NONDET**: run 0 differs, runs 1–4 byte-identical → **cold-start** artifact
   (`reset_flashmla_slot` lazy first-use draw), not MoE nondeterminism.

## Probe protocol for Stage-B (the existing needle_gate.py can't grade this ckpt)
- `needle_gate.py` (max_tokens=16, exact-`738291` match) false-fails on BOTH the
  tail artifact and the cold start. Do not use it as the Stage-B gate here.
- Working gate: **discard run 0 (cold start), then warm-run byte-hash A/B vs the
  baseline binary** on the same prompts. Byte-equal-to-baseline is the gate, not
  absolute needle retrieval.

## Measured budget decomposition (the Stage-B payoff target)
`DSv4 KV budget: free 22219MB, per_slot 924MB (arena×2 784MB + rotated 21MB +
state caches 118MB)`, TP=4 max_seq=16384:
- arena×2 = 784MB = **85% of per_slot** → this is what Blocks B/C move into the
  shared pool / make incrementally-drawn.
- per_slot_remaining after decoupling = 21 + 118 = **139MB**.
- Stage-A ceiling: `(22219×0.9 − ~40) / 924 ≈ **21 slots**` (verified: requested
  32 → WARN clamp to 21 → boots, no OOM).

## Concurrency before/after (projection — payoff NOT yet measured)
- **Before (Stage-A, measured)**: 21 slots @ TP=4 max_seq=16384.
- **After (Block B+C projection)**: pool fixed, per_slot 139MB → conservative
  `~11GB / 139MB ≈ 79 slots` ≈ **3.7×**. This is a projection off the measured
  budget line; the real payoff requires Block C (incremental per-request draw, not
  the current `reset_flashmla_slot` whole-band draw) and is pending pod measurement.

## Honest remaining work (Block B/C)
- Block A alone is byte-equal and carries NO concurrency change — it is the
  precondition only.
- Block C is bigger than "size pool once": `reset_flashmla_slot:980` still draws
  the full `max_seq` band per slot (`want_tokens = slot_pages × page_size`). Real
  payoff needs **incremental per-request draw** + real `free_slot` on release, pool
  sized for aggregate real tokens. Cross-rank consistency is already handled
  (`num_slots` is the post-`all_reduce_min` value).

## Rule
A read-path addressing change that routes an identity table is byte-equal by
construction, but PROVE it with a same-binary baseline A/B before stacking the
fragmenting flip — and decode the actual cases: this checkpoint's needle tail +
cold-start run-0 are artifacts that will false-fail a naive exact-match gate.
