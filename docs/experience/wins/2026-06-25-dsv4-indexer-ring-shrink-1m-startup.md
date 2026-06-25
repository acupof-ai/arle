# DSv4 indexer.compressed → ring staging — 1M startup-budget fix — pending-remote

`pending-remote`: 8×H20 DSv4 (TP8). Commit `80aae0fa` on main; this entry is the §Benchmarks gate.

## Context
ckl tried `max_total_tokens=1048576` (1M) and DSv4 engine **build rejected at startup**: not a
throughput issue — `kv_budget_plan` computed `per_slot` so large that 0 slots are affordable. Root
cause: the per-slot state that scales linearly with `max_seq_len`. The audit (2026-06-25) found the
MLA latent + DSA key-cache/rotated-keys are ALREADY paged (Phase-3), but the **indexer + compressor
compressed caches** were left as per-slot `max_seq`-length buffers. For GLM `SparseIndexed`
(`index_ratio=1`, full length) the indexer term dominates: `max_seq × index_head_dim × 2` per layer
≈ 256 MiB/layer @ 1M; summed over all indexer layers ≈ **59 GB/slot** → startup rejection.
(dense Qwen3 / Qwen3.6 / Metal are 1M-safe: KV paged, recurrent state fixed-size.)

## What Worked
`indexer.compressed` is **write-once / drain-immediately**, not retained history: `csa_select_official`
reads ONLY the contiguous delta tail `[packed_rows..rows_after]` each forward and drains it into the
already-paged DSA pools (`dsa_kv_pool`/`dsa_rotated_pool`), advancing `packed_rows`; the guard
`ensure!(packed_rows <= indexer_rows_before)` proves the unpacked lag ≤ one forward. Old rows are
never re-read — the history lives in the paged pools, so the full-length buffer was **redundant**.

Shrink it to a **staging ring** (`2 × prefill chunk = 8192` rows, fixed regardless of `max_seq`):
- Indexer term: **256 MiB → 2 MiB per layer (−99.2%)**; for all-SparseIndexed GLM the dominant
  `state_caches_per_slot` contributor collapses to a constant → `per_slot@1M ≈ 0`.
- **Offset-0 window-relative staging** (window base = `start_pos`) keeps the delta contiguous for
  non-chunk-aligned MTP/decode → **NO kernel change** (the read recovers row `r` at `r − base`).
- Decoupled `compressed_capacity()` so the selector's logical bound (`available = min(key_count,
  abs_pos/ratio)`) is unchanged after the physical buffer shrank (a landmine: the graph-replay path
  inferred capacity from buffer length).
- Gated to `SparseIndexed` only; `CompressedSparse` + `compressor.compressed` (CSA/HCA, need full
  retention) are **byte-for-byte untouched**. Selection output (`selected[]`, DSA block selection)
  byte-identical — the ring only relocates *where* the delta stages, not the values read.

`dsa_batched_per_slot` evaluated separately: max_seq-linear only in its `logits` term (`cc×4`,
per-slot one row, NOT × layers) ≈ 4-5 MiB/slot @ 1M → negligible, not a blocker.

## Rule
- **The 1M wall is per-slot `max_seq`-linear allocation, not spill/concurrency.** Allocation happens
  at startup, before any runtime spill — a 59 GB/slot term rejects the budget upfront; whole-slot
  spill (G3) cannot help (it swaps already-allocated state). The fix is paging/shrinking the
  allocation itself, model-by-model. dense/Qwen3.6/Metal already comply; DSv4 was the violator.
- **A write-once/drain-immediately staging buffer sized at `max_seq` is redundant — shrink to a ring,
  don't page it.** The indexer's history is in the paged DSA pools; the staging only needs the
  current forward's delta. This is the *cheaper* fix than paging (no kernel change) and is the
  asymmetry that distinguishes the indexer (shrinkable) from the compressor (needs full retention →
  block-table kernel change, separate follow-up).
- Verify the per-slot term actually dropped in `kv_budget_plan`, not just the buffer — the budget
  rejection is the symptom.

## Open
- Compressor `compressed` (CSA/HCA) — full-retention, needs block-table kernel args (`compressed_page_
  table`): in progress separately.
- Pod verify: 1M startup no longer rejected (`per_slot` collapses) + DSv4 needle byte-unchanged (8×H20).
