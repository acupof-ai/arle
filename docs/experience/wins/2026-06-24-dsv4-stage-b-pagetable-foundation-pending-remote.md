# DSv4 Stage-B — page-table kernel foundation (pending-remote needle; activation deferred)

Status: foundation committed + unit-proven; end-to-end needle pending-remote; the
dynamic-pool ACTIVATION (the num_slots payoff) NOT yet achieved. Honest state.

## Context
DSv4 high concurrency is capped because the MLA latent KV + DSA caches are reserved
per-slot (`num_slots × max_seq`) — num_slots=16 OOMs (measured). The architecture fix
([design](../../plans/2026-06-24-unified-kv-memory-architecture.md),
[spec](../../plans/2026-06-24-dsv4-stage-b-kernel-spec.md)) is a dynamic shared pool:
slots draw pages from one free-VRAM-sized pool via a device page table.

## What's done (committed, unit-proven)
- **Pack WRITE kernel** (`dsv4_fp8_kv_pack.cu`, `5eab59ad`): optional device page-table
  lookup; null = band, non-null = `page_table[logical]`. **12/12 bit-identity tests
  pass on the pod** (identity table == band, nvcc green).
- **build_indices READ kernel (batched)** (`dsv4_flashmla_decode_build_indices.cu`,
  `3020145b`): same page-table lookup; emits physical indices. Bit-identity test
  (identity == band, algebraically proven). The decode shim + vendor kernel + batched
  path read these indices off one global pool base — no change; CUDA-graph-safe.
- Callers currently pass identity/contiguous tables → **byte-equal to Stage-A**.

## What's NOT done (the activation = the num_slots payoff)
1. **Single-row eager decode** still band-relative (`attention.rs:6496` calls
   build_indices with `None`, `:6507` slices a per-slot band) — must flip to global
   base + physical indices.
2. **Graph-safe PERSISTENT page table** — `flashmla_device_page_table` does a fresh
   `clone_htod` per call; the decode is CUDA-graph-captured, so it needs a persistent
   device table whose contents are stamped before replay.
3. **Pool sized ONCE from free-VRAM** (currently `num_slots × tokens_per_slot`).
4. **Coherent budget** — the naive shortcut (drop the MLA arena from `per_slot`, keep
   the pool `= tokens_per_slot × num_slots`) **OVER-allocates → OOM** (verified by
   arithmetic: total ≈ `0.9·free + arena·num_slots > free`). The pool must be a FIXED
   shared buffer; `num_slots = (free − weights − pool − fixed) / per_slot_remaining`
   where `per_slot_remaining` = DSA + workspace only (small → high num_slots).
5. **Fragmentation** — non-identity draw from the shared free-stack.

## Bench
- Default = byte-equal Stage-A (page-table path active with identity tables); no perf
  change yet. End-to-end needle ×3 DET at the OLD num_slots = **pending-remote** (DSv4
  serve needs 8×H20; GPUs 0,1 were busy). num_slots-scaling payoff = pending the
  activation above.

## Rule
Two activation shortcuts failed: Phase-2 kept contiguous bands (byte-equal, no payoff);
the budget tweak over-allocates (OOM). The real activation is the SGLang shared-pool
design — fixed pool + small per-slot index/state + fragmented draw + a graph-safe
persistent device table. Don't ship a budget that isn't VRAM-coherent at high num_slots.
