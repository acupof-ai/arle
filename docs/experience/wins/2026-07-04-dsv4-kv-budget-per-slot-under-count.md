# DSv4 KV-budget per-slot under-count (43→382 MB) — pending-remote

> Status: pending-remote (CUDA-only; Mac typecheck clean, pod re-verify separate)

## Context

DSv4 `kv_budget_plan` (`crates/infer-cuda/src/dsv4.rs`) sized the per-slot KV
divisor by a hand-rolled `dsa_rotated + state_caches + dsa_batched` ≈ **43 MB**.
The REAL slot allocation (`Dsv4SlotState::device_bytes`) is **382 MB** at
seq_len 5120, MTP-on, TP=4/EP=4 — dominated by `spec_verify` (~282 MB), which the
budget missed entirely (so were `spec_normed`, `spec_rings`, `sw_window`, and the
FlashMLA/fused-wqkv scratch). Consequence: `affordable = (0.9×free − shared)/per_slot`
ran ~9× high (~471), the 256-slot request was never clamped, and the executor's
slot loop OOMed (`CUDA_OUT_OF_MEMORY`) at ~slot 31 during engine build.

## What Worked

Made the divisor the TRUE per-slot bytes via a single source of truth:
`Dsv4Model::per_slot_device_bytes(max_seq_len)` mirrors `Dsv4SlotState::new`
statically from config (the budget runs before any slot/`kv_adapter` exists), each
sub-component gaining a `device_bytes_for`/`estimate` adjacent to its `::new`
(`Dsv4LayerAttentionState`, `Dsv4CompressorState`, `Dsv4FlashMlaDecodeState`,
`Dsv4FusedWqkvDecodeScratch`, `Dsv4SpecRingSnapshot`). Budget `per_slot =
per_slot_device_bytes + dsa_key_cache_band + dsa_batched` (the two per-slot terms
that live outside the slot struct but scale with num_slots); FP8 arena stays in
the shared pool. A runtime drift guard in `executor.rs` warns if
`slots[0].device_bytes()` diverges from `per_slot_device_bytes` by >5%, so this
under-count class can't silently return.

**Expected on pod:** per_slot ~382 MB, affordable ~30 (down from ~471), engine
build no longer OOMs; drift-guard silent (≤5%). Bench/needle re-verify remote.

## Rule

The KV-budget per-slot divisor MUST equal the real `Dsv4SlotState` allocation,
computed from the same config the constructor uses — a hand-rolled subset silently
drops the dominant term (`spec_verify`) and inflates `affordable` ~9×. Enforce
with a runtime drift guard against `slots[0].device_bytes()`.
