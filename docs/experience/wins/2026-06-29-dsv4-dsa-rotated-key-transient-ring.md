# DSv4 DSA rotated-key cache — full-history mirror → transient drain-immediate ring (pending-remote)

## SLO-shape probed? — N (Mac type-check + clippy only; needle-gate + VRAM-budget validation pending-remote on H20)

## Context

The official DSA indexer kept **two** copies of every index key per (slot,
CSA-layer): the FP8 paged `dsa_key_cache` (132 B/tok, what the paged-MQA-logits
kernel actually reads) **and** a bf16 `rotated_keys` full-history mirror
(256 B/tok). A static audit (this session) of all 37 `rotated_keys` references in
`crates/infer-cuda/src/` found the bf16 mirror is **drain-immediate, delta-only**:

- **Write (W1)** `dsv4_dsa_hadamard128_bf16_cuda` writes only the per-forward
  delta `[packed_rows .. +newly_packed)`.
- **Read (R1)** `dsv4_dsa_fused_store_index_k_cache_cuda` reads that **same delta
  in the same forward** to produce the FP8 cache.
- **No other consumer reads it.** The paged-MQA-logits scoring kernel
  (`dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda`) reads the FP8 cache, not
  the bf16 mirror.
- The only full-length access was the LRU-swap snapshot (`Dsv4DsaOfficialImage`),
  whose own comment admitted it snapshotted full "because old packed rows are
  **not proven reconstructible**" — a conservative hedge, not a real read
  dependency. The full key history is already in the FP8 `key_cache_slot` band
  the image also captures.
- Precedent: the indexer's *source* staging (`Dsv4CompressorState`) is **already**
  a bounded ring (`DSV4_INDEXER_STAGING_RING_ROWS`); the rotated dst was the lone
  asymmetric full-history holdover.

This full-history bf16 mirror is one of the per-slot DSA state caches that scale
linearly with `max_seq` and are **not pooled** into the FlashMLA `TokenKVPool` —
flagged as a 1M-context startup-OOM contributor.

## What Worked

Deletion-style downgrade of `rotated_keys` to a transient drain-immediate ring
(`crates/infer-cuda/src/attention.rs`, `dsv4.rs`):

- New `dsv4_dsa_rotated_ring_rows(cc) = DSV4_INDEXER_STAGING_RING_ROWS.min(cc)`
  (= 8192 rows max); `Dsv4DsaOfficialState::rotated_keys` allocated at that size
  instead of `compressed_capacity`.
- `csa_select_official` (single-row) + `dsv4_dsa_cache_write_gather_row` (batched
  lane #60): Hadamard dst and fused-store src are now **ring-relative 0** (was
  absolute `packed_rows`); `cache_locs` / FP8 band stay absolute. Batched
  `dst_row` pushed to the kernel ptr table is now `0`. Verified against the CUDA
  batched kernels (`hadamard128_bf16_batched_kernel` reads `dst_row+r`,
  `fused_store_indexer_cache_batched_kernel` reads `key_arr[slot]` at slot-local
  0-based rows) — both consistent with `dst_row=0` + ring-relative src.
- Added an explicit `newly_packed <= rotated_ring_rows` dst-side precondition
  (the prior assert only bounded the source staging ring).
- **Deleted** the `rotated_keys` field from `Dsv4DsaOfficialImage` entirely
  (capture D2H, restore len-assert + H2D, host_bytes term) — LRU swap no longer
  copies it; the FP8 `key_cache_slot` band carries the full history. Net simpler
  rollback path.
- Budget `dsv4_dsa_rotated_keys_bytes` now returns the ring-bounded size, so
  `kv_budget_num_slots`'s `dsa_rotated_per_slot` term collapses from O(max_seq)
  to O(chunk).

### Memory delta (per slot, per CSA layer)

| ctx (ratio=1) | old rotated_keys | new rotated_keys | DSA state /tok |
|---|---|---|---|
| any | `cc × 256 B` | `min(8192, cc) × 256 B` | 388 → 132 B/tok beyond ring (**−66%**) |
| 1M | **256 MiB** | **2 MiB** | −254 MiB/slot/layer |

At 1M ctx the bf16 mirror dominated; capping it removes the linear term from the
per-slot DSA cost across every CSA layer × num_slots.

### Verification (local, Mac, no nvcc)

- `CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` — clean.
- `cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib` — no new warnings in changed code (the prior unused-`dst_row` is removed).

## Pending-remote

CUDA can't run on this Mac. **Needs on H20 before default-trust:**

1. `scripts/needle_gate.py` ×3 same-config repeats on a **CSA decode** workload
   (DeepSeek-V4-Flash) — confirm ring-relative addressing produces identical
   retrieval vs the pre-change envelope (MoE non-determinism floor).
2. The **batched decode lane (#60)** specifically — the per-row ptr-table path is
   the highest-risk change (`dst_row=0` across n slots). Run a multi-slot CSA
   decode and confirm no cross-slot rotated-key aliasing.
3. **1M-context boot** under the TP=4 budget — confirm the freed per-slot bytes
   raise `num_slots` (or unblock the 1M startup OOM) as the budget math predicts.
4. A `bench_guidellm.sh` decode run vs baseline to confirm no ITL regression (the
   ring is smaller, so cache locality should be neutral-to-better).

## Rule

A "stateful mirror" that is only ever written-then-read-back in the same forward
(drain-immediate) is a transient ring, not history — size it by the delta bound,
not `max_seq`. The real history lives in the downstream cache the consumer
kernel reads (here the FP8 `dsa_key_cache`); a parallel bf16 full-length copy is
redundant storage. Audit "snapshot full because not proven reconstructible"
comments — they are hedges to verify, not read dependencies to preserve.
