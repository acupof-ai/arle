# Prefix-Cache CUDA-Graph Page-Table UAF Fix (#8)

> Status: Shipped
> Date: 2026-07-07
> Env: 4×H20, TP4/EP4, DeepSeek-V4-Flash-FP8, FlashMLA on, prefix cache ON (default)

## Context

Repeatedly sending the same prompt triggered 100% deterministic corruption.
`ARLE_DISABLE_PREFIX_CACHE=1` completely fixed it. This is the "#8" defect
tracked in the #146 comment thread — independent of the DSA top-k content bug.

## Root Cause

CUDA graph use-after-free in the FlashMLA decode path:

1. `flashmla_device_page_table()` (`kv_layout.rs:1181`) allocates a temporary
   `CudaSlice<i32>` and returns it. The caller passes `Some(&page_table)` to
   the pack/index kernels.
2. When the decode attention path is captured into a CUDA graph (first eager
   step after `rearm_for_new_request`), the device pointer of this temporary
   slice gets baked into the graph node.
3. After the function returns, `page_table` is dropped → device memory freed.
4. On 1st request: freed memory not yet overwritten → replay "works".
5. On 2nd request (prefix cache hit): `mirror_restore_pages` + `swap_in_image`
   issue H2D copies that overwrite the freed region → graph replay reads
   garbage page table → corrupted attention output.

`rearm_warm(1)` only protects the 1st generated token (eager step, no capture).
Token 2+ replays the graph with the stale pointer.

## Fix

Make the device page table **persistent** in `Dsv4FlashMlaDecodeState` instead
of a per-call temporary:

- **`flashmla.rs`**: Added `device_page_table: CudaSlice<i32>` field, allocated
  once in `new()`. Added `refresh_device_page_table()` to re-sync from the host
  page table via H2D copy.
- **`dsa.rs`**: In `swap_in_image` (called during prefix-cache restore), call
  `flash.refresh_device_page_table(ctx, pool)?` after the flashmla match block
  — `mirror_restore_pages` may have changed the host table.
- **`attention.rs`**: Replaced all 6 `pool.flashmla_device_page_table(...)`
  temporaries with `&flash.device_page_table`.
- **`dsv4.rs`**: Batched path uses `flash.device_page_table` directly; removed
  the `pack_page_tables` ownership vec (flash state already owns it).
- **`kv_layout.rs`**: Removed dead `flashmla_device_page_table()`.

## What Worked

- The persistent table is slot-scoped (lives as long as the slot), so the
  captured graph pointer stays valid for all replays across cache hits.
- `refresh_device_page_table` re-syncs after `mirror_restore_pages`, so the
  device table matches the (possibly changed) host table.
- Baseline byte-identical: the device table content is the same as the old
  temporary; only the allocation lifetime changed.

## Verification (H20 pod, 2026-07-07)

**Qwen3.6-27B-FP8, TP=1, prefix cache ON, decode graph ON:**

Sent identical prompt 3× via `/v1/chat/completions`. All 3 responses correct
("2 + 2 equals 4." / "2+2 equals 4."). No corruption.

**Caveat:** Qwen3.6 TP=1 uses the full-attention path, not FlashMLA. The
specific device page table allocation the fix makes persistent is only used
by DSv4 and Qwen3.5/3.6 MoE with TP>1. The Qwen3.6 test validates that
prefix cache + CUDA graphs + the rebuilt binary produce correct output, but
does not directly exercise the FlashMLA page table path.

**DSv4 test blocked:** model >97GB/GPU at TP=1/2; TP=4 OOM at weight upload;
TP=8 OOM on GPUs 2-5 (residual CUDA context from prior attempts + other
users). Pending a clean 8×H20 window for full FlashMLA-path verification.

## Rule

- Any device buffer whose pointer may be captured by a CUDA graph must be
  **persistent** (state-owned), not a per-call temporary. Graph capture bakes
  raw device pointers; `CUDA_GRAPH_AUTO_FREE_ON_LAUNCH` only handles
  allocations made *during* capture, not alloc-then-free.
- Prefix-cache restore must re-sync ALL device-resident state derived from the
  host page table, not just the sw_window + compressor buffers.
