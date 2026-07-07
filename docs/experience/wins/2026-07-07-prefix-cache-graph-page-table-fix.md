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

**DSv4 test blocked (superseded below):** model >97GB/GPU at TP=1/2; TP=4 OOM
at weight upload; TP=8 OOM on GPUs 2-5 (residual CUDA context from prior
attempts + other users). Pending a clean 8×H20 window for full FlashMLA-path
verification.

## Follow-up: construction-time regression + missing first-draw sync (2026-07-07)

This commit (`36835179f`) broke **all** DSv4 FlashMLA boots. Root cause and fix
below; this is what finally closed the "DSv4 test blocked" gap above.

### The regression

`Dsv4FlashMlaDecodeState::new()` (`flashmla.rs:211`, called once per
layer/slot at engine-startup slot-pool construction, before any request is
ever admitted) called `refresh_device_page_table(ctx, pool)` immediately after
allocating `device_page_table`. But per the pre-existing #85 P2 Stage B
comment right above it (`flashmla.rs:155-158`): slots start EMPTY at
construction — the band is drawn on first admission. `refresh_device_page_table`
`ensure!`s the host table's length (0 at construction) matches
`device_page_table`'s fixed `shape.total_blocks` length — **fails 100% of the
time, for every slot, at startup.** DSv4 FlashMLA could not boot at all.

### The fix

1. **`flashmla.rs:211`** — deleted the premature construction-time call. The
   zeroed `alloc_zeros` placeholder is safe until a real band exists; no
   FlashMLA kernel reads `device_page_table` before a slot is admitted a
   request.
2. **New sync point** — the host page table transitions empty→populated
   exactly once per slot lifetime: inside `prepare_kv_batch`
   (`kv_layout.rs:920-955`), `mirror_band` draws the full fixed-size identity
   band on the slot's **first prefill row** (`row.start_pos == 0`, right after
   `submit_prefill_row`'s reset+free_slot). Later prefill/decode rows
   re-mirror the *same* pages (the band is fixed for the slot's lifetime, only
   the cursor advances via `flashmla_alloc_append`) — so **one** refresh at
   first-draw suffices; refreshing every decode step would itself be a
   CUDA-graph capture hazard (`rearm_warm(1)` only protects token 1; replay
   from token 2+ uses the captured graph — an eager H2D write into a
   graph-captured source under replay is the very class of hazard #8 fixed).
   New wiring, since `prepare_kv_batch` (on `Dsv4KvAdapter`/`Dsv4LayerKvLayout`)
   has no access to the per-slot `Dsv4FlashMlaDecodeState` (lives in
   `slot.attention`, private to `dsv4.rs`):
   - `Dsv4LayerAttentionState::refresh_flashmla_device_page_table` (`dsa.rs`,
     next to `flashmla_slot_idx()`) — no-op wrapper for layers without
     FlashMLA.
   - `Dsv4SlotState::refresh_flashmla_device_page_tables` (`dsv4.rs`, next to
     `reset()`) — loops all layers, pairing each with `kv_adapter.layer(i)`.
   - Called from `Dsv4CudaExecutor::submit_prefill_row` (`executor.rs`) inside
     the existing `if row.start_pos == 0 { ... }` block (alongside
     `zero_slot_band`).
3. The pre-existing restore-path call (`dsa.rs:1384`, inside `swap_in_image`,
   for the whole-slot demote/promote park path) is unchanged — it's the OTHER
   sync point, after `mirror_restore_pages` changes the host table.

### DSv4 FlashMLA-path verification (H20 pod, 2026-07-07 — closes the gap above)

**Env:** 4×H20 (GPUs 2/3/4/5, clean — 0 to avoid, 1 held by a foreign PID),
TP=4, DeepSeek-V4-Flash-FP8, FlashMLA on (default), `cargo build --release
--features cuda,nccl`.

- **Boot:** `all 4 worker engines ready; opening HTTP` — no `ensure!` panic
  (proves the construction-time fix).
- **Single request** ("What is 2+2?", greedy): `"content":"4"` — correct
  beyond token 1 (proves the first-draw sync; a missing sync would zero
  `device_page_table` forever and corrupt token 2+).
- **Repeat same prompt 3× serially:** all 3 byte-identical, correct.
- **Original #8 scenario — forced whole-slot park/promote** (`--kv-
  oversubscription --max-running-requests 1`, 2 concurrent requests each
  triggering the other's park under lockstep contention): `demoted_slots:18,
  promoted_slots:18` in `/v1/stats` (18 `swap_in_image`/
  `refresh_device_page_table` restore cycles) across mixed-prompt and 3×
  concurrent-identical-prompt trials — all outputs correct, zero corruption.
  (Note: DSv4's `reusable_prefix_blocks()` is hardcoded `0` —
  `executor.rs:343` — so plain serial repeats never exercise RadixCache-style
  publish/restore for DSv4; the restore path this bug targets is the
  whole-slot KV-tier park/promote, gated behind `--kv-oversubscription`.)
- **n=2 concurrent-decode needle-gate sanity** (task #6's known unrelated
  bug, `concurrent_needle_v3.py`, len=500, 3 trials): 5/6 exact, 1 miss
  (pure-truncation signature, matches the documented pattern exactly) — in
  line with the established 20-57% baseline miss rates, not worse.

## Rule

- Any device buffer whose pointer may be captured by a CUDA graph must be
  **persistent** (state-owned), not a per-call temporary. Graph capture bakes
  raw device pointers; `CUDA_GRAPH_AUTO_FREE_ON_LAUNCH` only handles
  allocations made *during* capture, not alloc-then-free.
- Prefix-cache restore must re-sync ALL device-resident state derived from the
  host page table, not just the sw_window + compressor buffers.
- A persistent device mirror of a host table needs a sync point for **every**
  transition the host table makes, not just the one the bug report named —
  enumerate all writers of the host table (construction / first-draw /
  restore) and place a sync at each real content change, none at pure
  no-op-content re-mirrors (would reintroduce the graph-capture hazard).
