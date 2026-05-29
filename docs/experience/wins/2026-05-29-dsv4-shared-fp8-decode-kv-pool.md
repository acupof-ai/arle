# DSv4 shared persistent FP8 decode KV pool — c≥8 OOM fix — 2026-05-29

> **pending-remote** — CUDA build + parity/bench run on the pod (Mac has no
> nvcc). This stub records the change, the design, and the exact pod commands;
> after-snapshot with numbers lands once the pod run completes.

## SLO-shape probed?  N (pending-remote)

Code lands on the FlashMLA FP8 decode path; the binding c≥8 SLO sweep
(c={1,4,8}, M≥4096 prefill) runs on the pod. Cannot claim PASS until that run
reports c=8 no-OOM + byte-identical greedy output. The whole point of the
change is reachability at c≥8 — correctness/perf verdict is pod-gated.

## Roofline check

N/A for this change — it is a memory-budget / ownership refactor, not a kernel
perf change. The per-row FlashMLA decode kernel + indices math are unchanged
(only the KV pool base device pointer moves to the slot's sub-range). Roofline
of the decode kernel is unaffected; the bench gate is "c=8 stops OOMing with
byte-identical output", not a FLOPS/bandwidth delta.

## Goal

- Capability/unblock: make DSv4 FlashMLA FP8 decode survive c≥8 so the
  beat-SGLang concurrency campaign (c=8/c=32) can run at all. Goal type =
  correctness-preserving capacity fix.

## Context — the bug (evidence)

- DSv4 decode allocated the FlashMLA FP8 KV pool **per-(slot, layer)**, lazily
  at first decode (`ensure_dsv4_flashmla_fp8_kv_pool`, old
  `weights.rs:4350`), sized to the full lifetime capacity:
  `comp_blocks = ceil(max_position_embeddings / compress_ratio / 64)`.
  With `max_position_embeddings = 1048576`, even at a moderate ratio the
  compressed sub-pool is ~tens of MB **per (slot, layer)**.
- At c≥8 → `num_slots × layers` (e.g. 8 × 61) separate **dynamic, unbudgeted**
  `alloc_zeros_traced` calls → ~18 GB → `DSv4 FlashMLA FP8 KV pool alloc
  failed: CUDA_ERROR_OUT_OF_MEMORY` (old `decode.rs:304` ≈ `weights.rs:4372`),
  even at `--mem-fraction-static 0.6`. c=1/c=4 worked (parity validated).

## What changed

ONE shared persistent pool, owned by the scheduler-side decode context,
allocated once and accounted in the static budget. Each (slot, layer) owns a
fixed byte sub-range; the per-row pack/decode logic is byte-identical — only
the base device pointer moves from an owned buffer's byte 0 to the slot's
sub-range start.

- `infer/src/model/deepseek/batch_decode.rs`
  - `DeepseekBatchDecodeBuffers` gains the shared `Option<CudaSlice<u8>>` pool +
    `fp8_kv_{slots,layers,slot_blocks,max_seq_len}`.
  - `ensure_fp8_kv_pool(num_slots, layers, slot_blocks)` — one monotonic alloc.
  - `fp8_kv_slot_layer_view(slot, layer, slot_blocks) -> (base_ptr, bytes)` —
    byte offset `(slot*layers + layer) * slot_blocks * 37376`.
  - `fp8_kv_pool_bytes(...)` (also drives the budget estimate) +
    `set/get fp8_kv_max_seq_len`.
- `infer/src/model/deepseek/state.rs`
  - `DeepseekAttentionRuntimeCache` drops the owned pool
    (`fp8_kv_pool` / `fp8_kv_pool_bytes`); holds only the bound view
    (`fp8_kv_pool_ptr: u64` + `fp8_kv_pool_view_bytes`) + the unchanged
    lifecycle flags (`fp8_kv_sw_bootstrapped`, `fp8_kv_comp_packed_rows`).
- `infer/src/model/deepseek/weights.rs`
  - `ensure_dsv4_flashmla_fp8_kv_pool` → `dsv4_flashmla_fp8_kv_pool_base_ptr`
    (reads the bound view + asserts capacity; no alloc).
  - Pack helpers (`dsv4_flashmla_bulk_pack_sw_ring_raw`,
    `dsv4_flashmla_pack_one_sw_token`, `dsv4_flashmla_pack_compressor_rows`)
    take a `fp8_pool_base_ptr: u64` instead of `&mut CudaSlice<u8>`.
  - The three comp-block sites (SW bootstrap, compressor pack, decode dispatch)
    read the uniform `(sw_blocks, comp_blocks)` stamped on the cache at bind
    time, bounding comp_blocks by served `max_seq_len`.
  - New model methods `dsv4_flashmla_pool_slot_blocks(max_seq_len)`,
    `bind_fp8_kv_pool_view(...)`, `loaded_layer_count()`;
    `dsv4_flashmla_decode_enabled()` now `pub(super)`.
- `infer/src/model/deepseek/forward.rs`
  - `create_decode_context` eagerly sizes the shared pool (knob-gated) and
    records `max_seq_len`.
  - `forward_decode_batch` binds every active (slot, layer) view at the top —
    single source of slot identity + decode context, so both the N≥2 batched
    path and the N==1 per-row fallback read pre-bound views with **no slot/ctx
    threading through the deep attention chain**.
  - `scheduler_runtime_workspace_bytes` reserves the pool in the static budget
    (`num_slots × layers × slot_blocks × 37376`, knob-gated).

### Pool sizing + slot→block-range scheme

- `slot_blocks = sw_blocks + comp_blocks` where
  `sw_blocks = ceil(sliding_window / 64)` and
  `comp_blocks = ceil(max_seq_len / min_nonzero_compress_ratio / 64)`
  (uniform worst case across layers so every layer's sub-range fits; lower-
  pressure layers carry zeroed slack the indices builder never references).
- Block bytes = `page_block_size(64) × bytes_per_token(584)` = **37376 B** per
  block per layer (confirmed `DSV4_FLASHMLA_MODEL1_*` in `weights.rs`;
  `csrc/attention/dsv4_fp8_kv_pack.cu` MODEL1 layout).
- Slot `s`, layer `l` byte offset =
  `(s*layers + l) * slot_blocks * 37376`. The pack kernels index `block_id ×
  37376` relative to this base — identical to the per-state buffer's byte 0.

### num_slots plumbing

`create_decode_context(max_batch_size = num_slots = states.len(),
max_seq_len = effective_max_seq_len, pool)` (warmup.rs:83). num_slots + layers
+ max_seq_len fully determine the allocation there. `forward_decode_batch`
(the sole scheduler decode entry, for both c=1 and c≥2) binds per-step.

### Slot-reuse reset handling

On slot release/reuse `GenerationState::reset()` → `incremental.clear()` drops
the per-layer caches; the next decode recreates them with
`fp8_kv_sw_bootstrapped = false` → the new sequence re-bootstraps (bulk-packs)
its sub-range, overwriting any prior KV. No leak across sequences.

## Typecheck

`CUDARC_CUDA_VERSION=12090 cargo check -p infer --lib --no-default-features
--features cuda,no-cuda,nccl` — clean (no new warnings; pre-existing
`argmax_batch_readback_into` / window-scratch / native-deepep warnings on main
are unrelated). `--features no-cuda` also clean.

## Pod commands (parity + bench)

```bash
# Build (pod, has nvcc)
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda,nccl

# Serve DSv4 with the FlashMLA decode knob on, e.g. --num-slots >= 8 and
# --mem-fraction-static 0.6 (the prior OOM config). Then validate against the
# running server on <port>: the script does c=1 (per-row reference), c=4
# (batched, must be byte-identical greedy), and a c=8 timing/no-hang sanity.
# Expect c=8 to STOP OOMing now (shared budgeted pool) with c=1==c=4 outputs.
python3 scripts/dsv4_batched_decode_validate.py <port>

# Static-budget headroom: confirm the scheduler envelope log reserves the
# shared FP8 pool (scheduler_runtime_workspace_bytes) and the KV pool shrinks
# accordingly instead of OOMing at first decode.
```

## Rule

- Per-sequence dynamic GPU pools sized to full *lifetime* capacity OOM at
  concurrency: a shared pool sized `num_slots × per-slot-capacity` and bounded
  by the served `max_seq_len` (not `max_position_embeddings`) must be accounted
  in the static budget — mirror qwen3's `PagedKVPool` ownership.
- When a per-state buffer must become shared but the `&self` attention chain
  can't carry a slot index, bind the per-(slot,layer) view **at the single
  decode entry that owns slot identity + the decode context**
  (`forward_decode_batch`), and pass the kernels a base **pointer** into the
  shared buffer — the per-row index math stays byte-identical.
