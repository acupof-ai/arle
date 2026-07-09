# DSv4 Route A: Cross-Request FlashMLA Page Sharing (MODEL1 only)

> Status: pending-remote

## Goal

Enable cross-request KV page sharing for DSv4-Flash (MODEL1, head_dim=512):
when request B reuses request A's prefix (matched via radix), B reads A's
physical FlashMLA KV pages instead of fresh ones.

## Hypothesis

The host `HostPagedKvPool` page IDs (managed by the engine's radix cache) are
disconnected from per-layer `TokenKVPool` FlashMLA physical page IDs. By
maintaining a `host_to_flashmla` mapping and refcounting physical pages via
`retain_pages`/`release_pages`, a second request can attach to already-written
FlashMLA pages, skipping prefix prefill.

V32/GLM (head_dim=576) uses band-base pack addressing
(`flashmla_pages_byte_range`) which requires contiguous identity-mapped bands.
Non-contiguous page sharing would break the V32 pack kernel, so it is gated
off (`supports_flashmla_page_sharing()` returns false for head_dim=576).

## Changes

- `crates/infer-cuda/src/attention/kv_layout.rs`:
  - Added `head_dim` to `Dsv4KvAdapter`
  - `host_to_flashmla: HashMap<u32, Vec<u32>>` maps host page ID → per-layer
    FlashMLA page IDs
  - `record_host_mapping()` records the identity convention used by fresh
    allocations (called from `mirror_slot_pages` / `prepare_kv_batch`)
  - `retain_flashmla_pages()` / `release_flashmla_pages()` bump refcounts on
    physical pages across all layers
  - `attach_slot_flashmla_pages()` wires a slot to existing physical pages via
    the host→FlashMLA mapping
  - `supports_flashmla_page_sharing()` returns `head_dim != 576`
  - `mirror_slot_pages` and `prepare_kv_batch` branch: MODEL1 records mapping +
    uses `layer_pages_from_host`; V32 uses `identity_layer_pages`
- `crates/infer-cuda/src/executor.rs`:
  - `save_prefix_sidecar` DSv4 arm calls `retain_flashmla_pages(prefix_pages)`
  - `release_prefix_pages` DSv4 arm calls `release_flashmla_pages(pages)`
  - `restore_route_a_prefix_state` calls `attach_slot_flashmla_pages` +
    `refresh_flashmla_device_page_tables`
  - `reusable_prefix_blocks` gated on `supports_flashmla_page_sharing()`

## Params

- GPU: 8×H20 (pending)
- Model: DeepSeek-V4-Flash-FP8 (MODEL1, head_dim=512)
- num_slots: 16
- Features: cuda, nccl, deepep

## Results

pending-remote

## Problems

pending-remote

## Learnings

- V32/GLM (head_dim=576) pack kernel requires contiguous identity-mapped bands;
  non-contiguous sharing is MODEL1-only.
- `retain_flashmla_pages`/`release_flashmla_pages` are naturally no-op for V32
  since `host_to_flashmla` stays empty (V32 path never calls
  `record_host_mapping`).
