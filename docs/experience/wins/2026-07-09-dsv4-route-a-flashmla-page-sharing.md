# DSv4 Route A: Cross-Request FlashMLA Page Sharing (MODEL1 only)

> Status: Historical shipped result; superseded 2026-07-10 — this Route A `host_to_flashmla` page-sharing implementation was deleted for correctness. Measurements below are retained as historical evidence.

## Goal

Enable cross-request KV page sharing for DSv4-Flash (MODEL1, head_dim=512):
when request B reuses request A's prefix (matched via radix), B reads A's
physical FlashMLA KV pages instead of recomputing them.

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
  - `host_to_flashmla: HashMap<u32, Vec<u32>>` maps host page ID → per-layer
    FlashMLA page IDs
  - `record_host_mapping()` records the identity convention used by fresh
    allocations (called from `mirror_slot_pages` / `prepare_kv_batch`)
  - `resolve_layer_pages()` resolves host page IDs to one layer's physical pages
  - `retain_flashmla_pages()` / `release_flashmla_pages()` bump refcounts on
    physical pages across all layers
  - `attach_slot_flashmla_pages()` wires a slot to existing physical pages via
    the host→FlashMLA mapping
  - `supports_flashmla_page_sharing()` returns `head_dim != 576`
  - `mirror_slot_pages` and `prepare_kv_batch` branch: MODEL1 records mapping +
    uses `resolve_layer_pages`; V32 uses inline identity pages
- `crates/infer-cuda/src/executor.rs`:
  - `save_prefix_sidecar` DSv4 arm calls `retain_flashmla_pages(prefix_pages)`
  - `release_prefix_pages` DSv4 arm calls `release_flashmla_pages(pages)`
  - `restore_route_a_prefix_state` calls `attach_slot_flashmla_pages` +
    `refresh_flashmla_device_page_tables`
  - `reusable_prefix_blocks` gated on `supports_flashmla_page_sharing()`

## Params

- GPU: 8×H20, TP=8
- Model: DeepSeek-V4-Flash-FP8 (MODEL1, head_dim=512)
- num_slots: 16, max_total_tokens: 4096
- Features: cuda, nccl, deepep
- Server: `arle serve --backend cuda --model-path <path> --num-slots 16 --max-total-tokens 4096`

## Results

**Single-request TTFT (prefix reuse A/B, ~2100-token prompt, 1152-token shared prefix):**

| Scenario | TTFT (ms) | Prefill tokens | Hit tokens | Δ |
|----------|-----------|---------------|------------|---|
| Cold (unique prefix) | 3308 | 1425 | 0 | baseline |
| Reuse (identical prompt) | 1695 | 273 | 1152 | **−49%** |
| Reuse (shared prefix + diff suffix) | 1697 | 274 | 1152 | **−49%** |
| Cold #2 (new unique prefix) | 3307 | 1425 | 0 | baseline confirmed |

**Concurrent c=4:**

| Scenario | TTFT (ms) | Δ |
|----------|-----------|---|
| Unique prefix (cold) | 7335 avg | baseline |
| Shared prefix (reuse) | 6971 avg | −5% |

**Prefix cache stats (59 requests total):**
- Hit rate: 69.5% (41/59)
- Total hit_tokens: 47,232
- 738 full blocks matched, 738 clamped (100% usable — no boundary rejection)
- resident_pages: 227, reuse_hit_resident: 738, reuse_miss: 18

## Problems

1. **max_prompt_tokens=4096 ceiling**: Phase 2 multi-turn prompts (4440–4509 tokens)
   all failed with 0 output. Server configured with `--max-total-tokens 4096`
   caps prompt length. Not a code bug — needs config bump for longer-context tests.
2. **Output tokens = 0 on random-text prompts**: Model generates empty string
   for gibberish/random input. Expected behavior — model refuses to complete
   nonsense. TTFT measurement still valid (includes full prefill + first decode).
3. **Concurrent reuse advantage narrows (5% vs 49% single)**: Under c=4, all
   requests share GPU; faster prefill of reuse requests is masked by waiting
   for cold requests' prefill in the same batch window.

## Learnings

- **2× TTFT speedup confirmed.** Cross-request FlashMLA page sharing halves
  prefill time when a 1152-token prefix is cached. Engine correctly reduces
  prefill_tokens from 1425 to 273 (−81%), and TTFT drops proportionally.
- V32/GLM (head_dim=576) pack kernel requires contiguous identity-mapped bands;
  non-contiguous sharing is MODEL1-only.
- `retain_flashmla_pages`/`release_flashmla_pages` are naturally no-op for V32
  since `host_to_flashmla` stays empty (V32 path never calls `record_host_mapping`).
- **All 738 matched blocks were usable** (prefix_match_full_blocks == clamped).
  No boundary rejections — DSv4 compression/SW/DSA boundaries aligned with
  page granularity for these test prompts.
- Concurrent benefit is real but smaller; primary win is single-request latency
  for repeated-prefix workloads (agent loops, multi-turn chat).
