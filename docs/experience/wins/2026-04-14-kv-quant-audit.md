# 2026-04-14 · CUDA KV Quant Audit

## What was checked

- `crates/infer-engine/src/backend/cuda/paged_kv.rs`
- `crates/infer-engine/src/ops/kv_quant.rs`
- `crates/infer-engine/src/model/{qwen3,qwen35,glm4}/batch_decode.rs`
- `infer/src/scheduler/cuda/prefill.rs`
- `ROADMAP.md`

## Current implementation

### Prefill

The production CUDA scheduler still does **not** use the `forward_prefill_with_pool()` dual-write hook today. The active path is now:

1. `model.forward_prefill()` writes contiguous KV
2. scheduler allocates paged token indices only for the range being materialized
3. `GenerationStateBase::migrate_kv_range_to_paged()` copies or quantizes only that range into the paged pool

Two important consequences:

- prefix-hit requests no longer re-migrate the already committed prefix
- prefill migration APIs now work on `start_pos + new_token_indices`, not only on the full `[0..seq_len)` prefix

For FP8 this means prefill migration is:

```text
BF16 contiguous KV[start_pos..] -> migrate_from_contiguous_fp8_range()
                                -> quantize_scatter_kv_fp8_range()
                                -> FP8 paged token pool
```

For TurboQuant the migrated contiguous range is first copied into the shared NHD work buffer at the new pool slots, then quantized into packed TurboQuant storage. This fixes the old gap where TurboQuant prefill migration still followed the BF16 copy helper.

### Decode

Decode is format-specific:

- `BF16` -> FlashInfer paged decode
- `FP8E4M3` -> custom split-KV decode kernel with fused FP8 cast (`decode_attention_fp8`)
- `INT8` -> custom split-KV decode kernel with fused INT8 dequant (`decode_attention_int8`)
- `TurboQuant` -> TurboQuant fused decode attention

This means ROADMAP and older planning notes that said “FP8 decode uses FlashInfer native FP8” were stale relative to the codebase.

### Quantized workspace

`PagedKVPool` allocates `int8_attn_workspace` for both `INT8` and `FP8E4M3`.

That is correct: FP8 decode reuses the same split-KV partial-reduction / merge scratch layout as INT8, so the shared workspace sizing is intentional.

### Server configuration surface

`infer --kv-cache-dtype` now explicitly supports:

- `bf16`
- `fp8`
- `int8`
- `tq2`
- `tq3`
- `tq4`

Mapping rules:

- `bf16` -> contiguous BF16 + paged BF16
- `int8` -> contiguous INT8 + paged INT8
- `fp8` -> contiguous BF16 + paged FP8
- `tq2/tq3/tq4` -> contiguous BF16 + paged TurboQuant

The important detail is that FP8 and TurboQuant are **paged-pool formats**, not contiguous-cache formats.

## Fixes made in this audit

1. `qwen35` decode planning now skips FlashInfer HD256 planning for non-BF16 KV formats.
2. `infer --kv-cache-dtype` help text and parsing now expose all supported quantized modes consistently.
3. Comments in `kv_cache.rs`, `paged_kv.rs`, and `ROADMAP.md` were updated to match the actual FP8/INT8/TurboQuant decode architecture.
4. `ModelForward::forward_prefill_with_pool()` documentation now reflects that incremental range migration is the active scheduler path.
5. Scheduler prefill now migrates only newly materialized KV ranges into the pool, instead of re-copying the whole prefix+suffix window on prefix hits.
6. TurboQuant prefill migration now quantizes into packed paged storage instead of falling through the BF16 helper.

## Takeaway

The CUDA KV quant stack is complete on the production path:

- prefill quantization happens during contiguous -> paged range migration
- decode attention is wired for FP8 / INT8 / TurboQuant
- FP8 and INT8 workspace allocation is already correct

The main issues found in the audit were:

- drift between implementation and documentation
- one `qwen35` planner path that still behaved as if all decode formats used BF16 FlashInfer planning
- prefill migration doing full-window copies instead of suffix-only copies on prefix hits
- TurboQuant prefill migration missing a real packed-quantized path
