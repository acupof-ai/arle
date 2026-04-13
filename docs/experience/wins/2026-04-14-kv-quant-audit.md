# 2026-04-14 · CUDA KV Quant Audit

## What was checked

- `crates/infer-engine/src/backend/cuda/paged_kv.rs`
- `crates/infer-engine/src/ops/kv_quant.rs`
- `crates/infer-engine/src/model/{qwen3,qwen35,glm4}/batch_decode.rs`
- `infer/src/scheduler/cuda/prefill.rs`
- `ROADMAP.md`

## Current implementation

### Prefill

The production CUDA scheduler does **not** use the `forward_prefill_with_pool()` dual-write hook today. The active path is:

1. `model.forward_prefill()` writes contiguous BF16 KV
2. scheduler allocates paged token indices
3. `GenerationStateBase::migrate_kv_to_paged()` quantizes and scatters into the paged pool

For FP8 this means prefill is already:

```text
BF16 contiguous KV -> migrate_from_contiguous_fp8()
                   -> quantize_scatter_kv_fp8()
                   -> FP8 paged token pool
```

TurboQuant follows the same pattern: contiguous BF16 prefill first, quantize on migration into the paged pool.

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
4. `ModelForward::forward_prefill_with_pool()` documentation now reflects that migration-from-contiguous is the active scheduler path.

## Takeaway

The CUDA KV quant stack is complete on the production path:

- prefill quantization happens during contiguous -> paged migration
- decode attention is wired for FP8 / INT8 / TurboQuant
- FP8 and INT8 workspace allocation is already correct

The main issue found in the audit was drift between implementation and documentation, plus one `qwen35` planner path that still behaved as if all decode formats used BF16 FlashInfer planning.
