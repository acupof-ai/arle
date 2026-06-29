# CUDA Qwen3.6-27B-FP8 (qwen35 hybrid): paged batched decode — concurrency now scales

## Context

Qwen3.6-27B-FP8 (qwen35 hybrid path) served on CUDA but **serialized** under
concurrency: measured 2026-06-28 flat **~21 tok/s from c=1..8**, latency stacking
(6→49 s). Its batched-decode path used a CONTIGUOUS kernel
(`fused_gqa_attention_decode_batched`) over per-slot `k_caches`/`v_caches` that the
always-on **paged** KV pool never populates, so the executor gate
(`executor.rs` ~4677, `… || full_attn_paged()`) skipped it → per-row sequential
decode. gap-2 of the unified model-support plan.

## What worked (clean wire — gap-1 pattern, no new kernel)

The batched **paged** HD256 decode kernels already existed and were batch-aware,
and the per-head gate is a **separable post-step** (not fused):
- `qwen35.rs`: `full_attention_paged_batch` — batched q/k/v GEMM →
  `decode_prep_paged_hd256_cuda(batch_size=B)` (writes each row's KV to its own
  slot's pages) → `resolve_paged_attn_v1(HD256, Decode)` at `(B,B,1)` →
  separable `attention_gate_paged_hd256_cuda(batch_size=B)` (already batch-aware) →
  o_proj + all-reduce. `forward_decode_batch_paged` routes full-attn rows here;
  `stage_recurrent_pointer_tables` stages only conv/GDR tables (paged default
  never allocates the contiguous KV tables).
- `executor.rs`: lifted `full_attn_paged()` from the skip gate; B>1 paged decode →
  `submit_decode_batch_paged`, which builds the B-row `kv_indptr` via the existing
  gap-1 `PageMeta::for_decode_batch` and runs one batched forward.

B==1 stays byte-identical (single-row `decode_row_paged_default` → `full_attention_paged`
untouched). recall / quant-KV (`--kv-cache-dtype fp8`) / TP fall to the per-row
floor. Qwen3.5 dense uses a separate path (`model.rs`), unaffected. FP8 weights
keep BF16 KV by default, so the FP8 model gets batched decode.

## Results — Qwen3.6-27B-FP8, 1×H20 (measured, llmbench)

| c | tok/s | baseline (flat) | lat p50 ms | lat p99 ms |
|---|---|---|---|---|
| 1 | 21.1 | 21.0 | 6058 | 6059 |
| 2 | 23.1 | 21.0 | 10795 | 16855 |
| 4 | 25.1 | 21.0 | 25483 | 25483 |
| 8 | **26.1** | 21.1 | 37580 | 53911 |

Throughput now **scales** (21→26 tok/s, +24% @c=8) where it was flat-serialized,
and p50 latency is lower than the serialized baseline at c=8 (37.6 s vs 49 s).
Coherent output (FP8 thinking-mode completion). Gain is modest — decode batches
but prefill stays sequential and the FP8 MoE grouped-GEMM bounds it — but it
converts 0 % concurrency scaling into real batched serving.

## Rule

A "batched decode" path that reads contiguous per-slot KV caches is **dead** under
a paged-default KV pool (those caches are never populated). Wire the batched
**paged** kernel (`batch_decode_paged_hd256` at bsz=B + per-row `kv_indptr`) and,
for a gated model, apply the gate as the same separable post-step the single-row
paged path uses — the HD256 paged + gate kernels were already batch-aware, so no
new kernel was needed. Verify by the c-sweep (throughput must rise) AND a coherent
completion (each row must attend only its own pages).
