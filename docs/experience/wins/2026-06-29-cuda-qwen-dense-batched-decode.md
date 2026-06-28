# CUDA Qwen3 dense (R6-clean): batched paged decode — c>1 serving unlocked

## Context

The rewrite CUDA Qwen3 dense path ("R6-clean") was **single-row only**:
`executor.rs` `submit()` hard-asserted `rows == 1`, so any concurrent request
made the scheduler batch ≥2 rows → engine thread died (`R6 clean CUDA forward is
single-row only`). Measured 2026-06-28: Qwen3-4B crashed at c=2. The batched
HD128 paged-decode kernels (`batch_decode_paged_hd128_q{16,32,40,64}_kv8`,
grid `(1,n_q_heads,B)`, per-row `KV_indptr`) **already existed** in
`cuda-kernels` — the executor just hardcoded `bsz=1`. Gap-1 of the unified
model-support plan ([`docs/plans/2026-06-28-cuda-unified-model-support.md`]).

## What worked

Wired batched decode through the R6-clean forward (5 files, `crates/infer-cuda/src/`):
- `loader.rs`: `PageMeta.batch` + `PageMeta::for_decode_batch` (per-row `kv_indptr`
  = prefix-sum of page counts, concatenated `kv_indices`, per-row positions).
- `attention.rs`: decode call passes `bsz=B`/`(B,B,1)` (was `(1,1,1)`); B==1 reduces
  to the original literals byte-identically. `decode_prep_paged.cu` was already batched.
- `model.rs`: `CudaModel::forward_decode_batch` — embed B → per-layer RMSNorm/QKV/RoPE
  + one batched paged attention + dense MLP/MoE over B → per-row sample.
- `executor.rs`: `submit` dispatch — `rows==1` keeps the verbatim single-row fast
  path (captured-graph + recall intact); `rows>1` → `submit_multi_row` =
  **sequential prefill sub-steps + one batched decode** (mixed plans, the
  continuous-batching case), via the existing `KvBatchDescriptor::subset`. Mirrors
  the dsv4 executor's mixed-plan structure. TP/quant-KV/recall at B>1 fall back to
  per-row sequential decode (correctness floor).

## Results — Qwen3-4B, 1×H20, fresh build (measured, llmbench two-point)

| c | reqs | tok/s | lat p50 ms | engine |
|---|---|---|---|---|
| 1 | 12 | 99.8 | 1283 | byte-identical to pre-gap-1 baseline |
| 2 | 14 | **107.0** | 2393 | **ok (was: crash)** |
| 4 | 17 | **114.2** | 4370 | **ok (was: crash)** |

**c>1 went from engine-crash (0 throughput) to batched serving**, throughput
scales 99.8→114.2 tok/s (c=1→4), c=1 unchanged, no `single-row` error. The first
impl (pure decode-batch only) still crashed on the **mixed** `1 prefill + 1 decode`
plans real concurrency emits; the GPU c>1 gate caught it (a typecheck couldn't),
and the mixed-plan extension fixed it — the value of verifying on hardware.

## Rule

The R6-clean CUDA batching bottleneck was the executor's `rows==1` gate +
hardcoded `bsz=1`, NOT the kernels (batched HD128 existed). Continuous batching
emits **mixed prefill+decode** steps — a batched-decode path that only handles
`prefill_rows.is_empty()` is insufficient; it must do sequential prefills + a
batched decode (dsv4-style) or it crashes on the first concurrent request.
Verify c>1 on hardware: a typecheck-clean batched path can still be a no-op or a
crash under the real mixed-plan schedule.
