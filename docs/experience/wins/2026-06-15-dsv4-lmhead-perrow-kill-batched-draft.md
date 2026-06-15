# DSv4 lm_head per-row kill — cuBLAS verify GEMM + batched-draft default-ON

**Date:** 2026-06-15 · **Backend/model:** CUDA / DeepSeek-V4-Flash · **SKU:** 8×H20
TP=8/EP=8, CUDA 12.9, sm_90a · **Track:** decode throughput (steady c≥4 MTP).

## Goal

Kill the per-row lm_head GEMV — nsys's #1 decode hog (19 %, `gemv_handwritten`
558 µs/call) — without quantizing lm_head or hand-writing a kernel. ckl: "逐行循环
本来就不应该存在吧 直接修复吧 … 先用 cuBLAS 吧 以及看看网络上有没有更好的 kernel".

## Hypothesis

The lm_head step is **weight-read bandwidth-bound** (reads the full ~1 GB bf16
vocab weight, `vocab 129 280 × hidden 4096 × 2 B`, **replicated** per rank). The
slowness is not the kernel — it is **how many times the 1 GB weight is read per
step**: the per-slot draft + per-row verify re-read it 16–48×. Collapse the reads
by batching; cuBLAS handles the batched bf16 GEMM at the HBM floor.

## What worked

Two stacked changes that together remove every per-row lm_head call on the
production MTP decode path:

1. **`5d6eb0da` — verify lm_head → one cuBLAS GEMM.** `lm_head_project_batch`'s
   DenseBf16 arm was a per-row GEMV loop (`for r { gemv }`), re-reading the 1 GB
   weight per row. Collapsed to `dsv4_linear → ops::gemm_batch → cuBLAS`
   (`gemm_cuda`), weight read once for all M rows. `gemm_batch` always routes
   bf16 to cuBLAS — never the hand GEMV — even at M=1. The verify
   (`forward_decode_batch_verify`) batches all slots' chains → **1 GEMM/step**.
2. **Batched draft default-ON (`ARLE_DSV4_BATCHED_MTP_DRAFT`, this commit).** The
   default `draft_chain` was per-slot serial → c=16 fired **16× (m=1)** lm_head
   GEMMs/step (16 full weight reads). `mtp_forward_level_batched` batches the N
   slots per depth-level → **1× (m=N)** GEMM (one weight read).

**Lever 2a's prior "marginal" verdict was stale** — it was measured *before*
`5d6eb0da`, while the batched lm_head still looped per-row internally, so batching
the slots saved nothing on the lm_head. Re-measured on the fixed binary it now
pays off.

## Params / Env

`arle serve --backend cuda --model-path DeepSeek-V4-Flash --port … --num-slots 16
--spec-type mtp --mtp-draft-tokens 2`, `ARLE_DSV4_BATCHED_MTP=1`,
`INFER_DSV4_MAX_SEQ_LEN=8192`, allreduce MoE + deepgemm experts. Same-binary A/B
(both arms carry the cuBLAS fix); only `ARLE_DSV4_BATCHED_MTP_DRAFT` flips.
Driver `dsv4_perf.py` (barrier-sync C generations, `/v1/stats` steady window),
coherence via `dsv4_needle_concurrent.py` (word needle, c=8, filler=20).

## Results

| arm | c=8 agg tok/s | c=16 agg tok/s | c=16 per-req | coherence |
|---|---|---|---|---|
| control — per-slot draft | 77.80 (act 8.0) | 86.25 (act 15.0) | 5.75 | cross-contam 0 |
| **treat — batched draft** | **83.09 (act 8.0)** | **95.82 (act 16.0)** | **5.99** | cross-contam 0 |
| **Δ** | **+6.8 %** | **+11.1 %** | **+4.2 %** | clean |

- c=8 is concurrency-matched (both avg_active 8.0) → **+6.8 % is a clean,
  confounder-free aggregate win**. c=16 per-req +4.2 % (compute), aggregate +11.1 %
  (compute + treat reaching the full 16th slot). The win **scales with
  concurrency**, exactly as the per-slot-weight-read model predicts.
- Coherence: **cross-contam = 0 in both arms** — batching the slots' drafts leaks
  no cross-slot state (the verify still validates each draft token against the
  target greedy argmax). The 3 needle "MISS" (`marig`/`narwh`/`quaser`) are
  char-level truncations at the generation boundary, **identical in both arms** —
  a test-harness artifact, not corruption (per `feedback_correct_inference_not_baseline_identity`).

## Problems / what is NOT the lever (online kernel survey)

See [`docs/research/2026-06-15-dsv4-lmhead-decode-kernel-survey.md`](../../research/2026-06-15-dsv4-lmhead-decode-kernel-survey.md).
- **Keep bf16.** SGLang/vLLM **explicitly skip lm_head in quantization** → FP8/FP4
  lm_head is off-policy/precision-risky. Rejected.
- **DeepGEMM is FP8-only** (bf16 WIP, M-axis grouping) → not usable for the bf16
  lm_head. cuBLAS is the right bf16 primitive.
- **Fused sampling (FlashInfer/FlashSampling)** saves only the ~0.26 MB logits/row
  = 0.05 % of the 1 GB weight traffic → wrong bottleneck.
- **Follow-on (not yet done):** vocab-parallel shard the lm_head (`ParallelLMHead`
  equiv) — ARLE's `load_dsv4_global_matrix` replicates the full vocab per rank;
  sharding → 1/8 weight/rank + a cheap logits all-gather. Marginal once the
  per-row reads are already down to ~2/step, so deferred.

## Rule

For a replicated large-vocab lm_head, decode cost is **(weight bytes) × (calls per
step)**, not kernel quality. Batch the calls before reaching for a fancier kernel
or quantization — and **re-test any lever whose verdict predates a prerequisite
fix** (Lever 2a was "marginal" only because the batched lm_head it depended on was
still looping per-row). Supersedes the marginal snapshot
[`2026-06-15-dsv4-batched-mtp-lever2a-draft-marginal.md`](2026-06-15-dsv4-batched-mtp-lever2a-draft-marginal.md).
