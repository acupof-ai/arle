# DSv4 decode lever #1 (projection GEMV): the existing MMA tensor-core GEMV is KILLED — pursue DeepGEMM fusion instead

## Context

The clean decode nsys breakdown ([`2026-06-07-dsv4-decode-nsys-real-breakdown.md`](2026-06-07-dsv4-decode-nsys-real-breakdown.md))
pinned the #1 decode GPU kernel as the **scalar projection GEMV**
`dsv4_fp8_gemv_batch_kernel` (3.62 ms/token, 18.9% GPU, <1% tensor pipe). In the
fastest decode variant (`flashmla_fused_wqkv`), `fused_wqkv` already moved
`wq_a|wkv` to DeepGEMM (+18.4%); the residual 3.62 ms is **wq_b, wo_a, wo_b**
(task #36's "剩 3/4 未融"), each `dsv4_linear → mla_linear → dsv4_fp8_gemv_batch`.

A tensor-core variant already exists in-tree: `quantized_gemv_mma.cu`
(`dsv4_fp8_gemv_batch_mma_launch`), gated behind `ARLE_DSV4_FP8_GEMV_MMA`,
written as "GAP-A" but **never licensed / default-off**. Cheapest lever-#1 test:
license-or-kill A/B it (no new code, true "先用最好的").

## What worked (the kill, with evidence)

Resident A/B, 8×H20 TP=8, DeepSeek-V4-Flash, `flashmla_fused_wqkv` variant,
default prompt `671,6102,294,8760,344`, vendored DeepGEMM, max_new=80 (63 steady):

| `ARLE_DSV4_FP8_GEMV_MMA` | steady tok/s | decode ms/tok | prefill ms | output |
|---|---:|---:|---:|---|
| `0` (scalar, default) | **38.76** | 25.8 | 2223 | coherent (`[11111,14,778,344,990,…]`) |
| `1` (MMA tensor-core)  | 26.28 | 38.1 | 4694 | **all-zero `[0,0,0,…]`** |

The existing MMA GEMV is **both broken (all-zero output) and slower (−32% decode,
−2× prefill)**. KILL: do not enable `ARLE_DSV4_FP8_GEMV_MMA`; the kernel is
unvalidated GAP-A cruft (candidate for deletion).

## Rule

- Lever #1 (projection GEMV → tensor-core) must go through **DeepGEMM** (the proven
  path: `dsv4_deepgemm_pack_quantize_bf16_to_fp8` + `dsv4_deepgemm_fp8_gemm_nt`, the
  same helpers `fused_wqkv` uses for `wq_a|wkv`), NOT the hand-rolled
  `quantized_gemv_mma.cu` (broken + slower). Mirrors `feedback_no_closed_door_solutions`.
- Implementation scope for wq_b/wo (the residual 3.62 ms): one-time repack of each
  FP8 weight to DeepGEMM block-scale layout at load (like the fused cache) +
  pre-allocated per-shape FP8 input scratch (no per-step alloc, per
  `reference_disabled_event_tracking_premature_buffer_free`) + per-call quantize +
  `dsv4_deepgemm_fp8_gemm_nt` at M=1, gated + same-load A/B + oracle/needle-gated
  before any default flip.
- A license-or-kill A/B is mandatory before trusting any default-off "tensor-core"
  kernel — `<1% tensor pipe` in the profile did not mean a ready replacement existed.
