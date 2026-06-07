# DSv4 decode gap PINPOINTED: DeepGEMM + NCCL host-DISPATCH (~396µs / ~1.36ms per call)

## Context

After confirming the decode is gap-dominated (kernels ~6-8ms vs ~26ms wall; production
c=1 33 tok/s ≈ harness, so real not harness-inflated) but every lever washed (graph,
mHC, comm-overlap, DSA-skip, GEMV), queried the nsys sqlite for the INTER-KERNEL GAP
timeline (which kernel pairs bracket the GPU-idle) — the analysis that finally locates it.

## The gap (nsys inter-kernel, decode window)

| total gap | count | per-gap | prev → next |
|---:|---:|---:|---|
| 1635ms | 4128 | ~396µs | dsv4_deepgemm_pack_quantize → sm90_fp8_gemm |
| 934ms | 688 | ~1.36ms | sm90_fp8_gemm → ncclAllReduce |
| 751ms | 1376 | ~545µs | dsv4_deepgemm_swiglu_quantize → sm90_fp8_gemm |
| 500ms | 336 | ~1.5ms | fused_q_indexer_rope → paged_mqa_logits_metadata |

The GPU idles **~396µs between the FP8 quantize and the DeepGEMM**, and **~1.36ms between
the MoE GEMM and the all-reduce**. These are **HOST-SIDE DISPATCH** gaps — the DeepGEMM
grouped-GEMM launch (JIT cache lookup + host m_indices + kernel config) and the NCCL
all-reduce launch — NOT kernel execution.

## Rule / why every prior lever washed (all explained)

- **The decode wall is HOST-DISPATCH-bound** (the per-call launch overhead of DeepGEMM +
  NCCL between the tiny kernels), not kernel/compute/comm-execution-bound.
- per-layer decode graph washed → **DeepGEMM/NCCL host-dispatch is NOT graph-capturable**
  as-built (JIT lookup, host m_indices, NCCL launch happen on the host between captures).
- comm-overlap washed → the gap is all-reduce *dispatch* (host launch), not comm execution.
- per-kernel opts (GEMV uint4 etc.) washed → the kernels are ~6% of the wall; the
  ~396µs-1.36ms host dispatch BETWEEN them is the wall.
- MTP (+71%) works → it's the only lever that amortizes the per-step dispatch (2 tok/step).
- **The 6ms lever:** kill the DeepGEMM + NCCL host-dispatch overhead — either make them
  graph-capturable (precompute m_indices, fixed-config DeepGEMM, capturable NCCL) so a
  whole-step graph replays with 0 host dispatch, OR slash the per-call dispatch cost
  (cache the JIT/kernel selection, persistent NCCL). ~396µs×43 + 1.36ms×43 ≈ 17-20ms/token
  of pure host dispatch — exactly the gap between the ~6-8ms GPU floor and the ~26ms wall.
- METHOD: to find a gap-dominated wall, query the nsys inter-kernel gap timeline (prev→next
  bracketing the idle), NOT the kernel sum (which hides the gaps) or synced stage_profile.
