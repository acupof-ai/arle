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

## CORRECTION: the 396µs "gaps" were HARNESS-confounded — the real gap is serial-dependency

Follow-up sqlite query: decode window has only **16 D2H** memcpy (0.04ms) + 4146 SYNC
events at **~2µs each** (8.4ms total). So the 396µs/1.36ms "per-call gaps" are NOT host
dispatch (which is ~µs) — they're the **resident_ab harness inter-run idle** (warmup/
prefill/rep boundaries) mis-attributed to whichever kernel pair brackets them. Another
harness confound (4th: synced-profile, load, window, now inter-run).

Re-derived cleanly from per-forward instance×duration: GPU kernels ≈ **~12ms/forward**
(sm90_gemm 3.6 + mhc 2.8 + AR 1.3 + deepgemm_pack 0.7 + flash_mla 0.6 + metadata 0.6 +
allgather 0.6 + nvjet 0.7 + rms 0.4 + …). Per-token wall ~26ms → **real ~14ms gap**.

**The ~14ms gap is the GPU SERIAL-DEPENDENCY CHAIN** — ~860 tiny dependent kernels/token,
each waiting on the prior's result + the per-layer all-reduce sync; the GPU idles between
them (launch+dependency latency). This is INHERENT to the forward's serial structure and
is why per-kernel opts, the per-layer graph, and comm-overlap all washed (none shortens
the dependency chain); MTP works because it amortizes the whole chain over ~1.85 tokens.

**The 6ms lever is KERNEL FUSION** — fewer, larger kernels per layer to collapse the
serial dependency chain (the SGLang mega-kernel approach: fuse the per-layer norm+proj+
attn-prep, the mHC, the MoE pack/gemm/unpack), cutting the ~860 kernels/token to a
fraction. That is a major architectural kernel effort, not a per-op tweak. Whole-step
graph alone won't suffice (the per-layer graph washed — the dependency latency, not the
launch, is the gap). DSv4-Flash B=1 decode = ~15ms (MTP); fusion is the gate to ~6ms.
