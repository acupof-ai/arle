# DSv4-Flash decode: FP8 fused wqkv_a linear → +5.07% on top of FlashMLA

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** landed, **gated** (`ARLE_DSV4_FUSED_WQKV_DECODE=1` + alloc gate),
correctness-verified, matched A/B. Endgame lever #2
([`2026-06-05-dsv4-endgame-architecture-adopt-best-first.md`](../../plans/2026-06-05-dsv4-endgame-architecture-adopt-best-first.md)).

## Context

Lever #2 = the FP8 attention-linear slice of decode. The prior per-projection
DeepGEMM attempt shipped ~0.8%
([[errors/2026-06-05-fp8-linear-per-projection-deepgemm-no-win]]) because 344
DeepGEMM calls/forward ate the WGMMA win on per-call launch + per-projection
quantize. The lesson: **adopt SGLang's fused call *structure*, not just the
kernel** (`先用最好的再自己写` — [[feedback_no_closed_door_solutions]]).

## What Worked

Reading SGLang's actual `dsv4`/`deepseek` backend revealed the real fusion is
**`wq_a + wkv → wqkv_a`** (the two down-projections concatenated into one GEMM),
*not* all MLA projections fused at once. ARLE now mirrors it for the
**seq_len==1 decode** path only (gated):

- **Load** (`loader.rs`): when the alloc gate is on, build
  `Dsv4Attention.wqkv_a_deepgemm: Option<Dsv4Fp8DeepGemmWeightCache>` via
  `from_dsv4_weight_pair_rows(wq_a, wkv)` — the concatenated FP8 DeepGEMM weight.
  `None` (default) → no extra VRAM.
- **Decode** (`attention.rs::run_fused_wqkv_decode`): activation quantized
  **once** (`dsv4_deepgemm_pack_quantize_bf16_to_fp8`), **one** dense
  `dsv4_deepgemm_fp8_gemm_nt`, then q/kv RMSNorm via pointer-offset slices of the
  fused output (`mla_rms_norm_decode_slice`) — no split kernel.
- Gate mirrors FlashMLA: `set_dsv4_fused_wqkv_decode_override(Option<bool>)`
  (`None`=env), inert by default → production path byte-identical.
- New `linear_profile.rs`: CUDA-event per-op timer (gated `ARLE_DSV4_LINEAR_PROFILE`,
  OnceLock zero-cost when off) for the linear-slice breakdown.

## Results (same-load, B=1, 64-tok decode, warmup=16, both orders × 3 reps)

| variant | mean tok/s | σ | vs flashmla |
|---|---:|---:|---|
| flashmla | 28.016 | 0.058 | baseline |
| flashmla + fused wqkv_a | **29.437** | 0.038 | **+5.07%** |

- Correctness: **16/16 oracle PASS**, 64-step both orders PASS.
- Cumulative arc: scalar 23.7 → FlashMLA 28.0 → +fused 29.4 tok/s.

## Honest read

- **Gated landing, not a default flip.** The fused DeepGEMM cache needs a
  load-time allocation gate and costs noticeably more VRAM; the runtime decode
  win is stable but unconditional-default needs the VRAM budget + a multi-shape
  check.
- +5.07% is the decode-only A/B at seq_len==1; the fused path is decode-specific
  (prefill keeps the existing path).

## Rule

When adopting an upstream fusion, **read which projections it actually fuses** —
SGLang fuses only `wq_a+wkv` (the down-projections), not the whole MLA linear
stack. The win comes from the *call structure* (quantize-once + one GEMM +
sliced-output RMSNorm), confirmed by a matched same-load A/B, not from the kernel
in isolation (per-projection was 0.8%).
