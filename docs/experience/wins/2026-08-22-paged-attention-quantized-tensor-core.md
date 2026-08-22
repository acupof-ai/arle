# Quantized paged attention on tensor cores — c=16 +33 %, c=32 +34 %, the only quantized decode path

> Status: Landed (`97d28ba2c`). Phase 1 of
> [the unification plan](../../plans/2026-08-22-quantized-kv-attention-unification.md);
> follows [the GQA-group kernel](2026-08-21-paged-attention-quantized-gqa-shared-dequant.md).

## Context

The GQA-group kernel still ran the dot products on CUDA cores: per token,
six warp-reductions and scalar FMAs per lane. B=32, ctx 32 K, fp8 KV measured
5.68 ms against a 0.5 ms bandwidth floor for the 2.1 GB it reads.

## What Worked

`paged_attention_quantized_fa3_partial_kernel` is now one CTA per (batch row,
kv-head, split). The kv-head's q-heads are the rows of a 16-row
`mma.sync.m16n8k16` tile (rows beyond the GQA ratio are zero); each warp
walks 16-token tiles, dequantises the K and V bytes once into a bf16 shared
tile (exact — FP8 e4m3 and INT8 are subsets of bf16), and runs S = Q·Kᵀ
then O = P·V on the MMA with the per-token scales applied to the S columns
and to P. The four warps merge (o, m, l) through shared memory at the end.
head_dim 128/256, FP8/INT8, sm_80+; 254 registers, no spills, 42.9 KB smem.

Standalone microbench, H20, Qwen3.8-27B shapes (24/4 heads, head_dim 256),
fp8 KV, ctx 32768, 30 launches, against the GQA-group kernel at its shipped
group size:

| B | GQA-group µs | MMA µs | speedup | max diff |
|---|---:|---:|---:|---|
| 1 | 522.6 | 134.3 | 3.89× | 4.9e-4 |
| 4 | 887.8 | 315.6 | 2.81× | 4.9e-4 |
| 16 | 3210.0 | 1100.3 | 2.92× | 9.8e-4 |
| 32 | 5683.1 | 2097.9 | 2.71× | 9.8e-4 |

Diffs are 1–2 bf16 ulp from P rounding to bf16 before P·V. B=32 is 4× off
the bandwidth floor (was 11×).

End-to-end, Qwen3.8-27B-NVFP4, 1×H20, fp8 KV, MTP on, 32 K agent prompts
×32, 214 output tokens, two interleaved trials per arm, base = `e3b9b0f81`:

| arm | c=1 decode tok/s | c=16 decode tok/s | c=32 decode tok/s |
| --- | ---: | ---: | ---: |
| base t1 / t2 | 83.1 / 82.7 | 9.9 / 10.2 | 6.0 / 5.9 |
| new t1 / t2 | 83.8 / 83.9 | 14.3 / 14.3 | 8.9 / 9.0 |

Per-request decode c=16 +40 %, c=32 +50 %, c=1 wash (B=1 decode is GEMM-bound). Needle ladder
×3 at 512/4096/16384/32768: 12/12 exact, DET. 200-item GSM8K-train greedy
eval: 177/200, against 179–180/200 on the two previous binaries and 177/200
for the same-base FP8 checkpoint (binomial spread at n=200 ≈ 4 items).

Deleted with this change: the scalar partial kernel, the `heads_per_cta`
picker, `decode_attention_varlen_quantized.cu` (443 lines, the FA3-disabled
fallback) and its branch; the `head_dim == 256` gate on the quantized decode
branch. Prefill rows over a quantized pool keep the FA3 shim.

## Rule

For decode over a quantized cache the kv-head is the unit of work and the
tensor core is the unit of compute: pad the GQA group to one MMA tile,
dequantise each KV tile once into shared memory, and keep the scales on the
S columns and P so the tiles stay exact. A kernel that is 10× off its
bandwidth floor on CUDA cores is not tuned, it is in the wrong form.
