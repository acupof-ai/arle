# 2026-05-07 · H_LP3 diagnosed — long-ctx 4k TTFT gap is cutlass small-tile GEMM, NOT per-chunk launch overhead

## Priority & ROI

**Priority**: P0 finding. Pivots the surviving long-ctx 4k/c=4
prefill gap hypothesis from "per-chunk launch overhead" to
"GEMM kernel shape selection" — these are different fixes with
different ROI.

**ROI evidence**:
- TTFT-window kernel breakdown extracted from M_nsys P1 trace:
  cutlass `Kernel2` (the small 16×16×128 wmma tensorop variant)
  = **1,417 ms / 2,500 ms = 56.7% of the TTFT window** at
  longctx 4k/c=4.
- 606 cutlass Kernel2 launches in 2.5 s = ~242 launches/sec, but
  EACH launch is 2.3 ms avg — so launch count is NOT the issue;
  per-launch GEMM work is.
- Prefill attention kernels (`prefill_attention_paged_qk_norm_rope_hd128`
  + `prefill_attention_paged_kv_write_hd128`) total only 45.4 ms
  / 2,500 ms = **1.8% of TTFT window**. H_LP3 (per-chunk attention
  launch overhead) is NOT the bottleneck.
- vLLM at the same shape: TTFT 1,177 ms vs ARLE 1,988 ms = vLLM
  1.69× faster. If we close the cutlass GEMM dispatch gap, even
  partial improvement (e.g. 30%) drops ARLE TTFT to ~1,400 ms,
  reaching parity with vLLM.

**Negative case**:
- "Pick a bigger tile" is not free — bigger tiles need more
  shared memory; may force per-SM kernels to lower occupancy.
- vLLM may not actually use a bigger tile; they may use a
  different GEMM library altogether (cuBLAS vs cutlass vs
  cuBLASLt) that has better dispatch heuristics.
- The small-tile variant may be correct for the chunk-2048 ×
  4-reqs varlen shape; the right fix may be increasing
  `chunked_prefill_size` so each launch has more work.

**Kill criteria**:
- If a single experiment (cuBLASLt fallback / TileLang prefill
  GEMM / chunk_size 4096) yields < 10% TTFT improvement, this
  finding still informs the M_b.2 prefill-axis kernel work
  (avoid small-tile cutlass under varlen).

## Goal

After M_nsys P1 validation captured 894 MB nsys-rep at longctx
4k/c=4, confirm or reject the surviving H_LP3 hypothesis (per-
chunk launch overhead) by mining the SQLite trace for first-
request-TTFT-window kernel breakdown.

## Method

1. Find the first real prefill window (post-warmup).
   `prefill_attention_paged_qk_norm_rope_hd128_kernel` first
   instances are at capture-relative ms 792-824 (warmup ~5.5 µs
   each) then JUMP to 13,376 ms when the bench's first real
   request prefills (1.77 ms each — full 2048-token chunk).

2. Sum kernel time for capture-relative ms 13,000–15,500 (the
   ~2.5 s window covering the first request's TTFT).

```sql
SELECT s.value AS kernel,
       COUNT(*) AS calls,
       SUM(k.end - k.start)/1e6 AS total_ms,
       AVG(k.end - k.start)/1e3 AS avg_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.start BETWEEN 13000000000 AND 15500000000
GROUP BY s.value
ORDER BY total_ms DESC;
```

## Results — first-request TTFT window kernel breakdown

| Kernel | Calls | Total ms | Avg µs | Notes |
|---|---:|---:|---:|---|
| `Kernel2` (cutlass 16×16×128 wmma bf16) | 606 | **1,417** | 2,339 | **56.7% of TTFT window** |
| `kernel_kernel` (TileLang prefill HD128 attn) | 102 | 456 | 4,467 | TileLang prefill kernel |
| `silu_mul_native_kernel` | 102 | 78 | 765 | FFN activation |
| `ampere_bf16_s1688gemm_128x128` | 36 | 42 | 1,161 | larger cutlass tile |
| `prefill_attention_paged_qk_norm_rope_hd128_kernel` | 300 | 35.6 | 119 | prefill prep |
| `add_native_kernel` | 203 | 29 | 143 | residuals |
| `ampere_bf16_s16816gemm_256x128` | 72 | 23 | 324 | larger cutlass tile |
| `rms_norm_batched_kernel` | 204 | 15 | 75 | per-layer norm |
| `quantize_paged_kv_fp8_kernel` | 204 | 14 | 69 | KV cache fill |
| `prefill_attention_paged_kv_write_hd128_kernel` | 300 | 9.7 | 32 | prefill KV write |

## Diagnosis

**The dominant cost in long-ctx 4k/c=4 prefill TTFT is cutlass
small-tile GEMM dispatch**:

- `Kernel2` is `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2`
  — a small 16×16 m,n tile size with K=128 stride. This is
  cutlass's wmma tensorop variant designed for varlen / small-batch
  shapes.
- 606 launches in the 2.5 s window. Per-launch overhead is NOT
  the issue (5–10 µs per `cuLaunchKernel` × 606 = ~6 ms total
  launch overhead). The issue is the kernel itself takes 2.3 ms
  per launch.
- Compare to the larger cutlass tile variants (`ampere_bf16_s1688gemm`
  128×128 = 1,161 µs, `ampere_bf16_s16816gemm` 256×128 = 324 µs)
  used much less frequently. These have higher per-call work
  but better arithmetic-to-launch-overhead ratio.

**Why is the small-tile variant being picked?**

cuBLASLt / cuBLAS dispatch for varlen prefill shapes:
- `B = 4 reqs × 2048 chunk tokens = 8192 batched rows`
- `M = 8192, N = hidden_dim, K = various`

For these shapes cuBLAS heuristics may select small-tile kernels
because the M dimension is "irregular" (not a multiple of 128
when split per-request). vLLM likely uses Triton/CUTLASS templates
that pre-tune for this shape, picking a larger tile.

**H_LP3 ruled out**:

- prefill_attention_paged_qk_norm_rope_hd128_kernel + kv_write =
  45.4 ms / 2,500 ms = 1.8% of TTFT window.
- Per-chunk attention launch overhead is real (3,924 launches
  over 60 s = ~10 ms total CPU launch cost) but immaterial.

## Next moves

### Immediate (P1)
- **Compare to vLLM trace at same shape** — does vLLM use a
  larger tile or different GEMM library? Run nsys trace on vLLM
  s8 longctx 4k/c=4 with same `--capture-range=cudaProfilerApi`
  pattern (vLLM has `cudaProfilerStart` PROFILER_START env / API
  hooks).
- **Try `chunked-prefill-size 4096`** — single chunk per request
  may shift cuBLAS dispatch to bigger tiles. (Note: codex `c219434`
  already license-killed this for TTFT/throughput tradeoff at
  longctx 4k/c=4, but the trace evidence may give a different
  answer at chunk-4096 specifically.)
- **Try cuBLASLt with explicit tile hints** — pin CUBLAS_GEMM_ALGO_*
  to a 128×128 or 256×128 variant for the prefill GEMMs.

### Medium-term (P2, gated on M_world1 Phase 0)
- **TileLang prefill GEMM port** — author dedicated TileLang
  scripts for prefill shapes (varlen M, contiguous K) similar to
  what's done for HD128 prefill attention. Probably 30-50% TTFT
  reduction if successful, given Kernel2 = 56.7% of window.
- **Combined: TileLang prefill GEMM + larger chunk size** —
  multiplicative.

### Reframe M_b.2 priority
- M_b.2 was "FP8 decode kernel" target (Phase 1 trace 41.6%
  decode). At long-ctx 4k/c=4 the TTFT gap is **prefill GEMM**,
  not decode. M_b.2 doesn't help long-ctx TTFT.
- The high-conc 1k/256/c=64 shape where ARLE leads vLLM by
  +30.3% is decode-dominated; M_b.2 would help there only if
  M_world1 Phase 0 confirms SGLang/TRT-LLM decode beats ARLE.
- **A new milestone M_pf-gemm (prefill GEMM optimization) may
  be higher priority than M_b.2.** Pending review.

## Cross-references

- M_nsys P1 validation: `f3ff34f`
- M_nsys P0 signal handler: `9b1fb8c`
- Codex H_LP1+H_LP2 license-kill: `c219434`
- Capture artifacts: `bench-output/2026-05-07-longctx-4k-c4-h_lp3-profile-nsys-signal/`
- vLLM control trace: TODO (next move P1)
- M_world1 roadmap: `docs/plans/M_world1-30-percent-lead-roadmap.md`

## Rule

- **Hypothesis ranking by trace evidence beats hypothesis ranking
  by intuition**. H_LP3 (per-chunk launch overhead) seemed plausible
  given the 800 ms TTFT gap; trace shows attention launches are
  1.8% of window and GEMM is 56.7%. Always trace before kernel
  work.
- **Prefill ≠ decode for kernel selection.** ARLE leads vLLM at
  decode-dominated shapes (high-conc +30%) but lags at prefill-
  dominated shapes (longctx 4k -3% out tok/s, 1.68× slower TTFT).
  These have different kernel hotpaths and benefit from
  different optimization tracks.
- **cutlass auto-dispatch is not always optimal at irregular
  varlen shapes.** Worth pinning tile heuristics or hand-rolling
  via TileLang.
