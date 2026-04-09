# 2026-04-09 · Native W4 GEMV Throughput Win

## Context

Re-enabled native W4A16 GEMV kernel for decode path, eliminating the INT4→INT8 unpack workaround.
L4 GPU (23GB VRAM), Qwen3-4B, num_slots=1, greedy decode, 100 token generation.

Previous state: W4 weights were unpacked to INT8 at load time and used W8 GEMV kernel.
This meant W4 decode speed = W8 decode speed (both memory-bandwidth-bound by INT8 data).

## Results

| Format | Decode tok/s | vs BF16 | vs W8 | Model Size | VRAM |
|--------|-------------|---------|-------|------------|------|
| BF16   | 30.4        | baseline | -     | 7.6 GB     | ~13 GB |
| W8A16  | 51.4        | +69%    | baseline | 4.2 GB | ~9 GB |
| W4A16 (native, g128) | 72.3 | +138% | +41% | 2.5 GB | ~7 GB |

Industry comparison (vLLM/Marlin benchmarks):
- W4 vs BF16: expected 1.5-2x, our result 2.38x ✅
- W4 vs W8: expected 1.3-1.5x, our result 1.41x ✅

## What Worked

1. **Root cause of original W4 failure found**: `decode.rs` used BF16-only `ops::gemv` and
   `fused_mlp_into` for single-request decode path, reading from a dummy 1-element BF16 buffer
   instead of quantized weights. Prefill worked because it went through `gemm_into` (quantized-aware).

2. **Made `ops::gemv` quantized-aware**: Added W2/W4/W8 dispatch at the top of `gemv()`,
   falling through to BF16 path when weights are not quantized.

3. **Made `fused_mlp_into` quantized-aware**: Added fallback for quantized weights:
   separate gate/up GEMVs + silu_mul + down GEMV (same approach as batch decode path).

4. **Native packed upload**: `from_quantized_int4` now uploads packed INT4 data directly
   (`quant_bits: 4`) instead of unpacking to INT8 at load time. Halves weight memory bandwidth
   during decode → 41% speedup over W8.

## Limitations

- Quality comparison between scheduler (batch decode / FlashInfer paged) and single-request
  engine shows divergent greedy output — pre-existing issue in paged attention path.
- Single W4 GEMV kernel vs batch W4 GEMV kernel: both use identical unpacking math.

## Rule

Every linear projection path (single decode, batch decode, prefill) must dispatch through
quantized-aware functions. When adding new BF16 fast-paths, check that quantized weights
don't silently fall through to dummy BF16 buffers.
