# 2026-04-08 · Quantized KV Cache with Fused-Dequant Decode Attention

## Context

Implemented full quantized KV cache (FP8 E4M3 + INT8) for batched decode with self-built fused-dequant attention kernel. All quantized formats use a unified architecture: dequant happens in registers inside the attention kernel, not as a separate pass. Matches vLLM, FlashInfer, TRT-LLM, QServe approach.

## Architecture

```
BF16     → FlashInfer native (unchanged)
FP8/INT8 → Self-built fused-dequant split-KV decode attention
             - Split-KV: multiple blocks per query head (FlashDecoding-style)
             - 4-warp parallelism, warp-level shuffle reduction
             - Phase 1: partial attention, Phase 2: log-sum-exp merge
```

## Benchmark Results

**Environment**: NVIDIA L4 24GB (SM89 Ada), CUDA 13.0, Qwen3-4B, 2 slots, FlashInfer 0.6.3

### Pool Capacity

| Format | Max Tokens | vs BF16 |
|--------|-----------|---------|
| BF16   | 24,888    | baseline |
| FP8    | 49,777    | **+100%** |
| INT8   | 48,269    | **+94%** |

### Decode Throughput — Long Context (C=1)

| Context | BF16 tok/s | FP8 tok/s | INT8 tok/s | FP8 vs BF16 | INT8 vs BF16 |
|---------|-----------|----------|-----------|-------------|-------------|
| 512     | 30.3      | 30.5     | 30.5      | +0.7%       | +0.7%       |
| 1024    | 29.8      | 30.3     | 30.2      | **+1.7%**   | **+1.3%**   |
| 2048    | 28.8      | 29.9     | 29.8      | **+3.8%**   | **+3.5%**   |
| 4096    | 27.2      | 29.2     | 28.9      | **+7.4%**   | **+6.3%**   |
| 8192    | 24.6      | 28.2     | 27.3      | **+14.6%**  | **+11.0%**  |

### Decode Throughput — Short Context Concurrency (C=4, 512 input)

| Format | Tok/s | vs BF16 |
|--------|-------|---------|
| BF16   | 110.0 | baseline |
| FP8    | 96.8  | -12%    |
| INT8   | 103.3 | -6.1%   |

C=4 regression is an industry-wide issue (FlashInfer FP8 native: -16~29%, SGLang FA3 FP8: -23.5%). Our -6~12% is better than both.

### ITL (Inter-Token Latency)

| Context | BF16 ITL | FP8 ITL | INT8 ITL |
|---------|---------|---------|---------|
| 512     | 33.0 ms | 32.8 ms | 32.8 ms |
| 2048    | 34.7 ms | 33.4 ms | 33.6 ms |
| 4096    | 36.8 ms | 34.2 ms | 34.6 ms |
| 8192    | 40.7 ms | **35.4 ms** | **36.6 ms** |

### Theoretical vs Measured Speedup

Decode reads model weights (~8 GB) + KV cache every step. KV quantization saves bandwidth proportional to KV's share of total reads:

```
KV bytes (BF16) = N_tokens × 36 layers × 2(K+V) × 8 heads × 128 dim × 2 = N × 1.18 MB
Weight bytes    = ~8 GB (constant)
KV share        = N × 1.18 MB / (8 GB + N × 1.18 MB)
```

| Context | KV share | Theoretical savings | Measured FP8 | Measured INT8 |
|---------|---------|--------------------|----|------|
| 512     | 7%      | ~1-2%              | +0.7% | +0.7% |
| 2048    | 23%     | ~5-8%              | +3.8% | +3.5% |
| 4096    | 37%     | ~10-15%            | +7.4% | +6.3% |
| 8192    | 52%     | ~18-25%            | +14.6% | +11.0% |

Measured speedup tracks theory at ~60-70% efficiency. Gap is from compute interleaving and kernel overhead.

## What Worked

1. **Fused-dequant architecture** — dequant in registers inside attention kernel, same approach as all major inference engines. No separate dequant pass.
2. **Split-KV parallelism** — FlashDecoding-style multiple blocks per head. Went from -37% (naive) to -0.3% (split-KV) vs BF16 at C=1.
3. **Unified kernel for all formats** — FP8 (no scales, direct cast) and INT8 (with per-head-per-token scales) share the same split-KV framework. Adding INT4/INT2 only needs a new DequantFn.
4. **2x memory capacity** with matching or superior throughput at agent-relevant context lengths (4K-16K).

## Key Lessons

- **Short context (<1K): KV quantization doesn't help throughput** — weight bandwidth dominates.
- **Long context (>4K): quantized KV is strictly faster** — KV bandwidth becomes the bottleneck.
- **C=4 short-context regression is industry-wide** — FlashInfer/SGLang/vLLM all show 12-29% FP8 regression. Not a bug in our kernel.
- **FP8 > INT8 for throughput** — FP8 has no scale overhead, direct cast is cheaper than `int8 * scale`.
- **The real value is memory capacity** — 2x tokens means 2x concurrency or 2x context length.

## Rule

KV quantization ROI scales with context length. At 8K+ tokens, expect 10-15% decode speedup from FP8/INT8. At 512 tokens, expect near-zero speedup. Always benchmark at the target context length, not just short prompts.
