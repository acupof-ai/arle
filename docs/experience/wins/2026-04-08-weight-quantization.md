# 2026-04-08 · Weight Quantization W8A16/W4A16

## Context

Implemented weight-only quantization (W8A16, W4A16) to reduce decode memory bandwidth. The decode bottleneck for small models (4B) is weight loading (~8GB per step), not KV cache.

## Results (L4 24GB, Qwen3-4B, 2 slots)

| Config | Decode tok/s | ITL | vs BF16 | Weight Size | Quality |
|--------|-------------|-----|---------|-------------|---------|
| BF16   | 31.1        | 32ms | baseline | 8 GB | ✅ |
| **W8A16** | **53.4** | **18.7ms** | **+72%** | 4 GB | ✅ |
| W4A16  | 43.1        | 23.2ms | +39% | 2 GB | ⚠️ naive symmetric |

### W8A16 Details
- Unified GEMV kernel: 256 threads, 4 rows/block, 128-bit vectorized loads
- Per-group-128 bf16 scales, symmetric quantization
- Quantization error: mean=0.000132, max=0.0062 (0.69% relative)
- Greedy decode output identical to BF16

### W4A16 Details
- Packed INT4 (2 values/byte), same kernel framework
- 43.1 tok/s slower than W8 because INT4 unpack overhead dominates at small K
- Quality degraded with naive symmetric quantization — needs GPTQ calibration

### Combined with KV Quantization

| Config | Weight | KV | Decode tok/s | Pool Capacity |
|--------|--------|-----|-------------|---------------|
| BF16 + BF16 KV | 8 GB | 2 bytes/elem | 31.1 | 24,888 tokens |
| **W8 + FP8 KV** | **4 GB** | **1 byte/elem** | **53.4** | **49,777 tokens** |
| W4 + FP8 KV | 2 GB | 1 byte/elem | 43.1 | ~60K tokens |

## What Worked

1. **Unified kernel template** — W8 and W4 share the same block/thread structure, only differ in unpack logic
2. **Single scale per vector load** — group_size=128 is always multiple of VEC_SIZE(16/32), eliminating per-element scale lookup
3. **Automatic dispatch** — `gemm_into` checks `qweight.is_some()` and routes to quantized GEMV transparently
4. **Offline quantization script** — one command: `python3 scripts/quantize_weights.py model --bits 8`

## Rule

W8A16 is the sweet spot for single-GPU decode: 72% speedup with zero quality loss. W4A16 needs calibrated quantization (GPTQ/AWQ) for acceptable quality.
