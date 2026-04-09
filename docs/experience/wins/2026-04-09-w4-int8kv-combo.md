# 2026-04-09 · W4 + INT8 KV Cache Optimal Configuration

## Context

Combined W4A16 native weight quantization with INT8 paged KV cache for maximum efficiency.
L4 GPU (23GB VRAM), Qwen3-4B, num_slots=1, greedy decode, 100 tokens.

## Results

| Config | Decode tok/s | vs BF16 | Max KV Tokens | VRAM |
|--------|-------------|---------|---------------|------|
| BF16 + BF16 KV | 30.4 | baseline | 37,176 | ~13 GB |
| W8 + BF16 KV | 51.4 | +69% | 37,176 | ~9 GB |
| W4-g128 + BF16 KV | 72.3 | +138% | ~100K* | ~7 GB |
| BF16 + INT8 KV | 30.1 | -1% | 72,100 | ~13 GB |
| **W4-g128 + INT8 KV** | **70.8** | **+133%** | **204,937** | **~5 GB** |

*Estimated from weight savings.

## What Worked

- INT8 paged KV was already fully implemented (fused-dequant attention kernel)
- 46% KV memory savings confirmed (contiguous: 604MB → 328MB per slot)
- W4 weights (~2.5GB) + INT8 KV leaves ~15.6 GB for paged pool → 205K tokens
- Combined: 2.33x throughput + 5.5x concurrent token capacity vs baseline

## Rule

For maximum L4 efficiency: W4-g128 weights + INT8 KV cache.
Only use BF16 KV if exact numerical parity with single-request engine is required.
