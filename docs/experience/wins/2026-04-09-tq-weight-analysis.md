# 2026-04-09 · TurboQuant Weight Quantization Analysis

## Context
End-to-end testing of TurboQuant weight quantization on Qwen3-4B (L4-24GB).
Investigated quality degradation (repeated tokens) and kernel performance.

## What Worked
- **Warp-level FWHT**: Replaced shared-memory FWHT (9 syncs/group) with
  `__shfl_xor_sync` for stride 1-16 (5/7 stages). Result: +24% decode throughput
  (15.6 → 19.3 tok/s), ITL 64ms → 52ms.
- **Memory compression**: 7.6 GB → 1.87 GB (3.9x), enabling 11 slots vs 7 (BF16).
- **Conversion pipeline**: Python quantization script + safetensors index rewrite works end-to-end.

## Key Finding: Quality
TQ rotation is **provably optimal** for its bit width — calibrated centroids
match analytical Beta-distribution centroids exactly (same SNR to 0.1 dB).
The quality issue is fundamental:

| Bits | SNR (dB) | NMSE/layer | 36-layer accumulated | Usable? |
|------|----------|------------|---------------------|---------|
| 3    | 14.7     | 3.4%       | ~120% noise         | No (4B) |
| 4    | 20.3     | 0.94%      | ~34% noise           | No (4B) |
| 8    | 39.0     | 0.01%      | ~0.4% noise          | Yes     |

TQ beats RTN by +2.6 dB at 4-bit, but the accumulated error across 36 layers
still destroys output for small models. GPTQ/AWQ avoid this via Hessian-aware
rounding that preserves cross-layer coherence.

## Rule
- **TQ for KV cache**: excellent (per-token error doesn't accumulate across layers)
- **TQ for weights on small models (4B-8B)**: not viable at 3-4 bit
- **TQ for weights on large models (70B+)**: potentially viable (more capacity per layer)
- **Weight quantization for small models**: use GPTQ/AWQ (already production-ready in engine)
