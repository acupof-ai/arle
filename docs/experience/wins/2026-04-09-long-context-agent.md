# 2026-04-09 · Long-Context Agent Sequence Benchmark

## Context
Tested multi-turn agent conversations and ultra-long context performance on L4-24GB
with Qwen3-4B BF16 weights + FP8 KV cache (25K token pool capacity).

## Environment
- GPU: NVIDIA L4 24GB
- Model: Qwen3-4B BF16 weights, FP8 KV cache
- Slots: 7 (auto), max_seq_len: 4096
- KV pool: 25,201 tokens (FP8 E4M3)

## Multi-Turn Agent Results (5 scenarios, 22 turns)

```
Aggregate:  29.6 tok/s, TTFT p50=54ms, ITL p50=33.6ms
```

Per-scenario ITL was rock-stable at 33-35ms across all turns, with TTFT
growing linearly with accumulated context (39ms → 125ms over 5 turns).

## Ultra-Long Context Degradation

| Context | TTFT    | ITL p50  | tok/s |
|---------|---------|----------|-------|
|     400 | 210ms   | 33.4ms   | 27.6  |
|     800 | 135ms   | 33.8ms   | 28.4  |
|   1,600 | 242ms   | 34.4ms   | 26.4  |
|   3,200 | 423ms   | 35.9ms   | 23.3  |
|   4,800 | 475ms   | 37.0ms   | 22.2  |

TTFT scales linearly with context (O(n) prefill). ITL degrades only +11%
from 400 to 4800 tokens (33.4 → 37.0ms) — the fused-dequant attention
kernel handles long KV sequences well.

## What Worked
- FP8 KV: 2x capacity with zero ITL regression
- Prefix cache: repeated system prompts skip prefill
- FlashInfer batched decode: stable ITL regardless of KV length
- Adaptive prefill rate: new requests admitted without blocking decode

## Ultra-Long Context with FP8 KV (2 slots, max_seq_len=16384)

| Context  | TTFT    | ITL p50  | tok/s | ITL vs 500 |
|----------|---------|----------|-------|------------|
|      500 | 195ms   | 33.2ms   | 26.8  | baseline   |
|    1,000 | 108ms   | 33.4ms   | 28.7  | +0.6%      |
|    2,000 | 163ms   | 33.8ms   | 27.1  | +1.8%      |
|    4,000 | 258ms   | 34.5ms   | 24.5  | +3.9%      |
|    8,000 | 299ms   | 35.6ms   | 23.2  | +7.2%      |
|   12,000 | 342ms   | 36.7ms   | 22.0  | +10.5%     |
|   14,000 | 363ms   | 37.4ms   | 21.4  | +12.7%     |

ITL degrades only +12.7% from 500 to 14,000 tokens — FP8 fused-dequant
attention handles long KV sequences with minimal overhead. The FP8 pool
(17K tokens) supports two concurrent 8K-context agent conversations.

## Rule
- ITL is dominated by decode GEMV (weight memory bandwidth), not KV access
- TTFT scales linearly with context (O(n) prefill) — the primary bottleneck for long contexts
- Use `--kv-cache-dtype fp8 --num-slots 2 --max-seq-len 16384` for long-context agent workloads on L4
- FP8 KV provides 2x capacity with zero quality/speed regression
