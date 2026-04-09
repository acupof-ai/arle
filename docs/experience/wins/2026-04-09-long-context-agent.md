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

## Rule
- ITL is dominated by decode GEMV (weight memory bandwidth), not KV access
- TTFT is the bottleneck for long contexts — chunked prefill helps but is O(n)
- For contexts > 4096, increase --max-seq-len (requires more KV memory per slot)
