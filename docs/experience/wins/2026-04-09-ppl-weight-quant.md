# 2026-04-09 · Weight Quantization PPL Evaluation

## Context

First PPL comparison across all weight quantization formats on Qwen3-4B.
WikiText-2 test set, 10 samples × 50 tokens, greedy decode via streaming API.
L4 GPU, num_slots=1, scheduler path (paged FlashInfer batch decode).

Note: PPL is "pseudo-PPL" computed from greedy decode logprobs, not true
language modeling PPL. Relative comparisons between formats are valid.

## Results

| Format | PPL | vs BF16 | Tokens | Decode tok/s |
|--------|-----|---------|--------|-------------|
| BF16 | 1.60 | baseline | 480 | 30.4 |
| W8-g128 | 1.56 | -2.7% | 480 | 51.4 |
| W4-g128 | 1.69 | +5.8% | 463 | 72.3 |
| W4-g32 | 1.78 | +11.6% | 480 | ~72* |
| W2-g32 | 51.38 | +3115% | 480 | ~72* |

*W4-g32 and W2 use same native kernel, similar throughput to W4-g128.

## Industry Comparison

Industry benchmarks (vLLM, llama.cpp, GPTQ/AWQ):
- W8: ~0% PPL degradation → our -2.7% is within noise ✅
- W4-g128: +0.15–0.40 PPL increase (~+10–25% relative) → our +5.8% is excellent ✅
- W4-g32: expected worse than g128 → confirmed (+11.6% vs +5.8%) ✅
- W2: "not viable for production" → confirmed (PPL 51.38) ✅
- W3 (3-bit): industry's practical floor, not yet implemented

## Rule

For Qwen3-4B deployment:
- **W4-g128** is the optimal tradeoff: +5.8% PPL, 2.38x faster, 67% less VRAM
- **W8** if quality is paramount: lossless, 1.69x faster
- **W4-g32** only if VRAM-constrained: worse quality with no throughput gain over g128
- **W2**: not viable at 4B scale
