# 2026-04-08 · INT8 KV Cache Batched Decode — Phase A Complete

## Context

Extended INT8 KV cache quantization from single-request to the full batched decode pipeline. TokenKVPool now supports dual-storage: INT8 persistent data (all layers) + shared bf16 working buffer (1 layer). Per-layer dequant→attention→quant cycle keeps FlashInfer unchanged.

## What Worked

**1.94x more tokens in the same memory budget:**

| Metric | BF16 | INT8 | Delta |
|--------|------|------|-------|
| Pool max tokens | 24,888 | 48,269 | **+94%** |
| Storage | 3.7 GB | 3.6 GB (data) + 0.1 GB (scales) + 0.2 GB (working) | ~same budget |

**Zero throughput regression** on L4 24GB with Qwen3-4B:

```
                     BF16          INT8          Delta
In=128 Out=128 C=1: 30.0 tok/s    29.8 tok/s    -0.7%
In=128 Out=512 C=1: 29.7 tok/s    29.6 tok/s    -0.3%
In=512 Out=256 C=1: 29.2 tok/s    29.1 tok/s    -0.3%
In=1024 Out=256 C=1: 28.3 tok/s   28.2 tok/s    -0.4%
In=2048 Out=256 C=1: 26.7 tok/s   26.7 tok/s    0.0%
In=512 Out=256 C=4: 109.9 tok/s   109.3 tok/s   -0.5%

ITL p50: 33.2-36.5ms (BF16) vs 33.4-36.6ms (INT8) — negligible difference
TTFT: identical across all configs
```

**Architecture**: Dual-storage pool design — INT8 buffers for all layers + 1-layer bf16 working buffer shared across layers. Per-layer: dequant INT8→bf16, FlashInfer reads bf16, quant new token bf16→INT8. CUDA Graph safe (all pointers pre-allocated).

**Environment**: NVIDIA L4 24GB, CUDA 13.0, SM 89, Qwen3-4B, 4 slots, FlashInfer 0.6.3

## Rule

INT8 KV quantization achieves ~1.94x memory savings with negligible throughput cost (<1%) because decode is compute-bound, not memory-bound at these sequence lengths. The dequant/quant overhead per layer is hidden by the dominant GEMV/attention computation. At longer contexts (>8K) the dequant bandwidth may become noticeable — Phase A2 (incremental dequant) should address this.
