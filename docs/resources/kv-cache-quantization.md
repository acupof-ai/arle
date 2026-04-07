# KV Cache Quantization Research

> Date: 2026-04-07 · Status: Active research

## 1. Problem

KV cache is the dominant memory consumer at inference time. For a 4B model with 36 layers, 8 KV heads, head_dim=128, BF16:

- Per token: `36 × 2 × 8 × 128 × 2B = 144 KB`
- 4K context × 4 slots = **2.25 GB** just for KV
- 128K context × 1 slot = **18 GB**

Quantizing KV cache from BF16 to INT8 halves this. INT4 quarters it.

## 2. Landscape of Methods

### 2.1 Uniform Quantization (Per-token Per-head)

Simplest approach. For each KV head at each token position, compute scale (and optionally zero-point) independently.

- **Symmetric**: `x_q = round(x / scale)`, `scale = max(|x|) / 127`
- **Asymmetric**: `x_q = round((x - zero) / scale)`, adds zero-point storage

**Frameworks using this**: LMDeploy (INT4/INT8 asymmetric), TensorRT-LLM (INT8/FP8)

### 2.2 Heterogeneous K/V Precision

Key insight: **Keys are more sensitive than Values** because they participate in softmax normalization (shared denominator). Quantization error in K distorts all attention weights.

Common configs: K8V4, K8V8, K4V4.

**Papers**: KVQuant (NeurIPS 2024)

### 2.3 Non-Uniform Quantization (NUQ)

Uses sensitivity-aware codebook instead of uniform grid. Achieves <0.1 PPL degradation at 3-bit.

**Papers**: KVQuant

### 2.4 Progressive Mixed-Precision

Different layers get different bit-widths. Sensitive layers (early, late) get higher precision. Middle layers tolerate lower precision.

**Papers**: PM-KVQ, KVTuner, MixKVQ

### 2.5 Low-Rank + Quantization

Decompose KV into low-rank approximation + residual. Quantize only the residual. Achieves 2-2.5 bits effective.

**Papers**: PALU (ICLR 2025), AQUA-KV

### 2.6 Outlier-Aware

Dynamically identify outlier tokens (high-magnitude activations) and keep them at full precision. Remaining tokens quantized aggressively.

**Papers**: OTT (ACL 2025)

### 2.7 Vector Quantization

Additive VQ with codebooks commutative with RoPE. 1-2 bit with minimal loss at 128K context.

**Papers**: CommVQ

## 3. Industrial Implementations

| Framework | Formats | Granularity | Notes |
|-----------|---------|-------------|-------|
| **vLLM** | FP8 (E4M3) | per-tensor / per-head | Needs llm-compressor calibration, Flash Attention backend |
| **TensorRT-LLM** | FP8 + INT8 | per-tensor | Decode throughput up to 1.45x |
| **LMDeploy** | INT4 / INT8 | per-head per-token (asymmetric) | Online quantization, no offline calibration |
| **SGLang** | — | — | Focuses on KV reuse (RadixAttention), not quantization |

**Hardware notes**:
- INT8: native since Pascal (SM60+), universal
- FP8: requires Ada/Hopper (SM89+). A100 emulates FP8 with 10-20% penalty
- NVFP4: Blackwell native, 50% memory reduction vs FP8

## 4. Evaluation Framework

Single PPL is insufficient. Best practice is a multi-dimensional matrix:

| Dimension | Benchmarks | Threshold |
|-----------|------------|-----------|
| **Language modeling** | WikiText-2 PPL, C4 PPL | <0.1 delta = excellent, <0.5 = acceptable |
| **Commonsense reasoning** | LM-Eval 6 tasks (ARC, HellaSwag, WinoGrande, PIQA, BoolQ, OBQA) | <1% accuracy drop |
| **Math reasoning** | GSM8K, MATH-500 | CoT chains are sensitive to KV precision |
| **Long context** | LongBench, Ruler, Needle-in-a-Haystack | Error accumulates with length |
| **Code generation** | HumanEval, LiveCodeBench | Token-level precision matters |
| **Scientific reasoning** | GPQA-Diamond, MMLU | Complex reasoning chains |
| **Attention diagnostics** | Layer-wise attention L2 error | Identifies most-affected layers |
| **Throughput & memory** | tokens/s, peak GPU mem, compression ratio | The whole point |

**Recommended eval flow**:
1. PPL quick-screen: WikiText-2 / C4, <0.1 delta → continue, >0.5 → reject
2. Reasoning validation: GSM8K + MMLU, check absolute accuracy drop
3. Long-context stress: NIAH sweep 4K→128K, LongBench multi-task
4. End-to-end throughput: fixed batch size, compare tokens/s and memory

## 5. Implementation Plan for agent-infer

### Phase 1: INT8 Per-token Per-head Symmetric (this PR)

Simplest viable approach:
- **Quantize on write**: After K/V projection, quantize to INT8 before storing in cache
- **Dequantize on read**: Before attention computation, dequantize back to BF16
- **Scale storage**: One FP32 scale per head per token position
- **Granularity**: Per-head per-token symmetric (`scale = absmax / 127`)

Memory layout change:
```
Before: k_cache[layer] = [max_seq_len, num_kv_heads * head_dim] as bf16
After:  k_cache[layer] = [max_seq_len, num_kv_heads * head_dim] as int8
        k_scales[layer] = [max_seq_len, num_kv_heads] as f32
```

Memory savings: 50% on KV data (bf16→int8), minus ~3% for scales overhead.

### Phase 2 (future): K8V4 Heterogeneous
### Phase 3 (future): Per-layer mixed precision

## References

- [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079)
- [PALU (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/7da6e0e00702c60607a6ae05c802ef85-Paper-Conference.pdf)
- [PM-KVQ](https://openreview.net/forum?id=Vem6FQvRvq)
- [OTT (ACL 2025)](https://aclanthology.org/2025.acl-long.631.pdf)
- [MixKVQ](https://arxiv.org/html/2512.19206)
- [KVTuner](https://arxiv.org/html/2502.04420v5)
- [vLLM KV Quantization](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [TensorRT-LLM vs vLLM KV Quant](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079)
- [LMDeploy INT4/INT8](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
- [NVIDIA NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [GPU-Accelerated INT8 Quantization](https://arxiv.org/abs/2601.04719)
- [HuggingFace KV Cache Blog](https://huggingface.co/blog/kv-cache-quantization)
