# 2026-04-10 · Metal MLX Optimization — Final State

## Result
- **Start**: 54.6 tok/s decode
- **Final**: 81.0 tok/s decode (98.4% of mlx_lm)
- **Target**: 82.3 tok/s (mlx_lm Python)
- **Total improvement**: +48.4%

## Performance Comparison

| Metric | Ours | mlx_lm | Ratio |
|--------|------|--------|-------|
| Decode throughput | 81.0 tok/s | 82.3 tok/s | 98.4% |
| Prompt throughput | 90.5 tok/s | 181.0 tok/s | 50% |
| TTFT (20 tokens) | 221ms | ~66ms | 3.3x |

Prompt speed gap is from sequential per-token prefill (our C++ forward
processes one token at a time) vs mlx_lm's batched prefill. This only
affects TTFT, not decode throughput.

## Key Optimizations (in order of impact)

### 1. Quantized lm_head as_linear (+25%, THE root cause)
Our `load_embed_tokens_from_tensors` dequantized the embedding matrix at
load time (248K × 2560 bf16 = 1.2GB). When `tie_word_embeddings=true`, the
lm_head reused this dense tensor for matmul — reading 1.2GB per step.

mlx_lm's `QuantizedEmbedding.as_linear()` uses `quantized_matmul` on the
original 4-bit packed weights (0.3GB read). At 120 GB/s memory bandwidth,
this saves **7.5ms per step**.

Fix: Store original quantized embed weights and use `quantized_matmul` for
the lm_head projection.

### 2. Double-buffered async_eval decode loop (+6%)
Pass the lazy (unevaluated) sample token directly to the next forward step.
The CPU builds step N+1's graph while the GPU executes step N.

### 3. mx::compile() for elementwise chains (+2%)
Match mlx_lm's compiled functions:
- `compute_g`: 10 ops → 1 compiled kernel (24 GDR layers)
- `silu`: 2 ops → 1 compiled kernel
- `swiglu`: 3 ops → 1 compiled kernel (32 MLP layers)

### 4. C++ full forward pass + decode loop (+1%)
All 32 layers in a single FFI call. Entire decode loop in C++.

### 5. Graph optimization: split(), 3D layout, intermediate retention (+1%)
- `mx::split()` instead of separate `slice()` calls
- 3D tensor layout `[1,1,H]` matching mlx_lm
- Retain intermediate arrays across steps for GPU buffer reuse

## What Didn't Help (verified experimentally)
- **Unfusing matmuls**: Fused (2 matmul + split) vs separate (4 matmul) = same perf
- **Linking system libmlx.dylib**: Same performance as source-built static lib
- **Metal stream management**: 0.09ms difference (negligible)
- **Array destruction overhead**: 0ms per 150 arrays (disproved GC theory)

## Architecture

```
Rust CLI (metal_bench/metal_request/metal_serve)
  → MetalBackend::generate()
    → CompiledQwen35::generate()  [Rust RAII wrapper]
      → qwen35_compiled_generate()  [C++ full decode loop]
        → forward()  [C++ 32-layer forward, all MLX ops]
          → quantized_matmul, rms_norm, rope, sdpa, Metal GDR kernel
          → compiled compute_g, silu, swiglu via mx::compile()
          → quantized lm_head via embed_as_linear
```

## Rule
- **ALWAYS** use `quantized_matmul` for tied lm_head — never dense matmul
- Profile with `async_eval` timing to separate CPU graph build from GPU eval
- Check memory bandwidth: `weight_bytes / bandwidth = minimum kernel time`
- `mx::compile(fn, shapeless=true)` for all elementwise chains
