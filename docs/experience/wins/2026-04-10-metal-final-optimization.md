# 2026-04-10 · Metal MLX Optimization — Final

## Result
| Metric | Start | Final | mlx_lm | Ratio |
|--------|-------|-------|--------|-------|
| Decode | 54.6 tok/s | 80.6 tok/s | 82.3 tok/s | **98%** |
| Prompt | 90.5 tok/s | 302 tok/s | 181 tok/s | **167%** |
| TTFT (20 tok) | 221ms | 66ms | ~66ms | **100%** |
| Peak RSS | 1.9GB | 2.5GB | 2.5GB | same |

## Key Optimizations

### 1. Quantized lm_head (+25% decode)
Root cause of the biggest gap. Our `load_embed_tokens_from_tensors` dequantized
the 248K×2560 embedding at load time. When `tie_word_embeddings=true`, lm_head
reused this 1.2GB bf16 matrix for dense matmul. mlx_lm's `as_linear()` uses
quantized_matmul on the 0.3GB packed weights. Saves 7.5ms/step.

### 2. Batch prefill (+234% prompt, 3.3x TTFT)
Process all prompt tokens in one forward() call with S > 1 instead of
sequential per-token. Changes: reshape dims parameterized by S, full attention
uses causal mask, GDR Metal kernel T=S, conv1d naturally handles multi-token.

### 3. Double-buffered decode (+6%)
Pass lazy sample token to next forward step. CPU graph build overlaps GPU.

### 4. mx::compile() (+2%)
Fuse compute_g (10→1 op), silu (2→1), swiglu (3→1).

### 5. C++ full forward + decode loop
32 layers in one FFI call. Intermediate retention for GPU buffer reuse.

## Architecture
```
Rust → CppQwen35Model::generate()
  → qwen35_compiled_generate() [C++]
    → forward(all_prompt_tokens, S=prompt_len)  [batch prefill]
    → decode loop:
        forward(lazy_token, S=1) + async_eval    [double-buffered]
```

## Rule
- ALWAYS use quantized_matmul for tied lm_head
- Batch prefill: forward() must be S-aware (reshape by S, causal mask, T=S)
- Check memory bandwidth: weight_bytes / bandwidth = minimum kernel time
