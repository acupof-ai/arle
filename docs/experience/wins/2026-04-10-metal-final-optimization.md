# 2026-04-10 · Metal MLX Optimization — Final State

## Result
- **Start**: 54.6 tok/s
- **Final**: 80.7 tok/s decode-only (75.7 including prefill)
- **Target**: 82 tok/s (mlx_lm Python)
- **Achievement**: 98% of Python decode performance (+48% from start)

## Key Optimizations (in order of impact)
1. **Quantized lm_head as_linear** (+15 tok/s, +25%): THE root cause — our lm_head
   did dense bf16 matmul (1.2GB weight read) while mlx_lm uses quantized_matmul
   (0.3GB). Saves ~7.5ms/step at 120 GB/s bandwidth.
2. **Double-buffered decode** (+3 tok/s): Pass lazy sample token to next forward step
3. **mx::compile() for elementwise chains** (+1 tok/s): Fuse compute_g, silu, swiglu
4. **C++ full forward pass + decode loop**: All 32 layers in one FFI call
5. **split() instead of slice()**: Reduce graph node count
6. **Intermediate array retention**: Keep forward() intermediates alive across steps

## Critical Lesson
The biggest performance issue (25% of total gap) was a **semantic difference**,
not a code quality issue. Our `load_embed_tokens_from_tensors` dequantized the
embed weights at load time. When `tie_word_embeddings=true`, the lm_head reused
this dequantized tensor — turning a quantized_matmul into a dense matmul.
Python's `QuantizedEmbedding.as_linear()` keeps the weights quantized.

## Rule
- When `tie_word_embeddings=true`, ALWAYS use quantized_matmul for lm_head
- NEVER dequantize large embedding matrices for matmul — keep quantized
- Profile with `async_eval` timing to measure GPU-side overhead
- Check memory bandwidth: weight_size / bandwidth = minimum kernel time
