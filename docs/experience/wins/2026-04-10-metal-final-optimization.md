# 2026-04-10 · Metal MLX Optimization — Final State

## Result
- **Start**: 54.6 tok/s
- **Final**: 63 tok/s decode-only (58 including prefill)
- **Target**: 82 tok/s (mlx_lm Python)
- **Achievement**: 77% of Python performance (+15% from start)

## Key Optimizations (in order of impact)
1. **Intermediate array retention** (+5 tok/s): Keep C++ forward() intermediates alive across steps via `intermediates` vector, preventing premature GPU buffer release
2. **Double-buffered decode** (+3 tok/s): Pass lazy (unevaluated) sample token to next forward step, overlapping CPU graph build with GPU execution
3. **mx::compile() for elementwise chains** (+1 tok/s): Fuse compute_g (10→1 op), silu (2→1), swiglu (3→1)
4. **split() instead of slice()** (+0.5 tok/s): Reduce graph node count
5. **C++ full forward pass** (0 tok/s net): Eliminates FFI overhead but same graph = same GPU time
6. **C++ full decode loop** (+1 tok/s): Entire decode loop in C++, all locals survive in loop scope

## Root Cause of Remaining 23% Gap
**C++ RAII vs Python lazy GC**: When `forward()` returns, ~300 local `mlx::core::array` variables are immediately destroyed by C++ RAII. Each destruction decrements a shared_ptr refcount. If refcount→0, MLX releases the GPU buffer. The next step must reallocate.

Python's lazy GC delays destruction — intermediate arrays survive until the next GC cycle. This allows MLX to reuse GPU buffers without reallocation, saving ~3.6ms/step.

Verified by:
- Forced GC in Python: 72→14 tok/s (5x slower!)
- Retaining intermediates in C++: 57→63 tok/s (recovered ~1.5ms/step)
- Remaining ~300 unreachable locals: ~2ms/step overhead

## What Won't Help
- mx.compile() on full forward (KV cache position-dependent slicing)
- Unfusing matmuls (net zero effect on graph node count)
- Different tensor layouts (2D vs 3D = negligible GPU kernel difference)
- Metal stream management (0.09ms difference in Python test)

## Rule
- NEVER use contiguous() carelessly — it caused 3x regression
- ALWAYS retain forward() intermediates via push_back to prevent GPU buffer churn
- Profile with `async_eval` timing to measure GPU-side overhead (not just CPU)
- C++ array destructor overhead is the #1 enemy for MLX performance from C++
