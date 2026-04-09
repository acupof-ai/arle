# 2026-04-09 · Metal MLX Optimization Progress

## Context
Optimizing Qwen3.5-4B-MLX-4bit decode throughput on Apple Silicon Metal/MLX.
Target: match mlx_lm's 78 tok/s.

## Progress

| Milestone | tok/s | Delta | Key Change |
|-----------|-------|-------|------------|
| Baseline (mlx-rs) | 54.6 | — | mlx-rs crate, no optimization |
| Custom mlx-sys | 51.3 | -6% | Direct C++ bridge, migration cost |
| Double-buffered decode | 55.5 | +8% | Lazy token → forward, CPU/GPU overlap |
| C++ full forward (fix contiguous) | 57.7 | +4% | 32 layers in 1 FFI call |
| Compiled compute_g + silu + swiglu | 58.3 | +1% | mx::compile on elementwise chains |
| **Current** | **58.3** | **+7% total** | |
| mlx_lm target | 78.0 | +36% needed | |

## What Worked
1. **Double-buffered async_eval** (+8%): Pass lazy (unevaluated) sample token to next forward step. CPU builds graph while GPU executes.
2. **mx::compile() on elementwise chains** (+1-3%): Fuses sigmoid+multiply (silu), compute_g (10 ops → 1 kernel).
3. **C++ full forward**: Eliminates ~1600 FFI calls per step. No speed gain vs Rust per-op (proves FFI is not the bottleneck).
4. **contiguous() is poison**: Adding contiguous() to conv state caused 3x regression (20 tok/s). Single line, massive impact.

## Remaining Gap Analysis (58 → 78, 25%)

### Confirmed NOT the cause:
- Per-op FFI overhead (C++ forward = same speed as Rust per-op)
- mx.compile() on full forward (mlx_lm doesn't use it)
- Graph node count (555 counted ops in mlx_lm, similar in our code)
- Weight loading (both use MLX mmap)

### Likely causes:
- **Python nanobind zero-overhead arrays**: No heap alloc per intermediate. Our C++ forward uses stack arrays (no heap) but from_arr for 65 outputs.
- **MLX internal optimization differences**: Python's pybind creates arrays differently from our `new array(std::move(result))`.
- **Graph scheduling**: MLX may schedule Python-constructed graphs more efficiently.
- **Remaining uncompiled ops**: ~100 elementwise ops still not fused.

## Rule
- NEVER use `contiguous()` in the forward path — it adds copy ops that serialize the graph.
- Always verify with A/B test (env var gate) before committing perf changes.
- `mx::compile(fn, shapeless=true)` is the RIGHT way to fuse elementwise ops.
