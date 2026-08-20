# Cold start 1.0s→0.62s: lazy merge eval + tokenizer parallelism + dead GDR merge skip — Metal, 2026-08-20

> Status: Shipped

## Context

After the paging-skip optimization (`wins/2026-08-20-cold-start-paging-skip.md`),
cold start was ~1.0s. The remaining critical path had three addressable chunks:
weight materialization (285ms, 80 blocking GPU syncs from per-merge `eval`),
tokenizer loading (197ms, sequential before engine spawn), and dead GDR merged
projections (48 lazy concat nodes never read under the default separate path).

## What worked

1. **Defer per-merge eval to warmup** (`weights.rs`). `merge_quantized_projection_rows`
   and `concat_weight_rows` called `eval()` after each `concatenate_axis`, creating
   ~80 GPU sync points for a 32-layer model (24 GDR × 2 + 32 MLP × 1). Removing
   the per-merge eval lets MLX batch all concatenations into one graph pass during
   warmup. Weight loading: 285ms → 2ms. Warmup absorbs the lazy eval: 211ms → 318ms.
   Net: −280ms.

2. **Parallelize tokenizer with engine startup** (`loaded.rs`). Tokenizer loading
   (~190ms) ran sequentially before resource planning and engine spawn. Moved to
   a background thread so it overlaps with the engine thread's resource guard +
   weight loading + warmup. Resource guard on the critical path: 255ms → 58ms.
   Net: −70ms (tokenizer finishes before engine, so no request delay).

3. **Skip dead GDR merged projections** (`qwen35.rs`). `build_qwen35_linear_attention`
   always computed `in_proj_qkvz`/`in_proj_ba`, but `AGENT_INFER_QWEN35_CPP_SEPARATE`
   defaults on, so the C++ forward uses the separate qkv/z/b/a weights and the merged
   ones are never read. Gate the merge on the env being off. 48 dead lazy concat
   nodes eliminated for a 32-layer model (60 for the 35B canonical).

## Result

M4 Pro 48GB, `mlx-community/Qwen3.5-9B-4bit` (5.6 GB), `--max-running-requests 1`.
Measured as process launch → `/health` 200, 5 runs each.

| Stage | Before (paging-skip) | After | Delta |
|---|---:|---:|---:|
| Weight loading (layer loop) | 285ms | 2ms | −283ms |
| Tokenizer (on critical path) | 197ms | 0ms (parallel) | −197ms |
| Resource guard | 84ms | 58ms | −26ms |
| Warmup | 211ms | 318ms | +107ms |
| **Cold start (measured)** | **0.95–1.04s** | **0.60–0.68s** | **−330ms** |

First-ever cold start (shader JIT): ~4.5s → unchanged (one-time MLX system cache).

Correctness: smoke test passed — model answers correctly after all three changes.
The dead GDR merge skip only affects weights that are never read under the default
separate path; the C++ forward uses the unchanged separate weights.

## Rule

A blocking `eval()` after each lazy op is a GPU sync point. When the ops are
independent (per-layer weight merges), batch them into one graph pass. And
work that doesn't depend on the engine thread (tokenizer loading) should not
block it.
