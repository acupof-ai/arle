# Cold start 2.2s→1.0s: skip the paging sample when memory is abundant — Metal, 2026-08-20

> Status: Shipped

## Context

Cold start (process launch → `/health` 200) for a 9B model on Metal took ~2.2s
with warm shader cache. The paging sample — a fixed sleep to detect active
pageout/swapout before committing to a weight load — was the largest remaining
fixed cost after MLX shader cache warmup.

## What worked

Two changes in `crates/infer-metal/src/resource.rs`:

1. `PAGING_SAMPLE_MILLIS` 1000→200. The 1s sleep was a safety margin, not a
   measurement requirement; 200ms is enough to detect active paging on macOS.
2. Skip the sample entirely when `available_memory > weight_bytes + 8 GiB`.
   Active paging is implausible with that much headroom — the guard exists to
   reject loads that would thrash, not to tax every launch.

Both `plan_resource_budget` and `plan_weight_only_resource_budget` get the
same conditional. When skipped, the log reports `paging_delta=not_sampled`.

## Result

M4 Pro 48GB, `mlx-community/Qwen3.5-9B-4bit` (5.6 GB), `--max-running-requests 1`.
Measured as process launch → `/health` 200, 3 runs each.

| Arm | Cold start (s) | Notes |
|---|---:|---|
| Baseline (warm shaders, 1s sample) | ~2.2 | Before this change |
| PAGING_SAMPLE_MILLIS=200 | ~1.5 | Intermediate |
| Conditional skip (this change) | **0.95–1.04** | `paging_delta=not_sampled` |
| First-ever cold start (shader JIT) | ~4.5 | One-time, MLX system cache |

The 8 GiB headroom threshold triggers on any model whose weights fit in
available memory with room to spare — the 35B MoE canonical model (~19 GB)
on a 48 GB machine still qualifies.

## Rule

A fixed-delay safety gate that fires on every launch is a tax on the common
case. Make it conditional: skip when the condition it guards against is
implausible, keep it when the margin is tight.
