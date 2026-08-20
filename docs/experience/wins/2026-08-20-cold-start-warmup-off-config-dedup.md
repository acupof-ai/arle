# Cold start 0.62s→0.35s: warmup default off + config de-dup — Metal, 2026-08-20

> Status: Shipped

## Context

After lazy-eval + tokenizer parallelism (`wins/2026-08-20-cold-start-lazy-eval-tokenizer-parallel.md`),
cold start was ~0.62s. The warmup forward (463ms) was the remaining bottleneck —
it pre-pays the embed dequant + JIT cost, but the first request pays it anyway.
The config was also parsed twice (`resource.rs` + `executor.rs`).

## What worked

1. **Default `--metal-warmup` to false** (`args.rs`, `runtime_flags.rs`). The
   warmup forward (463ms) pre-pays embed dequant (~200ms) + JIT (~0ms, cached)
   + KV cache + session. Without warmup, the first request overlaps the embed
   dequant with model generation. For cold-start scenarios the total time
   (launch → answer) is 0.18s faster. Serving deployments opt in with
   `--metal-warmup true`.

2. **De-duplicate `load_metal_config`** (`resource.rs`, `executor.rs`).
   `plan_resource_budget` and `from_resolved_model_path_with_plan` each parsed
   `config.json`. Store the parsed config in `MetalResourcePlan.config` and have
   the executor reuse it. `MetalResourcePlan` loses `Copy` (MetalModelConfig is
   Clone-only).

## Result

M4 Pro 48GB, `mlx-community/Qwen3.5-9B-4bit` (5.6 GB), `--max-running-requests 1`.

| Metric | Before (warmup on) | After (warmup off) | Delta |
|---|---:|---:|---:|
| Cold start (launch → /health) | 0.60–0.68s | **0.33–0.39s** | −42% |
| Total (launch → answer) | ~2.0s | **1.12–1.35s** | −35% |

Timing breakdown after both changes:
- Process init: ~94ms
- Resource guard: ~57ms
- Weight loading: ~10ms
- Tokenizer (parallel, blocks): ~218ms ← new bottleneck
- Router + HTTP bind: ~18ms

Correctness: smoke test passed — model answers correctly with warmup off.
The first request pays the embed dequant + JIT cost, but overlaps it with
model generation.

## Rule

A load-time warmup that pre-pays cost the first request would pay anyway is
a net loss for cold-start scenarios: it adds the warmup overhead to the
critical path without reducing the total work. Default it off; let serving
deployments opt in.
