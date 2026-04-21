# Metal Qwen3 scheduler simplification kept the low-pressure path flat, but canonical pressure still hits MLX resource_limit

## Context

- I simplified the Metal scheduler/runtime path for `Qwen3-0.6B-4bit` by keeping the new lightweight waiting-to-activation flow, but deleting the extra long-prompt running-budget policy and the Qwen3 prompt-cache size heuristic that did not produce a measurable fix.
- The verification target was the same long-prompt HTTP shape as the earlier canonical baseline:
  - canonical attempt: `scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-scheduler-simplify --target http://127.0.0.1:8019 --model mlx-community/Qwen3-0.6B-4bit --processor models/Qwen3-0.6B`
  - clean sync comparison: `scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-scheduler-simplify-sync-fresh --profile synchronous --max-seconds 30 --target http://127.0.0.1:8019 --model mlx-community/Qwen3-0.6B-4bit --processor models/Qwen3-0.6B`
- Reference baseline: [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md](../wins/2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md)

## Root Cause

- Simplifying the scheduler did not remove the underlying MLX/Metal allocator failure. The canonical `4096 / 256` sweep still reproduced:
  - `mlx_array_from_data returned a null MLX handle: [metal::malloc] Resource limit (499000) exceeded.`
  - the runtime caught it as `Metal prefill chunk panicked for RequestId(5)`.
- The failed canonical run also pushed the HTTP layer into repeated `Scheduler unavailable or full` errors during the pressure leg, which is consistent with the runtime surviving the panic but not eliminating the allocator-pressure source.
- The clean low-pressure sync rerun stayed close to the earlier baseline:
  - previous canonical sync: `TTFT p50 1135.4 ms`, `ITL p50 6.07 ms`, `out 90.42 tok/s`
  - simplified fresh sync: `TTFT p50 1080.7 ms`, `ITL p50 5.91 ms`, `out 80.06 tok/s`
- That pattern points away from scheduler complexity as the primary cause. The remaining issue is still in the long-prompt MLX allocation lifetime / object-count path under pressure, not in the existence of the deleted admission heuristics.

## Fix

- Kept:
  - lightweight `PendingMetalRequest` waiting state
  - activation-time creation of heavy `ActiveMetalRequest`
  - unified scheduler-owned waiting capacity via `max_waiting_requests`
- Removed:
  - long-prompt running-budget knobs in `MetalSchedulerConfig`
  - the Qwen3 `should_cache_prompt_len()` prompt-cache heuristic
  - the request-finish `clear_metal_cache()` heuristic
- Result:
  - code is simpler and the low-pressure sync path remains healthy
  - canonical pressure still fails for the same allocator reason

## Rule

- Keep the Metal scheduler simple unless a policy change has bench evidence.
- If a canonical long-prompt run still dies with `metal::malloc Resource limit (...) exceeded` after scheduler simplification, stop tuning admission heuristics and inspect MLX allocation lifetime / resource-count behavior directly.
