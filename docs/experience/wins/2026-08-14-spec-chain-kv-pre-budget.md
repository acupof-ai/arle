# Engine pre-budgets the spec chain KV — kills the mid-submit out-of-pages error

## Context

Issue #197: serving ThinkingCap-Qwen3.6-27B-FP8 with DSpark on one H20, 33k-token
prompts failed mid-run with `HostPagedKvPool out of pages: slot 0 needs 1, free 0`
— a step-loop error, not a preemption. The non-spec arm on the same binary and
prompts completed 8/8.

Root cause: the engine budgeted one KV token per decode row
(`plan_new_pages_needed`, `allocate_for_plan`), but the speculative executor
grows the host pool by the whole chain inside `submit`
(`grow_host_slot_to` → `host_kv.alloc`, no radix eviction) — MTP to
`start + depth + 1`, DSpark to `start + chain.len()`. When the pool's free
pages were all in the prefix cache, the direct alloc failed where the engine's
own `alloc_with_prefix_reclaim` would have evicted and succeeded.

## What worked

`BackendExecutor::spec_row_tokens` (seam, default 1) reports the total KV tokens
a decode row reaches in one submit. The engine budgets and pre-allocates that
many per decode row through its reclaim/preempt path, so the executor's
`set_host_slot_to` only truncates a warm/short row's over-budget and never
grows into an empty pool. The qwen35 executor reports `depth + 1` (MTP) or
`block_size` (DSpark); DSv4 MTP grows its own device bands and stays at 1.

The executor half (`set_host_slot_to` truncate-or-grow) landed earlier in
c5802bc9b but was reverted from the serving path by 8ad726e1c — without the
engine pre-budget it truncated host slots nobody had funded
(`materialized state len 29 != DecodeRow.kv_seq_len 34`). This change lands
both halves together.

## Measured

- Typecheck: `CUDARC_CUDA_VERSION=12080 cargo check -p arle --features cuda,no-cuda`
  green; clippy clean on infer-seam/infer-core/infer-cuda.
- Correctness/perf: `pending-remote` — re-run the #197 repro
  (`/host/dspark_on.sh`, 33k prompts, 8 requests, c=1) on the H20: expect
  8/8 completions with `KV-overflow preempt` lines in the log instead of the
  step error. No throughput delta expected at c=1 (the pre-budget is the same
  pages, allocated one tick earlier).

## Rule

A backend that grows KV inside `submit` must report its growth to the engine
(`spec_row_tokens`) so the budget and pre-allocation happen on the engine's
reclaim path, not the executor's direct alloc.
