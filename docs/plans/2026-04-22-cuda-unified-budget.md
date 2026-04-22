# CUDA Scheduler Unified Budget Plan

**Status:** completed locally (2026-04-22)
**Commissioned by:** post-trace follow-up on `c16` backlog / refill analysis
**Bench anchor:** `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`
**External references:** vLLM scheduler token budget, SGLang conservative KV fit

## Goal

Unify CUDA scheduler budgeting so `runtime.rs`, `execution.rs`, and `core.rs`
consume one canonical set of token/page/headroom calculations instead of each
maintaining a local variant.

## Problem

Current CUDA scheduling still spreads the budget surface across three places:

- `runtime.rs` owns admission-time page reservation for full ISL / active-tail fit
- `execution.rs` owns step-time prefill token budget and per-slot page fit
- `core.rs` owns reclaim pressure, waiting-shortage math, and host/store queue headroom

Each piece is locally reasonable, but the overall contract is not encoded in one
shared module. That makes it harder to reason about correctness and easier for
future drift to reintroduce over-admission or refill stalls.

## Design

Introduce one shared CUDA scheduler budget module with three responsibilities:

1. **Step token budget**
   - Canonical prefill token budget:
     `min(max_num_batched_tokens - running_decode_rows, max_prefill_tokens)`
2. **Page budget**
   - One `PageBudget` implementation tracks:
     - `remaining_free_pages`
     - `planned_seq_lens`
     - `page_size`
     - `active`
   - Both admission and step planning express their fit check as a slot-local
     target sequence length plus an optional prefix floor.
3. **Pressure budget helpers**
   - `full_request_pages(...)`
   - `waiting_admission_shortage_pages(...)`
   - `prefix_cache_reclaim_goal_pages(...)`
   - `coordinator_submit_headroom(...)`

## Rules

- `runtime.rs` must stop carrying a bespoke admission page-budget type.
- `execution.rs` must stop carrying a separate page-budget implementation.
- `core.rs` may keep policy decisions, but not duplicate page-count formulas.
- Prefix floors (`reserved_prefix_tokens`) remain explicit; they are part of
  the shared budget API, not inferred ad hoc by callers.
- Host-tier/store-queue headroom stays in reclaim policy because it is a policy
  decision, but the queue-capacity arithmetic is shared.

## External alignment

- **vLLM:** one step token budget and greedy consumption against that budget.
- **SGLang:** conservative future-tail fit for KV / memory, not just immediate
  chunk fit.

The target shape here is: **vLLM-style step token budget + SGLang-style future
page reservation**, expressed through one repo-local module.

## Acceptance

- One shared budget module is the only home for page-count formulas.
- `runtime.rs` and `execution.rs` both consume the same page-budget type.
- Existing scheduler tests still pass after migration.
- Runtime behavior remains bench-verifiable through a dated win entry.

## Follow-up

Once the refactor compiles and tests pass locally, rerun a targeted `c16`
bench/trace to confirm the refactor preserves or improves the latest post-
reclaim throughput and does not regress active-set occupancy.

## Outcome

- Local refactor + targeted `c16` regression check:
  `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c16-4eddda8-unified-budget.md`
- Unified-budget bottleneck trace on the rebased/pushed tree:
  `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-unified-budget-bottleneck.md`
