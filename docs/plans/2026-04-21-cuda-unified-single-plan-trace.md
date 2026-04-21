# CUDA Unified Single-Plan Trace — 2026-04-21

## Goal

把 CUDA scheduler 当前一个 tick 内真正由 scheduler 自己拥有的耗资源阶段梳理清楚，并明确标出只服务于旧“两段式 mixed + extra serial prefill”路径的状态，作为这次删除式重构的核对表。

## Scheduler-Owned Resource Stages

### 1. Emit / Tokenizer

- Waiting admission tokenization:
  [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs)
  `admit_waiting_prefill_batch()`
- Incremental stream emit: [`infer/src/scheduler/cuda/request.rs`](../../infer/src/scheduler/cuda/request.rs) `ActiveRequest::emit_delta()`
- Final flush + usage accounting: [`infer/src/scheduler/cuda/request.rs`](../../infer/src/scheduler/cuda/request.rs) `ActiveRequest::finish()`

Owned state:

- `QueuedRequest.prompt_tokens`
- `ActiveRequest.generated_tokens`
- `ActiveRequest.full_decoded`
- `ActiveRequest.decoded_token_count`
- `ActiveRequest.sent_len`
- `ActiveRequest.latest_logprob`

### 2. Admission / Prefix / Spill Classification

- Step planning entry: [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs) `plan_step()`
- Existing prefill batch selection: `queued_prefill_batch()`
- Waiting-queue admission + radix decision:
  `admit_waiting_prefill_batch()`
- Slot materialization: [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs) `materialize_waiting_request()`
- Token budget: `chunked_prefill_size` is the total prefill-token budget for one planned tick; `max_prefill_tokens` may only tighten it.

Owned state:

- `Scheduler.waiting`
- `Scheduler.prefill_queue`
- `Scheduler.active`
- `Scheduler.block_owner_slots`
- `Scheduler.slot_materialized_prompt_lens`
- `QueuedRequest`
- `ActiveRequest.reusable_prefix_len`
- `ActiveRequest.reusable_cached_prompt_len`

Note: the in-tree `lookup_or_stage` surface is a tier-aware classification helper only; it does not imply live staged readmission.

### 3. GPU Launch

- Decode-only launch: [`infer/src/scheduler/cuda/decode.rs`](../../infer/src/scheduler/cuda/decode.rs) `step_decode_launch()`
- Mixed launch: `step_decode_launch_mixed()`
- Prefill-only launch: [`infer/src/scheduler/cuda/prefill.rs`](../../infer/src/scheduler/cuda/prefill.rs) `step_prefill_batch()`

Owned state:

- `Scheduler.running_batch`
- `Scheduler.decode_bufs`
- `Scheduler.paged_kv_pool`
- `Scheduler.states`
- `PendingDecode`

### 4. Readback / Sampling

- Decode readback + token mutation: [`infer/src/scheduler/cuda/decode.rs`](../../infer/src/scheduler/cuda/decode.rs) `step_decode_readback()`
- Mixed completion handoff back into decode also lives here

Owned state:

- `PendingDecode.decode_indices`
- `PendingDecode.slot_indices`
- `PendingDecode.greedy_launched`
- `PendingDecode.mixed_prefill`

### 5. Cleanup / Release / Publish

- Per-tick cleanup: [`infer/src/scheduler/cuda/runtime.rs`](../../infer/src/scheduler/cuda/runtime.rs) `cleanup()`
- Slot release helpers: [`infer/src/scheduler/cuda/core.rs`](../../infer/src/scheduler/cuda/core.rs) `finish_slot()`, `move_to_decode()`
- Prefix publication: [`infer/src/scheduler/cuda/core.rs`](../../infer/src/scheduler/cuda/core.rs) `publish_to_prefix_cache()`

Owned state:

- `Scheduler.active`
- `Scheduler.prefix_cache`
- `Scheduler.block_to_pages`
- `Scheduler.slot_owned_blocks`
- `Scheduler.metrics`

## State Deleted With The Two-Stage Path

These existed only because one tick could first launch a mixed batch and then continue issuing extra serial prefills:

- `Scheduler.pending_mixed_prefill_idx`
- `PendingDecode.mixed_prefill_tokens`
- `PrefillBudget::reserve_mixed_prefill()`
- `step()` locals `already_mixed` and `mixed_prefill_tokens`
- the post-mixed serial prefill loop over `prefill_queue`

## State That Remains And Why

These are still needed after the refactor because a mixed batch still needs one completion handoff:

- `PendingDecode.mixed_prefill`
  - tells readback which prefill slots were merged into the decode launch and whether each chunk completed the prompt
- `Scheduler.prefill_queue`
  - still the single source of truth for prefilling requests; it is no longer scanned for extra launches after a mixed tick
- `Scheduler.waiting`
  - owns unslotted requests until the planner admits them inside `step()`, where tokenization, length gating, radix classification, and slot materialization now happen

## Resulting Tick Contract

The GPU execution part of one scheduler tick now plans exactly one path:

- `Idle`
- `DecodeOnly`
- `Mixed`
- `PrefillOnly`

This matches the intended sglang-aligned contract more closely:

- one admission decision
- one GPU launch path
- one readback path
- one cleanup path

`runtime.rs::assign_slots()` is deleted from this contract. Waiting requests are only materialized through the single planner path, and `lookup_or_stage` in this flow remains classification-only rather than a live staging/readmission API.
