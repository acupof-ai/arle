# Scheduler GPU/CPU Overlap — Deferred Future Work

> **Superseded (2026-04-17):** the launch/readback split is already in tree
> (`infer/src/scheduler/cuda/execution.rs:step()` Phase 1 `step_decode_launch`
> → Phase 2 CPU overlap → Phase 3 `step_decode_readback`). Any further
> scheduler restructuring rolls into
> [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md) §Phase 2/3.

## Status

Deferred on 2026-04-16. This note records the current analysis so the
next implementation starts from the actual CUDA scheduler structure
rather than from a vague "overlap CPU and GPU more" idea.

## Current structure observed in tree

Main CUDA scheduler loop in `infer/src/scheduler/cuda/runtime.rs` is
serial at the top level:

1. receive requests from `request_rx`
2. `drain_coordinator_events()`
3. `assign_slots()`
4. `step()`
5. `cleanup()`
6. update metrics / throttled memory query

Inside `infer/src/scheduler/cuda/execution.rs`, `step()` already has a
limited overlap optimization:

- Phase 1: `step_decode_batch()` for active decode requests
- Phase 2: `emit_delta()` tokenizer work for already-generated tokens
- Phase 3: admit `Phase::New` requests via `step_new(...)`
- Phase 4: prefill chunks via `step_prefill_chunk(...)`

The important detail is that `emit_delta()` only consumes tokens from
the previous step. That means the existing overlap only covers CPU
tokenizer work after decode returns; it does not overlap CPU scheduler
work with the current decode batch's GPU forward path.

## Actual split point in current code

`infer/src/scheduler/cuda/decode.rs::step_decode_batch()` currently
packages decode as one synchronous unit:

1. gather decode requests / allocate pool tokens
2. `decode_ctx.upload_token_ids(...)`
3. `decode_ctx.update_metadata(...)`
4. `decode_ctx.plan_attention(...)`
5. `self.model.forward_decode_batch(...)`
6. sample:
   - greedy fast path: `self.model.sample_batch_greedy(...)`
   - fallback: `prepare_batch_sampling_fallback(...)` then
     `select_tokens_batch(...)`
7. push sampled tokens into each active request and apply stop / length

For the greedy fast path, the concrete boundary is inside
`sample_batch_greedy()`:

- `infer/src/model/qwen3/forward.rs` and
  `infer/src/model/qwen35/forward.rs` call
  `argmax_batch_logprob_launch(...)`, then `self.ctx.sync()?`, then
  `argmax_batch_readback_into(...)`, then D2H-copy logprobs.
- `infer/src/model/glm4/forward.rs` does the same pattern with
  `argmax_batch_launch(...)`, then `self.ctx.sync()?`, then
  `argmax_batch_readback_into(...)`.

That means the future split point is not just "somewhere in
`step_decode_batch()`". It is specifically:

- launch forward decode
- leave the GPU work in flight
- run independent CPU work
- only then synchronize and perform argmax/token/logprob readback

In other words, the future scheduler shape is roughly
`step_launch()` + CPU work + `step_readback()`, with the readback phase
starting at the current boundary between
`forward_decode_batch(...)` completion requirements and
`sample_batch_greedy(...)` readback.

## Why this is deferred

### 1. Sampling depends on forward logits

The scheduler cannot simply move sampling earlier or in parallel with
forward. `sample_batch_greedy()` and `select_tokens_batch()` both depend
on logits produced by `forward_decode_batch(...)`. Any real overlap has
to preserve that dependency and only insert CPU work between launch and
the later sync/readback point.

### 2. The current model contract is synchronous at the wrong level

`ModelForward::forward_decode_batch(...)` is a synchronous trait method.
Today the scheduler calls it and only regains control after the model
path has completed enough work for sampling to proceed. Implementing
true overlap likely requires a contract split such as:

- launch/encode the decode batch without forcing immediate readback
- finalize/sync/read back sampled outputs later

That change is cross-cutting because it touches:

- CUDA scheduler execution flow
- the model trait surface in `infer/src/model.rs`
- per-model greedy fast paths
- the fallback sampling path
- CUDA Graph capture / replay assumptions in batched decode

### 3. The greedy path itself is fused around launch + sync + readback

The fast path is intentionally compact today: launch argmax, synchronize,
read back tokens, read back logprobs. Splitting around that requires
either:

- new launch/readback APIs for batched argmax/logprob, or
- a different decode-context-owned state machine for pending decode work

This is more invasive than moving `emit_delta()` or `assign_slots()`
around in the loop.

### 4. Simple loop reordering is not real overlap

Reordering the top-level loop to do more CPU work before or after
`self.step()` does not create overlap if `step_decode_batch()` still
waits inside `sample_batch_greedy()` before returning. The scheduler
thread only gets back to CPU work after the current decode batch has
already synchronized.

## Required future implementation shape

Minimum shape for a real overlap implementation:

1. Split the decode step into launch and completion phases.
2. Keep `plan_attention(...)` in the launch phase, because it prepares
   the decode context before the forward pass.
3. Ensure `forward_decode_batch(...)` can return with GPU work still
   pending instead of forcing the scheduler into immediate sampling.
4. Split batched greedy sampling into launch and readback or provide an
   equivalent decode-context abstraction that preserves the same effect.
5. Insert only CPU work that is independent of current-step logits
   between launch and readback.
6. Leave request mutation (`generated_tokens.push`, stop checks,
   `latest_logprob`, finish handling) in the completion phase after
   readback.

## Acceptance criteria for future implementation

- `runtime.rs` no longer executes decode as one blocking `self.step()`
  subphase when a decode batch is present; there is an explicit launch /
  completion split.
- The split point is after forward launch and before token/logprob
  readback, not merely before `plan_attention(...)` or after sampling is
  already complete.
- Greedy fast paths for Qwen3, Qwen3.5, and GLM4 preserve current output
  parity and logprob behavior.
- Non-greedy / fallback sampling still works correctly with the new
  phase structure.
- CUDA Graph replay still functions for decode batches after the split.
- `cargo check -p infer --no-default-features --features cuda,no-cuda`
  remains green.
- Any performance claim is backed by a before/after scheduler benchmark,
  because correctness alone does not prove that useful overlap was
  created.
