# Bench Stub — Delete Mixed-Batch Legacy Surface

## Context

This change deletes the remaining CUDA scheduler/model mixed-prefill legacy
surface that no longer participates in the shipped scheduler flow:

- `ModelForward::{supports_mixed_batch, supports_mixed_prefill_batch}`
- `ModelForward::{forward_mixed_batch, forward_mixed_batch_with_prefill}`
- Qwen3's eager mixed-prefill override and mixed workspace reservation
- scheduler startup's `mixed_prefill_enabled` / `prepare_mixed_decode_context`
- CLI/config surface `--enable-mixed-chunk`

The runtime now keeps one canonical scheduler shape: prefill batches run
through `forward_prefill_batch`, decode batches run through
`forward_decode_batch`, and the scheduler no longer advertises or pre-reserves
the deleted mixed-prefill path.

## What Worked

- The dead mixed-prefill trait/scheduler surface was removed without changing
  the live decode launch/readback flow.
- Qwen3 no longer reserves eager mixed decode+prefill workspace at startup.
- Scheduler docs/config now describe the actual token-budget knobs that remain.

## Rule

If the scheduler no longer calls a model/runtime entrypoint, delete the trait
surface and startup reservation path rather than leaving a dead compatibility
branch.

## Status

`pending-remote`

CUDA GuideLLM regression check still needs to run on a CUDA host because this
change can move startup memory reservation and high-concurrency scheduling
behavior.
