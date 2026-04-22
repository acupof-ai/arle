# Scheduler Prefill Phase Dead-Field Removal

## Context

After the overlap/admission cleanup, `Phase::Prefilling.materialized_prefix_len`
was no longer read anywhere in the CUDA scheduler. The field had become stale
state that only widened the request phase enum and confused follow-on refactors.

## What Worked

- Removed `materialized_prefix_len` from `Phase::Prefilling`.
- Updated all CUDA scheduler construction and match sites to use the smaller
  phase shape.
- Local no-GPU scheduler validation stayed green.

## Rule

When a scheduler phase field is only written and never read, delete it instead
of keeping it as historical bookkeeping.

## Status

`pending-remote` — cleanup only. Include this change in the next CUDA scheduler
GuideLLM sweep rather than claiming any performance effect from local checks.
