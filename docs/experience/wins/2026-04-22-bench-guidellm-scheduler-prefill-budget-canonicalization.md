# Scheduler Prefill Budget Canonicalization

## Context

CUDA scheduler overlap/admission cleanup continued on 2026-04-22. This tranche
did not change model or kernel code; it deleted scheduler-side dual-budget drift
 in `execution.rs` and made the prefill planner explicitly score-then-fit while
 keeping the existing queue-order ranking.

## What Worked

- `execution.rs` now mutates one canonical prefill token budget instead of
  tracking both `remaining_step_tokens` and `remaining_prefill_tokens`.
- Prefill planning is now structurally two-pass: collect/score candidates,
  then fit them against the token/page/request budget.
- Local no-GPU scheduler validation remained green after the cleanup.

## Rule

Status: `pending-remote`

- Any performance claim still requires a CUDA before/after GuideLLM snapshot.
- This scheduler-only cleanup should be included in the next CUDA scheduler
  parity run rather than claimed from local correctness checks alone.
