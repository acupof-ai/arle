# Scheduler Incremental Priority Queue

## Context

CUDA scheduler cleanup continued on 2026-04-22. The runtime previously sorted
the entire waiting queue on every `assign_slots()` pass even when the queue
shape had not changed.

## What Worked

- The waiting queue now stays priority-ordered on ingress and requeue.
- `assign_slots()` no longer performs a whole-queue sort on every scheduler
  iteration.
- Requeued decode victims still get equal-priority preference, while fresh
  arrivals preserve FIFO order within the same priority.

## Rule

Status: `pending-remote`

- This is a scheduler-only hot-path cleanup; any throughput or TTFT claim still
  requires a CUDA GuideLLM before/after snapshot.
