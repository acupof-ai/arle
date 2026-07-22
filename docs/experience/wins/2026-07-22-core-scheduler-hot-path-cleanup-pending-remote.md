# Core scheduler hot-path cleanup — CUDA L4, 2026-07-22

> Status: pending-remote

## Goal

Preserve fixed-concurrency L4 throughput and latency while removing one per-step stop-token allocation and one per-step backend capability query.

## Hypothesis

Borrowing model stops at the two checks and snapshotting the immutable plan-token cap at engine construction preserve plan semantics with less scheduler work.

## Parameters

Final matched L4 parameters are pending remote execution under `docs/bench-and-trace-spec.md`.

- Baseline: latest archived L4 champion before these two core commits
- Treatment: commits containing this entry
- Workload / trials: final combined L4 gate, matched A/B, at least three trials per arm if delta is within 5%

## Environment

CUDA host, GPU, driver, model, dtype, TP/EP, slots, KV, and server flags are deferred to the final combined L4 gate. Local verification covers backend-neutral scheduler semantics only.

## Results

No wall-clock metrics claimed. The mock-executor gate proves every submitted mixed decode+prefill plan stays within a 3-token backend capability, the capability is read once, and a 17-token prompt completes.

## Problems

CUDA execution is unavailable locally.

## Learnings

`pending-remote`. Run the final combined L4 matched benchmark; keep only with stable non-negative throughput/latency results.
