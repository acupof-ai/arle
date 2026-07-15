# DSv4 unbounded sweep exhausts KV pages

## Goal

Re-run the old canonical 4096-input/256-output saturation workload on DSv4 TP=4.

## Hypothesis

The saturation client would find a stable throughput point within the 16-slot server envelope.

## Environment

- Binary commit: `91d105f3f618`; 4× NVIDIA H20, TP=4
- Server: 16 slots, 400 KV pages, 25,600 compressed-context tokens
- Workload: 4096 input tokens, 256 output tokens, 60-second saturation points

## Result

KILL. At tick 2060 the client had `active=4` and `queue_depth=500`. Every TP rank then failed:

```text
KV alloc retry failed after reclaiming 1 pages: first error: HostPagedKvPool out of pages: slot 0 needs 1, free 0; retry error: HostPagedKvPool out of pages: slot 0 needs 1, free 0
```

No throughput number from this run is valid.

## Problems

The client controlled arrival rate, not in-flight concurrency. A 500-request queue exceeded the server's KV admission envelope and turned a load curve into an allocator failure.

## Learnings

- Canonical serving benchmarks use explicit fixed concurrency.
- A sweep that can grow an unbounded queue is invalid for capacity-limited paged KV.
- Record queue depth and free KV pages beside client throughput.

## Artefacts

- `/host/arle-megamoe-t1/bench-output/2026-07-15-dsv4-h20-tp4-allreduce-canonical-rerun/`
