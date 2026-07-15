# DSv4 L2 and L3 real-hit throughput

> Status: Shipped

## Goal

Measure real prefix-state reuse and its c=1/4/8/16 throughput with L2-only and
L2+L3 storage.

## Hypothesis

NVMe spill extends a small L2 working set at a bounded high-concurrency cost.

## Parameters

- Workload: one coherent 1,649-token README prompt, 96 output tokens, repeated
  20 times per point after a prime request.
- L2-only: 8 GiB deployment-total host budget.
- L2+L3: 512 MiB host plus 4 GiB NVMe deployment-total.
- Greedy, c=1/4/8/16, 20 requests/point; c16 repeated three times per arm.

## Environment

- 4x H20, TP=4, CUDA 12.9, allreduce, 16 running requests.
- DeepSeek-V4-Flash-FP8; local NVMe checkpoint and L3 root on `/data00`.
- Binary SHA256: `04ad2408db357ac68e2ca2ae7b619dc6df8e1fe616f51152f56f4ac5ca89dddb`.

## Results

| c | L2-only tok/s | L2+L3 tok/s | delta |
|---:|---:|---:|---:|
| 1 | 39.76 | 40.00 | +0.61% |
| 4 | 74.74 | 72.86 | -2.52% |
| 8 | 108.88 | 104.60 | -3.93% |
| 16 | 125.20 median | 119.96 median | **-4.19%** |

At c16, TTFT p50 was 3,196.8 ms for L2 and 3,578.4 ms for L2+L3
(+11.9%). All three c16 runs per arm completed 20/20 with no errors or
correctness failures.

Mechanism proof after the first long prompt:

- 51 prefix pages published; 4 remained in L2 and 47 spilled to L3.
- The second request hit 3,264 tokens and read 51 disk entries.
- Later service counters reached 3,865 disk reads and 316 host reads.
- Four sparse L3 files used 1.2 GiB of real disk space.

Raw artifacts: `/host/arle-megamoe-t1/bench-output/2026-07-15-kv-{l2-only,l2l3}-readme*/bench.{json,csv}`.

## Problems

An earlier 3,311-token contract prompt produced a degenerate decoded case and
was discarded. One L2 c16 run had an SSE event/token mismatch and was replaced
by three clean runs.

## Learnings

L3 buys working-set capacity with about 4.2% c16 throughput and 11.9% TTFT cost.
Use fixed byte budgets; concurrent percentage-based L2 allocation can overcommit
host memory.
