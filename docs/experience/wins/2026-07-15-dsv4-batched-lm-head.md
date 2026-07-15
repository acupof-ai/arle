# DSv4 batched greedy lm_head

> Status: Shipped

## Goal

Increase TP=4 output throughput at serving concurrency 4-16.

## Hypothesis

One batched lm_head GEMM and argmax should replace one lm_head pass per row.

## Parameters

- Same binary, GPUs, model, prompts, order, and server lifecycle per arm.
- Workload: 20 natural prompts, 128 output tokens, greedy; 20 requests/point.
- Baseline: per-row lm_head. Treatment: batched lm_head for unsharded greedy
  decode with probes off. Sampling and sharded-head paths keep the old flow.
- Trials: three per arm at c=4/8/16; one at c=1.

## Environment

- 4x H20, TP=4, CUDA 12.9, allreduce, 16 running requests, L2/L3 off.
- DeepSeek-V4-Flash-FP8; binary SHA256
  `04ad2408db357ac68e2ca2ae7b619dc6df8e1fe616f51152f56f4ac5ca89dddb`.

## Results

| c | baseline median tok/s | batched median tok/s | delta |
|---:|---:|---:|---:|
| 1 | 42.22 | 42.97 | +1.77% |
| 4 | 81.41 | 82.65 | **+1.53%** |
| 8 | 122.96 | 125.74 | **+2.25%** |
| 16 | 144.54 | 148.18 | **+2.52%** |

All reported runs completed 20/20 with zero errors and zero correctness
failures. c4 baseline trial one had a startup stall; the three-trial median
keeps it from driving the result.

Raw artifacts: `/host/arle-megamoe-t1/bench-output/2026-07-15-lmhead-{off,on}-r{1,2,3}/bench.{json,csv}`.

Final built-in smoke rebuilt SHA256
`07aba935be745c52d95b51d5c3235ba200061af07a03c9d37943488869af24c8` with the
old environment gate absent. A greedy request completed normally with `2+2`
decoded as `4`. Log:
`/host/arle-megamoe-t1/logs/final-builtin-smoke.log`.

## Problems

The gain is small but positive at every measured concurrency.

## Learnings

Keep the fast path built in; it reuses existing batched primitives and adds no
user-facing configuration. The project now has no minimum gain threshold: a
positive three-trial target wall-clock median is sufficient when correctness
and SLOs hold.
