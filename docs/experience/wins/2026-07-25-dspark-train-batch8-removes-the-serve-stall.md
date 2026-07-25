# `--dspark-train` batch 8 removes the 25 s serve stall; the residual 26% is the capture sync

## Context

The self-RL runs (srl1–srl6) were driven at **c=1** — `drive.sh` sends one request
at a time and waits — against a serve with `--max-running-requests 8`, so the
`slot=0..7` round-robin in the logs is slot assignment, not concurrency. No
throughput was recorded at the time; this entry recovers it.

Mining srl6's nanosecond log timestamps (`prefix-attach` lines, 352 requests over
1298 s) found the real story: **43 stalls, median 25.6 s, totalling 1139 s — 88% of
wall clock.** Periodicity was exactly 8 requests, and at the time the trainer
batch defaulted to 64: 8 requests × ~8 draft blocks each = 64 = one full batch.

## What Worked

Measured c=1 on 8×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash
draft, 48 requests × 64 max_tokens, temperature 0, `--max-running-requests 8`:

| | stalls >5 s | median latency | p90 | max | median tok/s | aggregate tok/s |
|---|---|---|---|---|---|---|
| training off | 0 | 0.846 s | 1.115 s | 1.163 s | **75.7** | 70.7 |
| `--dspark-train` (batch 8) | **0** | 1.142 s | 1.490 s | 1.546 s | 56.1 | 53.3 |

**The batch-64 → 8 default flip (382cfda3e) eliminated the stall entirely** — 43
stalls to 0, max latency 1.55 s. That commit was filed as "11× the optimizer steps
at the same data rate"; its actual value is that a 25 s CPU step no longer parks
inference.

The residual cost of training-on is a flat **−26% tok/s with no stalls**, and it is
not the optimizer: +0.296 s per request over ~20 spec steps ≈ **15 ms per spec
step**. Scaling with spec-step count is the signature of the per-step full stream
sync in the capture path (`infer-cuda/src/executor/dspark_train.rs` — its own
module doc says the sync "does add per-step latency"), not of an off-thread
single-core optimizer competing for 1 of 180 cores.

## Rule

A background trainer sharing a process with a serve needs its per-step cost
compared against request latency, not just against the data rate — a step longer
than a request converts the serve into a duty cycle. And when instrumentation is
added to a hot loop, budget the sync it introduces per step: 15 ms × 20 steps is a
26% throughput tax that no loss curve will ever show. Same defect class as #183.
