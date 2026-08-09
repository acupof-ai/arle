# Full-run ledger could not observe forward-busy time

> Status: pending-remote

## Context

The Qwen3.6-27B tranche-1 sweep completed 640/640 requests with
`ARLE_STEP_PHASE=1`. Request, token, prefix, and speculative-decode counters
reconciled exactly, and instrumentation changed throughput by -2.19% to +2.54%
without a consistent sign.

The run could not pass its 95% attribution gate. `/v1/stats` did not export the
existing submit-to-ready timer, and the log printed decode-only phase averages
at 500-step checkpoints rather than exact cumulative counters. The 1-second
GPU utilization samples cannot replace forward wall.

## Root Cause

`infer-core` accumulated process-global forward-busy and decode phase counters
for logs and an internal API, while `ThroughputStats` exposed only steps and
tokens. The server and relay therefore discarded the timing evidence required
by the benchmark contract.

## Fix

`ThroughputStats` now records submit-to-ready micros and step counts split into
prefill-only, decode-only, and mixed plans. It snapshots the existing exact
decode phase counters. `/v1/stats` exposes all fields through both local and
multi-process relay paths. Decode phase `submit_micros` is contained within
forward-busy and must not be added to it.

Pending remote: CUDA release build, stats field smoke, count/busy invariants,
and a replacement tranche-1 ledger reaching the 95% gate.

## Rule

A phase model is valid only when its exact cumulative counters are externally
observable and its overlap rules are explicit.
