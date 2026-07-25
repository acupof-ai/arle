# DSpark spec decode does not batch across slots — a 2.2× win at c=1 becomes a 39% loss at c=16

## Context

Following the c=1 block-size sweep
([block truncation is not a throughput win](2026-07-25-dspark-block-truncation-is-not-a-throughput-win.md)),
the open question was whether a smaller block wins at concurrency: at c=N the
verify forward carries N slots × block rows, so once rows saturate the GEMM the
low-yield tail positions should start costing real time.

Measured on 8×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash draft,
`--max-running-requests 16`, 8 requests per concurrency level × 64 max_tokens,
temperature 0, training OFF, one serve per block size:

| block | c=1 tok/s | c=8 tok/s | c=16 tok/s | accept_rate @ c=16 |
|-------|-----------|-----------|------------|--------------------|
| 16 (default) | **80.7** | 69.4 | **64.7** | 0.242 |
| 8  | 78.1 | **71.6** | 64.1 | 0.457 |
| 6  | 73.8 | 68.9 | 64.0 | 0.560 |

Block 8 is +3.2% at c=8 and the three configs converge to one line at c=16. The
hypothesis is not confirmed — and the reason is that the effect it predicted
cannot occur.

## Root Cause

**Aggregate throughput falls as concurrency rises** (80.7 → 69.4 → 64.7) and
latency scales linearly (0.79 s → 6.9 s → 15.5 s ≈ 1× / 8× / 16×). 8 requests at
c=1 take 6.3 s; 64 requests at c=8 take 59.0 s — 8× the requests, 9.4× the wall
clock. That is full serialization plus ~18% contention.

The control isolates it. Same model, same slot cap, spec decode OFF:

| | c=1 | c=8 | c=16 |
|---|---|---|---|
| no spec | 36.0 | 96.2 | **106.0** |
| DSpark (best block) | **80.7** | 69.4 | 64.7 |
| verdict | spec **2.24×** | no-spec **+34%** | no-spec **+64%** |

Plain decode scales 2.9× from c=1 to c=16, so the engine's batched decode is
healthy. The serialization is inside the spec path: DSpark draft generation runs
per request, not batched across slots, so concurrency adds latency without adding
throughput. Concurrent requests never share one verify forward, which is exactly
why block size stops mattering at c=16.

## Fix

None applied — this is a measurement that re-scopes the feature.

- DSpark is a **low-concurrency latency feature** as it stands: worth enabling at
  c=1 (2.24×), a net loss from roughly c≥4 upward, −39% at c=16.
- The real lever is batched draft generation across slots
  (`infer-cuda/src/executor/dsv4.rs:2003`), not block size. Until that lands, no
  block-size tuning at concurrency can pay.
- The block-size flag stays an instrument with default 16, unchanged.

## Rule

Before tuning a knob inside a feature, measure the feature against **not using
it** at the target operating point. A 2.2× win at c=1 said nothing about c=8, and
three block sizes converging to one line was the symptom that the axis under test
had already stopped being connected to the cost.
