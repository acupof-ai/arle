# Longctx Phase 1 S5 missed SGLang catch-up

## Context

Phase 1 S5 compares ARLE and pinned SGLang on the local L4 W1 gate:
Qwen3-4B, FP8 KV, prompt=32768, output=256, c=4. The Phase 1 entry target is
`ARLE/SGLang >= 0.95`.

Measured rows:

| system | c=1 out tok/s | c=4 out tok/s | c=4 TTFT p50 | c=4 ITL p50 |
|---|---:|---:|---:|---:|
| ARLE | 9.99 | 9.96 | 39535.2 ms | 96.82 ms |
| SGLang | 11.67 | 16.27 | 24182.25 ms | 119.43 ms |

Ratios:

| gate | ratio | required lift to 0.95x | required lift to 1.30x |
|---|---:|---:|---:|
| c=1 | 0.856x | +11.0% | +51.9% |
| c=4 | 0.612x | +55.2% | +112.4% |

## Root Cause

The immediate miss is c=4 prompt throughput and overlap, not decode ITL:
ARLE c=4 ITL p50 is `18.9%` lower than SGLang, but TTFT p50 is `63.5%`
higher and output throughput does not scale from c=1 to c=4.

This matches the mission risk in
`docs/projects/2026-04-30-longctx-32k-128k-leadership.md` §7.4: long FP8
prefill likely lacks the same tensor-core/chunked-overlap efficiency that the
pinned SGLang path gets from FlashInfer/FlashAttention-style kernels.

There is also a measurement gap: the current S5 ARLE entry records scheduler
activity and KV utilization but does not persist a `plan_label` counter proving
`Mixed > 0` and `Split = 0` as a machine-checkable acceptance row.

## Fix

Do not proceed to Phase 2 as if Phase 1 is green. Open the next tranche as a
gap-closure pass:

1. Add persistent plan-label counters to the ARLE stats/trace path so S5 can
   prove `Mixed > 0` and `Split = 0`.
2. Profile c=4 long prefill to separate kernel time, scheduler admission, and
   decode overlap.
3. Target the first +55% lift at W1/c4 before claiming Phase 1 catch-up.

## Rule

When `ARLE/SGLang < 0.95` on Phase 1 S5, record the failed gate as an error
entry and keep the mission in Phase 1 gap closure. Decode ITL wins do not
offset a long-prompt TTFT/throughput miss.
