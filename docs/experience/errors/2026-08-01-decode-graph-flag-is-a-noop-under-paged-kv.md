# `--qwen35-decode-graph` logs ARMED and captures nothing — the lane is unreachable under paged KV

## Context

Profiling where a Qwen3.6-27B-FP8 decode step actually goes (H20, 59 steps under
`nsys`) showed **1094 `cudaLaunchKernel` calls per step** and ~4 ms of GPU idle
out of a 25 ms step. The whole-step decode graph exists
(`Qwen35DecodeGraph`, `try_graph_decode`) and is opt-in behind
`--qwen35-decode-graph`, described in its own help text as "opt-in until the pod
license". Licensing it looked like a free 16%.

## Root Cause

The flag arms the lane and logs

```
Qwen3.5 whole-step decode graph ARMED (256 slots; lazy capture per slot, ...)
```

but `try_graph_decode` has exactly one call site
(`executor/qwen35.rs:3071`) and it sits **below** an unconditional early return:

```rust
if self.full_attn_paged() {
    return self.decode_row_paged_default(row, position, host_kv);
}
```

Paged KV is the serving default, so the graph lane is reachable only on the
legacy contiguous path (OPD weight offload). `submit_decode_batch` (rows > 1)
documents that batched steps never capture either. In every serving
configuration the flag is a no-op.

The comment above the return states the real blocker honestly — the capture
bakes device addresses and the page table grows each step — but nothing
downgrades the ARMED log, and no failure warning fires because the failure path
is never entered.

## Evidence

Two `nsys` captures, same binary, same model, flag off then on:

| | flag off | flag on |
|---|---:|---:|
| steps captured | 59 | 80 |
| `cudaLaunchKernel` | 64560 (1094/step) | 86080 (1076/step) |
| any `cuGraph*` / `cudaGraph*` API call | **0** | **0** |
| `graph failed ... downgrading to eager` in log | absent | absent |

Serve A/B, ThinkingCap-27B-FP8, plain decode, arms swapped across GPU 6/7:

| arm | GPU | c=1 tok/s | c=8 tok/s |
|---|---|---:|---:|
| on | 6 | 33.4 | 95.7 |
| off | 7 | 34.0 | 97.8 |
| on | 7 | 34.0 | 96.6 |
| off | 6 | 32.4 | 87.6 |

Between-GPU spread (87.6 vs 97.8 on the same arm) exceeds the on/off spread —
consistent with a flag that changes nothing.

## Fix

**Fixed 2026-08-03 by `cb6b3389d`** ("whole-step decode CUDA graph under paged
KV — capture the serving default"): the unconditional paged-KV early return was
removed and `try_graph_decode_paged` now captures the serving default. The flag
went default-on the same day. 35B c=1 TPOT 16.22 → 6.70 ms (−58.7%); see the
35B SOTA row in `docs/baselines.md`.

## Rule

**A feature flag's log line is not evidence the feature ran.** `ARMED` was
printed by the warmup path; the dispatch path returned before ever reaching it.
Confirm an opt-in actually engages by counting the API calls it is supposed to
produce — here, zero `cuGraph*` calls with the flag on settled it in one
capture, after a four-arm serve A/B had only produced noise. Related:
[the storage-precision flag that was inert on the checkpoint path](2026-07-27-tape-bf16-noop-on-checkpoint-path.md).
