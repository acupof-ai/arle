# Mixed Workspace Cap Pending Remote

## Context

Follow-up review found that mixed decode+prefill launches were capped to
`min(max_prefill_tokens, long_prefill_token_threshold)` at scheduling time, but
Qwen3 runtime workspace sizing still reserved mixed buffers from the full
`max_prefill_tokens`. With the CUDA serving defaults that meant sizing mixed
scratch for 16384 prefill rows while actual mixed launches could only admit
4096 prefill rows.

## What Worked

Centralized the mixed prefill cap on `SchedulerConfig::mixed_prefill_token_budget`
and passed both standalone prefill tokens and mixed prefill tokens into the
model workspace estimate. Qwen3 now sizes ordinary paged-prefill workspace from
`max_prefill_tokens`, mixed workspace from the same cap used by the mixed
launcher, and reserves both because the lazy mixed scratch and async prefill
pending buffers can coexist.

## Bench

Status: `pending-remote`.

Local machine has no NVIDIA CUDA runtime. Run the canonical regression check on
the CUDA host:

```bash
scripts/bench_guidellm.sh cuda-mixed-workspace-cap
```

Before snapshot: `docs/experience/wins/2026-04-26-bench-guidellm-cuda-mixed-packed-prefill-review-fixes-pending-remote.md`

Expected measurement: no throughput regression; available KV pool should only
increase by over-reserved mixed activation/logit rows after accounting for the
simultaneously live standalone prefill workspace.

## Rule

Any launch-time cap that changes maximum admitted packed rows must also feed
runtime workspace sizing. Also reserve lazy persistent mixed scratch alongside
async prefill pending scratch; they are not mutually exclusive lifetimes.
