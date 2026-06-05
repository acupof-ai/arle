# DSv4 #24 on-device MoE router: route-math licensed, async wiring blocked on cross-stream dep

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash TP=8/EP=8, 8×H20.
**Model:** DeepSeek-V4-Flash FP8. **Commissioned by:**
[`docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`](../../plans/2026-06-04-dsv4-decode-sglang-class-perf.md)
Step 2 (on-device routing). **Status:** GPU route kernel licensed; async
production wiring **NOT** licensed — gated off by default (`ARLE_DSV4_GPU_ROUTER`
unset), checkpointed as infrastructure pending #29.

## Context

DSv4 decode/prefill is host-route-bound: every MoE layer every step does
`clone_dtoh(router_logits)` → `infer_moe::route` (CPU) → H2D. 2048-token scaling
confirms it is the dominant prefill cost — `sync_ms` 2.62 s → 11.54 s and
`dsv4_route` CPU 319 ms → 1.25 s, both near-linear in token count. Step 2 moves
routing on-GPU (reuse `dsv4_route_cuda`; hash `tid2eid` now a device-resident
table), wired behind `ARLE_DSV4_GPU_ROUTER=1` into **both** MoE transports
(allreduce + DeepEP), device-side i32→i64 cast for DeepEP (no new D2H), host route
preserved as fallback. Rust-only rebuild 9 s.

## What Worked (and what didn't)

Same-binary, env-flip A/B at the production decode shape (prompt
`671,6102,294,8760,344`, max_new=16), oracle = the established host-route
`clean_tokens`:

| Mode | Result | Note |
|---|---|---|
| Route kernel semantics | **PASS** | rank0 first 6 routes (3 hash + 3 learned-bias) **0-diff** vs host oracle on indices *and* weights |
| `CUDA_LAUNCH_BLOCKING=1` token1 | PASS `[11111]` | serializing every launch → correct |
| async, no sync | **FAIL** `[0,0,…]`/`[1,1,…]` | not a route-math bug (math passed above) |
| `SYNC_AFTER_ROUTE=1` | token1 PASS, **drifts at token 9** | route output alone isn't the dependency |
| `SYNC_AFTER_MOE=1` | **16/16 PASS** | `[11111,603,671,6102,294,8760,344,11111,603,671,6102,294,8760,344,11111,603]` == oracle |

Decode throughput (`SYNC_AFTER_MOE`, correctness-bridge mode): **6.245 tok/s vs
4.365 baseline = +43%** — but with **43 sync/token**, so this is a *correctness
bridge, not the perf solution*. The async (sync-free) path is the real win and is
not yet correct.

**Root cause of the async failure (isolated, not yet fixed):** not route math
(0-diff), not a missing Rust `CudaSlice` keepalive (DeepGEMM scratch already
keeps its buffers; the GPU-route index/weight/i64 buffers are kept via
`Dsv4ForwardKeepalive::keep_route_*`, deliberately ungated so they survive
prefill too), not `tp.all_reduce_sum` reading early (confirmed compute-stream).
`SYNC_AFTER_ROUTE` fixing token1 but drifting at token 9 points at a
**cross-stream / DeepGEMM-native-bridge dependency** that the host-route sync
used to mask — the same class as
[`reference_disabled_event_tracking_premature_buffer_free`](../../../.claude/projects/-Users-bytedance-code-agent-infer/memory/reference_disabled_event_tracking_premature_buffer_free.md):
with cudarc event-tracking disabled, the MoE-output→downstream-consumer ordering
isn't enforced unless something syncs. `SYNC_AFTER_MOE` substitutes globally.

## Decision

GPU route kernel is **licensed** (math 0-diff). The async production wiring is
**not** — landing `SYNC_AFTER_MOE` as a default flip would be a fake optimization
(43 sync/token). Bring **#29 (scratch / dependency pool)** forward as #24's
companion: it must add the targeted cross-stream event/wait that `SYNC_AFTER_MOE`
currently fakes, so the sync-free path becomes correct. Checkpointed gated-off so
the verified route kernel + device hash table aren't lost; default path
unchanged.

## Rule

A GPU operator passing component parity (route indices/weights 0-diff vs host
oracle) is **kernel-licensed but not pipeline-licensed**: removing the host-side
sync that used to serialize it can expose a cross-stream dependency that produces
*accumulating* decode drift (correct token1, garbage by token 9) — distinct from
a math bug, which is wrong from token1. Bisect the sync point
(`SYNC_AFTER_ROUTE` vs `SYNC_AFTER_MOE`) to localize the unenforced dependency
before declaring the async path correct. Never default-flip a router that only
passes under a per-layer sync.
