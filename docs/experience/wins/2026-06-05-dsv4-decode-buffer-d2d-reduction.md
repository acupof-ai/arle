# DSv4 decode 21.615 → 25.129 tok/s: D2D/zeroing/alloc reduction (keepalive-clone-is-a-D2D-copy)

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Shape:** parity harness, prompt `671,6102,294,8760,344`, 16 decode, greedy.
**Commissioned by:**
[`docs/plans/2026-06-04-dsv4-decode-sglang-class-perf.md`](../../plans/2026-06-04-dsv4-decode-sglang-class-perf.md)
§6 (the post-#29 lever ranking). **Status:** landed.

## Context

The post-#29 wall-clock breakdown (50.70 ms/token) showed the largest remaining
bucket was *not* launch overhead but memory-management ops: `cuMemcpyDtoD`
9.03 + `cuMemset` 5.29 + alloc/free 5.02 = ~19.3 ms/token (38% wall). Bigger than
launch (22.5%), lower-risk than full graph → the evidence-licensed next lever.

## What Worked

Source-localized each bucket from the nsys-rep, then eliminated:

- **D2D root cause: the #23 forward keepalive used `CudaSlice::clone()`, which is a
  device-to-device *copy*, not a handle clone.** Every kept buffer was being
  deep-copied on device every step (the bulk of the 9 ms D2D). The #29 persistent
  scratch pool (fixed addresses, reused) already guarantees lifetime
  structurally, so the deep-copy keepalive is redundant → **default-off**
  (diagnostic fallback `ARLE_DSV4_DEEP_COPY_KEEPALIVE=1` retained).
- `shared_hc` was cloning the pooled shared-scratch padded output into a fresh
  `HiddenStates` (large D2D); now writes a caller-provided output and copies only
  the valid `[seq_len, hidden]` view.
- `HiddenStates::zeros` / HC `alloc_zeros` → `uninit` where the kernel
  full-writes the buffer (no need to pre-zero).

| Metric | before | after | Δ |
|---|---|---|---|
| decode tok/s (non-nsys) | 21.615 | **25.129** | +16% |
| decode tok/s (nsys) | 19.724 | 24.269 | +23% |
| wall/token | 50.70 ms | **41.2 ms** | −19% |
| `cuMemcpyDtoDAsync` | 9.03 ms/tok | 1.29 | −86% |
| `cuMemsetD8Async` | 5.29 ms/tok | 0.62 | −88% |
| `cuMemAllocAsync`+`cuMemFreeAsync` | 5.02 ms/tok | 1.76 | −65% |

Correctness: `clean_tokens` == oracle, 16/16. Cumulative from the 4.365 baseline:
**~5.75×.**

**Next lever (now #1):** launch overhead — `cudaLaunchKernel` + `cuLaunchKernelEx`
≈ 13.36 ms/token (32% of 41.2 ms) → #25 full decode graph. Residual D2D/memset are
now small (per-layer scratch resets). MLA attn (~11.8 ms) remains the kernel floor.

## Rule

`CudaSlice::clone()` in cudarc is a **device-to-device memcpy**, not an Arc-style
handle bump — using it for a "keepalive" (to defer a free) silently pays a full
D2D copy per buffer per step. Once a persistent scratch pool gives buffers stable
addresses, the clone-keepalive is both redundant *and* expensive; delete it.
When `zeros`/`alloc_zeros` feed a kernel that writes the whole buffer, use
`uninit` — the memset is dead work.
