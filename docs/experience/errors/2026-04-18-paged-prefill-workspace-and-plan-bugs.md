# Paged prefill (Phase 1A/3a) — three concrete bugs surfaced by bench

**Date:** 2026-04-18
**Branch:** main
**Commits introducing the work:** `859c3d2` (Qwen3.5 Phase 1A), `9821cb2`/`fb8f3a4` (Qwen3 Phase 3a)

## Context

Phase 1A migrated Qwen3.5 full-attn layers to paged-KV prefill and Phase 3a
did the same for Qwen3. The `prefill_uses_paged_pool() → true` flip
lifts the `CONTIGUOUS_KV_TOKENS = 512` chunk cap, so prefill chunks go up
to `prefill_chunk_size()` (4096 by default). This was committed and
merged, but the first real guidellm sweep against it on L4 24GB killed the
process immediately. Debug surface:

1. Server launches on `Qwen3.5-4B` with `num_slots=10`, `max_seq_len=5120`.
2. guidellm `sweep` profile fires a 4k-token prompt.
3. Scheduler chunks it `4096 + N` where `N` is the tail.
4. Process aborts with one of:
   - `fatal runtime error: Rust cannot catch foreign exceptions, aborting`
   - `Alloc failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)` (context poisoned)

## Root causes — three independent bugs stacked

### 1. FlashInfer HD256 `float_workspace` underallocated

At `cta_tile_q=64`, `padded_batch_size=256`, 4096-row split-KV, the
`tmp_v` partial-output buffer consumes all 256 MiB of
`FLOAT_WORKSPACE_BYTES`, then `tmp_s` (2 MiB at HD128, larger at HD256)
can't allocate and FlashInfer throws:

```
Error in function 'aligned_alloc' at allocator.h:49: Buffer overflow when
allocating memory for batch_prefill_tmp_s with size 2097152 and alignment
16, but only 0 bytes available in AlignedAllocator.
```

The same pattern hits the HD128 path under the sweep's concurrent load —
256 MiB is simply insufficient for paged prefill at 4k+ tokens × 10
slots.

### 2. No C++ → Rust exception translation on paged-prefill FFI

`flashinfer_batch_prefill_paged_hd{128,256}_{plan,run}` are `extern "C"`
but the FlashInfer `PrefillPlan` path throws `std::runtime_error` (via
`TORCH_CHECK`) on workspace exhaustion. Without a `try { … } catch (…)`
at the FFI boundary, the exception propagates through the C ABI and the
Rust runtime aborts with "cannot catch foreign exceptions" — the server
process dies, not just the single request.

### 3. `total_num_rows` device-pointer wire missing in HD128/HD256 `_run`

When `PrefillPlan` is built with `enable_cuda_graph=true` (our case — we
want graph-compatible dispatch), the kernel reads
`params.total_num_rows` as a **device pointer** into
`int_workspace[plan_info.total_num_rows_offset]`. HD256 `_run` only set
`params.max_total_num_rows = plan_info.total_num_rows` (the i32) and
never populated the device pointer, so the dispatch did OOB reads on Q.
HD128 `_run` had the same omission.

### Scheduler-level fourth issue (not a kernel bug, tracked separately)

Even with the three kernel fixes applied, a different crash appears when
the slot is reused with a non-zero radix hit for a
`supports_partial_prefix() = false` model (Qwen3.5, GLM4): the scheduler
classifies the hit as MISS and does a full recompute, but the paged-pool
seq_len from the prior request carries forward and the next
`alloc_pool_tokens_with_retry` hits the poisoned state. Reverting
`prefill_uses_paged_pool() → false` for Qwen3.5 sidesteps this until the
scheduler change lands.

### 5th issue — `enable_cuda_graph=true` inflates FlashInfer workspace

Confirmed via FlashInfer source (`scheduler.cuh`): `PrefillPlan` picks
`padded_batch_size = new_batch_size` when `enable_cuda_graph=false`,
but `max(max_batch_size_if_split, total_num_tiles_q)` when true —
the latter is meant for CUDA-graph-captured prefill where the shape
must be stable across invocations. We never graph-capture prefill
(only decode), so keeping the flag `true` was blowing
`batch_prefill_tmp_v = num_qo_heads × padded_batch_size × cta_tile_q × head_dim × sizeof(float)`
past the 512 MiB workspace.

**Fix:** flip `enable_cuda_graph=true` → `false` in both
`flashinfer_prefill_paged{,_hd256}.cu` plan calls. This is what
sglang does in its non-graph wrappers
([sglang/flashinfer_backend.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention)).

With that flip in, a single-request paged prefill on a 4096-token
prompt completes cleanly and the HD128 workspace can go back to
the 256 MiB default.

### 6th issue — concurrent paged prefill still poisons the CUDA context

With fixes 1–5 applied, Qwen3 paged prefill works for sequential
requests (3× 4096-token `/v1/completions` calls in a row succeed)
but the guidellm sweep's concurrent 10-slot load crashes the
scheduler thread on the very first batch:

```
thread '<unnamed>' panicked at infer/src/ops/linear.rs:513:14:
gemm_cuda failed: DriverError(CUDA_ERROR_UNKNOWN, "unknown error")
```

`CUDA_ERROR_UNKNOWN` inside `gemm_cuda` on the scheduler thread means
the CUDA context is already poisoned by an earlier operation —
classic OOB-from-a-prior-kernel symptom. Candidates for that OOB:

- Paged-prep kernel reads past the `slot_page_indices` buffer when
  `page_indices(slot).len()` > `num_pages` (residual pages from pool
  reuse), and the stale tail page IDs point into unmapped regions.
- Page-table H2D upload (`clone_htod`) has an async-drop / reuse
  race we haven't ruled out at the Rust side.
- cuBLAS workspace pressure in the next GEMM call (less likely —
  memory fraction is well under budget).

This is what keeps both Qwen3 and Qwen3.5 on the contiguous path for
now. Tracking as Phase 1C.

## Fixes

- `crates/cuda-kernels/src/flashinfer.rs`
  - Add `FlashInferWorkspace::new_with_float_bytes` — callers choose the
    workspace size; default constructor keeps 256 MiB for decode paths.
  - Expose `HD256_FLOAT_WORKSPACE_BYTES = 512 MiB` as the known-safe size.
  - `BatchPrefillPagedPlan::new` (HD128) now uses 512 MiB.
  - `BatchPrefillPagedPlan::new_hd256` added for the Qwen3.5 HD256 path.
- `crates/cuda-kernels/csrc/attention/flashinfer_prefill_paged{,_hd256}.cu`
  - Wrap `_plan` and `_run` in `try { … } catch (const std::exception& e) { fprintf; return -1; }`
    so workspace overflows surface as a FlashInfer return code, not a
    process abort.
  - Wire `params.total_num_rows = GetPtrFromBaseOffset(int_workspace,
    plan_info.total_num_rows_offset)` when
    `plan_info.enable_cuda_graph` is set.
- `infer/src/model/qwen35/forward.rs`
  - `prefill_uses_paged_pool() → false` (revert Phase 1A path selection)
    until the scheduler slot-reuse interaction is fixed.
    Tracking: `docs/plans/p99-unified-mixed-batch.md` §Phase 1C.
- `infer/src/model/qwen35/prefill.rs`
  - Use `BatchPrefillPagedPlan::new_hd256` (not `::new`) when the path
    is re-enabled, so HD256 gets the 512 MiB workspace explicitly.

## Rule

Any CUDA kernel wrapped in `extern "C"` that calls into FlashInfer (or
any C++ library that can throw) **must** wrap the body in
`try { … } catch (const std::exception&) { return -1; } catch (...) { return -1; }`.
Foreign exceptions through a C ABI are undefined behaviour on our
toolchain and manifest as `fatal runtime error: Rust cannot catch
foreign exceptions, aborting`. The try/catch is cheap and it
converts `std::runtime_error` into a numeric error code that Rust can
handle.

## Rule

`FlashInferWorkspace` default sizing of 256 MiB is only correct for the
single-request decode path. **Every paged prefill plan needs 512 MiB**
(the sglang-published value). Don't rely on the default — ask for the
size explicitly, and use `new_with_float_bytes` when you know the
workload will be larger than decode.
