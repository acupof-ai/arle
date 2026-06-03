# R6 clean CUDA Phase-0 hang: TileLang paged-attention num_pages/total_pages arg swap

**Status:** pending-remote (H20 clean greedy-parity re-run in flight on the pod)
**Track:** R6 clean-CUDA rewrite (`crates/infer-cuda`), Phase 0 (CUDA greedy parity)
**Commit:** `db85d56e` (fix) on `arch/ideal-inference-engine`

## Context

The clean `infer-cuda` BF16 Qwen3 forward (rewrite) had never run on a real GPU.
First H20 bring-up (Qwen3-0.6B, prompt ids `[785,6722,315,9625,374]`, MAX_NEW=16)
surfaced three bugs in sequence — two already fixed:

1. `SafetensorLoader` O(N²) re-read per tensor → read-once `RefCell` shard cache (`3f5f2ece`).
2. Wrong `hidden_size == heads*head_dim` config assertion (Qwen3 decouples head_dim) → removed (`fe841c62`).
3. **This entry:** the forward launched but never returned — GPU pinned at 100%
   util with no `clean_tokens`, dmesg `Xid 43` (GPU stopped processing) under
   `name=r6-qwen3-parity`.

## What Worked — localization

- Source inspection cleared the pre-GEMM kernels (embedding, rms_norm) and proved
  the GEMM layout matched the row-major `HiddenStates` convention. A non-LAUNCH_BLOCKING
  host backtrace pointed at `cublasGemmEx` (the o_proj GEMM) — but that was a red
  herring: it was just the **first device-sync after an async fault**.
- **`CUDA_LAUNCH_BLOCKING=1` was decisive.** Serializing every kernel pinned the true
  faulting op:
  ```
  cuLaunchKernel → tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda
    → infer_cuda::attention::run_tilelang_paged → paged_attention → forward_tokens
  ```

## Root Cause

`run_tilelang_paged` passed the two TileLang symbolic-shape args **swapped** vs the
legacy `infer/src/ops/attention.rs` contract (and its explicit comment), in all 8
prefill/decode HD128 arms:

| FFI arg | Correct (legacy) | Clean (buggy) |
|---|---|---|
| `num_pages` (arg 12) | `pool.max_total_pages` (K/V pool **capacity** = k/v_pool first-dim extent) | `meta.num_pages` |
| `total_pages` (arg 13) | page-table length (valid `kv_indices` entries = `meta.num_pages`) | `pool.max_total_pages` |

With the swap, the kernel computed K/V-pool strides as if only `meta.num_pages`
(=1 for a 5-token prompt) pages existed, then walked `max_total_pages` (thousands)
entries over a 1-entry `kv_indices` buffer → out-of-bounds read → illegal memory
access that launches but never returns (Xid 43).

## Fix

Swap the two args back to capacity-first / page-table-length-second in all 8 arms;
add a comment documenting the non-obvious naming. `page_size==16` is separately
enforced by the executor (`SUPPORTED_PAGE_SIZE`), so it is not a co-factor.
Typechecks under `cuda,no-cuda`.

## Rule

- A TileLang AOT kernel arg **named** `num_pages` is the pool capacity, not the
  current request's page count — the AOT wrapper promotes pool/tensor extents into
  kernel args. Always mirror the legacy call site arg-for-arg when porting a paged
  kernel; the param name is not the semantics.
- A host backtrace stuck in a CUDA API after an async launch names the **first sync
  point**, not the faulting kernel. `CUDA_LAUNCH_BLOCKING=1` is the cheap, decisive
  localizer (no rebuild) — reach for it before host-backtrace theorizing.

## Verification (pending-remote)

- [ ] clean `clean_tokens` == HF gold `[12095,13,576,6722,315,9625,374,1083,279,6722,315,279,5429,315,9625,13]` (Qwen3-0.6B, greedy, 16 new).
- [ ] Then: longer prompt + multi-shape greedy parity before declaring Phase 0 closed.
