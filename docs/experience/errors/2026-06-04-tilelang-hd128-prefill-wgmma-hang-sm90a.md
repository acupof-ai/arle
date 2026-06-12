# TileLang HD128 paged-prefill WGMMA spin on sm_90a — divergent barrier from fragment hoist; fixed by inlining the page lookup

**Status:** ROOT-CAUSED + FIXED (2026-06-12). The 06-04 "hard TileLang codegen bug"
classification was wrong — the bug was *triggered by our own kernel source* (the
page-lookup fragment hoist, `526515bd`), and reverting the hoist fixes it. The
falsification table below never tested removing the hoist; every killed row kept it.
**SKU:** H20 / sm_90a, CUDA 12.9, `batch_prefill_paged_hd128_q16_kv8`, Qwen3-0.6B BF16, clean `infer-cuda` (R6).

## Context

R6 clean-CUDA bring-up: the HD128 q16_kv8 **batched prefill** TileLang cubin spins at
layer-0 (GPU 100 % util, request never returns, **no Xid** → device-side spin, not
OOB). All host args verified. The HD128 q16_kv8 prefill kernel had **never run on
sm_90 before R6** (legacy benched it on sm_80, and ran the *HD256* prefill on sm_90).

## Falsified hypotheses — the full chain (each killed by a controlled experiment)

| Hypothesis | Experiment | Result |
|---|---|---|
| `num_pages`/`total_pages` arg swap | fix `db85d56e` | real earlier bug (removed Xid 43); spin remained |
| trip-count-1 < num_stages-2 pipeline deadlock | seq_len 5/64/**70** sweep | all hang (70 = trip 2, bx 2) → killed |
| partial-tile / unguarded-`exp2` NaN | seq_len=64 (FULL tile, 0 padding) | hangs → killed (separate latent NaN exists, not this) |
| dyn-shmem mis-sizing | cuobjdump launcher | 49152 B = `q+k+v` tiles, correct → killed |
| TileLang 0.1.10 FullRow defect | install 0.1.9 + forced regen (new sha) | seq_len=64 still hangs → version killed |
| build emitted plain sm_90 (no WGMMA) | cuobjdump cubin | `sm=90a`, WGMMA count 12 → killed |
| `BLOCK_N=64` tile width | set `BLOCK_N=32` (= working HD256), regen | still hangs → killed |
| FFI scalar arg order | diff Rust call vs generated signature | exact match → killed |
| `GemmWarpPolicy.FullRow` vs `Square` | set both gemms `Square`, regen | TileLang accepted it but **lowered to the IDENTICAL device-source sha** (`5ccc…`) → the knob is a **no-op** → killed |

Also ruled out earlier: host stream/sync omission (symptom is device spin, not host idle).

## The decisive positive — decode works, so it's prefill-cubin-specific

A 1-token prompt routes `seq_len==1 → decode` kernel and **ran cleanly through all 28
layers** via the clean R6 launch path. So the rewrite's launch path + the decode cubin
are sound; the spin is specific to the **HD128 multi-row prefill** path (`BLOCK_M=64`
q-tile, FullRow-WGMMA over 64 rows). The legacy 2026-05-30 H20 win ran the **HD256**
prefill FullRow-WGMMA on sm_90 correctly — so the defect is HD128-shape-specific.

## Root cause (found 2026-06-12, debug session with CUDA_LAUNCH_BLOCKING + gdb + lowered-source read)

**Divergent `__syncthreads()` from the page-lookup fragment hoist.** Commit
`526515bd` (2026-04-28, never validated on H20) hoisted the per-column page lookup
into `(BLOCK_N=64,)` fragments (`page_idx_j/in_page_j/valid_j`). With
NUM_THREADS=128, TileLang lowers a 64-wide fragment loop into a **half-warpgroup
predicated region** in the generated device source:

```c
if ((((int)threadIdx.x) >> 6) == 0) {
  __syncthreads();          // threads 0-63 arrive here
  /* K/V tile loads, page_idx_j thread-local */
}
/* ... */
{ tl::GmmaDescriptor desc_a, desc_b;
  __syncthreads();          // threads 64-127 arrive HERE instead
  tl::warpgroup_arrive();   // wgmma is warpgroup-WIDE (all 128 threads)
}
```

Threads 64–127 skip the predicated barrier and pair with the wgmma block's barrier
→ **barrier slip** (64+64 arrivals at different sites both "satisfy" the same
barrier) → warps 2–3 enter warpgroup-wide `wgmma.mma_async` while warps 0–1 are
still in the K/V load region → UB → device spin.

This explains every row of the falsification table: **all experiments kept the
hoist** (BLOCK_N=32 just changes the predicate to `tid>>5==0`, still divergent;
version/policy/shmem knobs don't touch the predicated-region lowering). It also
explains the arch asymmetry: sm_89/sm_70 lower `T.gemm` to per-warp `mma.sync`,
which tolerates the slip — only sm_90's warpgroup-wide wgmma deadlocks. And the
HD256 prefill kernel "working" on sm_90 with the same hoist is a different-lowering
accident, not a counter-example (different tile geometry → different predication).

**Attribution chain:** correct PageMeta dumped at layer 0 (`R6_ATTN_DEBUG=1`
clone_dtoh-synced) → host thread blocked in `cuLaunchKernel` under
`CUDA_LAUNCH_BLOCKING=1` (driver-API = the TileLang wrapper; prep/cuBLAS use
runtime API) → divergent barrier visible in the generated
`tilelang_batch_prefill_paged_hd128_q16_kv8_sm90_device_kernel.cu`.

## Fix

Revert the hoist: inline the divmod + `KV_indices` gather back into the `(j, d)`
copy loop (the pre-`526515bd` form — the same pattern the **working** decode kernel
`batch_decode_paged_hd128.py` uses). The duplicate divmod is the price of
correctness; a `(BLOCK_N,) < NUM_THREADS` fragment write between shared-memory
barriers is structurally unsafe under TileLang's sm_90 lowering.

## Resolution

**Interim (06-04, superseded):** sequential 1-token forwards through the proven
decode kernel (`chunk_size=1`) closed greedy parity vs HF gold
(`wins/2026-06-04-r6-cuda-eager-parity-verified.md`) — but was never wired into the
serve path, so serve-driven prefill stayed broken until 06-12.

**Final (06-12):** hoist reverted in `batch_prefill_paged_hd128.py`; batched prefill
runs clean on H20 (see the 06-12 wins entry for serve-path verification + needle
gate).

## Rule

- **The falsification table must include "revert the last untested commit touching
  this kernel".** `526515bd` landed `pending-remote` and was never run on H20; the
  06-04 session A/B'd eight knobs but never the one source change that had never
  been validated on the failing arch. Diff the kernel against its last
  known-good-on-this-arch sha *first*.
- **For device spins, read the generated `.cu` for divergent barriers before
  blaming the compiler.** Grep for `__syncthreads` inside `if ((threadIdx.x >> N)
  == 0)`-style predicated regions; on sm_90 any barrier slip feeds warpgroup-wide
  wgmma and deadlocks. "Hard codegen bug, route around" was a premature
  classification — the lowered source had the smoking gun all along.
- **In TileLang sm_90 kernels, never write `(W,)` fragments with `W <
  NUM_THREADS` between shared-memory barriers** — the partial-thread predicated
  region TileLang emits can capture a `__syncthreads()`. Keep per-column
  precomputes inline in the full-width copy loop (as the decode kernel does).
- **"Knob leaves device-source sha unchanged" means the knob is inert, not that
  the bug is upstream.** The `Square`==`FullRow` identical-sha killed the policy
  hypothesis only; the 06-04 session over-generalized it to "not reachable from
  source", which the hoist revert disproves.
- **A working decode kernel + `chunk_size=1` is a correctness fallback that decouples
  "forward verified" from "batched-prefill kernel fixed."** Don't let a hard
  perf-path codegen bug block the correctness gate.
- **Verify pod-binary freshness (git HEAD + `strings`/symbol) before attributing a
  host-side error to current code** — the sibling `cache_len != kv_seq_len` decode
  error was a stale-pod-`infer-core` artifact, not a code bug (current planner.rs
  captures `kv_seq_len` pre-allocate; 5 regression tests added in `8388fc64`).
- **Bisect kernel-bug vs harness-bug with a 1-token forward** (routes to a *different*
  cubin through the *same* launch path) before deep kernel spelunking — it proved the
  R6 launch path sound in one cheap run.
