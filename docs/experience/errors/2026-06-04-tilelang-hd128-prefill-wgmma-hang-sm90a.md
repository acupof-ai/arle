# TileLang HD128 paged-prefill WGMMA spin on sm_90a — robust to every tile/policy/version knob; decode-kernel correctness path

**Status:** root cause classified (hard TileLang codegen bug); correctness closed via chunk=1 (verification in flight); batched HD128 prefill cubin = perf-only follow-up.
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

## Root cause

A **hard TileLang codegen bug** in lowering the HD128 multi-row prefill FullRow-WGMMA
gemm for sm_90a — not reachable by any source-level knob the kernel exposes (the
`Square`==`FullRow` identical-sha result proves the policy is lowered away; version,
arch, BLOCK_N, trip-count, dyn-shmem all independently ruled out). It is an
upstream-TileLang / generated-`.cu` defect, fixable only upstream or by route-around.

## Resolution

**Correctness (closes Phase 0):** process the prompt as **sequential 1-token forwards
through the proven decode kernel** (`chunk_size=1`; each forward `seq_len==1 → decode`,
accumulating KV — causally identical to batched prefill, same logits). End-to-end
greedy parity vs HF gold verifies the clean R6 CUDA forward is numerically correct
without the batched-prefill cubin. **Perf follow-up:** the batched HD128 prefill cubin
(needed for fast long-prompt prefill) stays a documented known-issue — candidate fixes:
upstream TileLang lowering fix, or migrate HD128 paged prefill to FlashInfer C++ (the
long-term rec in `errors/2026-05-27-tilelang-0110-fullrow-warp23-nan-sm80.md`).

## Rule

- **Once version + arch + tile-size + warp-policy are ruled out AND a knob leaves the
  generated device-source sha unchanged, stop blind kernel A/Bs** — the bug is in the
  compiler's lowering, not your template; pursue route-around or upstream. The
  `Square`==`FullRow` identical-sha was the signal that the policy knob was inert.
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
