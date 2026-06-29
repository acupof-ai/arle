# DSv4 DeepEP Intranode Fix — mk_align=1 Bug, CUDA TP=4, 2026-06-29

## Context

`dsv4_moe_forward_deepep` passed `mk_align=1` to grouped-GEMM. DeepGEMM has rejected `mk_align < 64` since commit `51f38d90` (2026-06-13). The intranode DeepEP path was silently broken for ~16 days.

## Root Cause

`moe.rs` (`dsv4_moe_forward_deepep`) hardcoded `mk_align=1`; DeepGEMM requires `mk_align ≥ 64` (decode) or `≥ 128` (prefill). The mismatch caused a runtime rejection on every intranode DeepEP dispatch.

## Fix

Commit `8e6a0350` fix(cuda): deepep intranode mk_align=1 → adaptive 64/128, aligned scan + packed_rows

- `contig_align`: adaptive — 64 for decode, 128 for prefill.
- `moe_exclusive_scan_aligned_i32`: aligned scan over expert counts.
- `packed_rows = deepgemm_contig_rows_cap(...)`: cap contiguous rows to DeepGEMM limits.
- Scatter over `packed_rows` instead of raw row count.
- File: `crates/infer-cuda/src/moe.rs`, function `dsv4_moe_forward_deepep`.

## Environment

- **Backend:** cuda
- **Model:** DeepSeek-V4-Flash-FP8
- **Hardware:** 8×H20 (97 GB each), sglang-test container
- **Code commit:** `8e6a0350`
- **GPUs:** 1,3,4,7, TP=4
- **Workload:** 500 in / 128 out, 120 s windows
- **Tool:** `bench_nonstream.py`
- **DeepEP flag:** `ARLE_DSV4_MOE_BACKEND=native-deepep`; `nvshmem=false` (intranode only)

## Results

| c | deepep tok/s | allreduce tok/s (ref) | Δ |
|---|---|---|---|
| 1 | 16.0 | 25.1 | −36% |
| 2 | 19.2 | 31.9 | −40% |
| 4 | 38.4 | 43.2 | −11% |
| 8 | 42.7 | — | — |

Allreduce reference from `2026-06-29-cuda-throughput-ceiling-three-models.md` (c=4 = 44.3 tok/s; post-fix allreduce ceiling run: c=4 = 45.1, c=8 = 45.4 tok/s).

## Analysis

- **Fix unblocks the path**: intranode DeepEP now executes without rejection; prior to fix all dispatch calls failed silently.
- **c=1/2 still −36–40% vs allreduce**: 172 protocol syncs × ~75 μs ≈ 12.9 ms fixed overhead per decode step dominates at low concurrency.
- **c=4 gap closes to −11%**: batch volume begins amortizing the sync overhead.
- **c=8 = 42.7 tok/s**: approaching allreduce parity but allreduce still wins on single-node.
- **DeepEP advantage is internode (NVSHMEM, multi-node)**: not exercised here (`nvshmem=false`); intranode allreduce is the cheaper path.

## Problems

- None post-fix. Pre-fix: every intranode DeepEP dispatch rejected at `mk_align` check.

## Learnings

- **Do not enable `ARLE_DSV4_MOE_BACKEND=native-deepep` for single-node serving** — allreduce is superior at all concurrency levels tested (c=1..8).
- DeepEP's competitive range is internode (multi-node NVSHMEM); benchmark there before considering a default flip.
- `mk_align` must track DeepGEMM's minimum constraint at the call site — a hardcoded `1` will silently break on any DeepGEMM version that enforces the minimum.

## Δ vs baseline

- **Baseline:** [`2026-06-29-cuda-throughput-ceiling-three-models.md`](2026-06-29-cuda-throughput-ceiling-three-models.md) — DSv4 allreduce path, c=4 peak 44.3 tok/s.
- This entry establishes the first valid intranode DeepEP measurement (path was broken prior to `8e6a0350`).
