# DSv4 c≥2 decode crash ROOT-CAUSED + fixed — batched pack on compress_ratio==0 layers

Status: VERIFIED on pod (TP=4, DeepSeek-V4-Flash-FP8, GPUs 1,2,3,5). The a+c+b prepare-batching
(`45599845`/`2e605c3c`/`1109738b`) crashed ALL concurrent decode (c≥2) with
`CUDA_ERROR_INVALID_VALUE`; root cause found + fixed; c=1/8/16/32 all correct, no crash, and the
batching now LIFTS throughput at c=8/16.

## Context
After a+c+b landed, c=1 decoded correctly but **c≥2 crashed at the first batched step, all TP ranks**:
`worker rank N step (tick #1): DriverError(CUDA_ERROR_INVALID_VALUE)`. Two earlier mis-diagnoses
(it is NOT a clean gate-off — the batched precompute is coupled to the batched select; and it is
NOT the cudarc memcpys) were both refuted by evidence. A debug `eprintln` op-trace (gated to n≥2)
pinpointed it: `prepare-row r=0 → prepare-row r=1 → pack → INVALID_VALUE`. The crash is the PACK op.

## Root Cause
DSv4-Flash `compress_ratios = [0, 0, 4, 128, 4, 128, …, 0]` — **layers 0, 1, and the last have NO
compressor (ratio 0)**. `flashmla_decode_pack_batched` UNCONDITIONALLY called the completed-compressor
FFI, whose launcher does `if (ratio <= 0) return cudaErrorInvalidValue` (surfaced through `.result()?`
as a cudarc `DriverError`). The single-row path skips it (passes `compressed = None` for these layers).
The batched lane is n≥2-only, so c=1 never reached it and the `n=1` bit-identity gate missed it.

Key insight that unblocked the hunt: the FFI launchers return `CUresult`/`cudaError` via `.result()?`,
so an FFI arg-check error DOES appear as a cudarc `INVALID_VALUE` — the "FFI ≠ cudarc" premise (mine
and an adversarial workflow's) was wrong. The decisive evidence was always the runtime op-trace, not
code reading (which looked plausible 4×).

## Fix
- `51cf2701` — guard the batched completed-compressor on `compress_ratio > 0` (matches the single-row
  `compressed = None` skip). One `if`.
- `00b1860f` (kept) — `pack_num_logical_pages = max` over slots, not the last row's value: at n≥2 with
  differing slot positions the per-slot page-table lengths differ; the kernel's `block_id <
  num_logical_pages` bound must cover every row (was a silent per-row skip, separate correctness bug).
- `d9739c0c` — strip the debug eprintln traces.

## Verified (pod, TP=4, GPUs 1,2,3,5)
- needle_gate len=115/180: **exact=3** (correctness; NONDET = MoE run-to-run floor).
- c=4 (initial): 4/4 ok, no crash.
- c-sweep tok/s (max_tokens=96): **c=1/8/16/32 = 30.1 / 62.4 / 77.9 / 68.3** — no crash at any c.
  Indicative vs the pre-a+c+b dynamic-pool baseline (31/43/57/66): c=8 +45%, c=16 +37%. Not a matched
  A/B (a+c+b-off crashes via the coupled precompute), so treat the lift as indicative, not licensed —
  the SOLID result is the **crash fix + correctness**, the throughput lift is a bonus to confirm later.

## Rule
A batched op that copies a single-row body must enumerate EVERY per-row conditional the single-row
path had — here `compressed: Option` (skip when None) became an unconditional FFI call that an
arg-check (`ratio<=0`) rejected. The `n=1` bit-identity gate cannot catch n≥2-only batched-lane bugs;
**a batched-decode change needs a c≥2 concurrency test, not just n=1**. When code-reading a launch-config
crash looks "plausible" repeatedly, stop reading and get the runtime op-trace (cheap eprintln gated to
the failing shape) — and FIRST verify the binary is actually rebuilt (`strings arle | grep <symbol>` /
mtime > source), because a chain that skips `ep_build.sh` silently serves the stale binary.
