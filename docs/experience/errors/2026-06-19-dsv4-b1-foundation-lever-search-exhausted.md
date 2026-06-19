# DSv4-Flash B=1 foundation lever search — exhausted (3 levers re-killed by adversarial workflow)

## Context

Goal: recover 55 t/s single-stream B=1 on DSv4-Flash (TP4≡TP8). Ran an
adversarial-verification workflow (`dsv4-b1-perf-levers`, run `wf_a55d8dcb-e9a`,
6 agents) to pin + skeptically verify the three remaining B=1 / c8 decode
levers against the physical memory/compute floor and git history. All three came
back **refuted**. This consolidates the kills so the search is not re-spun.

Measured B=1 today: no-spec 39 t/s, MTP typical 34, MTP counting 41.2; c8 MTP 62.8.

## Root Cause (why each lever is dead — all measured, not inferred)

1. **foundation-gemv** (scalar M=1 FP8 GEMV 10.6ms vs 0.5ms floor, ~20×) →
   **DROP, recoverable@wall = 0.** The 10.6ms is the *deepgemm-OFF non-default*
   path. Default decode already routes wq_a|wkv (fused), wq_b, wo_a, wo_b through
   tensor-core DeepGEMM (`attention.rs:5476` `dsv4_decode_proj_deepgemm_enabled`,
   default-ON when DeepGEMM compiled); the 39 t/s already includes it. The
   uint4-vectorized M=1 GEMV (1.8× isolated, ncu-proven, byte-identical) was
   wired into full decode and washed (38.16≈38.5,
   `errors/2026-06-08-dsv4-gemv-1.8x-isolated-but-wall-neutral-overlapped.md`);
   the tensor-core mma variant was killed (−32%, all-zero,
   `2026-06-07-dsv4-mma-fp8-gemv-killed.md`). Decode is GPU-bound on the serial
   per-layer chain (attn→AR→MoE→AR ×43); these GEMVs overlap the prior layer's
   ncclAllReduce. 8 per-kernel/host/graph levers total washed (`6ms-FINAL`).

2. **c8-batched-decode** (claimed 127ms→6ms, 40ms recoverable) →
   **POD-MEASURE-FIRST, refuted as a *new* opportunity.** The full-flatten
   compressor/indexer batched prepass is *already default-on* — the
   `ARLE_DSV4_DECODE_COMPRESSOR_BATCH` gate was deleted in `ee8cc355`; it now runs
   unconditionally on the B>1 lane for non-SparseIndexed layers (`dsv4.rs:2254`
   `full_flatten = layer.mode != SparseIndexed`). The author's own clean c8 A/B is
   +4.9% (gate ON 77.6 vs OFF 74.0,
   `2026-06-16-dsv4-c1-8-baseline-clean-ab.md`), NOT 40ms; the +58% is at c4 where
   per-row GEMVs still dominate, narrowing to ~+5% by c8 once GEMM/MoE +
   batched-FlashMLA saturate. Batched attention alone = +3% @c8
   (`2026-06-15-...phaseB-fixed.md`). The 127ms-vs-6ms framing mixes a c8 aggregate
   against the H20-unreachable B=1 MTP floor (the §0 narrow-window trap). The real
   structural ceiling is **DP-attention (#89)** — multi-day, separate project,
   does not touch the MoE-∝-active-expert floor (fundamental, EP=8-mitigated 1.4×).

3. **hc-moe-pertoken** (hc_params Sinkhorn single-block @B=1, 8.51µs×86=0.73ms;
   claimed 0.33ms recoverable via hc_enter fusion) → **DROP.** The exact fusion was
   already *shipped* (`d457ad1b`: params+pre-mix+rms_norm in one kernel, stream row
   staged in 32KB smem, hc_mult==4 fast path) and *reverted* (`1554f734`) after a
   matched md5-verified pod A/B: −4.9% (41.63→39.61), the 40KB dynamic smem forces
   per-launch L1/SMEM carveout reconfiguration across ~172 transitions/token — a
   launch-neighborhood penalty invisible to the same-kernel microbench
   (`2026-06-11-dsv4-hc-enter-fusion-kill-smem-carveout.md`). Sinkhorn is already
   warp-parallel (negligible). The whole HC-fusion ladder is exhausted (`2e96526a`).

4. **WS3 commit-9ms** (not in the workflow; checked here) → **not licensed.** The
   9ms MTP "commit" is `commit_accepted_fold` (`dsv4.rs:1721`) — genuine per-layer
   compressor/indexer re-ingestion of accepted tokens over 43 layers, same inherent
   m-rows×layers structure as verify, NOT a wasteful whole-ring copy. The spec-ring
   *snapshot* (`capture_spec_rings`) is only ~1.6ms and overlapped. A copy-one-slot
   trim targets the small overlapped part → another B=1 overhead-removal candidate
   → wash by the 8-wash record.

## Fix

No code. The B=1 foundation lever search is **CLOSED**. The wall is the serial
per-layer dependency chain; B=1 per-kernel / overhead-removal / graph levers are
empirically dead (now 8+3 kills). The only levers that ever moved the B=1 wall:
**amortization** (MTP/EAGLE, +71% historically) — capped at ~52% accept by the
1-head NextN arch, so 55 t/s needs a **2-head MTP draft head (training, out of
inference scope)**; or **batching** (c>1, the c8 lane, already +4.9% from the
default-on prepass, structural ceiling = DP-attn #89 multi-day). Realistic B=1
ceiling on this config/checkpoint ≈ 45.

## Rule

A B=1 decode lever justified by a synced per-kernel CUDA-event number
(`linear_profile`/`stage_profile` `stop.synchronize`) is **isolated-active time,
not critical-path time**. On the GPU-bound serial per-layer chain it washes unless
it removes a serial *stall* (the warp-tail did, +5.9%; everything else, ×11, did
not). Before proposing any "next B=1 lever," grep the errors log first — it is
almost certainly already shipped (default-on) or already wall-A/B-killed. Verdicts
require a matched same-session wall-clock A/B, never the component number.
