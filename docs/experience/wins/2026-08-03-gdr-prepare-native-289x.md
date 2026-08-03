# GDR chunk-prepare goes native CUDA — 289× per launch, −19 s/step, losses bit-identical

**Date:** 2026-08-03 · **Commit:** 3d80dd473 · **Pod:** 8×H20, real 27B · **Verdict: ACCEPT**

## Context

nsys attribution (seq=8192 cp=2) exposed `kernel_kernel` at 12.3 s/step (13%):
the TileLang AOT GDR chunk-prepare, a flat 66 ms per GDN-layer forward at grid
(seq_len, num_value_heads). The stage's bandwidth roofline is ~0.1 ms — 600× off.

## Root cause

The TileLang lowering replicated the full q/k row into **every** thread's
registers (`float q_frag[128]; float k_frag[128]` per thread → local-memory
spill at 256 f32/thread), had all 128 threads redundantly load all 128 dims,
and ran the L2-norm sum-of-squares as a 128-step serial scalar loop per thread.
`T.serial` reads over a distributed fragment force per-thread replication.

## What worked

Native CUDA replacement (`csrc/recurrent/gdr_prefill_prepare.cu`), following
the solve-stage precedent: warp per (token, v_head), lane covers 4 dims,
`__shfl_xor_sync` reduction, fully coalesced IO, same public symbol — zero
wiring changes. TileLang entry removed from `kernels.toml`; kernel kept in
`KERNELS` as a lowering reference.

| Metric | TileLang | Native | Δ |
|---|---|---|---|
| ncu single launch (seq=8192) | 66 ms | 228.29 µs | **289×** |
| G2 fwd (cp=2, seq=32768) | 102.0–102.6 s | 91.65/91.73 s | **−10.1%** |
| G2 bwd | 384.2–386.5 s | 375.4 s | −8.8 s |
| G2 losses | 4.805783/6.064485 | identical to 6 decimals | bit-exact |
| ncu occupancy / regs | spill-bound | 88.2% ach., 31 regs/thread, DRAM 49.8% | — |

Gates: autograd linear_attention 14/14, cph_parity ce=8.534e-5 (bf16 floor).
Build self-healed the stale vendored artifact (source-hash rejection path fired).

## Rule

- A "prepare/glue" stage costing as much as the main kernel is a lowering
  pathology, not a workload property — read the generated code before ncu.
- TileLang `T.serial` over a distributed fragment = per-thread replication +
  spill. Elementwise + small-reduction stages belong in native CUDA (solve,
  now prepare); TileLang earns its keep only where T.gemm/T.Pipelined do.
