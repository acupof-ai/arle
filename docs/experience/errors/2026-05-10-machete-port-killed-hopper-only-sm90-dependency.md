# Machete W4 kernel port from vLLM — KILLED at Phase 2 hardware survey: HOPPER-ONLY (sm_90+) dependency

## Context

Date: 2026-05-10 (cron-loop tick 107 KST)
Source: user directive "**当前主轴:Machete W4 kernel 移植 from vLLM** —
port machete from vllm-project/vllm to ARLE crates/cuda-kernels for
sm_89 W4A8 优化 (预估 -20-40% ITL vs current Marlin)"

Phase 2 (hardware constraint sheet, per `kernel-optimization` skill)
survey of vLLM `csrc/quantization/machete/` source via gh API.

## Root Cause

### §1 Machete is architecturally Hopper-bound

vLLM `csrc/quantization/machete/Readme.md` literal text:
> "Machete is a spiritual successor to the Marlin kernel **but
> optimized for Hopper architectures** and based on Cutlass."

`machete_mainloop.cuh` source evidence:
- L43: `#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"`
- L92, L157: `using ArchTag = arch::Sm90;` (hard-coded)
- L162-164: `sm90_cluster_shape_to_tma_atom(...)` (TMA = Tensor Memory
  Accelerator, sm_90+ only)
- L199-201: `#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED ...
  "Unsupported Toolkit for SM90 Collective Builder"`
- L248: `using GmemTiledCopyScale = cute::SM90_TMA_LOAD;`
- L326-330: static_assert on `SM90_TMA_LOAD` / `SM90_TMA_LOAD_MULTICAST`

### §2 WGMMA = sm_90+ instruction (the core dependency)

`machete_mm_kernel.cuh` design rationale (L37-44):
> "the wgmma instructions only support sourcing from registers for the
> left-hand operand, we want to upconvert/decompress the quantized
> operand in register"

WGMMA (Warp Group Matrix Multiply-Accumulate) is a Hopper-specific
PTX instruction that operates on 128×N×K shapes per warp group. Ada
(sm_89) does NOT have WGMMA — it uses standard MMA atoms with
different fragment layouts.

The entire Machete kernel design is BUILT AROUND WGMMA's register-source
semantics for the left operand. There is no graceful sm_89 backport.

### §3 Hardware mismatch with ARLE primary target

| SM | WGMMA | TMA | Smem/SM | Machete-compatible? |
|---|---|---|---:|---|
| sm_80 (A100) | NO | NO | 164 KB | NO |
| sm_86 (RTX 30xx) | NO | NO | 100 KB | NO |
| **sm_89 (RTX 4070 Ti SUPER, ARLE primary)** | **NO** | **NO** | **100 KB** | **NO** |
| sm_90 (H100) | YES | YES | 228 KB | YES (target) |
| sm_100 (B100/B200) | YES (next-gen) | YES | TBD | likely with port |

ARLE's CLAUDE.md `kernel-optimization` skill Phase 2 hardware constraint
sheet identifies sm_89 as primary. Machete cannot run on sm_89 without
a fundamental rewrite that drops WGMMA + TMA — at which point it is
no longer Machete, just another Marlin variant.

### §4 User estimate (-20-40% ITL) was Hopper-anchored

The user's "预估 -20-40% ITL vs current Marlin" likely came from
literature numbers measured on H100 (sm_90). On Ada (sm_89), Machete
provides 0% benefit because it cannot compile, let alone run.

## Fix

### §1 KILL the port axis on current hardware

Per `kernel-optimization` skill Phase 2 ("Hardware constraint sheet")
+ Phase 8 ("License-or-kill"): a kernel that requires architectural
features the target hardware lacks is an immediate KILL with no
investment. Cheapest possible kill — saved ~1-2 weeks of futile port
work.

### §2 Pivot to sm_89-compatible Machete-class alternatives

Per `9735b47` REFUTATION + `bccf1bd` consistency audit, Machete-class
threshold (-20-40%) on sm_89 requires:

| Option | Mechanism | Status |
|---|---|---|
| A: Medusa (Option A) | 2.25× tok/s via 5 trained heads | READY, gated on user GO (`f0c7561`) |
| B: Hybrid W4A16/W4A8 | -14% E2E latency stacking with A | READY, validated by REFUTATION (`bccf1bd`) |
| C: Cutlass FP8 direct mma sm_89 | replace cuBLASLt heuristic; 1.88× → ?× | needs Phase 0 spike (per `kernel-optimization` skill anti-pattern #7) |
| D: Custom W4A8 kernel improvements | tile/stage tuning on existing Marlin | needs Phase 0 nsys binding-constraint evidence first |
| E: Wait for sm_100 hardware | NVFP4 native | not on near-term roadmap |

Recommended: A + B combined (per `bccf1bd` §4.2) for ~2.61× tok/s +
-14% latency, ~4-5 days wall-clock. Option C is open if A+B
under-deliver.

### §3 What Machete WOULD give us if hardware changed

If ARLE adds an H100/H200 secondary backend in future, Machete becomes
relevant. At that point, the port is mechanical (cutlass kernels are
designed to be portable across Hopper SMs). Until then, parking the
port is the SOLID call.

## Rule

**Always read kernel README + arch-tag definitions BEFORE planning a
port.** This kill came from 2 gh API calls + 5 minutes of grep. The
alternative path (start porting, hit WGMMA compile errors days in,
re-evaluate) would have wasted 1-2 weeks.

For ARLE specifically: any kernel from "modern" (post-2024) ML infra
sources MUST be checked for Hopper/Blackwell-only dependencies before
adoption. Default assumption: new high-perf GEMM kernels target
sm_90+. Ada compatibility requires explicit verification.

This adds a NEW SKILL anti-pattern candidate:
- **#43 (or next available): pre-port arch-tag survey mandatory** —
  read `arch::Sm9X` / `wgmma` / `TMA` / `cluster_shape` markers
  in kernel headers BEFORE estimating port effort. n=1 evidence
  (this Machete KILL).

## Cross-references

- vLLM `csrc/quantization/machete/Readme.md`: "optimized for Hopper architectures"
- vLLM `csrc/quantization/machete/machete_mainloop.cuh`: hard `arch::Sm90` + WGMMA
- ARLE `kernel-optimization` skill Phase 2: SM table places sm_89 (no WGMMA, no TMA)
- `9735b47` REFUTATION wins entry (Hybrid B caps at -14%, NOT Machete-class)
- `bccf1bd` consistency audit (Hybrid B is auxiliary -14% that stacks with Medusa)
- `f0c7561` Medusa Phase 1.B substrate brief (ready, gated on user GO)
- CLAUDE.md `crates/cuda-kernels/AGENTS.md` references existing Marlin path
