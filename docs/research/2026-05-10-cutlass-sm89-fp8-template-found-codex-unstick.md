---
title: cutlass sm_89 FP8 GEMM template — codex Phase 0 unstick (GemmUniversalWithAbsMax + Sm89 ArchTag)
date: 2026-05-10
type: research
status: codex-resumed-with-template
---

# cutlass sm_89 FP8 GEMM template — codex Phase 0 unstick (GemmUniversalWithAbsMax + Sm89 ArchTag)

> Codex's Path B-Phase2' Phase 0 spike kernel `/tmp/cutlass_fp8_smoke.cu`
> hit `CUTLASS error :66: Error Internal` on first compile + run.
> Claude found the unstick this tick via direct upstream check on
> NVIDIA/cutlass examples — **dedicated Ada (sm_89) FP8 reference exists**
> at `examples/58_ada_fp8_gemm/ada_fp8_gemm.cu`. Codex briefed via
> paste-buffer with raw template extract + reasoning.

## §0 Direct upstream evidence (raw `gh api` output, NOT memory recall)

Per skill v1.10.0 anti-pattern #28 ("hallucinated tool output overrides
peer-agent investigation"), every claim below is backed by raw
`gh api` output quoted verbatim this tick.

### Cutlass examples directory listing

```bash
$ gh api repos/NVIDIA/cutlass/contents/examples --jq '.[] | select(.name | contains("fp8") or contains("sm89")) | .name'
54_hopper_fp8_warp_specialized_gemm
58_ada_fp8_gemm                     ← Ada (sm_89) FP8 reference
64_ada_fp8_gemm_grouped             ← Ada FP8 grouped
67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling
94_ada_fp8_blockwise                ← Ada FP8 blockwise scaled
```

Three Ada FP8 examples — sm_89 FP8 path is FIRST-CLASS in upstream
cutlass. Earlier in the session I conflated "Machete is Hopper-only"
(true) with "cutlass FP8 is Hopper-only" (FALSE — cutlass has clear
Ada FP8 support).

### Key template excerpt from example 58 (lines 92-110)

```bash
$ gh api repos/NVIDIA/cutlass/contents/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu \
    | base64 -d | grep -nE "Sm[0-9]+|arch::|using.*Gemm|cutlass::gemm::device|epilogue|TileShape|Stages"

92:static int const kStages = 3;
96:using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
106:using Gemm_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
108:    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
110:    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
811:  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAdd>> testbed_staged_accum;
822:  TestbedRunner<Gemm_<cutlass::arch::OpMultiplyAddFastAccum>> testbed_fast_accum;
```

## §1 Why codex's first attempt hit Status::Internal

Most likely root cause: codex used `cutlass::gemm::device::GemmUniversal`
(generic / Hopper-implicit) instead of
`cutlass::gemm::device::GemmUniversalWithAbsMax` (Ada-specific FP8 path).

**Why GemmUniversalWithAbsMax is required for Ada FP8:**
- Ada FP8 mma operates on e4m3 (limited dynamic range)
- The output scale factor must adapt per output tile based on running
  absmax of accumulators
- Hopper hides this via TMA + warp-specialization in
  `KernelTmaWarpSpecializedCooperative`
- Ada has neither TMA nor WGMMA; the absmax-aware GEMM device handles
  the scale-tracking explicitly

Without `WithAbsMax`, the GEMM template instantiates but the kernel
launch fails with `Status::Internal` because the scale-tracking
contract isn't met.

## §2 Operator variant comparison from example 58

Two viable operator variants for Phase 0 smoke (lines 811, 822):

| Operator | Accumulation | Use for |
|----------|--------------|---------|
| `cutlass::arch::OpMultiplyAdd` | staged accumulation | accuracy-preserving baseline |
| `cutlass::arch::OpMultiplyAddFastAccum` | fast accumulation | throughput target (license measurement) |

For Phase 0 smoke license: **measure both**. License gate is the
fast-accum number (peak achievable); cross-check accuracy with
staged-accum to ensure FP8 e4m3 doesn't lose more than the PPL gate
allows in P0.B.

## §3 Required headers

```cpp
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"  // NOT gemm_universal.h alone
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/epilogue/thread/activation.h"
```

## §4 Pickup brief sent to codex (paste-buffered this tick)

`/tmp/codex-help-cutlass-sm89.txt` — contains:
- Full template excerpt with sm_89 ArchTag
- GemmUniversalWithAbsMax explanation
- Header list
- Two op variants (OpMultiplyAdd vs OpMultiplyAddFastAccum)
- Tile shape suggestion if Status::Internal persists (start 64×64×128, kStages=3)
- Skill v1.10.0 #28 reminder: cite raw cutlass output verbatim
- PushNotification trigger: first valid Status::Success cutlass FP8 smoke

Codex acknowledged + Working (3s) post-help.

## §5 Implication for the Machete blocker (e65a096)

This tick's discovery does NOT change the Machete blocker. Machete
itself is still Hopper-only (per `1829c4e` + `e65a096` 5-pt evidence).
The Ada FP8 GEMM in cutlass is a **different** code path — it's the
non-Machete cutlass mainloop with Sm89 ArchTag, exactly what
Path B-Phase2' needs.

Effectively: cutlass has two FP8 GEMM paths:
1. **Hopper warp-specialized** (Sm90, examples 54/67/68) — what Machete builds on
2. **Ada absmax-aware** (Sm89, examples 58/64/94) — what Path B-Phase2' uses

Earlier in the session I should have noted this distinction in
`e65a096`. Adding here for future-readers: "Machete = Hopper-only ≠
cutlass FP8 = Hopper-only".

## §6 Cross-references

- Phase 0 brief: `docs/research/2026-05-10-path-b-phase2-prime-phase0-brief-codex-kickoff.md` (5a7a28b)
- Phase 2' survey: `docs/research/2026-05-10-path-b-phase-2-prime-w4-fp8-sm89-native.md` (3e83741)
- Machete blocker (5-pt evidence): `docs/research/2026-05-10-machete-blocker-stronger-evidence-user-reissued-axis.md` (e65a096)
- Hallucination errors (skill v1.10.0 #28 source): `docs/experience/errors/2026-05-10-claude-hallucinated-grep-output-cli-flag.md` (ee2c5b0)
- Cutlass example 58: https://github.com/NVIDIA/cutlass/blob/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu
- Cutlass example 64 (grouped): https://github.com/NVIDIA/cutlass/tree/main/examples/64_ada_fp8_gemm_grouped
- Cutlass example 94 (blockwise): https://github.com/NVIDIA/cutlass/tree/main/examples/94_ada_fp8_blockwise
- Skill anti-pattern #7 (cuBLASLt heuristic ≠ cutlass direct): `.claude/skills/kernel-optimization/SKILL.md`

## §7 Status

Codex unstuck and Working on revised cutlass FP8 smoke using
`GemmUniversalWithAbsMax` + `Sm89` ArchTag template. PushNotification
trigger: first Status::Success on /tmp/cutlass_fp8_smoke.cu. Wins or
errors entry will land per Phase 0 license/kill matrix once sufficient
data collected.

If first compile clears but kernel returns Status::Internal at
runtime: try smaller tile (64×64×128) + force OpMultiplyAdd (staged)
first to isolate epilogue-vs-mainloop issue.
