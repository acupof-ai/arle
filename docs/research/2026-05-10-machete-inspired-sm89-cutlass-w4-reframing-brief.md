---
title: 2026-05-10 Machete-inspired sm_89 cutlass W4 kernel — reframing of user "Machete W4 移植 from vLLM" directive after architectural KILL
date: 2026-05-10
type: research
status: open (gives user coherent forward path on Machete directive intent)
related_docs: [`fc33cfb` Machete KILL, `2b956ce` sm_89 alternatives, `89a04d7` loop-arg staleness audit, `bccf1bd` consistency audit]
---

# Machete-inspired sm_89 cutlass W4 kernel — reframing the user "port Machete" directive

> **Why this brief**: User's /loop directive persistently states
> "**当前主轴: Machete W4 kernel 移植 from vLLM** ... port machete from
> vllm-project/vllm to ARLE crates/cuda-kernels for sm_89 W4A8 优化
> (预估 -20-40% ITL vs current Marlin)". This is architecturally
> impossible — `fc33cfb` confirmed Machete is HOPPER-ONLY (sm_90+ WGMMA
> + TMA), 0% benefit on sm_89 ARLE primary hardware.
>
> This brief offers a coherent reframing of the directive's INTENT
> (what optimization the user actually wants) into an sm_89-feasible
> work plan. Outcome: rename the axis to "Machete-inspired sm_89
> cutlass W4 kernel" with clear LOC + risk + ROI estimates.

## §1 What user likely wants (charitable interpretation)

The "-20-40% ITL vs current Marlin" estimate suggests user is targeting
a Machete-class W4 kernel improvement. The literal Machete kernel is
not the only path to that — Machete's KEY INNOVATIONS that COULD be
backported to sm_89 are:

1. **Cutlass-based templated kernel** (vs Marlin's hand-tuned PTX)
   - Trade off: more flexible per-shape tuning, but heavier compile time
2. **Prepacked weight layout** matching tensor core fragment alignment
   - Marlin already does this for sm_75+ mma; Machete extends to wgmma
3. **ScheduleConfig pattern** for per-problem-shape tile selection
   - Could auto-tune Marlin's tiles per workload shape
4. **Cleaner add-new-types path** (from Cutlass templates)
   - Lower friction to add W3/W2/W6/etc. quant variants

These are sm_89-COMPATIBLE concepts. WGMMA + TMA are NOT.

## §2 Existing ARLE Marlin path (the candidate-replace target)

`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (41 KB):
- Adapted from HandH1998's W4A8 mods to IST-DASLab Marlin
- Uses raw PTX `mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32`
  (sm_80+ tensor core mma)
- Hand-tuned tile config (BLOCK_M, BLOCK_N, NUM_STAGES, NUM_THREADS)
- Already at-par with vLLM Marlin on sm_89 per `bccf1bd` measurements

Per `kernel-optimization` skill Phase 3 binding-constraint: Marlin
W4A16 + W4A8 paths on sm_89 currently HEALTHY (Arms C + D in PF8.5
4-arm A/B). Wall-clock TTFT/ITL is good enough to be the perf ceiling.

## §3 Reframing options (Machete-inspired alternatives)

### §3.1 Option M' (full cutlass rewrite)

Port ARLE Marlin to cutlass templates with sm_89-compatible mma atoms
(NOT WGMMA). Estimated work:
- LOC: ~1500-2000 (cutlass kernel + tile configs + Rust dispatcher)
- Wall-clock: 2-3 weeks (codex GPU work)
- Risk: HIGH — cutlass template depth is hard to get right; regression
  vs current hand-tuned Marlin likely on small shapes
- Expected gain on sm_89: **5-15% on best-case shapes, possibly 0%
  or regression on others**
- Why limited: The big Machete wins on sm_90 come from WGMMA 128×N
  shape + TMA bandwidth, neither available on sm_89. Without those,
  cutlass templates compete head-to-head with hand-tuned PTX, which
  is what Marlin is.

### §3.2 Option M'' (cutlass-style schedule auto-tune ON existing Marlin)

Apply Machete's ScheduleConfig pattern to ARLE's existing Marlin —
auto-pick BLOCK_M/N/STAGES per problem shape. Estimated work:
- LOC: ~200-400 (heuristic dispatcher + new template instantiations)
- Wall-clock: 3-5 days (codex)
- Risk: LOW-MEDIUM — adds dispatch logic, kernel core unchanged
- Expected gain: **2-8% on shape-mismatched paths** (most ARLE prefill
  shapes already match the chosen tile)

### §3.3 Option M''' (just port the W4-FP8 preprocess from vLLM Marlin)

vLLM `marlin_int4_fp8_preprocess.cu` is sm_75+ compatible AND addresses
the W4-FP8 PF8 path that ARLE PF8.3 broke. This would directly help
Task #47 redesign (per `494ad3a`):
- LOC: ~100 (preprocess kernel + dispatch wiring)
- Wall-clock: 1-2 days (codex)
- Risk: LOW
- Expected gain: validates one of the 3 PF8.3 failure-mode hypotheses
  (per `M_pf83_h1prime_v2_redesign_brief.md` §1)

## §4 Priority placement (refines `2b956ce` §5)

| Priority | Option | Wall-clock | Expected | Risk |
|---|---|---:|---|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | 2.61× tok/s + -14% latency | LOW |
| P2 | vLLM upstream Marlin diff-port | 1-2 days | 2-5% improvement | LOW |
| P3 | Task #47 H1' v2 (per `494ad3a`) | 1 day | unblocks PF8 path | LOW-MEDIUM |
| P3.5 | **Option M''' (W4-FP8 preprocess port)** | **1-2 days** | **complements P3** | **LOW** |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | 2-8% conditional | LOW-MEDIUM |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | 5-15% on best shapes, possibly 0% | HIGH |
| P6 | Wait sm_100 (NVFP4 native) | months | new hardware path | LOW |
| KILLED | Literal Machete port (sm_90+) | impossible | 0% on sm_89 | NONE |

### §4.1 Why M'/M'' rank LOW vs A+B

ARLE's existing Marlin is highly optimized for sm_89. The Machete-class
"-20-40% ITL" gain comes from architectural features sm_89 LACKS
(WGMMA, TMA). Without those, kernel-level rewrites compete on
diminishing returns axis (~5-15% best-case).

Meanwhile A+B exploit ORTHOGONAL axes (model architecture / dispatch
phase) where ARLE has clear gaps to fill (no Medusa heads exist; no
phase-routed dispatch exists). Expected gain there is fundamental
multiplicative (2.61×), not incremental (5-15%).

**Recommendation**: do A+B first. Re-evaluate M'/M'' only if A+B
under-delivers AND user confirms wall-clock budget for ~2-3 weeks of
risky kernel work.

## §5 What this brief gives user

If user's intent in repeating "Machete W4 移植" is "I want Machete-class
gains on sm_89", the SOLID answer is:
- **Literal port**: impossible (architecturally killed)
- **A+B combined**: the actual sm_89-feasible Machete-class path
  (~2.61× tok/s, ready for pickup, 4-5 days)
- **M'/M''**: lower-priority kernel-axis fallbacks (5-15% gain, weeks)
- **M'''**: complements Task #47 PF8 redesign (1-2 days)

If user actually wants kernel-axis exploration despite the lower ROI,
M'' is the lowest-risk entry point. M' is high-risk / multi-week and
should not start without explicit time-budget approval.

## §6 Cross-references

- `fc33cfb` Machete KILL (the architectural impossibility)
- `2b956ce` sm_89 W4 alternatives (this brief refines §5 priority table)
- `bccf1bd` consistency audit (validates A+B as Machete-class path)
- `494ad3a` Task #47 H1' v2 redesign brief (parallel to M''')
- `89a04d7` cron-loop arg staleness audit
- `9735b47` REFUTATION (the original measurement-based finding)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (current path)
- vLLM `csrc/quantization/marlin/marlin.cu` (sm_75+ compatible reference)
- vLLM `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu` (M''' source)
- vLLM `csrc/quantization/machete/` (KILLED — Hopper-only)
- `kernel-optimization` skill Phase 3 (binding-constraint discipline)
- `kernel-optimization` skill anti-pattern #7 (cuBLASLt vs cutlass direct)

## §7 Loop directive note

The repeating "**当前主轴: Machete W4 kernel 移植 from vLLM**" line in
recent /loop firings was written before `fc33cfb` Machete KILL +
`2b956ce` alternatives + `89a04d7` staleness audit + this brief.
Per `89a04d7` §3 recommendation, refresh the loop prompt to:

```
当前主轴: A+B 双轴推进 (Medusa + Hybrid, 4-5 days, 2.61× tok/s + -14% latency)
  + Task #47 PF8.3 H1' v2 (parallel, 1 day, ~50 LOC, P3 fallback)
Machete literal port KILLED (sm_90+ only). Machete-inspired sm_89
options (M'/M''/M''') documented in this brief, all LOW priority vs A+B.
```
