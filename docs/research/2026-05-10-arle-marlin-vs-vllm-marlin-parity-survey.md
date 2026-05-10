---
title: 2026-05-10 ARLE marlin_kernel.cu vs vLLM marlin.cu — parity survey (P2 priority sizing)
date: 2026-05-10
type: research
status: open (sizes P2 vLLM Marlin diff-port effort)
related_docs: [`86b28c7` M''' completion correction, `2b956ce` sm_89 alternatives, `d8ebe73` Machete-inspired reframing]
---

# ARLE marlin_kernel.cu vs vLLM marlin.cu — parity survey

> **Why now**: `86b28c7` confirmed M''' (W4-FP8 preprocess) DONE.
> Refined priority table puts P2 (vLLM upstream Marlin diff-port) at
> 1-2 days for 2-5% gain. This survey sizes whether P2 actually has
> ~~5% diff to port, or whether ARLE Marlin is already at-par.

## §1 Header attribution

**ARLE `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (L1-15)**:
```c
/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 * Licensed under the Apache License, Version 2.0 ...
 */
```

NO vLLM attribution. ARLE forked directly from IST-DASLab upstream
(Frantar 2024), NOT from vLLM's adapted version.

**vLLM `csrc/quantization/marlin/marlin.cu` (L1-3)**:
```c
/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 */
```

Same upstream root (Frantar 2024), but vLLM has Neural Magic
modifications layered on top.

## §2 File-level comparison

| Metric | ARLE marlin_kernel.cu | vLLM marlin/marlin.cu |
|---|---:|---:|
| Lines | 828 | (gh API content fetch failed; estimated similar) |
| Bytes | 33,821 | 32,798 |
| Arch guards | none explicit (sm_75+ implicit via mma PTX) | `#if __CUDA_ARCH__ < 750` (L40) |
| Internal namespace | `arle::marlin::vllm::kU4B8` (L132) | `marlin::` |

**Size delta is ~3% (ARLE +1 KB)** — ARLE has slightly more code,
likely from FFI shim + cudarc adaptations.

## §3 Partial vLLM-derived helpers in ARLE

ARLE `marlin_kernel.cu` line 132:
```c
arle::marlin::vllm::kU4B8.id()
```

The presence of `arle::marlin::vllm` namespace indicates ARLE has
already **selectively pulled in vLLM-modified types/helpers** beyond
the bare IST-DASLab fork. This includes the W4 ID enum (`kU4B8`)
which vLLM defines for its quant-type registry.

Other ARLE marlin files with vLLM attribution (per `86b28c7` survey):
- `marlin_int4_fp8_preprocess.cu` (PF8.2): "Verbatim port of vLLM's ..."
- `marlin_w4a8_kernel.cu`: "Adapted from HandH1998's W4A8 mods to
  IST-DASLab Marlin" (HandH1998 = the original W4A8 contributor that
  vLLM also picked up)

## §4 What's the actual delta?

Without performing a full textual diff (gh API content fetch failed
this session — would need next-tick), the EVIDENCE-BASED conclusion:

- **ARLE is NOT directly tracking vLLM commits**. ARLE forked from
  IST-DASLab + selectively pulled vLLM modifications (PF8.2 verbatim,
  W4A8 via HandH1998).
- **Likely deltas** vLLM has that ARLE doesn't:
  - Multi-shape ScheduleConfig dispatch (per Machete README pattern,
    vLLM may have backported similar to gptq_marlin)
  - Continuous tile-config updates from Neural Magic upstream
  - Possible additional quant type IDs (kU4, kU8 variants beyond kU4B8)
- **Likely deltas** ARLE has that vLLM doesn't:
  - cudarc FFI shim (extern "C" + cudaStream_t pattern)
  - sm_89-tuned stage/tile choices (per CLAUDE.md `kernel-optimization`
    skill Phase 2 hardware constraint sheet)

## §5 P2 effort sizing — REVISED estimate

`d8ebe73` + `86b28c7` cited P2 as "1-2 days, 2-5% gain". Based on this
parity survey:

- **Lower-bound effort (just diff against current vLLM)**: 0.5 day
  (codex reads vLLM file, identifies new tile configs, ports 1-2)
- **Upper-bound effort (full track-vLLM)**: 2-3 days
  (requires testing all new tile configs, may regress on ARLE's
  sm_89-specific path)
- **Expected gain**: 0-5% (could be 0% if vLLM upstream hasn't moved
  much since ARLE's last sync). Risk-adjusted expected ~2%.

### §5.1 Recommended P2 approach

1. **Diagnostic step** (0.5 day Claude): full textual diff of ARLE
   `marlin_kernel.cu` vs current vLLM `marlin.cu` to enumerate
   actual delta
2. **License-or-kill** (per `kernel-optimization` skill Phase 8):
   if delta < 50 LOC, do P2 (small-effort, easy gain)
   if delta > 200 LOC, defer P2 (high-effort, uncertain gain;
   investment better spent on A+B)
3. **P2 codex pickup** (1 day): port the identified delta, A/B bench
   on conc=1+conc=4 W4A16 sustained 60s

## §6 Updated priority table (refines `86b28c7` §2)

| Priority | Path | Wall-clock | Status | Expected |
|---|---|---:|---|---|
| P1 | A+B combined | 4-5 days | gated on user GO | 2.61× tok/s + -14% latency |
| P2 | vLLM Marlin diff-port | **0.5d diagnostic + 1d port = 1.5d** | open, **needs diff first** | 0-5% (~2% risk-adjusted) |
| P3 | Task #47 H1' v2 | 1 day | gated on diagnostic logging | unblocks PF8 path |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE | PF8.2 in production | already integrated |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | open | 2-8% conditional |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open | 5-15% best-case, HIGH risk |
| KILLED | Literal Machete port | impossible | KILLED `fc33cfb` | 0% on sm_89 |

## §7 Cross-references

- `86b28c7` vLLM W4 port completion correction (this entry refines §2 P2 row)
- `d8ebe73` Machete-inspired reframing brief
- `2b956ce` sm_89 W4 alternatives
- `fc33cfb` Machete KILL
- `494ad3a` Task #47 H1' v2 redesign brief
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (828 lines, IST-DASLab fork)
- vLLM `csrc/quantization/marlin/marlin.cu` (~32.8 KB, Neural Magic mod)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (HandH1998 W4A8 mod)

## §8 Rule

For "track upstream X" pickups, the SOLID first step is a diff-driven
diagnostic, NOT immediate porting. Without diff data, "1-2 days, 2-5%
gain" estimates are vibes. Diff first → estimate effort second →
license-or-kill third. This adds another data point to the SKILL
candidate "always-source-survey-before-pending-list" — n=3 evidence
now (this entry + `e021026` Alpaca + `86b28c7` M''').
