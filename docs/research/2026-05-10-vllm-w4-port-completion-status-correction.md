---
title: 2026-05-10 vLLM W4 kernel port completion status — M''' (W4-FP8 preprocess) ALREADY DONE; refines d8ebe73 priority table
date: 2026-05-10
type: research
status: open (corrects d8ebe73 §3.3 P3.5 stale claim)
related_docs: [`d8ebe73` Machete-inspired reframing brief, `2b956ce` sm_89 alternatives, `89a04d7` loop-arg staleness audit]
---

# vLLM W4 kernel port completion status — M''' already done

> **Why this**: `d8ebe73` §3.3 listed "P3.5 Option M''' (W4-FP8
> preprocess port from vLLM): 1-2 days, complements P3". Direct file
> survey shows ARLE has ALREADY ported this kernel as PF8.2 substep.
> M''' is NOT pending — it's DONE. This entry corrects the priority
> table.

## §1 Evidence: PF8.2 IS the M''' port

ARLE file: `crates/cuda-kernels/csrc/gemm/marlin_int4_fp8_preprocess.cu`
(70 LOC).

Header literal text (L3-4):
> "Verbatim port of vLLM's
> `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu`
> `_without_zp` kernel under Apache 2.0 license."

L21:
> "Adapted from:
> https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"

Wired up at:
- `crates/cuda-kernels/src/ffi/gemm.rs:141` (extern "C" declaration)
- `crates/cuda-kernels/src/tensor.rs:888` (call site)

### §1.1 Differences from upstream

| Aspect | vLLM | ARLE PF8.2 | Why |
|---|---|---|---|
| `without_zp` kernel | YES | YES | needed for GPTQ checkpoints |
| AWQ variant | YES | **deliberately SKIPPED** | ARLE has no AWQ checkpoint loader |
| Torch FFI | YES | replaced with extern "C" + cudaStream_t | matches `crates/cuda-kernels/src/ffi/gemm.rs` pattern |
| Apache 2.0 attribution | YES | YES (preserved) | license compliance |

The skip of AWQ is deliberate, not a gap. ARLE consumes GPTQ
checkpoints only.

## §2 Refined post-Machete-KILL priority table

This supersedes `d8ebe73` §4 and `2b956ce` §5:

| Priority | Path | Wall-clock | Status | Expected |
|---|---|---:|---|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | gated on user GO | 2.61× tok/s + -14% latency |
| P2 | vLLM upstream Marlin diff-port | 1-2 days | open | 2-5% improvement |
| P3 | Task #47 H1' v2 (per `494ad3a`) | 1 day | gated on diagnostic logging | unblocks PF8 path |
| ~~P3.5~~ | ~~Option M''' (W4-FP8 preprocess port)~~ | **DONE** | **PF8.2 in production** | already integrated |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | open | 2-8% conditional |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open | 5-15% best-case, HIGH risk |
| P6 | Wait sm_100 (NVFP4 native) | months | hardware | new path |
| KILLED | Literal Machete port (sm_90+) | impossible | KILLED `fc33cfb` | 0% on sm_89 |

## §3 What this means for user "port Machete" directive

User's intent ("get Machete-class W4 wins on sm_89") has TWO layers:

### §3.1 Layer 1 — sm_89-compatible vLLM Marlin kernels

ALREADY MOSTLY DONE:
- `marlin.cu` (sm_75+ main kernel) → ARLE `marlin_kernel.cu` (33.8 KB)
- `marlin_dequant.cuh` → ARLE `marlin_dequant.cuh` (23.4 KB)
- `marlin_int4_fp8_preprocess.cu` (W4-FP8 preprocess) → **DONE PF8.2**
- `marlin_repack.cu` (weight repack) → ARLE `marlin_repack.cu` (5.8 KB)
- `marlin_w4a8_kernel.cu` (HandH1998 W4A8 mods) → ARLE `marlin_w4a8_kernel.cu` (41.4 KB)

ARLE Marlin path is at-par with vLLM v0.x Marlin path. Only delta is:
- AWQ variant (deliberately skipped)
- Possible recent vLLM upstream improvements (P2, 2-5% gain)

### §3.2 Layer 2 — Machete's architectural advantages

These come from sm_90+ features (WGMMA, TMA, 228 KB smem) that sm_89
LACKS. No port can backfill these. Closest approximation:
- A+B combined (orthogonal multiplicative gain, ~2.61×)
- M'' / M' (cutlass-template kernel rewrite, 5-15% best-case, weeks)

### §3.3 Recommendation (refined)

**Pickup order** (sm_89-feasible, post-this-correction):

1. **P1: A+B combined** — biggest single ROI (4-5 days, 2.61× tok/s)
2. **P2: vLLM Marlin diff-port** — quick win (1-2 days, 2-5%)
3. **P3: Task #47 H1' v2** — unblocks PF8 path (1 day)
4. ~~~~ M''' DONE ~~~~
5. **P4-P5**: only if P1-P3 underdeliver

Total wall-clock for P1+P2+P3 = ~7 days for compound: 2.61× tok/s
+ -14% latency + 2-5% Marlin + PF8 path unblock.

## §4 Cross-references

- `d8ebe73` Machete-inspired reframing brief (this entry refines §3.3 P3.5)
- `2b956ce` sm_89 W4 alternatives (this entry refines §5 priority)
- `fc33cfb` Machete KILL errors entry
- `494ad3a` Task #47 H1' v2 redesign brief
- `89a04d7` loop-arg staleness audit
- `bccf1bd` Hybrid plan consistency audit
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_int4_fp8_preprocess.cu` (PF8.2, 70 LOC)
- ARLE `crates/cuda-kernels/src/ffi/gemm.rs:141` (FFI declaration)
- ARLE `crates/cuda-kernels/src/tensor.rs:888` (call site)
- vLLM `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu` (upstream source)

## §5 Rule

When briefing pickup options, ALWAYS file-survey existing source
BEFORE listing items as "pending". Saved 1-2 days false-pending
work in this audit; would have wasted codex time if user picked it
up under the d8ebe73 stale claim.

This is SKILL candidate "always-source-survey-before-pending-list" —
n=2 evidence (this entry + `e021026` Alpaca-already-done discovery).
Recommend graduating at next SKILL bump.
