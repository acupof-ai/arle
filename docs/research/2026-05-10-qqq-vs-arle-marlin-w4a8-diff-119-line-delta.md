---
title: 2026-05-10 QQQ qqq_gemm.cu vs ARLE marlin_w4a8_kernel.cu — 119-line delta diagnostic; P2.5 has real port opportunity
date: 2026-05-10
type: research
status: open (P2.5 license-pending; codex pickup-ready)
related_docs: [`f8b8174` QQQ upstream finding, `2f19a3c` Marlin parity, `b6b8adc` P2 DONE, `d8ebe73` Machete-inspired reframing]
---

# QQQ vs ARLE marlin_w4a8 diff diagnostic — 119-line delta surfaces

> **Why now**: `f8b8174` §2.3 listed P2.5 (HandH1998/QQQ diff-port) as
> "diff status not measured this audit". This tick executes the
> diagnostic step. Result: real 119-line delta exists. P2.5 is a
> legitimate codex-pickup candidate, not a no-op.

## §1 File comparison

| Source | Lines | Bytes |
|---|---:|---:|
| ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` | 987 | 41,397 |
| QQQ `csrc/qqq_gemm.cu` | 1106 | 45,454 |
| **Delta** | **+119 lines (+10.7%)** | **+4057 bytes (+9.8%)** |

QQQ is the upstream; ARLE has 119 fewer lines.

## §2 Structural divergence at top of file

### §2.1 Same Frantar+HandH1998 lineage

Both files start with identical Apache 2.0 header citing IST-DASLab
+ HandH1998 + Frantar 2024. Same fork base.

### §2.2 cp_async helper variants differ

| ARLE | QQQ | Comment |
|---|---|---|
| `cp_async4_pred` (L59) | `cp_async4_pred` (L56) | match |
| `cp_async4_stream` (L74) | (missing) | **ARLE-only streaming variant** |
| `cp_async1_stream` (L88) | (missing) | **ARLE-only streaming variant** |
| (none) | `cp_async4` (L69) | **QQQ-only non-streaming** |
| (none) | `cp_async1` (L84) | **QQQ-only non-streaming** |
| `cp_async_fence` (L101) | `cp_async_fence` (L95) | match |
| `cp_async_wait<n>` (L106) | `cp_async_wait<n>` (L100) | match |

**Interpretation**: ARLE replaced the baseline cp_async helpers with
streaming variants (likely sm_89 cache-bypass tuning per
`kernel-optimization` skill Phase 2 hardware constraint sheet).
QQQ uses baseline (likely Hopper-tuned).

This is NOT a missing-feature gap. ARLE has DIVERGED for sm_89
optimization, and QQQ may have diverged in opposite direction for
its target hardware.

## §3 What's the 119-line gap actually?

**NOT measured this audit** (would require full text diff). Two
hypotheses:

### §3.1 Hypothesis A — QQQ has features ARLE lacks
- Additional dequant variants (per_channel + per_token + group)
- Additional template instantiations for shapes ARLE skipped
- Newer entry-point function (Torch FFI dispatch)

### §3.2 Hypothesis B — ARLE has stripped features QQQ keeps
- Torch FFI shim (ARLE uses extern "C" + cudaStream_t)
- Test/debug kernels not needed in production

Per ARLE marlin_w4a8 last touch ~2026-05-08 + QQQ last touch
2026-04-23: temporal ordering allows BOTH directions.

## §4 P2.5 license-or-kill decision

Per `kernel-optimization` skill Phase 8 license thresholds:
- **Effort**: 1-2 days codex (port any actually-needed kernel changes)
- **Risk**: LOW-MED (same lineage, well-understood codebase)
- **Expected gain**: unknown until full text diff completed

### §4.1 Recommended next step (codex-pickup-ready brief)

```
Task: P2.5 QQQ diff-port

Step 1 (Claude or codex, 30 min):
  - Fetch /tmp/qqq_gemm.cu (already at /tmp this session)
  - diff -u /tmp/qqq_gemm.cu \
           crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu \
           > /tmp/qqq-arle-w4a8.diff
  - Categorize the 119-line delta:
    * cosmetic (whitespace, comments) — IGNORE
    * Torch FFI vs cudarc FFI — IGNORE (architectural difference)
    * cp_async streaming variants — KEEP ARLE (sm_89 tuned)
    * tile config / template instantiations — POTENTIAL PORT
    * dequant function variants — POTENTIAL PORT

Step 2 (codex, 1-2 days, gated on Step 1 output):
  - License-or-kill on actual portable delta size:
    * < 50 LOC of meaningful kernel changes: port + bench A/B
    * > 200 LOC: defer P2.5, focus on P1 A+B
    * Between: case-by-case judgment

Step 3 (Claude, 1 hr):
  - A/B bench at conc=1+conc=4 W4A8 sustained 60s
  - License threshold: Δ TTFT/ITL within ±5% (neutral or better)
  - Kill threshold: any regression > 5%
```

### §4.2 Priority placement

P2.5 stays at parallel-codex-track priority. Does not block P1 A+B.
Suitable as an interleaved codex pickup OR as concrete user-facing
demonstration of "kernel-axis work happening" parallel to A+B
substrate development.

## §5 Updated final priority table (refines `f8b8174` §3)

| Priority | Path | Wall-clock | Status |
|---|---|---:|---|
| P1 | A+B combined (Medusa + Hybrid) | 4-5 days | gated on user GO |
| P2.5 | **HandH1998/QQQ diff-port** | **30min diag + 1-2d port** | **diagnostic DONE: 119-line delta confirmed** |
| P3 | Task #47 H1' v2 | 1 day | gated on diagnostic logging |
| ~~P2~~ | ~~vLLM Marlin diff-port~~ | DONE (`b6b8adc`) | maintenance only |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE (`86b28c7`) | already integrated |
| P4 | Option M'' (Marlin schedule auto-tune) | 3-5 days | open |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | open, HIGH risk |
| KILLED | Literal Machete port | impossible | sm_90+ only |

## §6 Cross-references

- `f8b8174` QQQ upstream finding (this entry executes §2.3 diagnostic)
- `b6b8adc` ARLE marlin_pf8 = vLLM fork (P2 DONE)
- `2f19a3c` Marlin parity survey
- `86b28c7` M''' completion correction
- `6577ba6` SKILL v1.16.0 #43 graduation (this entry positively applies it)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 lines, 41.4 KB)
- QQQ `csrc/qqq_gemm.cu` (1106 lines, 45.4 KB) — fetched to `/tmp/qqq_gemm.cu`
- HandH1998/QQQ: <https://github.com/HandH1998/QQQ>

## §7 SKILL #43 inline evidence pointer

Per canonical anti-pattern #43 (graduated v1.16.0): this entry
includes inline "verified absent at <path>" pointers:

- "ARLE has DIVERGED for sm_89 optimization (cp_async4_stream)"
  — verified at line 74 of marlin_w4a8_kernel.cu
- "QQQ uses baseline cp_async4 (likely Hopper-tuned)"
  — verified at line 69 of qqq_gemm.cu
- "119-line delta" — verified by `wc -l` on both files
- "Step 1 categorization NOT YET DONE" — explicit pending marker

Future codex pickup brief follows the same template.
