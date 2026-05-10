---
title: 2026-05-10 P2.5 Step 1 LICENSE NUANCE correction — ARLE has REAL sm_89 advantage (cache-hint cp_async); QQQ "auto-tune" is actually a tradeoff, not strict upgrade
date: 2026-05-10
type: research
status: open (refines 6024d2b LICENSE decision; reduces P2.5 expected gain)
related_docs: [`6024d2b` P2.5 Step 1 LICENSE, `b2cccb9` P2.5 diagnostic, `f8b8174` QQQ upstream finding]
---

# P2.5 Step 1 LICENSE — NUANCE CORRECTION after full hunk audit

> **Why this entry**: `6024d2b` LICENSEd the P2.5/M'' merged port at
> "2-8% TTFT expected gain". This tick completed the audit of all 12
> hunks (was previously only hunks 1, 8, 12). Found 2 critical
> nuances that REDUCE the expected gain and INCREASE the risk profile.

## §1 Nuance 1 — ARLE has REAL sm_89 cache-hint advantage in cp_async (Hunks 3, 6, 7, 8)

ARLE uses **PTX `createpolicy.fractional.L2::evict_first`** in
`cp_async4_stream` (and `cp_async1_stream`):

```c
// ARLE (lines 74-83 of marlin_w4a8_kernel.cu)
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}
```

QQQ uses the **simpler baseline**:

```c
// QQQ (lines 69-79 of qqq_gemm.cu)
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}
```

ARLE's variant tells the GPU to evict B (W4 quantized weights) from L2
immediately — they're accessed once per layer and shouldn't pollute
L2 cache reserved for activations A and outputs C.

This pattern is consistently applied across ARLE Hunks 6, 7, 8 (all
B/scale loads use `_stream` variants).

**Implication**: ARLE has a real sm_89-tuned advantage QQQ lacks.
**MUST PRESERVE in any port.**

## §2 Nuance 2 — QQQ schedule "auto-tune" is actually a TRADEOFF (Hunks 8, 10, 11)

### §2.1 What QQQ does (Hunks 8 + 11)

QQQ replaces ARLE's `CALL_IF` macro (4 args) with `CALL_IF` (3 args)
+ a `thread_config_t` runtime struct + `determine_thread_config()`
runtime function:

```c
// QQQ Hunk 11 — only 4 explicit configs:
CALL_IF(8, 8, 256)
CALL_IF(16, 4, 256)
CALL_IF(8, 4, 128)
CALL_IF(4, 8, 128)
```

ARLE has **10 compile-time-listed configs**:

```c
// ARLE Hunk 11 — 10 explicit configs:
CALL_IF(1,  8,  8, -1)
CALL_IF(1,  8,  8,  8)
CALL_IF(1, 16,  4, -1)
CALL_IF(1, 16,  4,  8)
CALL_IF(2, 16,  4, -1)
CALL_IF(2, 16,  4,  8)
CALL_IF(3, 16,  4, -1)
CALL_IF(3, 16,  4,  8)
CALL_IF(4, 16,  4, -1)
CALL_IF(4, 16,  4,  8)
```

ARLE tracks `(thread_m_blocks, thread_n_blocks, thread_k_blocks,
group_blocks)` — 4-D config space. QQQ tracks
`(thread_n_blocks, thread_k_blocks, num_threads)` — 3-D, dropped
group_blocks specialization.

### §2.2 Tradeoff direction

**QQQ wins**: cleaner runtime adaptability, fewer kernel template
instantiations to compile, cleaner code.

**ARLE wins**: more compile-time-specialized kernels covering more
shapes optimally; likely better perf at the 10 listed shapes.

Per `kernel-optimization` skill anti-pattern #4 (Hopper defaults on
Ada): QQQ may have stripped configs that mattered for sm_89 shapes
ARLE actively uses.

## §3 Refined LICENSE assessment

`6024d2b` listed P2.5/M'' merged at "2-8% TTFT expected gain". After
full hunk audit, this is OPTIMISTIC. Refined estimate:

### §3.1 Possible outcomes

| Outcome | Probability | Reason |
|---|---|---|
| Net gain 2-8% | LOW (~20%) | requires QQQ shapes to match production better than ARLE's 10 configs |
| Net regression < 5% | MEDIUM (~40%) | likely if ARLE's 10 configs cover production shapes well |
| Net regression > 5% | LOW (~10%) | only if ARLE-specific cache-hint stripped accidentally |
| **Neutral ±2%** | **HIGH (~30%)** | **most likely outcome — both kernels mature, similar perf at production shapes** |

### §3.2 Refined LICENSE conditions

ORIGINAL (`6024d2b`):
- License: TTFT/ITL Δ within ±5% (neutral or better) AND greedy 0.0% diff
- Soft win: TTFT/ITL Δ ≥ -2% with σ < 5% across n=3
- Kill: any regression > +5% OR greedy diff > 0.5%

REFINED (with nuance):
- License: same threshold, but EXPECT neutral outcome (~30% prob)
- Add EXPLICIT preservation requirement: ARLE cp_async4_stream +
  cp_async1_stream MUST stay in ported kernel
- Add EXPLICIT diff-config A/B: bench BOTH
  `(QQQ-style auto-config)` and `(ARLE-style 10 configs preserved)`
  to attribute any gain to the right axis

### §3.3 Updated effort estimate

ORIGINAL (`6024d2b`): 3-5 hr (Step 2.A-D codex)

REFINED:
- Step 2.A: 1-2 hr (write code) — same
- Step 2.B: 1-2 hr (Rust dispatcher) — same
- Step 2.C: 1 hr (build + correctness test) — same
- Step 2.D-revised: 1.5 hr (TWO A/B bench runs, not one):
  - Bench A: ARLE current (baseline)
  - Bench B: ARLE + QQQ thread_config_t (port outcome)
  - Bench C (NEW): ARLE + QQQ thread_config_t WITH ARLE's 10 configs
    backported (to attribute gain/loss to dispatcher vs config-space)

**Total revised: 4.5-6.5 hr** (was 3-5 hr).

## §4 Recommendation refresh

P2.5/M'' is STILL the cheapest pickup on the queue (4.5-6.5 hr is
still less than P3 Task #47 v2's 1 day). But expected gain is now
**~30% probability of neutral outcome, ~20% probability of 2-8% gain**.

This shifts P2.5 from "obvious win" to "moderate-risk experiment with
unclear ROI". Codex pickup is still worth doing — even a neutral
result + correctness validation provides reference implementation
for future shape-tuning work — but should be labeled accordingly.

### §4.1 If user is bandwidth-constrained

Consider DEFERRING P2.5/M'' until A+B (P1) lands. The 4.5-6.5 hr
investment could be redirected to A+B Substrate Phase 1.B (per
`f0c7561`), which has higher expected ROI per hour.

### §4.2 If multiple codex tracks available

Run P2.5/M'' as parallel track to A+B. Even neutral outcome adds:
- Validated reference for QQQ-style auto-tune pattern on sm_89
- Potential gain on shapes not covered by ARLE's 10 configs
- SKILL #43 application: confirms upstream-tracking discipline

## §5 Final priority table (refined)

| Priority | Path | Wall-clock | Expected | Probability |
|---|---|---:|---|---|
| P1 | A+B combined | 4-5 days | 2.61× tok/s + -14% latency | HIGH (>70%) |
| P2.5/M'' | QQQ Hunk 8 port (REVISED) | 4.5-6.5 hr | 2-8% OR neutral | 20% gain / 30% neutral / 40% regress < 5% |
| P3 | Task #47 H1' v2 | 1 day | unblocks PF8 path | LOW-MED (~50%) |
| ~~P2~~ | ~~vLLM Marlin diff-port~~ | DONE | maintenance only | |
| ~~P3.5~~ | ~~M''' (W4-FP8 preprocess)~~ | DONE | already integrated | |
| ~~P4 M''~~ | ~~Marlin schedule auto-tune~~ | MERGED INTO P2.5 | | |
| P5 | Option M' (full cutlass rewrite) | 2-3 weeks | 5-15% best-case | LOW |
| KILLED | Literal Machete port | impossible | sm_90+ only | NONE |

## §6 SKILL #43 evidence growing

n=5 evidence now (was n=4 at v1.16.0 graduation per `6577ba6`):
1. `e021026` Alpaca data already downloaded
2. `86b28c7` M''' (W4-FP8 preprocess) already DONE
3. `2f19a3c` ARLE marlin_kernel.cu at-par with vLLM
4. `b6b8adc` ARLE marlin_pf8/ = COMPLETE vLLM fork
5. **THIS entry**: full hunk audit reveals ARLE has REAL sm_89
   advantages (cache-hint cp_async, broader config space) NOT visible
   from file-size comparison alone — survey depth matters

Detection-rule strengthening: source survey must include at least
ONE deep hunk-level read, not just file-size + structural diff.
Otherwise risk shipping "this looks done at file level" judgments
that miss real architectural advantages.

## §7 Cross-references

- `6024d2b` P2.5 Step 1 LICENSE (this entry refines optimism)
- `b2cccb9` P2.5 diagnostic
- `f8b8174` QQQ upstream finding
- `d8ebe73` Machete-inspired reframing §3.2 M''
- `6577ba6` SKILL v1.16.0 #43 graduation (this entry adds n=5 evidence)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`
  - Lines 74-83: cp_async4_stream with L2 cache hint
  - Lines 86-99: cp_async1_stream with L2 cache hint
  - Lines 936-955: 10 CALL_IF entries (broader config space)
- QQQ `csrc/qqq_gemm.cu`
  - Lines 69-79: cp_async4 baseline (no cache hint)
  - Lines 84-93: cp_async1 baseline
  - Lines 1030-1040: 4 CALL_IF entries (narrower config space)
- `/tmp/qqq-arle-w4a8.diff` (full 446-line diff)
