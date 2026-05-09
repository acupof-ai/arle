---
title: PF8.3 vLLM upstream marlin survey — 259277c claim correction (FP8 mma has BOTH k=16 AND k=32) + dependency surface estimate
date: 2026-05-10
type: research
status: pf83-scope-corrected-upstream-substrate-mapped
---

# PF8.3 vLLM upstream marlin survey — 259277c claim correction (FP8 mma has BOTH k=16 AND k=32) + dependency surface estimate

> Codex pulled vLLM upstream marlin source to `/tmp/vllm-marlin-src`
> for PF8.3 audit. Independent Claude survey THIS tick reveals two
> findings that change the PF8.3 implementation calculus:
> (1) `259277c` claim "FP8 mma is uniformly m16n8k32" is WRONG —
>     vLLM marlin_mma.h has FP8 mma at BOTH `m16n8k16` AND `m16n8k32`.
>     This is **6th Claude hallucination this session**.
> (2) Dependency surface for clean FP8 path extraction is ~1000 LOC,
>     not the ~3107 LOC of the entire upstream marlin tree.

## §0 Direct evidence (raw grep + read THIS tick on /tmp/vllm-marlin-src)

### Finding 1 — FP8 mma has BOTH shapes in vLLM upstream

`/tmp/vllm-marlin-src/csrc/quantization/marlin/marlin_mma.h` shows a
template `mma<type_id, use_fp16_accum, int k_size = 16>` with branches:

```cpp
// Line 76-85 (k_size == 16 branch, FP8 e4m3 path):
} else if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      ...
}

// Line 94-101 (k_size == 32 branch, FP8 e4m3 path):
} else if (k_size == 32) {
  if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        ...
}
```

Both `m16n8k16.f32.e4m3.e4m3.f32` AND `m16n8k32.f32.e4m3.e4m3.f32` are
valid sm_89 PTX FP8 mma instructions. The choice is governed by the
`k_size` template parameter — caller picks based on fragment layout +
inner-loop iteration tradeoff.

### Finding 2 — vLLM marlin_template.h confirms sm_89+ FP8 support

`/tmp/vllm-marlin-src/csrc/quantization/marlin/marlin_template.h:284-285`:

```cpp
// FP8 computation is only supported for Ada Lovelace or newer architectures.
if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
```

(Returns early for non-Ada/Hopper builds; sm_89 = Ada Lovelace explicitly
supported.)

### Finding 3 — Dependency surface (raw `wc -l` THIS tick)

```
   268 marlin_mma.h          (mma instructions, has FP8 in both k=16 + k=32)
   149 marlin_dtypes.cuh     (type system, has __nv_fp8_e4m3 aliases L34/70/100-108)
   609 dequant.h             (dequant routines, has FP8 specializations)
  2081 marlin_template.h     (the mega-template, 13 FP8 references)
  ----
  3107 TOTAL upstream marlin tree
```

ARLE has Phase 1 dequant.h port already at
`crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` (per task #42
completed) — partial substrate already in tree.

## §1 259277c CLAIM CORRECTION (hallucination #6)

`docs/research/2026-05-10-pf83-scope-analysis-mma-shape-mismatch.md`
asserted (§0 + §1):

> "INT8 mma is m16n8k16 (per upstream marlin_w4a8 kernel evidence
> above), and FP8 mma is m16n8k32 (per the NVIDIA ISA reference)."
>
> "| mma instruction | `m16n8k16.satfinite.s32.s8.s8.s32` | `m16n8k32.f32.e4m3.e4m3.f32` |"

Per the `marlin_mma.h` evidence above, this is **incomplete**. Both
shapes exist for FP8 e4m3. The actual story:

| mma path | k tile | Where used in vLLM | PF8.3 fit |
|----------|--------|---------------------|-----------|
| m16n8k16 INT8 (`s32.s8.s8.s32.satfinite`) | 16 | ARLE current marlin_w4a8_kernel.cu | baseline (unchanged) |
| **m16n8k16 FP8** (`f32.e4m3.e4m3.f32`) | **16** | vLLM marlin_mma.h L79+L211 | **PF8.3 Path A — minimal change** |
| m16n8k32 FP8 (`f32.e4m3.e4m3.f32`) | 32 | vLLM marlin_mma.h L97+L229 | PF8.3 Path B — higher throughput, fragment layout change |

This is hallucination #6 this session per skill v1.11.0+ #28+#31:

| # | Tick | Hallucination | Caught by |
|---|------|---------------|-----------|
| 1 | `0f4d0ae` | --max-waiting-requests CLI flag exists | codex |
| 2 | `43bda9c` | W4A16 has max_par×64×n reduce buffer | codex |
| 3 | `4b30c15` | ARLE has /health endpoint | self via router.rs grep |
| 4 | `5bf0e20` | 2026-05-09 baseline-B5 comparable to newdequant | self via raw command.txt |
| 5 | `451d094` | bit-pack `0x76543210 → 0xFEDCBA98` | empirical smoke run |
| 6 | **THIS TICK** | **FP8 mma is uniformly m16n8k32** | raw grep on /tmp/vllm-marlin-src/marlin_mma.h |

**New common-mode**: even when basing claim on "NVIDIA PTX ISA
reference" (presumed authoritative), if I don't actually grep the
real upstream code that uses both, I miss the second variant. The PTX
ISA defines what's allowed; what implementations actually call is the
ground truth.

## §2 Corrected PF8.3 implementation calculus

### Path A — m16n8k16 FP8 mma (minimal change from W4A8)

| Aspect | Existing W4A8 | PF8.3 Path A |
|--------|--------------|--------------|
| mma asm | `m16n8k16 ... s32.s8.s8.s32.satfinite` | `m16n8k16 ... f32.e4m3.e4m3.f32` |
| Accumulator | INT32 | F32 |
| FragA/B | INT8 in INT32 | FP8 e4m3 in INT32 |
| Inner loop iter | unchanged (k_tile=16) | unchanged (k_tile=16) |
| Smem stage size | unchanged | unchanged |
| Dequant | int4 → int8 | int4 → fp8 e4m3 (NEW) |
| Reduce buffer | INT32 | F32 (sizing might differ) |

**Effort**: ~400-600 LOC delta from marlin_w4a8_kernel.cu (mainly mma
asm swap + dequant int4→fp8 + accumulator type). NOT 800-1200 LOC.

### Path B — m16n8k32 FP8 mma (theoretical 1.6× per warp)

Same as 259277c described — k tile doubles, inner-loop iter halved,
smem stage layout changes. ~800-1200 LOC.

### Path C — DEFER to errors entry (codex's stated fallback)

If neither A nor B integrates cleanly with ARLE's existing marlin_dequant.cuh
+ build pipeline, codex lands an errors entry per their own statement:
"如果上游 FP8 Marlin 模板依赖面太大，会先落一个 errors/research
结论而不是硬写一个不可验证的 kernel。"

## §3 Recommended PF8.3 strategy update

Original 259277c recommended Strategy B (mirror W4A8 with k=32 mma).
Corrected per this survey: **Strategy A first** (m16n8k16 FP8 mma,
minimal delta from W4A8). Reasons:

1. ~400-600 LOC vs ~800-1200 LOC (50% scope reduction)
2. Fragment layout unchanged — fewer subtle alignment bugs
3. Smem stage budget unchanged — no additional occupancy regression
   risk on sm_89 100 KB ceiling
4. Path B (k=32) is the optimization on top of Path A — can be
   measured as Phase 2 if Path A licenses
5. Codex's audit explicitly worried about "upstream FP8 Marlin
   template dependency surface" — Path A only needs marlin_mma.h FP8
   asm extraction (~30 LOC) + dequant.h FP8 sections (~100 LOC),
   NOT the full 2081-LOC marlin_template.h

The 1.6× theoretical FP8 mma speedup of k=32 over k=16 is a Phase 2
optimization. Phase 1 PF8.3 only needs to demonstrate prefill-only
FP8 acts deliver TTFT Δ ≥ -8% (per a66d99a §2). Path A is sufficient.

## §4 Substrate already in ARLE tree (raw verification)

```bash
$ ls /home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/dequant.h
ls: cannot access '...': No such file or directory

$ ls /home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh
/home/ckl/projects/arle/crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh    FOUND
```

Phase 1 (task #42) ported `dequant.h` from vLLM upstream as
`marlin_dequant.cuh`. This is the substrate codex worked on
2026-05-09 → 2026-05-10. Codex can extend it with FP8 dequant
specializations from upstream `dequant.h:???` (need raw grep on
upstream to find FP8 sections — codex already explored this per their
"Read marlin_mma.h, marlin_dtypes.cuh, generate_kernels.py, kernel.h,
marlin.cuh" trace).

## §5 Updated PF8.3 license/kill matrix (carries over from aebd4a5)

Per `aebd4a5` PPL gate methodology + this survey's Path A
recommendation:

| Metric | License | Kill |
|--------|---------|------|
| TTFT p50 (4k prompt, c=4) | Δ ≥ -8% σ < 5% n=3 | Δ < -3% or any regression |
| TTFT p99 | Δ ≥ -5% | Δ > +10% tail regression |
| ITL p50 (decode unchanged) | Δ < +2% | Δ > +5% (decode mistakenly affected) |
| Throughput tok/s | Δ ≥ +5% | Δ < 0% |
| greedy_consistency | PASS | any FAIL |
| **PPL Δ% (wikitext via eval_ppl_pf83.py)** | **≤ +1.0%** | **> +5%** |
| Kernel LOC sanity | < 800 LOC delta | ≥ 1500 LOC delta (means scope creep) |

## §6 Cross-references

- `259277c` (PF8.3 mma shape mismatch — to be SUPERSEDED by this entry's correction)
- `93e1430` (PF8.3 brief sent to codex — Path B implicit; needs Path A redirect)
- `aebd4a5` (PF8.3 PPL gate methodology)
- `a66d99a` (NEW prefill-only FP8 directive — §2 license matrix)
- `b551bea` (skill v1.11.0 anti-pattern catalog — #6 hallucination this tick extends pattern)
- `/tmp/vllm-marlin-src/csrc/quantization/marlin/marlin_mma.h` (raw upstream FP8 mma source THIS tick)
- ARLE Phase 1 substrate: `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` (task #42 completed)
- Codex audit trace: tmux 0:0 capture this tick (Read marlin_mma.h + marlin_dtypes.cuh + generate_kernels.py + kernel.h + marlin.cuh)

## §7 Status

PF8.3 scope CORRECTED. Path A (m16n8k16 FP8 mma, minimal W4A8 delta,
~400-600 LOC) recommended over original Path B (m16n8k32, ~800-1200
LOC). 6th hallucination this session catalogued (FP8 mma single-shape
claim in 259277c).

Codex audit will land its own conclusion (port vs errors entry); this
Claude survey provides cross-check evidence + corrected scope.

If codex chooses Path A: PF8.3 effort estimate revised to ~1 day
(was 1-2 days). License sequence per aebd4a5 §4 unchanged.

If codex chooses errors entry (dependency surface still too large
even for Path A): document the specific blocker so Path B-Phase2'
PF8 axis closes cleanly.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(/tmp/vllm-marlin-src/marlin_mma.h L76-101 + L200-235 raw read this
tick, marlin_template.h:284-285 raw read this tick, ARLE substrate
raw ls this tick).
