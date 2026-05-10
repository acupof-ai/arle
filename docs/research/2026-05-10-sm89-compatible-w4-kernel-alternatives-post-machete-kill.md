---
title: 2026-05-10 sm_89-compatible W4 GEMM alternatives — post-Machete-KILL pivot survey
date: 2026-05-10
type: research
status: open (concrete kernel-axis pivot recommendations)
related_docs: [`fc33cfb` Machete KILL errors entry, `9735b47` REFUTATION, `bccf1bd` consistency audit, `f0c7561` Phase 1.B Medusa brief]
---

# sm_89-compatible W4 GEMM alternatives in vLLM — pivot recommendations after Machete KILL

> **2026-05-10 later update**: references to the Medusa brief as ready
> for Option A are historical for Qwen3/Qwen3.6. Active Qwen3.5 Medusa
> is blocked on recurrent-state rollback. The sm_89 W4 alternatives
> remain independent.

> **Why now**: `fc33cfb` killed Machete port axis (Hopper-only WGMMA/TMA
> dependency). User-stated main axis "Machete W4 kernel 移植 from
> vLLM ... 预估 -20-40% ITL" needs re-grounded sm_89-compatible kernel
> path. This survey identifies what vLLM has that DOES work on sm_89.

## §1 vLLM `csrc/quantization/` directory survey

| Subdir | Size | sm_89-compatible? | ARLE-relevant? |
|---|---|---|---|
| `awq/` | small | sm_75+ (Turing+) likely | YES (W4A16 alt) |
| `fused_kernels/` | mixed | check per-file | varies |
| `gguf/` | mixed | sm_60+ | no (CPU-side) |
| `gptq/` | mixed | sm_75+ likely | YES (existing path) |
| `gptq_allspark/` | 60 KB | unknown — survey | possibly (W8A16) |
| `hadamard/` | small | sm_70+ | maybe (rotation) |
| **`machete/`** | **150+ KB** | **NO — sm_90+ ONLY** (KILLED `fc33cfb`) | **NO** |
| `marlin/` | 80+ KB | **YES — sm_75+** (verified L40) | **YES (primary path)** |
| `w8a8/` | mixed | sm_75+ | indirect (FP8 path) |

## §2 vLLM Marlin (the actual sm_89-compatible W4 kernel)

`csrc/quantization/marlin/marlin.cu` (32.8 KB) — the IST-DASLab Marlin
kernel + Neural Magic modifications. Arch check (L40):

```c
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
  // ... fallback or error ...
#endif
```

Requires sm_75+ (Turing+). sm_89 = 890, well above. **COMPATIBLE.**

### §2.1 Marlin file structure (12 files)

| File | Role |
|---|---|
| `marlin.cu` | Main kernel implementation |
| `marlin.cuh` | Public header |
| `marlin_template.h` | Templated GEMM body |
| `marlin_mma.h` | mma instruction abstraction |
| `marlin_dtypes.cuh` | dtype trait definitions |
| `dequant.h` | unpack-dequant helpers |
| `kernel.h` | dispatch entry points |
| `marlin_int4_fp8_preprocess.cu` | **W4-FP8 preprocess (PF8 analog)** |
| `gptq_marlin_repack.cu` | weight repack utility |
| `awq_marlin_repack.cu` | AWQ weight repack utility |
| `generate_kernels.py` | code generation |
| `.gitignore` | (codegen artifacts) |

## §3 ARLE's current Marlin status (per CLAUDE.md `crates/cuda-kernels/`)

ARLE already has Marlin path for both W4A16 and W4A8 (per `b5889b3`
+ `eab166d`). Per `1ccb41f` audit + `bccf1bd` REFUTATION:
- W4A16-marlin-zpfix: working, perf ceiling at conc=1 prompt=512
- W4A8-marlin (qzeros-fixed): working, accuracy validated by codex `8d1caad`
- PF8.3 (W4A8 + FP8 activation marlin variant): per `0be278f` PF8.5 KILL,
  has substrate problem on sm_89 16GB at runtime

**ARLE Marlin is at-par with vLLM Marlin's main W4 paths.** The only gap
is `marlin_int4_fp8_preprocess.cu` (PF8 preprocessing) which ARLE
attempted in PF8.3 substrate but encountered runtime failures.

## §4 Pivot recommendations (sm_89-compatible Machete-class alternatives)

### §4.1 PRIMARY: A + B combined (per `bccf1bd` strategic matrix)

Already validated, ready for pickup, sm_89-compatible:
- **B (Hybrid Option B)**: 1.5 days, -14% E2E latency (REFUTATION-validated)
- **A (Medusa)**: 2.5-3 days, 2.25× tok/s (per `f0c7561` substrate brief)
- **Combined**: ~2.61× tok/s + -14% latency, 4-5 days

This is the dominant path. No new kernel work required for B; A is a
new model component (no kernel needed per `1ccb41f` vLLM prior-art).

### §4.2 SECONDARY: vLLM upstream Marlin diff-port

If A+B falls short of user's "Machete-class" expectations, evaluate:
- ARLE Marlin vs vLLM v0.x Marlin code-diff
- Identify any optimizations ARLE has missed since last sync
- Likely small wins (2-5%); kernel is mature
- Effort: 1-2 days code-read + bench

### §4.3 TERTIARY: PF8.3 substrate redesign (per Task #47 BLOCKED)

PF8.3 (W4-FP8 marlin) is the closest ARLE has to a "Machete-class"
W4-with-low-precision-activation kernel. Current PF8.3 substrate is
broken (per `0be278f` PF8.5 KILL).

vLLM has `marlin_int4_fp8_preprocess.cu` reference — could be ported
to fix the per-call workspace alloc issue identified in PF8.5.
- Effort: 2-4 days substrate fix + bench
- Expected gain: 8-16% TTFT (per Task #44 PF8 spec)

### §4.4 QUATERNARY: Cutlass FP8 direct mma sm_89 (per skill anti-pattern #7)

ARLE's existing FP8 path uses cuBLASLt heuristic. Per
`kernel-optimization` skill anti-pattern #7: cuBLASLt may be
suboptimal vs cutlass direct mma. Earlier `/tmp/fp8_smoke.cu` showed
~1.88× cuBLASLt utilization; cutlass direct may hit higher.
- Effort: 3-5 days kernel work
- Expected: open question, requires Phase 0 spike

### §4.5 NOT RECOMMENDED: Wait for sm_100

NVFP4 native sm_100 is not on near-term ARLE hardware roadmap.

## §5 Strategic recommendation order

| Priority | Path | Wall-clock | Risk | Expected |
|---|---|---:|---|---|
| P1 | A + B combined | 4-5 days | LOW | 2.61× tok/s + -14% latency |
| P2 | vLLM upstream Marlin diff-port | 1-2 days | LOW | 2-5% improvement |
| P3 | PF8.3 substrate redesign | 2-4 days | MEDIUM | 8-16% TTFT (gates on substrate fix) |
| P4 | Cutlass FP8 direct mma sm_89 | 3-5 days | MEDIUM | open |
| P5 | Wait sm_100 | months | LOW | NVFP4 native |
| KILLED | Machete port (sm_90+) | impossible | KILLED | 0% on sm_89 |

## §6 Cross-references

- `fc33cfb` Machete KILL errors entry (KILLED main axis premise)
- `9735b47` REFUTATION wins entry (-14% Hybrid measured)
- `bccf1bd` consistency audit (Hybrid plan was correct)
- `f0c7561` Phase 1.B Medusa brief (substrate ready for A)
- `e021026` Alpaca data ready (training-prep done)
- `0be278f` PF8.5 KILL errors (PF8.3 substrate broken)
- vLLM `csrc/quantization/marlin/marlin.cu` — sm_75+ compatible (CONFIRMED L40)
- vLLM `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu` — W4-FP8 reference
- vLLM `csrc/quantization/machete/Readme.md` — "optimized for Hopper architectures" (KILLED)
- ARLE `crates/cuda-kernels/csrc/gemm/marlin_*.cu` (existing analogs)
- `kernel-optimization` skill Phase 2 hardware constraint sheet
