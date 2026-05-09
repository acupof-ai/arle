---
title: Phase 1 dequant.h port — scope note + dependency map (pre-staged for codex)
date: 2026-05-10
type: research
status: codex-Working-on-phase1
---

# Phase 1 dequant.h port — scope note + dependency map (pre-staged for codex)

> Codex briefed (this tick) on Path B Phase 1 dequant.h port and is
> reading source. While codex investigates, Claude pre-stages the
> upstream files locally + maps the dependency cascade so the port
> path is deterministic.

## Pre-staged upstream files (raw `gh api` this tick)

```
/tmp/upstream-marlin/dequant.h              609 LOC  (verified raw line count)
/tmp/upstream-marlin/marlin_dtypes.cuh      149 LOC  (verified raw line count)
```

Both fetched verbatim from `vllm-project/vllm:main` via gh api this
tick. Apache 2.0 license per `marlin.cu` upstream attribution
(prior d5a6679 verification).

## Dependency cascade

Verbatim port of dequant.h pulls in a small cascade:

```
dequant.h (609 LOC)
├── marlin_dtypes.cuh (149 LOC)
│   ├── core/scalar_type.hpp  (vllm general ScalarType)
│   └── marlin.cuh           (Vec<T,N> template + arch constants)
└── marlin.cuh               (same as above)
```

dequant.h template signature uses `vllm::ScalarTypeId` with concrete
specializations for `kU4B8`, `kU4`, `kU8B128`, `kU8` (4 quant types
× 2 has_zp variants × 2 dtypes (half2/bf16) = 16 specializations).

## Two viable porting strategies for codex

### Strategy A — Verbatim cascade port (~900 LOC, more upstream-fidelity)

Port dequant.h + marlin_dtypes.cuh + minimal subset of scalar_type.hpp.
Keeps signature shape identical to upstream so future updates can be
cherry-picked easily. Phase 2 multi-shape spec (later directive) drops
in cleanly.

LOC delta:
- `crates/cuda-kernels/csrc/gemm/marlin_dequant.h` (verbatim ~609 LOC)
- `crates/cuda-kernels/csrc/gemm/marlin_dtypes.cuh` (verbatim ~149 LOC)
- `crates/cuda-kernels/csrc/gemm/marlin_scalar_type.h` (minimal subset
  ~80 LOC — only U4B8/U4/U8B128/U8 enum values + ScalarTypeId typedef)
- Total ~840 LOC

### Strategy B — Stripped port (~300-400 LOC, ARLE-side simplification)

Port dequant.h FUNCTIONS only, replace vllm template machinery with
arle-side simple int constants. Phase 2 multi-shape would need to
re-do the template machinery later.

LOC delta:
- `crates/cuda-kernels/csrc/gemm/marlin_dequant.h` (~300 LOC, only
  the actual dequant logic for the cases ARLE uses today: U4B8 + bf16)
- Adapter changes in `marlin_kernel.cu` to call new dequant funcs

## Recommendation

**Strategy A (verbatim cascade)** for the following reasons:

1. e59beb5 Phase 1 brief explicitly said "verbatim port keeps Phase 2
   multi-shape drop-in friendly". Strategy B undoes that.
2. Marginal LOC cost (840 vs 300-400) is acceptable for a 1-day port.
3. Future upstream cherry-picks become mechanical.
4. The 16 specializations are mostly empty-or-delegate (e.g. kU4 with_zp
   → calls kU4B8 with_zp, just FP16↔BF16 dispatch wraps). The "verbatim
   line count" overstates the unique logic.

If codex prefers Strategy B for simplicity, that's also valid — kill
gate is the same (ITL Δ ≥ -3% with σ < 5%).

## What codex should NOT need to fetch upstream

Pre-staged files cover the immediate scope. Codex doesn't need to
gh api during the port — direct file references at `/tmp/upstream-marlin/`.
Optional if Strategy A: codex may also want `core/scalar_type.hpp`
upstream (in `csrc/`, not `csrc/quantization/marlin/`); fetch on
demand if Strategy A trips at compile time.

## ARLE substrate to integrate with (verified raw grep this tick)

```bash
$ grep -nE "^[a-zA-Z_]+ \w+\(|__device__|__global__|__forceinline__" \
    crates/cuda-kernels/csrc/gemm/marlin_kernel.cu | head -20

131: __device__ inline FragB dequant(int q) {     ← REPLACE THIS
197: __global__ void Marlin(...)                  ← unchanged
731: int marlin_cuda(...)                          ← may need atomic_add flag (Substep 1.2)
```

ARLE's existing `dequant(int q)` is a 23-LOC inline. Strategy A
replacement: include marlin_dequant.h, call
`marlin::dequant<scalar_t2, ScalarTypeId::kU4B8, has_zp>(q, frag_b)`
or similar templated form per ARLE's chosen template binding.

## License/kill gates (unchanged from e59beb5)

- ITL Δ ≥ -3% with σ < 5% n=3 → license dequant.h replacement
- Greedy_consistency PASS required
- TTFT regression > +2% → KILL specific change

## Cross-references

- Phase 1 survey: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- Phase 0 KILL: `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md` (61c9666)
- Codex Phase 0 errors entry: `docs/experience/errors/2026-05-10-...` (67f18b9)
- Pre-staged: `/tmp/upstream-marlin/dequant.h`, `/tmp/upstream-marlin/marlin_dtypes.cuh`
- Brief: `/tmp/codex-brief-phase1-dequant.txt` (sent prior tick)
- ARLE Marlin substrate: `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (844 LOC)

## 状态

Codex Working (40s at tick capture) on Phase 1 source reading. Two
strategies surfaced (verbatim A vs stripped B). Strategy A
recommended per e59beb5 brief intent; B acceptable if codex prefers
simplicity. Pre-staged files reduce porting friction. License gates
unchanged.
