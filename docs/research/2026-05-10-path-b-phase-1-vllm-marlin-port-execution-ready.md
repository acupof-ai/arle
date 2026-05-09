---
title: Path B Phase 1 (vLLM-current Marlin port) — execution-ready P0 survey
date: 2026-05-10
type: research
status: execution-ready-pending-user-confirm
---

# Path B Phase 1 (vLLM-current Marlin port) — execution-ready P0 survey

> Prepared while codex builds for #36 PrefixAware bench.
> Machete-vs-Marlin user decision still pending (1829c4e PushNotification
> dispatched, no reply yet). This entry pre-locks Phase 1 of Path B
> (vLLM-current Marlin port) so it's execution-ready the moment user
> confirms — saves a tick of survey delay.

## Fresh upstream verification (2026-05-10)

`gh api repos/vllm-project/vllm/contents/csrc/quantization/marlin`:

```
.gitignore
awq_marlin_repack.cu
dequant.h                       ← 609 LOC, NO CHURN since 2026-05-09 survey
generate_kernels.py
gptq_marlin_repack.cu
kernel.h
marlin.cu                       (main dispatcher, Apache 2.0)
marlin.cuh                      (default_threads=256, pipe_stages=4)
marlin_dtypes.cuh               (scalar_type abstraction)
marlin_int4_fp8_preprocess.cu   ← W4+FP8 sm_89 native FP8 path!
marlin_mma.h                    (mma instruction wrappers)
marlin_template.h               ← 81,605 bytes ≈ 2000-3000 LOC, multi-shape kernels
```

12 files total. License Apache 2.0 (compatible with cherry-pick).
Stable target: dequant.h matches prior survey count exactly.

## ARLE current Marlin substrate

`crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (844 LOC, single file
from PR #31 cherry-pick `a019a0e`):

Key existing inline functions:
- `__device__ inline FragB dequant(int q)` at line 131 (~23 LOC)
- `__device__ inline int lop3(int a, int b, int c)` at line 119
- `__device__ inline void mma(...)` at line 93
- `__device__ inline void cp_async4_*` at lines 54/69/82
- `__device__ inline void scale(FragB&, FragS&, int)` at line 154
- `__global__ void Marlin(...)` at line 197
- `int marlin_cuda(...)` at line 731 (host dispatcher)

The 23-LOC inline `dequant()` is the single point being replaced by
the 609-LOC `dequant.h` module.

## Phase 1 scope (concrete execution plan)

### Substep 1.1 — Extract dequant to standalone header (~50 LOC delta)

Create `crates/cuda-kernels/csrc/gemm/marlin_dequant.h` mirroring vLLM's
`dequant.h` structure but ported to ARLE's namespace + symbol conventions.

Replace inline `dequant()` at marlin_kernel.cu:131 with:

```cpp
#include "marlin_dequant.h"
// ... use marlin::dequant_4bit_into<scalar_type>(...)
```

vLLM's `dequant.h` covers:
- INT4 → FP16/BF16 (with/without zero-point/float-zero-point)
- INT8 → FP16/BF16
- FP4 → FP16/BF16 (decode-only, sm_89 has no native FP4)
- FP8 → FP16/BF16
- BF16 conversion subtleties (`__hsub2` fast path)

ARLE's current path uses INT4 → FP16 only (per `marlin_kernel.cu:131`
inline scope). Phase 1 ports the full dequant.h verbatim — even unused
paths come along to keep the header drop-in for Phase 2 multi-shape.

LOC delta:
- New file: ~609 LOC (verbatim port of dequant.h)
- marlin_kernel.cu: -23 LOC (inline dequant removed) + 1 LOC include
- Net: +587 LOC (one new file, one small edit)

### Substep 1.2 — Add atomic_add reduce path (~100 LOC delta)

Current ARLE Marlin allocates `max_par × 64 × n` FP32 reduce buffer
per call (`marlin_cuda(...)` allocates `alloc_zeros(...)`).
vLLM-current version supports an `use_atomic_add` template parameter
that uses atomic FP16/BF16 add directly into the output, saving the
reduce buffer allocation entirely.

Phase 1.2 adds:
- New `template<bool USE_ATOMIC_ADD>` parameter on the Marlin kernel
- Atomic-add path inside the global reduce step
- Host-side flag in `marlin_cuda(...)` to opt in (default false for
  Phase 1 to preserve numerical baseline; opt-in via env var
  `INFER_MARLIN_ATOMIC_REDUCE=1` for A/B testing)

LOC delta: ~100 LOC (kernel template branch + host wiring + env var read).

### Substep 1.3 — A/B bench + greedy gate

```bash
# Baseline (current, allocate reduce buffer)
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh path-b-p1-baseline \
    --concurrencies 4 --max-seconds 120 --warmup 10 \
    --data 'prompt_tokens=4096,...,output_tokens=256,...'

# Treatment A — new dequant.h, no atomic
INFER_MARLIN_ATOMIC_REDUCE=0 ... bench_guidellm.sh path-b-p1-newdequant

# Treatment B — new dequant.h + atomic
INFER_MARLIN_ATOMIC_REDUCE=1 ... bench_guidellm.sh path-b-p1-newdequant-atomic
```

Plus `cargo test --release --features cuda --test greedy_consistency`
to ensure new dequant.h preserves numerical output.

License gates per kernel-optimization skill v1.9.0 Phase 8:
- ITL Δ ≥ -3% with σ < 5% n=3 → license dequant.h replacement
- ITL Δ ≥ -2% additional with σ < 5% n=3 → license atomic-add opt-in
- TTFT Δ regression > +2% → KILL specific change, keep baseline behavior
- greedy_consistency PASS required for both treatments

Conservative gain estimate (per 2026-05-09 survey):
- dequant.h alone: ITL -3-8%, TTFT -1-3%
- + atomic_add: TTFT additional -2-5% (saves alloc_zeros)
- Combined: ITL -3-8%, TTFT -3-8%

If user's "-20-40% ITL" target is the goal: Phase 1 won't reach
single-handedly — Phase 2 (multi-shape specialization from
marlin_template.h) is required for the larger window. Phase 1 is
the minimum-risk first proof.

## Substep totals

| Substep | LOC delta | Effort | Risk | Expected gain |
|---------|-----------|--------|------|---------------|
| 1.1 dequant.h port | +587 | 0.5-1 day | low (verbatim + namespace rename) | ITL -3-8% |
| 1.2 atomic_add opt-in | +100 | 0.5 day | low-medium (atomic semantics, opt-in default) | TTFT -2-5% |
| 1.3 bench + gate | 0 | 0.5 day | low | gate decisions |
| **Phase 1 total** | **+687** | **1.5-2 days** | **low** | **ITL -3-8%, TTFT -3-8%** |

If license: open Phase 2 directive (multi-shape specialization,
~2000 LOC, ~2-3 days, Phase 2 directive will be drafted post-Phase 1
license per "license-or-kill at every phase" rule).

## Bonus path identified: marlin_int4_fp8_preprocess.cu

vLLM csrc has `marlin_int4_fp8_preprocess.cu` (NEW since prior 2026-05-09
survey may have noted it). This is a **W4 + FP8 activation** path —
sm_89 has native FP8 mma! For ARLE's existing W4A8 path
(`marlin_w4a8_kernel.cu` 987 LOC), this could be a higher-ROI Phase 2'
than multi-shape specialization. Worth fetching content + comparing
against ARLE W4A8 in a follow-up survey before kicking off Phase 2.

## License/kill decision tree

1. User confirms Path B → execute Phase 1 substeps 1.1 → 1.2 → 1.3
2. User picks Path A (Machete sm_89 backport) → reject, request
   sticking with Path B per 1829c4e SOLID evidence
3. User picks Path C (different repo) → fetch + survey, may reuse
   Phase 1 substep template
4. User no reply by next 2 ticks → default to Path B per 1829c4e
   PushNotification commitment

## Briefing pre-draft for codex (when user confirms)

`/tmp/codex-brief-path-b-p1.txt` (DRAFT — not sent yet):

> Pickup: Path B Phase 1.1 — port vLLM-current marlin/dequant.h to
> ARLE marlin_kernel.cu.
>
> Source: gh api repos/vllm-project/vllm/contents/csrc/quantization/marlin/dequant.h
> (609 LOC, Apache 2.0, no churn since 2026-05-09).
>
> Target file: crates/cuda-kernels/csrc/gemm/marlin_dequant.h (NEW)
>
> Replace marlin_kernel.cu:131 inline dequant() with include + call.
>
> Validation:
> - cargo build --release --features cuda must succeed
> - cargo test --release --features cuda --test greedy_consistency must PASS
> - cargo test --release --features cuda --test e2e for prefill+decode parity
>
> A/B bench after build clears: scripts/bench_guidellm.sh path-b-p1-newdequant
> --concurrencies 4 --max-seconds 120 --warmup 10
>
> Then write wins or errors entry per kernel-optimization skill v1.9.0
> license/kill matrix.
>
> Wall time estimate: 0.5-1 day. Push when done.

## Cross-references

- Machete blocker (Path A KILL): `docs/research/2026-05-10-machete-sm89-port-blocker-confirmed-upstream-still-hopper-only.md`
- Prior 2026-05-09 industry survey: `docs/research/2026-05-09-w4a8-industry-kernel-survey.md`
- ARLE Marlin substrate: `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` (844 LOC),
  `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC)
- Existing dispatch: `infer/src/ops/linear.rs` (Marlin/Hybrid/W4A16Gemv enum)
- vLLM upstream Marlin: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin
- vLLM Apache 2.0 license: https://github.com/vllm-project/vllm/blob/main/LICENSE
- Skill v1.9.0 anti-patterns + Phase 8 license matrix:
  `.claude/skills/kernel-optimization/SKILL.md`

## 状态

Path B Phase 1 substeps 1.1 + 1.2 + 1.3 are execution-ready. LOC budget
tight (~687 LOC), gates clear, bonus W4+FP8 path identified for Phase 2'.
Awaiting user reply on Machete-vs-Marlin decision (1829c4e). Default
Path B confirmation by next 2 ticks → brief codex from /tmp/codex-brief-path-b-p1.txt.
