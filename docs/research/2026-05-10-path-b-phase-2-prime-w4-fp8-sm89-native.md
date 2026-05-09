---
title: Path B Phase 2' — W4+FP8 sm_89 native FP8 mma path (the real -20-40% ITL source)
date: 2026-05-10
type: research
status: phase-2-prime-survey-pending-phase-1-license
---

# Path B Phase 2' — W4+FP8 sm_89 native FP8 mma path (the real -20-40% ITL source)

> Survey done while codex's #36 PrefixAware bench arm A runs (GPU 100%
> at 15.5GB; 5m 34s into 120s window). Investigates the bonus path
> flagged in `e59beb5` Path B Phase 1 survey: vLLM's
> `marlin_int4_fp8_preprocess.cu`. Finding: it's just a 100-LOC weight
> preprocess, but it points at the **real ROI mechanism** for the
> user's "-20-40% ITL vs current Marlin" target — switching ARLE's
> W4 path from INT8 activations to native sm_89 FP8 mma.

## §0 Initial assumption corrected

Initial assumption in `e59beb5`: `marlin_int4_fp8_preprocess.cu` is a
"W4+FP8 GEMM kernel". **Wrong.**

Actual content (3,710 bytes, ~100 LOC):
- `__global__ void marlin_int4_fp8_preprocess_kernel_without_zp(...)`
  for GPTQ format (subtraction-merge: bake zero-point=8 into weight)
- `__global__ void marlin_int4_fp8_preprocess_kernel_awq(...)` for AWQ
  format (bake AWQ zero-point into weight, removes runtime subtract)
- `marlin_int4_fp8_preprocess(...)` host wrapper

This is an **offline weight transformation** that lets the W4+FP8 GEMM
skip the per-element zero-point subtract during runtime. The actual W4+FP8
GEMM lives inside `marlin_template.h` (~2000-3000 LOC) instantiated with
FP8 activation type.

## §1 Real finding — ARLE W4A8 is W4+INT8, NOT W4+FP8

Direct grep evidence:

```bash
grep -nE "fp8|FP8|__nv_fp8|float8|8e4m3|e4m3" \
  crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu \
  crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu
# (no output — zero FP8 references)
```

ARLE's W4A8 stack:
- `marlin_w4a8_kernel.cu` (987 LOC) — W4 weights + **INT8 activations**, uses sm_89 INT8 mma
- `w4a8_activation_quant.cu` (59 LOC) — `quantize_bf16_rows_to_int8_kernel` (line 6) + `quantize_bf16_rows_to_int8_cuda` (line 45) — runtime BF16→INT8 quant
- `marlin_repack.cu` (151 LOC) — GPTQ marlin repack only, **no AWQ repack, no FP8 preprocess**

vLLM's W4+FP8 stack (current main):
- `marlin_template.h` (~2000-3000 LOC) — multi-shape templated GEMM, instantiated with FP8 type for native sm_89 FP8 mma
- `marlin_int4_fp8_preprocess.cu` (100 LOC) — offline weight prep
- BF16→FP8 activation quant — done outside the marlin csrc, typically by upstream tensor-quant ops

**The actual ITL gap mechanism**: ARLE uses sm_89 INT8 mma path; vLLM's
production W4 path uses sm_89 NATIVE FP8 mma path. Different hardware
units with different throughput.

## §2 Hardware constraint sheet (sm_89 4070 Ti SUPER)

Per kernel-optimization skill v1.9.0 §Phase 2:

| Op | TFLOPS peak (sm_89) | ARLE current path | vLLM W4+FP8 path |
|---|---:|---|---|
| BF16 mma | 88.5 | (not used in W4) | (not used in W4) |
| **INT8 mma** | ~440 (4× BF16) | **ARLE W4A8 (current)** | — |
| **FP8 mma** | **706** (8× BF16) | — | **vLLM W4+FP8 (target)** |
| FP4 mma | (none, sm_100+ only) | n/a | n/a |

Theoretical FP8 vs INT8 ratio: 706 / 440 ≈ **1.6×** for the GEMM phase.
If current ARLE W4A8 GEMM is the binding constraint of the W4 path,
switching to FP8 could plausibly hit 1.5-1.7× ITL = **−33-41% ITL**.
That maps cleanly to the user's "-20-40% ITL vs current Marlin" target.

But: per M_quant Phase 0 v2 (`docs/plans/M_quant-fp8-w4-magnitude-path.md`),
sm_89 cuBLASLt FP8 hit only ~24% of theoretical peak via heuristic
dispatch (skill anti-pattern #7: "cuBLASLt heuristic ≠ cutlass direct
mma"). So real-world delta may be smaller. Phase 0 spike must verify
direct cutlass FP8 path before any architectural commitment.

## §3 Phase 2' substep breakdown (NEW — supersedes Phase 2 multi-shape priority)

### Sub-condition: Phase 2' assumes Phase 1 (dequant.h port + atomic_add) licensed

Phase 1 is independent and lower-risk. Phase 2' extends ARLE's W4A8
path with a parallel W4+FP8 path; both can coexist via env-var or
config-driven dispatch.

### Substep 2'.1 — BF16→FP8 activation quant kernel (~60 LOC)

Mirror `w4a8_activation_quant.cu` (59 LOC current INT8 version) with
FP8 output:

```cpp
// new file: crates/cuda-kernels/csrc/gemm/w4_fp8_activation_quant.cu
__global__ void quantize_bf16_rows_to_fp8_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, ...);
extern "C" cudaError_t quantize_bf16_rows_to_fp8_cuda(...);
```

Use NVIDIA's `__nv_fp8_e4m3` type (sm_89 native, single-conversion
intrinsic). Per-row scaling factor stored in FP32 sidecar tensor for
later dequant.

Risk: low (verbatim mirror of existing INT8 version with type swap).
LOC: ~60.

### Substep 2'.2 — Port marlin_int4_fp8_preprocess.cu (~120 LOC)

Verbatim port of vLLM's 100 LOC into
`crates/cuda-kernels/csrc/gemm/marlin_int4_fp8_preprocess.cu`. Adapt
torch→cudarc FFI (extern "C" wrappers per ARLE convention; ARLE has
no torch dep).

Skip `marlin_int4_fp8_preprocess_kernel_awq` for Phase 2'.2 (we don't
have AWQ checkpoints today; add later if AWQ checkpoint loader lands).
Phase 2'.2 = `marlin_int4_fp8_preprocess_kernel_without_zp` only,
matches our existing GPTQ format.

Risk: low (verbatim algorithm, FFI shim is mechanical).
LOC: ~120 (port + extern "C" wrappers + tests).

### Substep 2'.3 — Port FP8 marlin GEMM (~700-1500 LOC, the real work)

This is the heavy lift. Two viable approaches:

**Approach A — Instantiate marlin_template.h with FP8 type**

Pull `marlin_template.h` (~2000-3000 LOC) wholesale, namespace it,
instantiate the FP8 specialization. Reuses upstream's tested kernel,
but pulls in the multi-shape template machinery whether we want it or
not.

LOC: ~2500 (header + adapter), 2-3 days.

**Approach B — Write a single-template W4+FP8 specialization**

Mirror our current `marlin_w4a8_kernel.cu` (987 LOC) structure but with
FP8 mma intrinsics replacing INT8. Reuses our existing dispatch surface,
single template = simpler verification.

LOC: ~700-1000, 1-2 days.

**Recommendation**: Approach B for Phase 2'.3 first. If it lands and
shows the predicted -33-41% ITL, then Phase 2 (multi-shape spec via
Approach A) becomes the natural follow-up.

### Substep 2'.4 — Linear-dispatch wiring (~50 LOC)

Add `MarlinW4FP8Gemm` variant to `infer/src/ops/linear.rs`
`SelectedW4Path` enum (currently has W4A16Gemv, W4A16BatchGemv,
MarlinW4Gemm, MarlinW4A8Gemm, MarlinW4Hybrid).

Env-var opt-in for Phase 2'.4 first cycle: `INFER_MARLIN_W4_FP8=1`.
Default off to preserve numerical baseline until license A/B clears.

### Substep 2'.5 — A/B bench + greedy gate

```bash
# Baseline (current ARLE W4A8 INT8 path)
scripts/bench_guidellm.sh path-b-p2prime-baseline-w4a8-int8 \
  --concurrencies 4 --max-seconds 120 ...

# Treatment (new W4+FP8 path)
INFER_MARLIN_W4_FP8=1 scripts/bench_guidellm.sh path-b-p2prime-w4-fp8 \
  --concurrencies 4 --max-seconds 120 ...
```

License gates:
- ITL Δ ≥ −20% with σ < 5% n=3 → license W4+FP8 default-on (matches
  user target floor)
- ITL Δ ≥ −33% → strong license, also pursue Approach A multi-shape
  for the upper -41% range
- Any greedy_consistency regression > 5% → KILL, FP8 quant accuracy
  too lossy
- Any TTFT regression > +5% → KILL specific change, investigate
  preprocess overhead

## §4 Phase priority comparison (post-survey reranking)

| Phase | LOC | Effort | Risk | Expected ITL Δ | Bench gate |
|-------|----:|--------|------|----------------|------------|
| Phase 1 (dequant.h port) | ~687 | 1.5-2 days | low | -3-8% | -3% min |
| Phase 2 (multi-shape spec) | ~2000 | 2-3 days | medium | -20-40% **on certain shapes only** | shape-conditional |
| **Phase 2' (W4+FP8 sm_89 native)** | **~900-1700** | **2-3 days** | **medium-high** (FP8 quant accuracy) | **-20-40% global** | **-20% min, -33% strong** |

**New priority order** (replaces e59beb5's Phase 1 → Phase 2 sequence):

1. **Phase 1** first (low risk, durable dequant.h substrate, independent)
2. **Phase 2' next** (matches user target globally, not shape-conditional)
3. **Phase 2** later (multi-shape spec is incremental on top of either above)

Reasoning: the user's "-20-40% ITL" target maps cleanly to Phase 2'
(global, FP8 mma 1.6× theoretical) but only conditionally to Phase 2
(certain N×K shapes). Phase 2' is the more direct path to the headline
goal even though risk is higher (FP8 quant accuracy, upstream cuBLASLt
heuristic trap).

## §5 Pre-conditions for Phase 2' Phase 0 spike

Before committing to Phase 2'.3 (the big port), spike with:

- Direct cutlass FP8 GEMM smoke (verify ARLE can hit > 50% of 706 TFLOPS
  theoretical via cutlass direct mma, not cuBLASLt heuristic)
- BF16→FP8 quant accuracy test on Qwen3-4B weights (PPL Δ vs INT8 quant)
- Existing FP8 spike at `/tmp/fp8_smoke.cu` (per skill anti-pattern #7
  reference) status check — was it ever followed up?

If Phase 0 spike shows < 30% theoretical FP8 utilization OR PPL Δ > 0.5
vs INT8: KILL Phase 2', revert to Phase 2 multi-shape path.

## §6 Cross-references

- e59beb5 Path B Phase 1 survey:
  `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md`
- 1829c4e Machete blocker:
  `docs/research/2026-05-10-machete-sm89-port-blocker-confirmed-upstream-still-hopper-only.md`
- M_quant FP8 path plan:
  `docs/plans/M_quant-fp8-w4-magnitude-path.md`
- ARLE W4A8 substrate:
  - `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC, INT8)
  - `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu` (59 LOC, BF16→INT8)
  - `crates/cuda-kernels/csrc/gemm/marlin_repack.cu` (151 LOC, GPTQ only)
- Linear dispatch enum:
  `infer/src/ops/linear.rs:32-92` (W4A16Gemv / W4A16BatchGemv / MarlinW4Gemm
  / MarlinW4A8Gemm / MarlinW4Hybrid — Phase 2'.4 adds MarlinW4FP8Gemm)
- vLLM upstream Marlin csrc:
  https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin
- Skill v1.9.0 anti-pattern #7 (cuBLASLt heuristic ≠ cutlass direct mma):
  `.claude/skills/kernel-optimization/SKILL.md`
- Hardware constraint sheet:
  same skill v1.9.0 §Phase 2 hardware reference table

## §7 Status

Phase 2' survey LANDED for the post-Phase-1 priority queue. Scope is
~900-1700 LOC over 2-3 days, predicted ITL -20-40% (matches user
target globally, not just shape-conditional). Phase 0 spike (cutlass
direct FP8 + PPL gate) must precede Phase 2'.3 commit.

Default sequencing absent further user input:
1. User confirms Path B → execute Phase 1 (dequant.h + atomic_add)
2. Phase 1 license PASS → spike Phase 2' Phase 0 (cutlass FP8 + PPL)
3. Phase 2' Phase 0 PASS → execute Phase 2'.1-2'.5
4. Phase 2' license PASS → execute Phase 2 multi-shape spec on top
5. Each phase produces wins or errors entry per kernel-optimization
   skill v1.9.0 license/kill matrix

PushNotification not dispatched — Machete decision still pending,
this is a pre-locked branch ready to ship the moment user confirms.
