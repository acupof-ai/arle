---
title: NEW prefill-only FP8 directive — execution-ready draft (P1 per de36538 revised priority)
date: 2026-05-10
type: research
status: directive-ready-pending-codex-pickup
---

# NEW prefill-only FP8 directive — execution-ready draft (P1 per de36538 revised priority)

> Per `09ae5a5` revised priority + `61c9666` Phase 0 P0.A KILL
> architectural synthesis: W4 decode is HBM-bound (FP8 mma is wrong
> lever for ITL), but W4 PREFILL is compute-bound (FP8 mma 1.6×
> theoretical over INT8 helps directly). Codex's P0.A smoke confirmed
> prefill 5.21× speedup at M=2048 N=4096 K=2560.
>
> This directive ports the existing W4+INT8 prefill (marlin_w4a8_kernel.cu)
> to W4+FP8 for the TTFT axis only — orthogonal to the killed
> ITL-axis work. Estimated -8-16% TTFT separate from ITL.

## §0 Why this axis is correct (architectural grounding per skill v1.10.0+ #28)

Per `61c9666` direct evidence:

- **Decode (M=1)**: HBM-bound on weight read. W4 weights already
  4× smaller than BF16; FP8 activation is 0.2% of memory traffic;
  FP8 mma helps compute not bandwidth → **0% decode ITL gain** for
  W4+FP8 path.
- **Prefill (M=2048)**: COMPUTE-bound. P0.A smoke achieved 22.6%
  of theoretical 706 TFLOPS = below 50% absolute gate but 5.21×
  speedup over BF16 baseline. INT8 mma peak = 440 TFLOPS; FP8 peak
  = 706 TFLOPS; ratio 1.6× theoretical. If current ARLE W4+INT8
  prefill achieves similar utilization, switching to FP8 can deliver
  **roughly 1.4-1.6× prefill GEMM speedup** = **-8-16% chunked-prefill TTFT
  contribution** depending on workload prefill-dominance.

The smoke evidence (codex 67f18b9) confirms FP8 mma works at sm_89
on the right shape. The remaining work is wiring + activation quant.

## §1 Substep breakdown (~700 LOC over 2 days codex)

### Substep PF8.1 — BF16→FP8 activation quant kernel (~60 LOC)

Mirror existing `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu`
(59 LOC, BF16→INT8) with FP8 e4m3 output:

```cpp
// new file: crates/cuda-kernels/csrc/gemm/w4_fp8_activation_quant.cu

__global__ void quantize_bf16_rows_to_fp8_e4m3_kernel(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* row_scales,         // per-row absmax / 448 (e4m3 max)
    int rows, int cols);

extern "C" cudaError_t quantize_bf16_rows_to_fp8_e4m3_cuda(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* row_scales,
    int rows, int cols,
    cudaStream_t stream);
```

Use `__nv_fp8_e4m3` type (sm_89 native conversion intrinsic). Per-row
scale stored in FP32 sidecar tensor for downstream dequant.

Risk: low (verbatim mirror of INT8 version with type swap + scale logic).

### Substep PF8.2 — Port marlin_int4_fp8_preprocess.cu (~120 LOC)

Verbatim port from vLLM upstream (`csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu`,
100 LOC, Apache 2.0) into ARLE:

```bash
$ gh api repos/vllm-project/vllm/contents/csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu \
    | base64 -d > /tmp/upstream-marlin/marlin_int4_fp8_preprocess.cu
$ wc -l /tmp/upstream-marlin/marlin_int4_fp8_preprocess.cu
~100 LOC
```

Function: subtraction-merging zero-point into INT4 weight (offline
preprocess, per `3e83741` survey). Adapt torch→cudarc FFI (extern "C"
wrappers per ARLE convention; ARLE has no torch dep).

Skip the `marlin_int4_fp8_preprocess_kernel_awq` variant for PF8.2
(no AWQ checkpoint loader yet); keep just `_without_zp` for GPTQ
which matches our existing format.

### Substep PF8.3 — Port FP8 marlin GEMM kernel (~400-600 LOC)

Two viable approaches per `3e83741` Phase 2' analysis:

**Approach A — Single-template specialization (RECOMMENDED)**
Mirror `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC,
INT8 mma) structure with FP8 mma replacing INT8. Reuses existing
dispatch surface, single template = simpler to verify against the
existing W4A8 path.

LOC: ~700-1000, 1-2 days. Focus on prefill shapes only (M ≥ 32);
decode shape (M=1) keeps the existing W4+INT8 path since FP8 doesn't
help decode ITL.

**Approach B — Pull cutlass marlin_template.h with FP8 instantiation**
~2000 LOC additional, also pulls multi-shape spec for free. Defer
to Phase 2 if Approach A licenses but multi-shape gain is needed.

**Recommendation**: Approach A first. Codex already has the cutlass
sm_89 FP8 template knowledge from P0.A spike (d5a6679 unstick:
`GemmUniversalWithAbsMax` + `arch::Sm89` + `LinearCombinationGenericWithScalingAndAbsMax`).

### Substep PF8.4 — Linear-dispatch wiring (~50 LOC)

Add `MarlinW4FP8Prefill` variant to `infer/src/ops/linear.rs`
`SelectedW4Path` enum (currently has `MarlinW4Gemm`, `MarlinW4A8Gemm`,
`MarlinW4Hybrid`, etc).

Dispatch logic:
- Decode (batch=1 or M ≤ small_threshold): keep `MarlinW4A8Gemm`
  (existing W4+INT8 path, decode ITL is HBM-bound)
- Prefill (M ≥ prefill_threshold): use `MarlinW4FP8Prefill` (new)
- Hybrid mode: env-var opt-in `INFER_MARLIN_W4_FP8_PREFILL=1` first
  cycle to preserve numerical baseline until license A/B clears

### Substep PF8.5 — A/B bench + greedy gate

```bash
# Baseline (current ARLE W4+INT8 prefill)
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh prefill-fp8-baseline-w4int8 \
    --model Qwen3-4B-W4A8-marlin --concurrencies 4 --max-seconds 120 \
    --warmup 10 \
    --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=128,output_tokens_min=128,output_tokens_max=128'

# Treatment (new W4+FP8 prefill, decode unchanged)
INFER_MARLIN_W4_FP8_PREFILL=1 \
  scripts/bench_guidellm.sh prefill-fp8-treatment ... [same params]
```

Server-restart between arms per `4b30c15` protocol (use
`/readyz` endpoint per `c3bb82b`, `setsid bash -c 'exec ...'`
pattern).

## §2 License/kill gates (per kernel-optimization skill v1.10.0 Phase 8)

| Metric | License threshold | Kill threshold |
|--------|-------------------|----------------|
| TTFT p50 | Δ ≥ -8% with σ < 5% n=3 | Δ < -3% or any regression |
| TTFT p99 | Δ ≥ -5% | Δ > +10% (tail regression) |
| ITL p50 | regression < +2% (decode unchanged) | Δ > +5% (mistakenly affected decode) |
| Throughput tok/s | Δ ≥ +5% | Δ < 0% |
| greedy_consistency | PASS required | any FAIL |

Conservative gain estimate (per `09ae5a5` revised priority):
- TTFT -8 to -16% prefill-dominance dependent
- Larger gain on prompt-heavy workloads (4k+ prompt tokens)
- No ITL change (decode path unchanged)

## §3 Phase 0 spike substrate (already done)

P0.A cutlass DIRECT FP8 GEMM smoke (codex `67f18b9`) verified:
- Cutlass headers via TileLang (per `crates/cuda-kernels/build.rs:512-547`)
- ARLE existing 431-LOC `decode_attention_varlen_fp8.cu` proves FP8 mma
  works in ARLE codebase
- Cutlass example 58 (`58_ada_fp8_gemm`) is the reference for
  `GemmUniversalWithAbsMax` + `arch::Sm89` + `LinearCombinationGenericWithScalingAndAbsMax`
- Smoke achieved 5.21× speedup at M=2048 N=4096 K=2560 (159.8 TFLOPS
  = 22.6% theoretical)

Phase 0 evidence preserved at `/tmp/cutlass_fp8_smoke.cu` (don't
delete — reference for PF8.3 Approach A implementation).

## §4 Boundaries (what this directive is NOT)

- NOT decode-axis improvement (decode stays W4+INT8; FP8 KILLED for ITL per `61c9666`)
- NOT multi-shape spec (Phase 2 work, defer until PF8 licenses)
- NOT a Machete port (Machete is Hopper-only per `e65a096` 5-pt evidence)
- NOT new quant calibration (uses existing GPTQ-W4 weights)
- NOT a P0 ITL win (that's #28 Medusa per de36538 priority — now unblocked via df37a68)

## §5 Pickup brief for codex (when this lands)

```
Codex pickup: NEW prefill-only FP8 directive (~700 LOC, 2 days)

Pull origin/main first:
  git pull --ff-only origin main

Where things stand:
- Phase 1 Substep 1.1 LICENSED (codex f86d0fd + Claude 4f1b036 σ-tight n=2)
- #34 RESOLVED via df37a68 (`arle model download <id>` CLI surface live)
- Phase 0 P0.A KILLED for ITL but prefill 5.21× signal retained
- This directive = TTFT-only axis using prefill 5.21× evidence

Substeps (per docs/research/2026-05-10-prefill-only-fp8-directive-draft.md
§1):
  PF8.1: BF16→FP8 e4m3 act quant kernel (~60 LOC)
  PF8.2: Port marlin_int4_fp8_preprocess.cu without_zp (~120 LOC)
  PF8.3: FP8 marlin GEMM kernel — Approach A single-template (~700 LOC)
  PF8.4: Linear dispatch MarlinW4FP8Prefill enum + INFER_MARLIN_W4_FP8_PREFILL env var (~50 LOC)
  PF8.5: A/B bench + greedy gate

Use cutlass sm_89 template from your P0.A spike (d5a6679 unstick):
  GemmUniversalWithAbsMax + arch::Sm89 + LinearCombinationGenericWithScalingAndAbsMax

License gates (per docs §2):
  TTFT p50 Δ ≥ -8% with σ < 5% n=3 → license
  Δ < -3% → kill
  greedy_consistency PASS required

What you should NOT do:
- Touch decode path (W4+INT8 for decode unchanged; FP8 doesn't help
  decode ITL)
- Pull marlin_template.h multi-shape spec (Phase 2 work)
- Delete /tmp/cutlass_fp8_smoke.cu (reference for PF8.3)

What you SHOULD do:
- Reuse cooperative discipline from this session: status-BEFORE-commit
  per ca09db0 + de36538 retrospective
- Cite raw bench output verbatim per skill v1.10.0+ #28
- PushNotification when first cargo build clears
- Push wins or errors entry per substep
```

## §6 Cross-references

- Architectural grounding: `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md` (61c9666)
- Phase 0 P0.A smoke: codex `67f18b9` errors entry + Claude `61c9666`
- Cutlass sm_89 template (from P0.A unstick): `docs/research/2026-05-10-cutlass-sm89-fp8-template-found-codex-unstick.md` (d5a6679)
- Phase 2' survey (W4+FP8 background): `docs/research/2026-05-10-path-b-phase-2-prime-w4-fp8-sm89-native.md` (3e83741)
- Strategic revision: `docs/research/2026-05-10-no-immediate-50pct-itl-path-revised-priority.md` (09ae5a5)
- Session retrospective: `docs/research/2026-05-10-session-retrospective-4-hallucinations-discipline-evolution.md` (de36538)
- Existing W4A8 substrate (mirror for PF8.3 Approach A):
  `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (987 LOC INT8)
- Existing INT8 act quant (mirror for PF8.1):
  `crates/cuda-kernels/csrc/gemm/w4a8_activation_quant.cu` (59 LOC)
- vLLM upstream Marlin: https://github.com/vllm-project/vllm/tree/main/csrc/quantization/marlin

## §7 Status

Directive ready for codex pickup. Estimated wall: ~2 days codex.
Outcome: TTFT -8 to -16% on prefill-dominant workloads (separate from
ITL axis). Substep breakdown is concrete + license gates documented.

If codex doesn't pick up in 2-3 ticks, Claude can self-implement
PF8.1 + PF8.2 (small substeps); PF8.3 (the GEMM kernel) is best
left to codex given P0.A spike experience.
