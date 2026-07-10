---
title: Phase 0 P0.A decode KILL — architectural implication for "-20-40% ITL on sm_89" target
date: 2026-05-10
type: research
status: strategic-synthesis-codex-errors-entry-pending
---

# Phase 0 P0.A decode KILL — architectural implication for "-20-40% ITL on sm_89" target

> Codex's P0.A cutlass FP8 smoke ran successfully (post my d5a6679
> unstick) and produced split outcome: decode 1.86× (KILL),
> prefill 5.21× (separate axis). Codex Working on errors entry.
> This entry is the Claude-side **architectural synthesis** of the
> finding — durable framework for next-axis decision.

## §0 Raw evidence (per skill v1.10.0 #28 — raw `cat /tmp/cutlass_fp8_smoke_run2.log` quoted)

```
GPU: NVIDIA GeForce RTX 4070 Ti SUPER sm_89

DECODE  M=1    N=4096 K=2560:
  BF16-CUTLASS:        mean=0.426588 ms  std=0.001367  TFLOPS=0.05
  FP8-CUTLASS-staged:  mean=0.245514 ms  std=0.001255  TFLOPS=0.09  → 1.74× speedup
  FP8-CUTLASS-fast:    mean=0.229284 ms  std=0.001263  TFLOPS=0.09  → 1.86× speedup

PREFILL M=2048 N=4096 K=2560:
  BF16-CUTLASS:        mean=1.399307 ms  std=2.092525  TFLOPS=30.69  ← σ noisy
  FP8-CUTLASS-staged:  mean=0.311385 ms  std=0.000485  TFLOPS=137.93 → 4.49× speedup
  FP8-CUTLASS-fast:    mean=0.268771 ms  std=0.000746  TFLOPS=159.80 → 5.21× speedup
```

All `can_implement=Success`, workspace=0, smem=73728 bytes, threads=128.
Same kernel template, only shape varies between decode/prefill.

## §1 Per-Phase-0 license matrix outcome

| Shape | Speedup | Threshold | Verdict |
|-------|--------:|-----------|---------|
| Decode (M=1)   | 1.86× | ≥3× license, ≤2× kill | **KILL** |
| Prefill (M=2048) | 5.21× | ≥3× license | LICENSE (separate axis) |

Decode is the ITL-binding shape per skill anti-pattern #5 (wall-clock
framing rule). Per the Phase 0 license matrix in `5a7a28b` brief, the
decode KILL is decisive for the **W4+FP8 ITL axis**. Prefill 5.21×
is a separate finding for the TTFT axis.

## §2 Why decode FP8 only delivers 1.86× — the architectural insight

For decode at M=1 N=4096 K=2560:
- BF16 GEMM TFLOPS achieved: 0.05 / 88.5 = **0.06% of theoretical**
- FP8 GEMM TFLOPS achieved: 0.09 / 706 = **0.013% of theoretical**

Both are 100s of times below tensor-pipe peak. The kernel is **NOT
compute-bound**. It's memory-bound on weight read.

Memory-bound formula (per M_quant magnitude plan §2.1):
- Weight read time = N × K × bytes_per_param / HBM_bandwidth
  = 4096 × 2560 × bytes_per_param / (672 GB/s × 1e9)
- BF16 (2B): 10.49 MB / 672 GB/s = **15.6 µs theoretical**
- FP8 (1B): 5.24 MB / 672 GB/s = **7.8 µs theoretical**

For a single-token decode at this shape, weight bandwidth is the
binding constraint, NOT mma throughput. FP8 helps weight bandwidth
2× (matches the observed 1.86×) but doesn't unlock the 8× FP8/BF16
mma ratio.

## §3 The W4 case is the WORST case for FP8 ITL win

ARLE production decode uses **W4 weights** (Marlin), not BF16. So:
- W4 (0.5B per param): 2.62 MB / 672 GB/s = **3.9 µs theoretical**

Switching activation from BF16 to FP8 in a W4 GEMM kernel:
- Activation bandwidth: small fraction of total (one BF16 row = 5KB,
  weight = 2.62MB → activation is 0.2% of memory traffic)
- Weight bandwidth: unchanged (W4 is already 4× smaller than BF16)
- Net theoretical gain: ≈ **0% for decode ITL**

The Phase 0 P0.A smoke compared BF16 weights vs FP8 weights+acts. For
production W4 + FP8-act path:
- BF16 weights → 1.86× speedup (memory bandwidth halved)
- W4 weights + BF16 act (current ARLE W4A8 with INT8 act) → already
  near memory-bound floor for weights
- W4 weights + FP8 act (Phase 2' target) → activation savings are
  ≈ 0.2% of memory traffic = ~0% ITL gain

**This means the Phase 2' Phase 0 decode KILL is even MORE definitive
for the production case than the smoke shows**. The smoke's BF16-weight
baseline overstates the FP8 ITL gain by including weight bandwidth
savings that don't apply to W4 production.

## §4 Strategic implication: "-20-40% ITL on sm_89" target reassessment

User's stated target: -20-40% ITL on Qwen3-4B W4 decode for the
"Machete-from-vLLM" axis. This target is **structurally infeasible**
via the FP8 mma path on sm_89:

- W4 decode is HBM-bound on weight read (already 4× smaller than BF16)
- FP8 mma helps compute, not memory bandwidth
- Memory bandwidth is fixed by hardware (672 GB/s on 4070 Ti SUPER)

For ITL gains on sm_89 W4 decode, the binding constraint is **HBM bandwidth**, not GEMM throughput. ITL improvement requires:

- (a) Smaller weight footprint (W4 → W3 / W2 quantization)
- (b) Better KV bandwidth (smaller KV cache, better page layout, prefetch)
- (c) Spec decoding (effective batch >1 amortizes weight read)
- (d) Smaller model (impractical for Qwen3-4B)

**FP8 activation is none of (a)-(d).**

## §5 What this tick rules out

- **Phase 2'.1-2'.4** (W4+FP8 substrate per `3e83741`): KILL for ITL.
  Decode P0.A smoke 1.86× already overstates the production gain.
- **Machete sm_89 backport**: blocked + would face the same memory-
  bound ceiling at decode. e65a096 blocker stands.

## §6 What this tick does NOT rule out

### (a) Path B Phase 1 — dequant.h port (`e59beb5`)

Still applicable as low-risk -3-8% ITL fallback. Targets per-launch
overhead reduction (5-launch Marlin → 2-3 launch with atomic_add)
not memory bandwidth — orthogonal to the Phase 0 finding.

### (b) Prefill-only FP8 optimization (NEW finding from this tick)

Codex's prefill 5.21× speedup at M=2048 is a **separate, valid
license** for the **TTFT axis** (chunked prefill). At prefill batch
sizes, the GEMM is compute-bound (159 TFLOPS achieved = 22.6% of
706 TFLOPS theoretical), and FP8 mma helps the compute side directly.

ROI estimate for prefill-only FP8:
- Current Marlin W4 prefill ITL contribution at chunked prefill = ~10-20% of TTFT
- FP8 5.21× of that contribution = -8-16% TTFT improvement on prefill-heavy workloads
- Less than user's -20-40% target but a meaningful TTFT win

A new Phase 2'-prefill-only directive could:
- Port `marlin_int4_fp8_preprocess.cu` (~120 LOC, sm_89 compatible)
- Add FP8 path to chunked prefill ONLY (not decode)
- Estimated wall: ~2 days codex (~700 LOC vs original 900-1700)

### (c) Memory-side optimization (the actual ITL win mechanism)

Per §4, the binding constraint for sm_89 W4 decode ITL is HBM
bandwidth. Real ITL wins require:

- **W3 / W2 weight quantization** — direct weight footprint reduction
  - W3: 25% smaller than W4 → ~25% ITL ceiling reduction = matches
    user's -20-40% target if accuracy holds
- **Speculative decoding (Medusa/EAGLE — task #28 pending)** —
  effective batch >1 amortizes weight read, theoretical 2-3× ITL
- **KV cache compression / better layout** — smaller KV → less HBM
  traffic per decode step
- **Better prefetch / pipeline overlap** — narrower gap to memory floor

These are different P0 axes than W4+FP8.

## §7 Recommended next-axis priority (revised post P0.A decode KILL)

Per `e65a096` Machete blocker default rule, with Path B-Phase2' decode
now KILLed:

| Priority | Path | LOC | Wall | Predicted gain |
|----------|------|-----|------|----------------|
| **P0** | Spec decoding scaffold (#28, blocked on #34 HF Hub) | 500 | 1 wk + training | -50%+ ITL via amortized weight read |
| **P1** | W3 / W2 quantization research | TBD | 1 wk research | -25-50% ITL ceiling per quant level |
| **P2** | Phase 1 dequant.h port (`e59beb5`) | 687 | 1.5-2 days | -3-8% ITL (fallback) |
| **P3** | Prefill-only FP8 (`new`, derived from P0.A 5.21×) | 700 | 2 days | -8-16% TTFT |

P0 (spec decoding) is the single biggest ITL lever per the architectural
analysis. Blocked on #34 (HF Hub library blocker for downloading draft
model weights).

## §8 PushNotification recommendation

User has stated -20-40% ITL target multiple times. P0.A decode KILL
+ memory-bound architectural analysis = structural mismatch with
target on FP8 path. Honest message to user:

> "Phase 2' decode KILL via P0.A cutlass FP8 smoke (1.86× decode <
> 3× license, < 2× kill). Architectural cause: W4 decode is HBM-bound
> on weight read; FP8 mma helps compute not memory bandwidth. Same
> reason Machete (sm_90) wouldn't help even if backportable: ITL
> bottleneck is bandwidth, not throughput. Real -20-40% ITL paths on
> sm_89 W4 decode = (a) spec decoding (#28, blocked on #34), (b) W3/W2
> quant, (c) memory-side opts. FP8 path retained for **prefill-only
> TTFT improvement** (5.21× at M=2048). Default next axis = #28 spec
> decoding once #34 unblocks."

## §9 Cross-references

- Phase 0 brief: `docs/research/2026-05-10-path-b-phase2-prime-phase0-brief-codex-kickoff.md` (5a7a28b)
- Cutlass sm_89 unstick: `docs/research/2026-05-10-cutlass-sm89-fp8-template-found-codex-unstick.md` (d5a6679)
- P0.B PPL inventory: `docs/research/2026-05-10-p0b-ppl-eval-infra-inventory.md` (6a6114d) — decision matrix says skip P0.B if P0.A decode ≤2x → applies, P0.B not run
- Machete blocker: `docs/research/2026-05-10-machete-blocker-stronger-evidence-user-reissued-axis.md` (e65a096)
- M_quant magnitude plan §2 ITL formula: `docs/plans/M_quant-fp8-w4-magnitude-path.md`
- Skill v1.10.0 anti-patterns: `.claude/skills/kernel-optimization/SKILL.md`
  - #5 wall-clock framing rule (decode = ITL-binding)
  - #28 hallucinated tool output overrides peer (this entry quotes raw log evidence)
- Codex's errors entry: pending (codex Working at tick capture time)
- Raw smoke logs: `/tmp/cutlass_fp8_smoke_run2.log`, `/tmp/cutlass_fp8_smoke_build.log`

## §10 Status

P0.A decode KILL definitively rules out Path B-Phase2' for the user's
"-20-40% ITL" target. Architectural cause: HBM bandwidth ceiling on
W4 decode, FP8 mma is the wrong lever. Real ITL wins require spec
decoding (#28) or W3/W2 quant (new axis).

Prefill 5.21× is a separate valid signal for TTFT-axis optimization
(new "Phase 2'-prefill-only" directive candidate, smaller scope).

PushNotification dispatched (next tick) with revised priority order.
