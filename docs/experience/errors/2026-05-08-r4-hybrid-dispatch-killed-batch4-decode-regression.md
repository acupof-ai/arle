# R4 #6 hybrid dispatch KILLED — `W4A16BatchGemv` slower than Marlin at batch=4 decode

> Hypothesis ENDED with hard data. R4 #6 patch (`f00ff8b`) reverted in this entry.
> The Marlin tensor-core advantage at batch=4 decode is REAL, exceeding the
> 5-launch overhead. Hybrid dispatch threshold is an anti-optimization for
> W4A16 path on sm_89.

## Phase 1 target recap

Decode ITL p50 on Qwen3-4B-W4A16-sym-g128-marlin at 4k/c=4, auto-FP8 KV
(per skill v1.2.0 isolation-motive callout — no `--kv-cache-dtype` override).

| Phase 8 threshold | Action |
|---|---|
| Δ ≥ −20% vs Arm B Marlin | LAND |
| Δ −5% to −20% | LAND with note |
| Δ ±5% NULL band | KILL or Phase 6 sweep |
| Δ > +5% regression | **KILL hard** |

## Phase 5 result — KILL hard

| Metric | Arm A BF16 baseline (`786a20a`) | Arm B Marlin all-batch (`f6f3af3`) | **Arm C R4 #6 hybrid (this run)** | Δ vs Arm B |
|---|---:|---:|---:|---:|
| TTFT p50 | 1976 ms | 2565 ms | 2394 ms | **−6.7% (improve)** |
| **ITL p50** | 19.27 ms | **11.76 ms** | **18.9 ms** | **+60.7% REGRESSION** |
| out tok/s | 153.83 | 191 | 143.52 | −24.9% |
| ITL std | n/a | n/a | 0.06 ms | tight σ — real signal |
| TTFT std | n/a | n/a | 94 ms | acceptable σ |
| greedy_consistency | n/a | n/a | 2/2 PASS | correctness preserved |

Δ ITL +60.7% > +5% → **KILL hard** per Phase 8.

## Root cause

The hypothesis (Round 4 prep `b3f22ea`) was:

> Marlin's per-call overhead (alloc + bf16_to_fp16 + Marlin GEMM + fp16_to_bf16
> = 6 launches) exceeds W4A16BatchGemv's single BF16-native launch at
> batch ≤ 8 decode. Predicted ITL 13-15 ms (1.23-1.47× vs BF16 baseline).

The hypothesis was **wrong on magnitude direction**. Actual:

- Marlin all-batch decode at batch=4: **11.76 ms ITL** (Arm B)
- W4A16BatchGemv decode at batch=4 (hybrid threshold=8): **18.9 ms ITL** (Arm C)

W4A16BatchGemv is **+61% slower than Marlin** at batch=4. The launch overhead
hypothesis was correct in magnitude (5+ launches per call vs 1) but wrong in
*sign of net effect*: Marlin's tensor-core throughput at batch=4 dominates
launch overhead. The W4A16BatchGemv kernel uses CUDA cores (no tensor mma) —
its single launch executes a slower kernel than Marlin's multi-launch pipeline.

In other words: the launch overhead is the cost of *amortizing tensor-core
compute*. The cost is real, but the benefit is even larger.

This refutes the framing in skill v1.2.0 anti-pattern #12 ("single kernel choice
not optimal at all batch sizes"). At least for W4A16 on sm_89:

- Decode batch=1: not tested in this experiment — `decode()` arm (W4A16Gemv)
  may still beat Marlin since GEMV at M=1 has tiny tensor-core advantage
- Decode batch=2-8: **Marlin wins** (this evidence)
- Decode batch>8 / Prefill batch=2048: Marlin wins (consensus)

## TTFT improvement is informative

Arm C TTFT 2394 ms < Arm B Marlin 2565 ms = -6.7% improvement. Why?

Hypothesis: at prefill batch=2048, both arms route to Marlin (threshold=8 still
triggers Marlin for batch>8). The TTFT difference is run-to-run variance, NOT
a hybrid-dispatch effect. σ TTFT 94 ms is within typical run variance; -171 ms
delta is ~1.8σ — not a clean signal.

Conclusion: the TTFT delta is noise. The ITL delta is signal. Both consistent
with "Marlin is correct dispatch for all batch>=2 in W4A16 path".

## What stays valid from R4 prep

- Round 4 prep `b3f22ea` survey of W4A16BatchGemv as BF16-native path: still
  factually correct (it IS BF16-native, 1 launch per call). The error was in
  predicting that NET savings exceed Marlin's compute advantage.
- R1 baseline correction `2853551`: still valid (production Marlin = 1.64×
  vs BF16, license fired).
- Skill v1.2.0 isolation-motive callout: still valid (avoid `--kv-cache-dtype`
  overrides). This experiment used auto-FP8 KV correctly.

## What changes for skill v1.3.0

Anti-pattern #12 ("single kernel choice not optimal at all batch sizes") needs
hardening:

> The decode-vs-prefill duality assumes that small batches are launch-overhead
> bound. **This is true for sufficiently small kernels** (e.g. norms, RoPE)
> but **NOT for compute-heavy ops like W4 GEMM with tensor cores**. Test the
> dual-kernel hypothesis with formula + bench BEFORE landing dispatch
> changes; do not assume the duality applies universally.

Will land in v1.3.0 alongside another skill update.

## Multi-shape defenses NOT run

Per Phase 8 "KILL hard" decision, the 3 defense benches (high-conc 1k/256/c=64,
multi-tenant prefix-cache, longctx-8k) are NOT run — primary regression is
conclusive.

If a future experiment wants to revisit hybrid dispatch (e.g. different
threshold, different GEMV kernel), defenses become mandatory again.

## Action — revert + this entry

`f00ff8b` reverted in companion git revert commit. R4 #6 axis CLOSED.

Surviving knowledge:
- W4A16 production Marlin all-batch dispatch is correct on sm_89
- Hybrid dispatch is anti-optimization for W4 GEMM at sm_89
- Skill anti-pattern #12 needs hardening (planned v1.3.0)

## Skill methodology applied

- ✅ Phase 1 target stated (decode ITL ≥ 1.5× vs BF16 baseline)
- ✅ Phase 2 hardware sheet (sm_89 Ada, 100KB smem, 706 FP8 / 88.5 BF16 TFLOPS)
- ✅ Phase 3 binding constraint formula-grounded (HBM bandwidth on weight read)
- ✅ Phase 4 magnitude formula (predicted 1.23-1.47×; actual 0.62× vs Arm B)
- ✅ Phase 5 single-variable A/B at production-default auto-FP8 KV (matched)
- ⏭ Phase 6 combo skipped (KILL hard at primary)
- ✅ Phase 7 tradeoff axis "tensor-core advantage at small batch" was the
  hypothesis-under-test — refuted as predicted possible outcome
- ✅ Phase 8 KILL hard threshold fired with σ < 5% confidence

NULL of opposite sign (hypothesis predicted +1.23-1.47× ITL improvement;
got -0.62× = +60% regression). σ tight, single arm conclusive — no need for
n=3 confirmation given σ < 0.5%.

## Cross-references

- R4 #6 plan: [`M_quant-marlin-round4-hybrid-dispatch.md`](../../plans/M_quant-marlin-round4-hybrid-dispatch.md) (`8adc1e1` plan, `0e7da0a` tick log, `5879103` post-correction update)
- R4 #6 patch (reverted): `f00ff8b`
- R1 baseline correction: [`2026-05-08-marlin-r1-baseline-correction.md`](2026-05-08-marlin-r1-baseline-correction.md) (`2853551`)
- W4A16 Marlin license bench: [`2026-05-08-m_quant-w4a16-marlin-bench.md`](../wins/2026-05-08-m_quant-w4a16-marlin-bench.md) (`f6f3af3`)
- Codex W4A8 substrate: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` + adapter (`a019a0e`)
- Skill v1.2.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../../.claude/skills/kernel-optimization/SKILL.md) (`4add8d7`) — anti-pattern #12 needs hardening
- Bench artifacts: `bench-output/2026-05-08-marlin-w4a16-r4-hybrid-c4-4k/`

## Rule

For W4 GEMM kernels on sm_89 (and likely sm_80+), Marlin's tensor-core
throughput dominates per-call launch overhead at decode batch ≥ 2. Do NOT
implement small-batch BF16-native fallback for W4A16 dispatch. Reserve
the hybrid-dispatch pattern for kernels where:
- Small-batch alternative actually has competitive throughput (not just
  fewer launches), AND
- Cost of multi-launch path exceeds the per-call kernel time of small
  alternative.

Round 4 #6 violated both criteria; the hypothesis was elegant but
empirically wrong. Skill rule #6 (License-or-kill σ < 5%) applied.
