---
title: Task #35 Pass 3 cap=8 first-burst — Claude formula prediction BEFORE codex bench numbers land (SKILL #34 mantra rule 1 verification setup)
date: 2026-05-10
type: research
status: open (prediction made BEFORE bench result; will be reconciled with codex's wins entry when it lands)
related_tasks: [#35 (cap=8 prefill warmup, codex Working ~36min)]
---

# Task #35 Pass 3 cap=8 first-burst — formula prediction PRE-bench

> **Purpose**: codex is currently running W4 c=8 agent trace cold-start
> bench for Task #35 acceptance. Per SKILL kernel-optimization v1.12.0
> mantra rule 1 ("Predict with formula, not vibes"), Claude commits a
> prediction NOW (before numbers come back) so reality vs prediction
> can be reconciled in the wins entry. This is the SOLID
> license-or-kill discipline applied to a substrate change.

## §1 What Pass 3 does

Per `M_warmup-prefill-pass-directive.md` + codex Task #35 implementation:

- **Pass 1**: model load + weights pinning (always)
- **Pass 2**: graph-capture mode only — re-captures graphs with autotuned algorithms
- **Pass 3** (NEW): cap=8 prefill batch pre-warm at server startup
  - default-on with `INFER_PREFILL_WARMUP=0` opt-out (per `60f114f`
    matched-control discipline doc)
  - Eager warmup of cap=8 prefill kernels so first cap=8 burst doesn't
    hit cold codegen / capture / JIT paths
  - Observed startup cost: +282.7ms (per codex's sustained-load A/B
    summary this session)
  - Pass 3 body itself: 308ms stable

## §2 Baseline (prior cap=8 bench numbers)

Per `docs/experience/wins/2026-05-08-ttft-p99-cap8-fix-86pct-reduction.md`
(Task #31 closure, `bench_agent_trace.py agent-w4-tool-resume`):

| Metric | cap=4 (pre-#31) | cap=8 (#31 default-on) | Δ |
|---|---|---|---|
| TTFT p50 | 11768 ms | 5868 ms | −50% |
| TTFT p99 | 72515 ms | 10259 ms | −86% |
| p99/p50 spread | 6.2× | 1.75× | −72% |

The cap=8 numbers above are the **without-Pass-3 baseline** for Task
#35. Codex's Task #35 "off" arm should reproduce ~5868 / 10259 ms
(within σ).

## §3 Formula prediction for Pass 3 cap=8 first-burst

### §3.1 Mechanism

First cap=8 burst (request set 1, fresh server, cold kernels):

```
TTFT_first_burst = T_kernel_cold + T_graph_capture + T_prefill_compute
                 + T_decode_first_token

Without Pass 3:
- T_kernel_cold = full JIT/codegen on first cap=8 prefill = ~hundreds of ms
- T_graph_capture = cuda graph capture for cap=8 shape = ~tens of ms
- T_prefill_compute = warm-path baseline (~hundreds of ms for 8×512=4096 tokens)
- T_decode_first_token = warm-path baseline (~10 ms)

With Pass 3:
- T_kernel_cold = ~0 (paid at server startup via Pass 3 +282.7ms)
- T_graph_capture = ~0 (already captured during Pass 3)
- T_prefill_compute = warm-path baseline (unchanged)
- T_decode_first_token = warm-path baseline (unchanged)
```

### §3.2 Magnitudes

The current cap=8 baseline (5868 ms p50 / 10259 ms p99) IS the
"steady-state cap=8" — these numbers are aggregated over many bursts
in the agent trace, including warm ones. The cold first-burst is
amortized.

If Pass 3 eliminates the cold first-burst overhead:
- p50 over the trace: should change LITTLE (most bursts are warm
  already in steady-state)
- p99 (the first burst tail): could drop more visibly since first
  burst's tail is the worst case
- p99/p50 spread: should tighten further

### §3.3 Predicted Δ ranges

| Metric | Predicted Δ vs `off` arm | Reasoning |
|---|---|---|
| TTFT p50 | **−2% to +2%** (essentially noise) | Most agent-trace bursts are warm; first-burst small fraction |
| TTFT p99 | **−5% to −20%** | First-burst tail is what Pass 3 eliminates |
| TTFT p99/p50 spread | **−5% to −15%** | Tight tail = better spread |
| First-burst TTFT (if measured separately) | **−30% to −60%** | This is what Pass 3 actually optimizes |
| Steady-state ITL p50 | **0% (unchanged)** | Pass 3 doesn't touch decode |
| Server startup time | **+282.7ms** (observed, paid once) | Already measured |

### §3.4 License threshold (Claude's pre-bench gate)

Per SKILL v1.12.0 Phase 8 license-or-kill:

| Outcome | Verdict |
|---|---|
| TTFT p99 Δ > −10% AND first-burst Δ > −30% AND ITL Δ ≈ 0 | LICENSE Pass 3 default-on |
| TTFT p99 Δ < −5% (regression in tail) OR first-burst Δ > −10% | KILL Pass 3 (effect too small to justify +282.7ms startup) |
| Between | REVIEW (might want to flip default-on/off based on workload) |

## §4 What could falsify this prediction

Per skill #1 "推断 ≠ SOLID — 推断 = hypothesis, evidence = bench numbers":

1. **bench_agent_trace.py shape ≠ first-burst-heavy**: if the agent
   trace replays many warm bursts and only 1-2 cold first-bursts, the
   first-burst Δ would be diluted in aggregates. **Check codex's
   wins entry for first-N-burst breakdown.**
2. **Pass 3 doesn't actually warm the right kernels**: implementation
   bug — Pass 3 might warm shape A but production hits shape B.
   Symptom: no cold-vs-warm difference in numbers. **Check codex
   wins for whether off/on first-burst differs at all.**
3. **Cold first-burst on this shape is already small**: maybe the
   agent trace's first burst doesn't trigger expensive JIT (e.g. TileLang
   AOT cache hit). Symptom: Δ ≈ 0 across all metrics.
4. **Variance dominates**: σ across N=3 runs > Δ → not a real win.

## §5 Reconciliation plan

When codex's wins entry lands:

1. Read the actual TTFT p50 / p99 / first-burst Δ
2. Compare to §3.3 predicted ranges
3. If WITHIN range → SKILL #1 mantra holds, formula was right
4. If OUTSIDE range:
   - Above range (over-delivered) → revise formula upward, learn
     that JIT cost was bigger than estimated
   - Below range (under-delivered) → check §4 falsification list
5. Either way, document the actual Δ in this doc's §6 (added by next
   tick)

## §6 Actual results (TBD — codex's bench in flight)

[To be filled in when codex's wins entry lands. Format:

| Metric | Predicted Δ | Actual Δ | Within range? |
|---|---|---|---|
| TTFT p50 | -2% to +2% | TBD | TBD |
| TTFT p99 | -5% to -20% | TBD | TBD |
| First-burst TTFT | -30% to -60% | TBD | TBD |
| ITL p50 | 0% | TBD | TBD |

Then either: ✅ prediction held → formula good / 🔄 prediction off
→ which §4 reason → revise.]

## §7 Cross-references

- `M_warmup-prefill-pass-directive.md` (the directive Task #35
  implements)
- `2026-05-08-ttft-p99-cap8-fix-86pct-reduction.md` (Task #31 cap=8
  baseline, the "off" arm reference)
- `60f114f` matched-control escape-hatch discipline (Task #35 design)
- `0be7220` SKILL kernel-optimization v1.12.0 mantra rule 1 +
  Phase 8 license-or-kill
- `infer/tests/greedy_consistency.rs:test_greedy_solo_vs_concurrent`
  (codex's correctness gate, PASSED)
- This doc: `2026-05-10-task35-pass3-cap8-prediction-pre-bench.md`

## §8 Status

**Prediction committed BEFORE codex bench result lands.** This is
the verification setup — formula vs reality. Outcome reconciled in
§6 next tick.
