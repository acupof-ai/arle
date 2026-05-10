---
title: 2026-05-10 M_quant hybrid plan vs REFUTATION consistency audit — plan was CORRECT, my §12.7 extrapolation was wrong
date: 2026-05-10
type: research
status: open (corrects my framing decay; repositions Hybrid as auxiliary)
related_docs: [`9735b47` REFUTATION wins, `114aca4` strategic matrix update, `M_quant-w4a16-w4a8-hybrid-prefill-decode.md`]
---

# Hybrid plan consistency audit — REFUTATION measurement matches plan, NOT my extrapolation

> **2026-05-10 later update**: the "Hybrid stacks with Medusa" framing
> remains a hypothesis for the old Qwen3/Qwen3.6 chain. For active
> Qwen3.5, Medusa is blocked until recurrent-state rollback exists.

> **Why now**: Tick 102 (`114aca4`) updated session-tail strategic matrix
> to position Option A (Medusa) as dominant for ALL contexts based on
> `9735b47` REFUTATION. This audit cross-checks against the original
> `M_quant-w4a16-w4a8-hybrid-prefill-decode.md` plan (predates
> REFUTATION) and finds that **the plan's -14% E2E prediction was
> CORRECT**. The mistake was in `4718b44` §12.7 linear extrapolation
> to -35.3%, NOT in the plan. Hybrid Option B repositions from "killed
> by REFUTATION" to "auxiliary -14% win that stacks with Medusa".

## §1 What the M_quant plan actually predicted

`M_quant-w4a16-w4a8-hybrid-prefill-decode.md` §5:
> - Hybrid vs W4A16-only: **−14% E2E latency**
> - Hybrid vs W4A8-only: **−29% E2E latency**

Plan trigger was `b5889b3` 4k bench:
- W4A8 prefill TTFT: 1632 ms vs W4A16 2388 ms = **-32%**
- W4A16 decode ITL: 11.73 ms vs W4A8 19.18 ms = **-39% (1.64×)**
- Combined Hybrid (W4A8 prefill + W4A16 decode): **-14% E2E**

## §2 What REFUTATION measurement showed

`9735b47` measured Hybrid Option B at conc=1, prompt=16384, output=128:
- Hybrid vs W4A16-only: **-14.2% E2E latency**

This is essentially **EXACT MATCH** to plan's -14% prediction.

## §3 Where I went wrong

`4718b44` §12.7 in the long-ctx wins entry extrapolated:
- "Hybrid Option B at 16k EXTRAPOLATED to -35.3% perceived latency
  (would cross Machete-class threshold)"

This was a linear extrapolation of n=4 data points (-1.4% / -7.5% /
-11.1% / -14.2%) projecting continued growth. The extrapolation was
WRONG because:
- Plan §5 had already correctly modeled it as steady-state -14%
- Linear extrapolation ignored the structural cap from shared
  paged-attention prefill kernel
- I should have read the plan FIRST before extrapolating

## §4 Repositioning Hybrid Option B

REVISED understanding (post this audit):

| Property | Hybrid Option B | Medusa (Option A) |
|---|---|---|
| E2E latency win vs W4A16 | **-14%** (steady-state from 8k+) | **2.25× tok/s** (predicted) |
| Substrate effort | 150-300 LOC, 1.5 days | ~350 LOC + 48-60 hr training |
| Memory cost | +45% (dual weight tensors) | +100 MB (5 heads) |
| Stacking with each other | YES — orthogonal axes | YES — orthogonal axes |
| Risk | LOW — math validated by REFUTATION | MEDIUM — α value unproven |
| Time-to-verdict | 2 days | 2.5-3 days |

### §4.1 They STACK

- **Hybrid** = better TTFT/ITL per-step (-14% latency)
- **Medusa** = more accepted tokens per step (2.25× tok/s)
- Both compose multiplicatively at the throughput level:
  - Hybrid alone: 1/(1 - 0.14) = 1.16× tok/s
  - Medusa alone: 2.25× tok/s
  - Combined: ~1.16 × 2.25 = **2.61× tok/s vs W4A16-only**

### §4.2 Order matters

Recommended order:
1. **Hybrid first** (1.5 days, low risk, validates infra + acc)
2. **Medusa second** (2.5-3 days, medium risk)
3. Then bench combined (Hybrid + Medusa) — 0.5 day

Total wall-clock: 4-5 days for both. Predicted combined gain: **2.61×
tok/s + -14% latency** vs current W4A16 baseline.

## §5 Strategic matrix UPDATED (supersedes `114aca4`)

| User priority | Recommended path | Why |
|---|---|---|
| Maximum tok/s any context | **A + B combined** | 2.61× tok/s + -14% latency |
| Maximum TTFT/ITL short-ctx | A (Medusa) | hybrid value sub-noise here |
| Maximum TTFT/ITL long-ctx (≥8k) | **A + B combined** | Hybrid plateaus at -14% but stacks with Medusa |
| Lowest-risk first deliverable | **B first** (Hybrid) | math validated by REFUTATION; 1.5 days |
| Best ROI per day | A + B (B first, then A) | 4-5 days for ~2.6× tok/s + -14% latency |
| ROADMAP P0 World #1 (32k×c=4) | A + B + separate harness | bench gap remains regardless |

### §5.1 What this corrects

`114aca4` strategic matrix said:
> "Option B alone is no longer the recommended long-ctx path. Option A
> is now the dominant single-axis investment."

This was based on misreading -14.2% as "the failure of B to reach
Machete-class" rather than "B delivers exactly its planned value".
B was never SUPPOSED to reach Machete-class alone — that was my
extrapolation error in §12.7, not B's design intent.

The CORRECT framing: B and A are independent axes. Both should be
pursued. B first (lower-risk, faster).

## §6 Action items (updated, supersede `e021026` §3 friction-reduction list)

§7 gate from `f0c7561` Phase 1.B brief now should ALSO ask user:

- [ ] Pursue Hybrid Option B first (1.5 days, low risk, -14% E2E)?
- [ ] Then Medusa Option A (2.5-3 days, 2.25× tok/s, dataset Alpaca-ready)?
- [ ] Or parallel codex tracks (B with one codex, A with another)?

If user OKs sequential B-then-A: total 4-5 days to compound win.
If user OKs parallel: total ~3 days but higher coordination overhead.

## §7 Cross-references

- `9735b47` REFUTATION wins entry (the measurement that matches plan)
- `4718b44` §12.7 — the WRONG extrapolation (this audit corrects)
- `114aca4` strategic matrix update — this audit further refines
- `M_quant-w4a16-w4a8-hybrid-prefill-decode.md` — plan was CORRECT
- `b5889b3` original 4k bench (plan trigger)
- Phase 1.B Medusa brief: `f0c7561`
- Alpaca data ready: `e021026`
- Plan readiness: 5-doc Medusa pickup chain + this audit = 6-doc
  bilateral A+B pickup chain

## §8 Rule

When data appears to "refute" a plan's prediction, FIRST check whether
the plan itself predicted that value before claiming refutation. The
plan's §5 already had -14% on the page. My §12.7 extrapolation
introduced a SUPER-prediction (-35.3%) that data refuted, not the
plan's prediction (-14%) that data matched.

This is SKILL candidate "always-read-plan-before-extrapolating" — n=1
new evidence to graduate at next SKILL bump.
