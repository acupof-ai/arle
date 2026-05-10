---
title: SKILL kernel-optimization candidate #37 — substrate change validation requires multi-shape bench (regression-guard + acceptance-target)
date: 2026-05-10
type: research
status: open (single evidence point — codex's Task #35 wins entry §Rule, awaiting n+1 occurrence before sedimenting)
related_tasks: [#35 (cap=8 prefill warmup, codex authored wins entry this tick)]
---

# SKILL kernel-optimization candidate #37 — multi-shape bench discipline

> **Purpose**: capture a discipline rule explicitly stated in codex's
> Task #35 wins entry as a SKILL anti-pattern candidate. Single
> evidence point — not sedimenting into SKILL.md yet per accumulation
> policy (n=2-3 needed across distinct sessions). Documenting now so
> future-self has the evidence point + the rule's wording when the
> next occurrence happens.

## §1 The rule (codex's wording)

From `docs/experience/wins/2026-05-10-bench-35-cap8-prefill-warmup.md`
§Rule (codex authored, codex's exact words):

> Startup warmup changes need two gates: a short sustained-load
> regression smoke for conc 1/2/4, and a separate full first-burst
> workload for the workload that originally exposed the bimodal
> failure. Do not substitute one for the other.

## §2 Generalized formulation

Substrate change validation is incomplete with a single bench shape.
It needs:

1. **Regression-guard shape**: any reasonable workload (typically the
   common-case shape, e.g. c=1/2/4 sustained-load) that confirms the
   change doesn't break the steady-state happy path. PASS = "no
   regression introduced". This is the cheap gate that runs on every
   substrate diff.

2. **Acceptance-target shape**: the specific workload the change was
   designed to optimize. PASS = "the optimization actually works".
   Often more expensive (longer trace, specific load pattern, harder
   to set up).

**Substituting one for the other is the anti-pattern.** Pass on
regression-guard alone is "no harm" but unproven benefit; license
gate requires the acceptance-target shape too.

## §3 Why this is a #34 generalization, not a duplicate

SKILL v1.12.0 #34 says:

> greedy_consistency single-request PASS NECESSARY but NOT SUFFICIENT
> for new GEMM kernel substrate. Pair with sustained-load bench at
> conc 1+2+4 BEFORE declaring license.

This is a specific case (greedy + sustained-load) of the more general
multi-shape rule. The Task #35 evidence shows the same pattern at a
different abstraction level (sustained-load + first-burst):

| Anti-pattern | Necessary | Not sufficient on its own |
|---|---|---|
| #34 specific | greedy_consistency conc=1 PASS | greedy_consistency conc=1 PASS for kernel substrate |
| #37 general | regression-guard shape PASS | regression-guard shape PASS for substrate change |

#37 subsumes #34 if codified — but skill anti-pattern accumulation
policy errs toward specific cases over general principles since
specific cases have testable criteria. Worth tracking the candidate
for now.

## §4 Why this is also a #29 generalization

SKILL v1.12.0 #29 says:

> default test fixtures may be known-broken (load-bearing 2026-05-10
> session: codex caught greedy_consistency PASS on W4A8 keeping PF8
> INACTIVE before re-running on hybrid checkpoint, saved false-license
> risk).

#29 is "the default test fixture / shape may not match production".
#37 is "no single shape can validate substrate change". They're
related — both warn against single-shape false confidence — but #29
focuses on the trap of inheriting bad defaults, #37 focuses on the
trap of stopping at one shape.

## §5 Evidence chain

### §5.1 First evidence (n=1, codex's Task #35 wins entry, 2026-05-10)

Codex's Pass 3 prefill warmup change had:
- regression-guard: c=1/2/4 short sustained-load smoke (n=3 each arm)
  → PASS
- acceptance-target: full W4 8k cap=8 first-burst trace (`Task #31`
  shape) → NOT MEASURED (codex stopped long-running attempt)

Codex's wins entry explicitly refuses to claim Task #35 acceptance
based on regression-guard alone. The §Rule is the lesson learned.

### §5.2 Watch-list for n+1 evidence

When other substrate changes land in future sessions, check whether
the wins entry has both shapes. If the next ≥1 substrate-change wins
entry independently produces this same multi-shape discipline → n=2
→ candidate #37 graduates to canonical SKILL anti-pattern.

Specific watch: PF8.3 H1' refactor (Task #47) when bench v11 licenses
PF8 and codex implements the static-scratch fix. Acceptance shapes:
- regression-guard: greedy_consistency at conc=1 (already PASS)
  + sustained-load at conc=1/2/4
- acceptance-target: sustained-load at conc≥4 60s+ (the shape that
  exposed the original PF8.3 RUNTIME KILL at 101380/101380 failures)

If H1' wins entry has both AND lists them both as gates → n=2 confirmed.

## §6 Cross-references

- `docs/experience/wins/2026-05-10-bench-35-cap8-prefill-warmup.md`
  §Rule (codex's exact wording, n=1 source)
- SKILL `kernel-optimization` v1.12.0 #34 (specific case for GEMM
  substrate; #37 generalizes)
- SKILL `kernel-optimization` v1.12.0 #29 (default fixture broken;
  #37 partially overlaps)
- `940c7cc` Claude Pre-bench prediction (predicted exactly this gap
  in §4.1 falsification list — "bench shape ≠ first-burst-heavy")
- `5f3f58f` reconciliation (confirms the gap empirically + flags
  this as a #37 candidate)
- Task #35 cap=8 prefill warmup
- Task #47 H1' refactor (next watch for n+1 evidence)

## §7 Status

**Single evidence point.** Awaiting n+1 occurrence before SKILL
sedimentation. This research note is the evidence ledger entry —
when the next substrate change explicitly applies multi-shape
discipline (with both gates documented in its wins entry), upgrade
to SKILL anti-pattern in v1.13.0+.

NOT actionable this session — pure documentation for future-self.
