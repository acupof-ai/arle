---
title: Task #43 INVERSE outcome — Claude dispatch-audit hypothesis OVERTURNED by codex's behavioral A/B
date: 2026-05-10
type: research
status: closed (Task #43 DISPROVEN per 83fc5d0; Claude self-correction sediment)
related_tasks: [#43 (codex completed DISPROVEN), #47 (H1' refactor — REMOVE Task #43 share claim), #44 (PF8 chain — H1' justification weakens)]
---

# Task #43 INVERSE outcome — Claude hypothesis OVERTURNED

> **Purpose**: Task #43 codex bench result (`83fc5d0` errors entry)
> didn't just disprove the hypothesis — it showed the OPPOSITE
> direction. This is a SOLID lesson worth sedimenting: dispatch-audit
> + grep evidence is hypothesis, not ground truth. Behavioral A/B
> tests are the only way to verify causal direction.

## §1 The actual numbers (per 83fc5d0)

| arm | env | guidellm successful | live kernel failures | TTFT p50 | ITL p50 | verdict |
|---|---|---:|---:|---:|---:|---|
| A | `INFER_PREFILL_GRAPH=1` | 71 | **36** (OOM!) | 834.6 ms | 7.35 ms | **substrate kill** |
| B | unset (eager fallback) | 56 | 0 | 2381.7 ms | 11.36 ms | **healthy** |

Representative Arm A live failure:
```text
Request 1: prefill batch failed: Alloc failed:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

## §2 Claude's hypothesis (from `1ba06f0` + `2cc608a`)

Based on dispatch audit at `linear.rs:2064-2095` + `qwen3/forward.rs:312-313`:
- Arm A (env on, marlin_scratch=Some, `_with_scratch` path) → predicted HEALTHY
- Arm B (env off, marlin_scratch=None, per-call alloc fallback) → predicted SUBSTRATE-KILL

Predicted root cause: per-call alloc fragmentation (analogous to PF8.3 KILL).

## §3 Reality: INVERSE direction

Arm A is the killing arm; Arm B is healthy. **Enabling the scratch
path causes OOM, not fixes it.** Per codex's analysis:

> The failing arm is the prefill-graph/scratch-enabled arm, not the
> eager fallback arm. This points to prefill-graph memory footprint
> or persistent graph-resource cache pressure, not per-call cudarc
> allocator fragmentation in the eager fallback.

The prefill graph + persistent scratch buffers compete with KV cache
budget under conc=4 4k-prompt sustained load. The eager fallback is
slower (TTFT p50 2382 vs 835 ms = 2.85× slower) but doesn't OOM.

## §4 SOLID lesson — n=2 evidence for SKILL candidate #36

Per skill candidate #36 (proposed in `2cc608a` §2.3): "before
designing a new substrate pattern, grep the file for existing
variants of the same shape. The pattern may already exist;
designing from scratch duplicates effort and risks divergence."

This case is the **dual** of #36's original evidence:

| Aspect | #36 n=1 (2cc608a) | #36 n=2 (this case, INVERSE) |
|---|---|---|
| Action taken | Designed PF8Scratch from scratch | Designed Task #43 hypothesis from dispatch audit |
| Missed | MarlinScratch already existed (saved 40 LOC by reusing) | Hypothesis was directionally wrong (saved nothing — wasted hypothesis effort) |
| Cure | grep for variants before designing | Grep alone insufficient — also need behavioral A/B |

Strengthened wording for SKILL v1.13.0+ #36 (when graduated):
> "Before designing OR hypothesizing about a substrate pattern, do
> BOTH (a) grep file for existing variants AND (b) cheap behavioral
> A/B to verify causal direction. Static code audit (dispatch-audit
> + grep) is hypothesis-grade evidence, not ground truth.
> Behavioral verification (n=2 here) catches inverted assumptions
> that grep cannot."

## §5 Implications for PF8 chain (Task #44/#47)

**The "two-tasks-one-PR" opportunity in `2cc608a` §2.2 is VOIDED**:
PF8.3 substrate fix (Task #47 H1' refactor) does NOT share root
cause with Task #43. Two separate problems with different causes:

- **PF8.3 RUNTIME KILL** (`0cde63d`): PF8 path lacks `_with_scratch`
  variant entirely (all per-call alloc), 101380/101380 fail at conc=4
- **Task #43 W4A16 sustained-load failure**: PRESENCE of scratch
  path (with persistent graph resources) competes with KV cache
  → OOM under conc=4 4k-prompt

**Task #47 H1' refactor justification is now SOLO** (not shared with
Task #43). Still valid (PF8.3 needs scratch variant for any conc≥2
production), but the LOC savings argument from `2cc608a` should be
reviewed — does extending MarlinScratch with PF8 fields exacerbate
the Task #43 OOM problem?

**Open question**: would adding PF8Scratch fields to MarlinScratch
make Task #43 OOM WORSE (more persistent buffers competing with KV)?
This needs cheap experiment BEFORE Task #47 implementation lands.

## §6 Updated forward path

Pickup queue (`f63838b`) needs revision:
- P1 LICENSE Task #47 H1' refactor — still valid for PF8 path, but:
  - Cannot bundle Task #43 fix in same PR (Task #43 needs different
    fix: reduce graph capture footprint OR config knob to skip
    persistent scratch when KV pressure high)
  - H1' implementation should A/B test: with vs without PF8 scratch
    fields added to MarlinScratch — confirm no Task #43-style OOM
    regression
- Task #43 next pickup options:
  - Investigate prefill-graph memory footprint (which buffers persist
    + how big)
  - Add knob to disable persistent scratch under KV pressure
  - Profile graph capture buffer growth at conc=4 4k

## §7 What I (Claude) should self-correct

1. **Update `cb86836` verdict implications**: §2 (CONFIRMED) and §3
   (DISPROVEN both arms HEALTHY) are both wrong framings. Add §3.5
   "INVERSE: Arm A KILL + Arm B HEALTHY" — overturns hypothesis,
   same dispatch routing produces OPPOSITE memory pressure
2. **Update Task #47 description**: REMOVE "two-tasks-one-PR" claim
3. **Update `2cc608a` H1' design REVISION §2.2**: remove "two-tasks-
   one-PR opportunity" + flag the OOM-regression risk for H1'
   implementation
4. **Update SKILL candidate #36** with n=2 evidence (this case)
5. **Reinforce SOLID rule 1** (推断 ≠ SOLID): dispatch-audit was
   推断, behavioral A/B was evidence — they disagreed, evidence won

## §8 Cross-references

- `83fc5d0` codex Task #43 errors entry (DISPROVEN, source of inverse data)
- `1ba06f0` Claude dispatch-audit hypothesis (now OVERTURNED)
- `2cc608a` H1' design REVISION §2.2 (two-tasks-one-PR opportunity VOIDED)
- `cb86836` verdict implications doc (needs §3.5 INVERSE addendum)
- `0cde63d` PF8.3 RUNTIME KILL (still valid; different cause from Task #43)
- `M_pf83_h1prime_static_scratch.md` Task #47 plan (still valid for PF8;
  add OOM-regression A/B gate)
- SKILL `kernel-optimization` v1.13.0 #38 (graduated this session)
- SKILL candidate #36 (now n=2 evidence — graduate next revision)
- SKILL `kernel-optimization` §0 SOLID rule 1 (推断 ≠ SOLID)

## §9 Status

**Closed** — Task #43 hypothesis DISPROVEN with inverse direction
(83fc5d0). Claude's dispatch-audit hypothesis is now a documented
self-correction case study. Skill candidate #36 reaches n=2 (grep
alone insufficient — need behavioral A/B too).

PF8 chain forward path adjusted: Task #47 H1' refactor remains valid
for PF8 specifically but cannot bundle Task #43 fix (different root
causes). H1' implementation needs OOM-regression A/B gate to
confirm adding PF8Scratch fields doesn't worsen Task #43 case.
