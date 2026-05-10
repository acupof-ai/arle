---
title: Task #43 verdict implications for PF8 chain forward path — pre-computed conditional next-round plan
date: 2026-05-10
type: research
status: open (pending Task #43 verdict; codex running scripts/task43_hypothesis_test.sh as of ~10:09 KST)
related_tasks: [#43 (codex in_progress), #44 (PF8 chain), #47 (H1' refactor)]
---

# Task #43 verdict implications for PF8 chain forward path

> **Purpose**: Task #43 is running NOW (codex dispatched per pickup
> queue P0). The verdict is highly informative for the PF8 chain
> (Task #44/#47) regardless of which way it goes. Pre-compute the
> conditional next-round plan so codex can pivot immediately when the
> verdict lands without re-discovery time.

## §1 The dispatch chain context

- `1ba06f0` (this session) Claude code audit: Task #43 W4A16 sustained-
  load failure may share root cause with PF8.3 KILL — both per-call
  alloc fragmentation when scratch fallback path is taken (linear.rs:
  2064-2095 + qwen3/forward.rs:312-313 dispatch evidence)
- `2cc608a` H1' design REVISION links Task #43 + Task #47 as
  "two-tasks-one-PR opportunity"
- `458394c` `scripts/task43_hypothesis_test.sh` 2-arm A/B (env on/off)
- `64d9b65` pickup queue dispatch log for this codex pickup

## §2 If Task #43 CONFIRMED (Arm A HEALTHY + Arm B SUBSTRATE-KILL)

**Strong implications:**

1. W4A16 sustained-load failure IS env-gated allocator fragmentation
2. Same fragmentation mechanism explains PF8.3 RUNTIME KILL
   (101380/101380 failures at conc=4)
3. Fix pattern is shared: extend `_with_scratch` variant to PF8 path
   per Task #47 H1' refactor (2cc608a) + ensure default routing
   uses scratch when available

**Forward path (codex pickup chain)**:

| Step | Action | Time |
|---|---|---|
| 1 | Codex commits Task #43 wins entry with verdict + arms data | 30 min |
| 2 | Codex picks up Task #47 H1' refactor (M_pf83_h1prime + 2cc608a REVISION) | 3-4 hours |
| 3 | Same PR: route any non-`_with_scratch` W4A16 callers through scratch variant (Task #43 fix) | included in step 2 |
| 4 | Re-bench W4A16 sustained-load to validate Task #43 fix | 30 min |
| 5 | Re-bench PF8 at conc 1+2+4 sustained-load to validate Task #47 H1' fix | 30 min |
| 6 | If both bench gates pass → close Task #43 + close Task #47 +
     update Task #44 status | 15 min |

**Skill candidate #38 reinforcement** (just graduated v1.13.0): if
H1' implementation properly clamps PF8Scratch sizing per #38, that's
n=3 evidence for the canonical anti-pattern. Watch for clamp/fallback
patterns in codex's H1' implementation.

**Skill candidate #36 watch**: if codex naturally greps marlin variant
family before designing PF8Scratch (per 2cc608a §2.3 self-audit
lesson) → n=2 evidence for "grep variants before designing from
scratch" candidate.

## §3 If Task #43 DISPROVEN (both arms HEALTHY)

**Implications:**

1. W4A16 sustained-load failure root cause is NOT env-gated scratch
   fragmentation
2. PF8.3 KILL hypothesis weakens (was anchored to same mechanism)
3. Need deeper investigation for both Task #43 + PF8.3

**Forward path (codex pickup chain)**:

| Step | Action | Time |
|---|---|---|
| 1 | Codex commits Task #43 errors entry with disproven verdict + arms data | 30 min |
| 2 | Both arms HEALTHY means scratch path works fine → re-investigate Task #43 with different hypothesis | unknown |
| 3 | PF8 chain Task #47 H1' design needs alternate justification (not "share Task #43 fix"); H1' may still be correct fix for PF8 specifically (per 0cde63d 101380/101380 failures + 57c37b5 H8 disproven) | implementation still 3-4 hours |
| 4 | OR pivot to Task #28 Medusa (per 63769be Alpaca recipe) since PF8 chain has weaker forward path without Task #43 confirmation | 2 hrs setup + 1 wk train |

**Skill candidate #36 strengthens**: "grep variants before designing"
becomes more valuable — design plans based on hypothesized shared
root cause should be validated cheaply BEFORE expensive
implementation.

## §4 If Task #43 AMBIGUOUS (both arms failed OR partial signal)

**Implications:**

1. Bench environment / measurement issue masks the real signal
2. Need to re-run with cleaner setup OR debug instrumentation

**Forward path**:

| Step | Action | Time |
|---|---|---|
| 1 | Codex inspects per-arm logs (`/tmp/task43-A-scratch-enabled.log`, `/tmp/task43-B-scratch-disabled.log`) | 15 min |
| 2 | Identify environmental issue (port conflict, GPU memory leak from prior run, etc.) | varies |
| 3 | Fix + re-run hypothesis test | 30 min |

**Skill #34b reinforcement**: AMBIGUOUS verdict triggers "check server
log first" discipline (per 868e147 `pf83_bench_health.sh` integration).

## §5 Bench v11 license decision intersection

User-runnable PF8.5 license decision (`bash scripts/pf85_bench_v11_user.sh`)
is INDEPENDENT of Task #43 outcome — bench v11 measures PF8 path at
conc=1 (where allocator fragmentation doesn't trigger; per 57c37b5
H8 verify). So bench v11 LICENSE/KILL outcome happens regardless of
Task #43 verdict.

But the COMBINATION matters:

| Task #43 verdict | bench v11 outcome | Implications |
|---|---|---|
| CONFIRMED | LICENSE | Strongest case for Task #47 H1' refactor — production conc≥2 fix needed AND validated mechanism |
| CONFIRMED | KILL | PF8 path KILLs at conc=1 (kernel-level issue beyond fragmentation); Task #47 H1' wouldn't help — pivot Task #28 Medusa |
| DISPROVEN | LICENSE | Task #47 H1' may still help PF8 conc≥2 (different mechanism than Task #43); proceed cautiously |
| DISPROVEN | KILL | Both PF8 + Task #43 mechanisms unclear; deeper investigation needed |

## §6 Recommended dispatch logic for next codex tick

```
codex_finishes_task_43():
    verdict = read_task_43_verdict()
    if verdict == "CONFIRMED":
        # Highest-ROI: Task #47 H1' refactor with Task #43 fix bundled
        if user_has_run_bench_v11():
            if pf8_licensed: pickup_task_47_h1_refactor_with_task_43_fix()
            else:           pickup_task_28_medusa_via_alpaca()
        else:
            # Wait for bench v11 OR pickup Task #48 W4A8 regression bisect
            pickup_task_48_w4a8_bisect()  # 1 hr, no user gate
    elif verdict == "DISPROVEN":
        # Task #43 needs new hypothesis; PF8 chain weakens
        if user_has_run_bench_v11():
            # Continue with H1' but flag as "speculative not Task #43 share"
            ...
        else:
            pickup_task_48_w4a8_bisect()  # safest pickup
    elif verdict == "AMBIGUOUS":
        debug_per_arm_logs()
        re_run_task_43()
```

## §7 Cross-references

- `1ba06f0` Task #43 hypothesis source
- `458394c` `scripts/task43_hypothesis_test.sh` (running now)
- `2cc608a` H1' design REVISION + two-tasks-one-PR
- `64d9b65` pickup queue §8 dispatch log
- `868e147` `pf83_bench_health.sh` (verdict tool)
- `ead46dc` `pf85_bench_v11_user.sh` (user PF8.5 license)
- `63769be` Medusa Alpaca cross-link (KILL pivot)
- `8b530ad` SKILL v1.13.0 #38 (canonical, watch for n=3 reinforcement)
- `M_pf83_h1prime_static_scratch.md` (Task #47 plan)

## §8 Status

**Pre-computed conditional plan** for next codex dispatch decision.
Awaits Task #43 verdict (~30 min). Whichever way verdict goes, the
next codex pickup is now zero-discovery-time.
