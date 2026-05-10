---
title: "Default-on with explicit escape hatch" matched-control discipline — codex Task #35 evidence point
date: 2026-05-10
type: research
status: open (evidence point for future SKILL kernel-optimization v1.13.0+ candidate, not sedimenting yet — single-occurrence evidence)
related_tasks: [#35 (cap=8 prefill warmup, codex in_progress)]
---

# "Default-on with explicit escape hatch" matched-control discipline — codex Task #35 evidence point

> **Purpose**: capture the discipline pattern codex demonstrated during
> Task #35 cap=8 prefill warmup implementation as a single evidence
> point. Not sedimenting into SKILL.md as anti-pattern yet — single
> occurrence is insufficient (per skill v1.7.0+ §accumulation rule:
> wait for n=2-3 evidence points across distinct sessions before
> codifying).

## §1 The pattern

Codex implementing Task #35 cap=8 prefill warmup (Pass 3 added on top
of existing Pass 1 + Pass 2) reasoned:

> 这个改动只加了默认开启的 warmup pass 和显式关闭开关；我再跑一次目标
> greedy，避免把已知 W4A8 accuracy failure 混进本次验证结论。
>
> [translation]: This change adds a default-on warmup pass and an
> explicit-off switch; I'll re-run the targeted greedy to avoid mixing
> the known W4A8 accuracy failure into this verification's conclusion.

Concretely codex added:

- **Pass 3**: cap=8 prefill warmup, default-on (per M_warmup directive)
- **`INFER_PREFILL_WARMUP=0` escape hatch**: explicit opt-out env var
- **`docs/environment.md` documentation**: documents the escape hatch

This enables single-binary single-variable A/B for the cold-start
bench:

```
baseline:  same binary, INFER_PREFILL_WARMUP=0 → Pass 1 + Pass 2 only
treatment: same binary, default            → Pass 1 + Pass 2 + Pass 3
```

vs. the alternative (without escape hatch):

```
baseline:  build N binary, no Pass 3 in source → Pass 1 + Pass 2
treatment: build N+1 binary, with Pass 3       → Pass 1 + Pass 2 + Pass 3
```

The without-escape-hatch path is multi-variable (binary identity +
feature behavior) — different binaries may have different inlining /
codegen / cache layouts even at the same Pass 3 default. The
with-escape-hatch path holds binary identity constant.

## §2 Why this is matched-control discipline

Per SKILL kernel-optimization v1.12.0 mantra rule 3 ("Single A/B
variable, matched controls") + Phase 5 matched-control checklist,
single-variable A/B requires holding all but one variable constant.

Codex's escape-hatch design isolates the Pass 3 variable from the
"compile-time vs runtime" variable. This is a stronger matched control
than the simpler "rebuild without feature" approach.

Adjacent example (skill anti-pattern #2): "Multi-variable change → can't
attribute" caught by `M_b.2.2 split-KV opt-in changed both kernel +
path + format simultaneously, regression couldn't be bisected." The
escape-hatch pattern is the prophylactic to that anti-pattern: build
the opt-out into the design from the start.

## §3 Why not codify yet

Per skill accumulation policy (informal but consistent across v1.5.0+
revisions):

- **Single evidence point** is insufficient. The pattern needs to recur
  across at least 2-3 distinct sessions / contexts before sedimenting
  as an anti-pattern. This avoids cluttering SKILL.md with
  one-off-isn't-it-nice patterns that don't generalize.
- **Better wait for the contradicting case**. The escape hatch costs:
  +1 env var maintenance, +1 doc entry, +1 surface for misuse (e.g.
  someone sets `INFER_PREFILL_WARMUP=0` in production thinking it's a
  perf flag). If a future session shows the cost > benefit on a
  different feature, the rule needs nuance ("escape hatch when... but
  not when...").
- **Skill v1.13.0+ #35 candidate already proposed** (W4A8 regression
  evidence in `e3e1ab5`): "Tasks closed `root cause TBD` need a
  regression test or canary assertion." Adding a second candidate from
  same session loses evidence-distinction signal.

## §4 Watch-list for n+1 evidence

Future sessions where escape-hatch discipline matters:

- **PF8.5 license decision**: `INFER_MARLIN_W4_FP8_PREFILL=1` is an
  opt-in env var (not default-on with escape-hatch), so the bench v11
  baseline is legitimately a different binary path (PF8 INACTIVE) vs.
  treatment (PF8 ACTIVE). Different design choice — opt-in instead of
  default-on-with-opt-out — but still gives single-variable A/B
  because the env var toggles between the two paths in the same
  binary.
- **H1' static-scratch refactor (Task #47)**: when codex implements
  per `M_pf83_h1prime` plan §4, consider whether PF8Scratch eager-init
  should be default-on with `INFER_PF8_SCRATCH_LAZY=1` opt-out, or
  default-lazy with `INFER_PF8_SCRATCH_EAGER=1` opt-in. Default-on
  with opt-out matches Task #35's pattern; lazy-with-opt-in matches
  PF8.4 dispatch's pattern.
- **Future Task #28 Medusa scaffold**: spec decode is naturally an
  opt-in path (different scheduler behavior + draft model loading).
  Default-on with opt-out would surprise users not running spec.

When 2 of those 3 land (or any other escape-hatch-style design choice
gets explicitly documented), this evidence point + the new ones can
sediment as SKILL v1.x.0 anti-pattern.

## §5 Cross-references

- Task #35 cap=8 prefill warmup (codex in_progress, this evidence)
- `M_warmup-prefill-pass-directive.md` (the directive being implemented)
- `2026-05-10-warmup-pass2-vs-current-state-reconciliation.md` (`58b0ac1` reconciliation)
- SKILL `kernel-optimization` mantra rule 3, Phase 5 matched-control
  checklist, anti-pattern #2 (`b551bea`+ `b96a1e7`)
- `e3e1ab5` W4A8 regression research (the OTHER skill v1.13.0+ #35
  candidate from this session)
- `0be7220` SKILL v1.12.0 (current latest)

## §6 Status

Open. Single evidence point. Awaiting n+1 occurrence in distinct
session before SKILL sedimentation. NOT actionable this session — pure
documentation for future-self.
