# Synthetic small-repo bug-fix corpora cannot produce a 27B capability curve — KILLED

## Context

The agentic-OPD capability-curve plan
([plans/2026-07-03-agentic-opd-27b-capability-curve.md](../../plans/2026-07-03-agentic-opd-27b-capability-curve.md))
needed a held-out task suite where the untrained 27B Qwen3.6-FP8 lands at a
10–40% pass-rate so an RFT curve has dynamic range. Real SWE-bench-Pro was
rejected up front (decoded 0-accept wall,
[errors/2026-06-29](2026-06-29-agent-opd-accept-wall-is-no-edit-exploration-not-wrong-dir.md)),
so a 36-task synthetic corpus was built: single injected bug per small Python
package, SWE-Pro schema, hidden pytest reward (`scripts/gen_agent_opd_tasks.py`).

## Root Cause

**Five escalations, five saturated baselines** (all greedy, n=24 held-out,
single H20, pod logs under `/host/aopd_curve/`):

| Escalation | Baseline |
|---|---|
| easy (repro + named module, 3 files) | 8/8 |
| hard v1 (prose symptom, 11 files) | 24/24; envelope over 3 repeats 22–24/24 |
| hard v2 (+ neutral docstrings, + 18 gold-module scenery, ~17–24 files) | 24/24 at ~9 s/task |
| turn budget 3 / 2 | 22/24 / 20/24 |
| turn budget 1 | 0/24 — only blind-edit wins exist; not a band, a cliff |

Two structural reasons, each sufficient:

1. **Classic single-line bugs are pattern-matched, not localized.** Inverted
   comparison / off-by-one / mutable default / `set()` order loss /
   string-compare versions are canonical idioms a 27B fixes on sight; v1 even
   self-annotated them (the docstring stated correct behavior directly above
   the contradicting line), but v2's neutral docstrings changed nothing —
   the function *name* plus symptom prose already pins the fix.
2. **The read→edit loop completes in 2 turns on any small repo**, so no
   turn-budget band exists between "impossible" (1 turn ⇒ must edit blind)
   and "saturated" (2 turns ⇒ 83%). Scenery volume (11→24 files) did not
   move a single held-out task.

The capability gap the curve was meant to show lives at REAL-repo scale
(ansible: explore-forever, tool-schema misuse, 0 accepts) — a regime where
RFT has **no gradient** (0 accepts ⇒ nothing to write back). Toy corpora
can't reach that regime by construction: any task small enough to stage
cheaply is easy enough to one-shot.

## Fix

Kill the toy-corpus capability lane. The licensed path is the plan's G6:
**teacher-in-the-loop agentic OPD on real SWE-Pro instances** — a think-on
teacher (same 27B weights, thinking enabled; zero extra VRAM, precedent:
the 2026-06-20 4B BFCL think-on win) generates rescue trajectories on
0-accept tasks, distilled via the existing masked-CE writeback. Prerequisite:
stage ≥12 real SWE-Pro instances (pod has 1 train + 3 eval ansible
instances, 269 MB, from the 06-27..30 campaign). The synthetic corpus and
harness stay: corpus as the smoke/regression gate (self-check + loop
stability), harness unchanged for the real lane (same schema).

Salvage from the campaign: the 16-round full run on the saturated corpus is
a clean **stability** read (loss 0.376→falling, pass-rate flat at baseline
0.958, zero OOM after the tape-margin fix) — wins entry, not a capability
claim.

## Rule

- A capability-curve corpus needs a **measured mid-band baseline before any
  training run** — probe the untrained model first; do not infer difficulty
  from task design (five design-level "this should be harder" judgments in
  a row were wrong).
- Difficulty axes that don't bind a strong model: surface cues (docstrings,
  repro snippets), scenery volume, turn budgets on small repos. The axis
  that binds is task regime (real-repo scale, spec-driven fixes) — and once
  there, rejection sampling alone starves (0 accepts); plan the teacher
  before the corpus.
