# Rubric primitives + Rubricator — train, 2026-08-25

> Status: pending-remote

## Goal

#98 host portion: the rubric judge primitives (parse/select) get cpu-lane
test coverage, and the Rubricator (rubric induction from contrasting
teacher/student samples) lands as pure host logic. The rubric-OPD
orchestration (cuda-gated, rubric_opd.rs) and the multi-seed training gate
are pod-only.

## What landed

- `Rubricator` (rubric.rs): `induce_prompt(problem, teacher_samples,
  student_samples)` renders the induction prompt; `parse_induction(output)`
  parses the JSON rubric (task + criteria with kind), rejecting empty or
  malformed inductions. Reuses `last_json_object`.
- 9 cpu-lane unit tests covering: verdict parse (accept/reject/process-only/
  malformed/missing-key), best-of-N selection (accepted/distinct/parse-errors),
  and Rubricator parse (valid/empty/malformed).

The existing primitives (`judge_prompt`, `solve_prompt`, `parse_verdict`,
`select`, `select_by_self_consistency`) were already in tree — this tranche
adds the missing Rubricator and the test gate.

## Parameters

```bash
# pending-remote: multi-seed ≥5 OPD training runs on H20
# - held-out lift on ≥2 open-ended dims (IFEval + agentic skill task)
# - reward-hack tripwire (in-loop score rises while held-out falls → revert)
```

- Baseline: (no Rubricator, no rubric tests)
- Treatment: this commit
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod (pending-remote)

## Rule

A rubric induction that rejects empty/malformed output is a retry, not a
silent default — the parse never returns a partial rubric.

## Correction — 2026-09-11

`Rubricator::induce_prompt` had zero callers and was deleted in #356;
`Rubricator` + `parse_induction` are retained as `#[cfg(test)]` (used only
by the three induction parser tests). The training binary never emits
induction prompts.
