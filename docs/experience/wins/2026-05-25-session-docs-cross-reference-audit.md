# Session Docs Cross-Reference Audit

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T13.

## Context

The 2026-05-24/25 OPD mainline session shipped many small commits from
`fdb021c` onwards: BBuf skill distillation, OPD CLI wiring, T3 deletion
cleanup, T4a/T5a code-only patches, T6/G6 validation, T7/T11 research/design,
T8/T9 cleanup, T10/G5 wireframe, and T12 capability-eval preflight.

The issue was not missing implementation evidence; it was navigation drift.
Several wins/errors/research entries were locally correct but did not point
back to the backlog, pivot, or gap-analysis docs, while top-level
`docs/index.md`, `ROADMAP.md`, and `CHANGELOG.md` still pointed readers at the
2026-05-15 status snapshot.

## What Worked

- Added `Related:` pointers to the session wins/errors/research entries that
  lacked a direct plan/project anchor.
- Added a session artifact ledger to
  `docs/projects/2026-05-24-opd-mainline-task-backlog.md`, mapping
  `fdb021c`→HEAD commits to their plan/project anchor and durable artifact.
- Refreshed `docs/index.md` so the OPD mainline backlog and OPD-only pivot are
  visible as active truth surfaces without deleting the DSv4 status snapshot.
- Refreshed `ROADMAP.md` P4 to point at the OPD mainline queue, capability
  eval preflight, and GPU-deferred gates.
- Added an Unreleased changelog note for the OPD mainline queue transition,
  code-only KL/KV work, SFT anchor corpus attribution, and capability-eval
  preflight.

## Verification

```bash
git diff --check
```

- Exit 0.
- Inline Python local-link check covered 18 changed Markdown files; all local
  Markdown links resolved.
- Scope is docs-only. No GPU, no server process, no P5 checkpoint mutation.

## Rule

High-velocity task queues need a commit-to-artifact ledger before the session
ends. Every wins/errors/research entry should name the plan/project anchor that
licensed it, and top-level docs should point to the live queue when execution
moves materially.

## Verdict

PASS. T13 is a docs-only cross-reference cleanup; no runtime benchmark is
required.
