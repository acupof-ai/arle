# bench_ab.sh --vs-no-spec: built-in control arm — tooling, 2026-09-09

> Status: Shipped (Mac syntax and unit checks; bench tooling, exempt from the
> bench-entry requirement per CLAUDE.md hard gates)

## Context

Spec-decode A/Bs need a no-spec control arm. The previous flow ran two
`bench_ab.sh` invocations and typed the spec flags twice, so the arms could
drift, and a missing arm's data could still produce a diff table with
ratio-shaped cells.

## What Worked

`scripts/bench_ab.sh` gains `--vs-no-spec`: given one treatment command, it
derives the control by stripping the spec flag set (`--spec-type`,
`--mtp-draft-model`, `--dspark-sps-bias-ms`, `--dspark-sps-row-ms`,
`--dspark-block-size`, `--dspark-markov-init`, each with its value). One
invocation runs control then treatment and emits the side-by-side diff.

Guards:

- The strip is validated: a treatment command with no spec flag exits 2
  instead of running two identical arms.
- The diff step exits non-zero when either arm's JSON is missing, so a ratio
  is never printed from one arm.

Checks run on Mac: `bash -n` syntax; the strip function against flag-bearing
and flag-free commands; argument-validation exit codes. No GPU involved.

## Rule

A comparison harness derives its control arm from the treatment by deletion,
never by re-typing: the flags that differ are then exactly the flags the
harness stripped, and a treatment with nothing to strip is an error.
