# Agent task harness: goals, gates, execution, report

Date: 2026-09-09 · Status: Done — built and in use (scripts/agenda.py ledger + docs/agenda.jsonl, board page, hygiene gate); indexed from docs/index.md · Owner: ckl

## Problem

Four agent sessions execute in this tree at once. Before this harness, what
each one was doing, which goal it served, and what it concluded lived in three
places that do not join: the controller session's context, `SendMessage`
transcripts, and — only after the fact — a wins or errors entry. A lane that
stalled was invisible until someone asked. A task that closed on an opinion
rather than a measurement looked identical to one that closed on a number.

The roadmap (`docs/plans/2026-08-24-roadmap.md`) ranks work but records no
state, by design. Nothing held the state.

## Three layers, one join key each

```
goal  --.
        `-- task (owner, exit gate) --.
                                      `-- run (scripts/prereg.py)
```

| Layer | Holds | Written by | Ledger |
|---|---|---|---|
| Goal | the outcome and the one metric that measures it | the controller | `docs/agenda.jsonl` |
| Task | what one agent executes, and the event that closes it | the controller, updated by the owner | `docs/agenda.jsonl` |
| Run | one measurement, hypothesis written before it starts | the executing agent | `docs/experience/prereg.jsonl` |

A task lists the run names it spawned; `report` joins the two files by name.
The wins or errors entry stays where it is and is linked from the closing row,
so the evidence corpus is unchanged.

## The exit gate

A task carries an `exit`: the named, measurable event that closes it. Never a
date, and never a deliverable that cannot be false. The pattern comes from
tileRL's `docs/roadmap.md`, whose phases exit the same way.

`--exit` is required by `task add`, and `check_agenda_ledger` fails on a row
that lost it. Three shapes appear in the seeded ledger:

- **Comparative.** "ms/committed-token at c=16 beats the no-spec control on
  the same binary, needle ×3 in envelope."
- **Disjunctive**, when the task may end by falsifying its own premise. "A
  DSv4 prefill at 4K and 16K tokens either reproduces the i32 overflow, or the
  in-repair claim is deleted from `architecture.md` with an entry saying why."
  This shape matters: without it a lane that finds nothing to fix has no way to
  finish, and reports as stalled.
- **Existence with a property.** "L2 and L3 single-page cold-read p50/p99
  measured, stated as a ratio to one decode step."

## Decomposition

The controller writes goals and tasks; owners update and close them. One goal
per standing objective, at most a handful of live tasks per goal, one owner per
task. A task is sized to one agent's brief — if it needs two agents it is two
tasks.

Assignment and the brief stay in `SendMessage`; the ledger holds only what
must survive the session. A brief is long and situational; a row is short and
durable.

## Execution

The owner calls `task update` when something changes, and names the prereg run
it started:

```bash
scripts/prereg.py start --name c16-mtp --cmd "..." --hypothesis "..."
scripts/agenda.py task update --id mtp-batch --run c16-mtp --note "op_timing says ..."
```

`--status blocked` marks a lane waiting on something it does not control. A
blocked task does not age into a defect; an open or active one does, after 72
hours.

## Close

```bash
scripts/agenda.py task close --id mtp-batch --status done \
    --verdict "accepted, -32% at c=16" \
    --entry docs/experience/wins/2026-09-10-batched-mtp-verify.md
```

`--status killed` closes a task whose answer is that the work should not
happen. A killed task is a result; it needs a verdict and, where a measurement
produced it, an entry.

## Report

`scripts/agenda.py report` renders the three layers as one page and is the only
thing a human reads. It is derived — nothing is written to it by hand — so it
cannot drift from the ledger. `--out` writes it to a file for publishing.

## What the harness checks

`check_agenda_ledger`, inside `scripts/check_repo_hygiene.py`, imports the
ledger's own reader so the format has one definition, and fails on:

| Defect | Why it is one |
|---|---|
| task with no `exit` | it closes on an opinion |
| task naming a goal that does not exist | the join is broken |
| `done` with no entry | a claim with no evidence |
| `open` or `active`, untouched for 72h | a lane nobody is running |

`--selftest` builds a world for this check by running the real writer and
back-dating one row, so a drift between the writer and the reader turns it red.
Seven checks now prove they can fail.

## Deliberately absent

- **No progress percentages and no dates.** A task is open, active, blocked,
  done or killed. A percentage on work whose exit is a measurement is a guess
  presented as a number.
- **No priority field.** The roadmap ranks; the ledger records. Two ranking
  surfaces drift.
- **No auto-close.** A gate the harness evaluates would need to run every
  bench; the owner reads the number and closes the row.
- **No second store.** Goals and tasks share one JSONL because they are joined
  on every read.

## Files

| Path | Role |
|---|---|
| `scripts/agenda.py` | goal / task writer, defect reader, report renderer |
| `scripts/prereg.py` | run-level ledger, hypothesis before the run |
| `docs/agenda.jsonl` | goals and tasks |
| `docs/experience/prereg.jsonl` | runs |
| `scripts/check_repo_hygiene.py` | `check_agenda_ledger`, `check_prereg_no_stale_running`, and the selftest worlds for both |
