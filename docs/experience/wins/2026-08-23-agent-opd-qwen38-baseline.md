# Agent-OPD baseline: Qwen3.8-27B-FP8 on localized swe-smith

> Status: Baseline recorded — the pre-training anchor for OPD on this corpus

## Setup

`Qwen3.8-27B-FP8` served on one H20, driven through cc-harness (Claude Code as
the agent scaffold, the local serve as the model). 10 tasks sampled `seed=11`
from the 218 localized swe-smith tasks
(`scripts/filter_swesmith_localized.py`), 5-way concurrent, 1800 s per task.

Each task is validated before it counts: the pristine repo must FAIL its own
`fail_to_pass`. All 10 did, so none of these numbers come from a task that
passes without a fix.

## Result

```
invalid = 0/10
edited  = 10/10
solved  = 4/10
partial = 0.617      mean fraction of fail_to_pass passing
secs      median 899, max 1800
```

| task | turns | passed | fraction |
|---|---|---|---|
| sqlparse `eywg3gkn` | 8 | 43/43 | 1.00 |
| sqlparse `es0exs2h` | 29 | 13/13 | 1.00 |
| flake8 `class_rm_base__z0gd60hx` | 37 | 8/8 | 1.00 |
| sqlparse `tac71rdk` | 41 | 53/53 | 1.00 |
| sqlparse `sbt2ztom` | 25 | 7/8 | 0.88 |
| flake8 `ypch9g8g` | 10 | 3/4 | 0.75 |
| flake8 `ycnuq1fj` | 19 | 17/38 | 0.45 |
| flake8 `ar9mwpdm` | 26 | 1/20 | 0.05 |
| flake8 `7e1ipwsu` | 10 | 1/22 | 0.05 |
| flake8 `remove_assign__5bz2ejw2` | 22 | 0/19 | 0.00 |

A continuous spread from 0.00 to 1.00 with nothing saturated at either end:
usable gradient, which the NVFP4 base could not produce at all
(`errors/2026-08-23-nvfp4-tool-calls-corrupt.md`).

## Reading it

**`partial`, not `solved`, is the metric to track.** 1/22 and 0/22 are
different signals to a policy-gradient update; `solved` scores both as zero and
throws that away.

**One run is not a baseline.** `7e1ipwsu` returned 3/22 serially and 1/22
concurrently — same model, same task, sampled rollouts. Comparing a trained
model against a single-sample anchor will read noise as effect. Average several
samples per task before claiming a delta.

**Two tasks ran to the 1800 s cap** (`sbt2ztom` at 0.88, `z0gd60hx` at 1.00 on
the line). A cap truncates a rollout mid-fix and the result reads as weak
capability rather than a short budget. Raised to 3000 s.

## What had to be fixed to get a real number first

Three separate faults each produced a plausible-looking result that was not
about the model:

1. **Scoring against a poisoned import.** An agent ran `pip install -e .`
   inside a rollout; the `.pth` landed in the shared site-packages and every
   later `import flake8` on the box resolved to that leftover tree. Six tasks
   scored 6/6 "solved" and the pristine repos passed their own tests. Fixed in
   `sandbox.rs::workdir_pythonpath` plus `PIP_REQUIRE_VIRTUALENV` on the agent.
2. **`git diff` on a repo with no `.git`.** The staged repos carry no git
   metadata, so the edit detector returned 0 for every task.
3. **`wc -l` on a file with no trailing newline**, which made `pass >= f2p`
   true by construction and every task "solved".

## Rule

Validate the control arm per task, not per suite. "The pristine repo fails its
own tests" is one cheap command and it is the only thing separating a reward
signal from a number that would look the same if the model did nothing.
