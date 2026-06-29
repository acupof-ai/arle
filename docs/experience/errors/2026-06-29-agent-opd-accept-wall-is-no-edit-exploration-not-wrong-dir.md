# Agent-OPD 0-accept wall is no-edit exploration, NOT the wrong-dir/staging bug

## Context

The frozen-prompt-KV writeback payoff was blocked behind a rollout-accept wall:
the agent-OPD train rollout on `ansible__ansible-f327e65` produced **0 accepts**,
so the frozen-KV writeback never fired and the payoff stayed unmeasurable. The
SAME task got 4/4 accepts earlier (`run_close`, `run_final`). The leading
hypothesis (from a prior run, `run_fkv5`) was a **staging/sandbox bug**: sample 0
had `cd`'d into a SIBLING task's repo (`/host/aopd_work_fkv5/ansible__ansible-0ea40e0`)
from turn 0 and wasted all 30 turns there, leaving its own diff empty.

Latest HEAD (`017a495c`, binary built on pod with the
`ARLE_OPD_WRITEBACK_FROZEN_PROMPT_KV` symbols verified present), generous budget:
`--max-turns 30 --max-tokens 4096 --samples-per-prompt 4 --rollout-temperature 1.0`,
GPU 4, KV BF16 (not FP8). Decoded the full per-sample trajectory (case-as-fact).

## Root Cause

The wrong-dir `cd` was a **one-off sampling artifact**, not the wall. On the
latest direct run it did NOT recur — and 0 accepts persisted anyway:

- **Sample 0 (30 turns, ALL in the correct workdir, ALL relative paths):
  0 edits.** It found the right file early (turn 14-16:
  `lib/ansible/galaxy/collection/__init__.py`, `validate_collection_name`), then
  WANDERED — turn 17-23 chasing an irrelevant `display.py`, turn 24-26 reading
  `lib/ansible/galaxy/api.py` and hallucinating a totally unrelated `__lt__` bug
  ("Found it! line 303..."), turn 28-29 drifting to `role.py`. It NEVER emitted a
  `replace` or `write` across all 30 turns → empty diff → no accept. The model
  gets lost in an exploration loop and never commits a fix.
- **Samples 1-3: `turns=0, Stop`** — emitted zero tool calls and stopped
  instantly (a separate empty-response degeneracy).

Tool-call tally for sample 0 across 30 turns: `function=bash` ×18,
`function=read` ×9, `function=grep` ×1, `replace`/`write` ×**0**. Every call —
including the 9 `read` and the 1 `grep` — used `<parameter=command>`. But the
`read` tool's schema is `path`/`start`/`end` (not `command`) and `grep` is not a
defined tool, so those 10 calls silently no-op'd; the model only ever saw output
from the 18 `bash` calls. The student treats every tool as a shell command and
never reaches the edit tools.

So the accept wall is a **rollout-policy degeneracy** (explore-forever / no-edit /
instant-stop), reproducible at temperature 1.0 on this task at this checkpoint —
NOT a staging bug and NOT FP8 (KV was BF16, verified `executor.rs:143`
Auto→Bf16). The earlier 4/4 accepts vs now-0 is sampling/checkpoint variance on a
single-task best-of-4, not a regression with a code root cause.

## Fix

Two parts:

1. **Hardened the latent reachability hole** (it WAS real, just not the dominant
   failure this run): all per-task workdirs are flat-siblings under one
   `--work-root`, and the eval pass leaves its 3 staged dirs there before the
   train rollout, so a hallucinated absolute `cd /abs/other-task-repo` resolves
   to a real foreign repo. Added `cd_escape_message()` in
   `crates/train/src/sandbox.rs`: `run_bash` now rejects a bash command that
   `cd`s to an absolute path escaping the workdir (chained `&& cd /abs` caught
   too), mirroring the `resolve()` jail the read/write/replace tools already
   enforce and the "use relative paths only" rule the system prompt already
   states. Relative `cd subdir` and absolute `cd` that stays inside the workdir
   remain allowed. Unit test `bash_rejects_absolute_cd_into_sibling_task`
   reproduces the exact decoded escape vector.

2. **Reported, not yet fixed (harness, separate change):** the student mis-formats
   `read` as `<parameter=command>` and never emits `replace`/`write`. The real
   accept wall needs a rollout-policy nudge (stronger edit-or-finish system-prompt
   pressure, a turn-budget "you must edit by turn N" reminder, and/or a louder
   tool-schema error when `read` gets no `path`), not a sandbox change. Left for a
   follow-up so this fix stays surgical.

## Rule

A wrong-dir / staging "bug" observed in ONE run is a case to decode, not a
structural root cause — re-run the latest directly and read the per-turn
trajectory before attributing. Here the wrong-dir `cd` was sampling variance; the
real 0-accept wall is the policy never emitting an edit tool call (explore-forever
+ instant-stop). Confirm the failure mode from the decoded tool-call tally
(`replace`/`write` count = 0) before changing the sandbox. Harden the latent
reachability hole regardless, but don't mistake it for the wall.
