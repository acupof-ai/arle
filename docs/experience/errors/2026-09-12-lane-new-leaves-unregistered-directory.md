# `lane.sh new` can leave an unregistered directory that blocks retries — cause unknown

Date: 2026-09-12. Observed during the `dspark-c8-branch-decider` work.

## Context

`scripts/lane.sh new <name>` is the only documented way to make a lane.
Its new-lane body is three steps (`scripts/lane.sh:62-66`):

```bash
share_target
path="$LANES/$name"
[ -e "$path" ] && { echo "lane: $path already exists" >&2; exit 2; }
git -C "$ROOT" fetch --quiet origin main 2>/dev/null || true
git -C "$ROOT" worktree add -b "lane/$name" "$path" main
```

There is no trap cleaning `$path` up if the `git worktree add` fails
partway, and no check afterward that the directory is a registered
worktree. `set -euo pipefail` exits on the failed command, leaving whatever
on-disk state git reached before failing.

## Observed state

Command run from the main checkout (output piped to `tail -3`):

```
scripts/lane.sh new dspark-c8-branch-decider 2>&1 | tail -3
```

It printed the normal trailing help lines, so it looked like success. Two
files were then written into `$LANES/dspark-c8-branch-decider/`. The next
git command in that directory failed with `fatal: not a git repository`.
Inspection showed all three at once:

- `$LANES/dspark-c8-branch-decider/` existed with the two newly written
  files, and **no `.git` file**;
- `git worktree list` did not contain the path;
- no `lane/dspark-c8-branch-decider` branch existed.

So the directory passed the `[ -e ]` precheck on every later attempt while
being neither a worktree nor removable by `git worktree remove`.

## Recovery

Move the written files aside, recreate the lane cleanly, move the files
back, delete the stranded directory:

```bash
mv $LANES/dspark-c8-branch-decider /tmp/c8-stranded
scripts/lane.sh new dspark-c8-branch-decider   # now verifies registered
cp /tmp/c8-stranded/... $LANES/dspark-c8-branch-decider/...
rm -rf /tmp/c8-stranded
```

No committed work was lost because the files were uncommitted and had just
been written; had the failure happened after an hour of edits, the path
precheck would have blocked the only documented recovery and the files
would read as part of a worktree that does not exist.

## Root cause

Cause unknown. One observation narrows it: `lane.sh new` printed its trailing
"Cost is ~1.5 GB" lines, which under `set -e` can only happen after
`git worktree add` returned 0 — so the worktree was registered at creation
and the registration (and the working dir's `.git` link) disappeared before
the diagnosis roughly half an hour later. A concurrent worktree removal or
`git worktree prune` by another session is the best fit (a bulk cleanup of
stale worktrees was in progress that session), but it was not observed and
is not proven; git's `prune` removes administrative metadata without
deleting the working directory, which matches the stranded files but not the
missing `.git` link. The trigger is left unassigned. The script defect is
independent of it: `new` reported "ready" without verifying the directory
was a registered worktree and had no way to detect or recover this state.

## Fix

Make `new` verify its outcome and run the same cleanup on three failure forms:

1. nonzero return from `worktree add`;
2. zero return but the path is not a registered linked worktree afterward
   (the observed form — `.git` link removed, directory present);
3. zero return but the directory is gone entirely, so the verification `cd`
   itself fails.

In every case remove the fresh name-validated directory, delete the branch
only when it did not pre-exist, and prune. Form 3 is the subtle one: a bare
`real_path="$(cd "$path" && pwd -P)"` under `set -e` aborts the script
BEFORE the rollback, leaving the branch behind — and then the name is
permanently unusable because the retry fails `-b` on the branch and treats
it as pre-existing. The `cd` and the `rev-parse` both run inside guarded
conditions (`if … ; then` / `|| true`), so a missing path flows through to
cleanup with `registered=0` instead of aborting.

The verification cannot be
`git -C "$path" rev-parse --is-inside-work-tree`: a stranded directory inside
the main checkout resolves to the PARENT repository and answers true. It
compares `pwd -P` of the path against
`git -C "$path" rev-parse --show-toplevel` — a linked worktree is its own
top level, the stranded directory reports the main checkout's. The `rm -rf`
target is safe because the pre-existing `[ -e "$path" ] && exit 2` precheck
proved the path absent immediately before the add, so cleanup can only
remove what the add created; the charset validator and `--` are secondary
defense, not the load-bearing one. A registration deleted by another session
after "ready" is a shared-worktree hygiene issue and is not preventable from
inside `new`.

The rollback is committed as a checked test,
`scripts/tests/test_lane_new_rollback.sh`, registered in
`scripts/pre_push_checks.sh` (the hook runs a hardcoded list, not a glob).
Five cases in throwaway repos, 19 assertions:

1. happy path registers a live worktree;
2. nonzero add (failing `post-checkout` hook): no dir, branch, or worktree;
3. the observed zero-but-unregistered form — a git wrapper removes the new
   worktree's `.git` link and prunes between add and check (a stand-in; the
   real race cannot be forced) — fully rolled back, and the name is reusable;
4. zero add with the directory removed entirely (verification `cd` fails):
   the branch is still deleted and the name is reusable — the regression
   guard for the form-3 `set -e` hole;
5. a pre-existing diverged `lane/<name>` branch is preserved (never deleted)
   while the stray directory is removed.

## Rule

A script that creates a directory and then runs a command that can fail
partway needs an outcome check and cleanup, and the check must itself be
safe under `set -e` — a verification step that aborts before the cleanup is
the same defect reached from another direction. The success-looking trailing
output is not evidence the registration happened; a checked, hook-run test
is what keeps it from regressing.
