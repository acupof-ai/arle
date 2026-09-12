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

Make `new` verify its outcome and run the same cleanup on both failure forms:

1. nonzero return from `worktree add`;
2. zero return but the path is not a registered linked worktree afterward
   (the observed form).

In both cases remove the fresh name-validated directory, delete the branch
only when it did not pre-exist, and prune. The verification cannot be
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

Verified in a throwaway repo: happy path; duplicate name; pre-existing
diverged lane branch (kept, stray dir removed); forced nonzero add via a
failing post-checkout hook; and the observed zero-but-unregistered form,
simulated with a git wrapper that removes the new worktree's `.git` link and
prunes between the add and the check (a stand-in for the real race, since
the race could not be forced). The fifth case leaves no directory, branch, or
worktree entry and exits 1.

## Rule

A script that creates a directory and then runs a command that can fail
partway needs an outcome check and cleanup: the success-looking trailing
output is not evidence the registration happened. Verify with
`git -C "$path" rev-parse --is-inside-work-tree` and `git worktree list`
before reporting "ready".
