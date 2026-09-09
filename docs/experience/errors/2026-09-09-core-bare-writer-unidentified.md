# core.bare keeps flipping true on the shared config; the writer is unidentified

## Context

`core.bare = true` has appeared on the shared `.git/config` at least six times
over two days. Every worktree then fails with `fatal: this operation must be
run in a work tree`, which stops all four lanes at once. It recurs during
pushes, and the sessions doing the pushing each report they did not write it.

## What was believed, and why it is wrong

The accepted explanation was that git exports `GIT_DIR` and `GIT_WORK_TREE` to
hooks, so a `git init` in a subprocess below the hook reinitializes the real
repository as bare. Three fixes shipped on that theory: `disown_ambient_git()`
in `scripts/check_repo_hygiene.py`, and `unset GIT_DIR GIT_WORK_TREE` at the
top of `scripts/tests/test_validate_release.sh` and
`scripts/tests/test_pod_flow.sh`.

The theory does not hold on this repository. A probe hook that dumps its own
environment during `git push --dry-run` receives exactly three git variables:

    GIT_EDITOR=true
    GIT_PREFIX=
    GIT_EXEC_PATH=/Applications/Xcode.app/Contents/Developer/usr/libexec/git-core

There is no `GIT_DIR` and no `GIT_WORK_TREE`, so nothing below the pre-push
hook can inherit them. The three fixes are harmless and they did not address
the cause; core.bare flipped again after all three had landed.

## What is reproducible

On git 2.50.1 (Apple Git-155), `git init` writes `core.bare = true` into an
existing repository when `GIT_DIR` names it **with a trailing slash**:

| Environment for `git init` | Resulting core.bare |
|---|---|
| `GIT_DIR=/path/v/.git` | false |
| `GIT_DIR=/path/v/.git/` | **true** |
| `GIT_DIR=/path/v/.git GIT_WORK_TREE=/path/v` | false |
| `GIT_DIR=/path/v/.git git init --bare .` | false |
| `GIT_DIR=/path/v/.git git clone --bare` | false |
| `GIT_DIR=/path/v/.git git worktree add` | false |

git decides bare-vs-not by comparing the `GIT_DIR` basename to `.git`; a
trailing slash makes the basename empty, the test fails, and the repository is
initialized bare. This is a real mechanism, but no caller in this tree is known
to build a `GIT_DIR` with a trailing slash, and the pre-push hook does not
receive `GIT_DIR` at all.

## Cause

Unknown. The trailing-slash mechanism above is reproducible in isolation and is
not shown to be the writer here.

## Detector, not a fix

`.githooks/pre-push` now appends `enter`/`exit` lines with the observed
`core.bare` and the hook pid to `${TMPDIR}/arle-core-bare.log`, outside the
repository so a checkout cannot lose it. The next occurrence is then bracketed
to before the hook, inside the hook, or outside pushes entirely, which is the
one fact that separates the remaining candidates.

`scripts/lane.sh doctor` continues to repair the flag.

## Rule

A root cause is a mechanism reproduced in the failing environment, not a
mechanism that is plausible and reproducible somewhere else. Three fixes
shipped against a theory that a five-second probe of the hook environment would
have falsified. Probe the environment the failure actually runs in before
writing the fix.
