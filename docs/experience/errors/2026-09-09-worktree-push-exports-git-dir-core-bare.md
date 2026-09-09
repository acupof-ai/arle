# core.bare kept flipping true on the shared config: only worktree pushes export GIT_DIR

## Context

`core.bare = true` appeared on the shared `.git/config` at least seven times
over two days. Every worktree then fails with `fatal: this operation must be
run in a work tree`, which stops all four lanes at once. It recurred during
lane pushes, and every session doing the pushing reported it had not written
it.

## Root cause

A push from a **worktree** exports `GIT_DIR` to the pre-push hook. A push from
the **main checkout** exports none. Measured with a probe hook that dumps its
own environment during `git push`:

| Push from | GIT_ variables the hook receives |
|---|---|
| main checkout | `GIT_EDITOR`, `GIT_PREFIX`, `GIT_EXEC_PATH` |
| worktree | the same three, plus `GIT_DIR=<repo>/.git/worktrees/<name>` |

The basename of that path is the worktree's name, not `.git`. git decides
bare-vs-not by comparing the `GIT_DIR` basename to `.git`, so a `git init` in
any subprocess below the hook reinitializes the shared repository **as bare**.
Reproduced end to end on git 2.50.1 (Apple Git-155):

    GIT_DIR=<repo>/.git/worktrees/wt  git init   ->  core.bare = true
    GIT_DIR=<repo>/.git               git init   ->  core.bare = false
    GIT_DIR=<repo>/.git/              git init   ->  core.bare = true   (empty basename)

Adopting lane worktrees on 2026-09-09 is what turned a dormant condition into a
daily outage: before that, every push came from the main checkout, where the
hook receives no `GIT_DIR` at all.

## The wrong turn

An intermediate finding recorded this cause as unknown, on the strength of a
probe hook that showed no `GIT_DIR`. That probe ran in the **main checkout** —
the one place where the variable is absent. The three fixes already shipped
against the inheritance theory (`disown_ambient_git()` in
`scripts/check_repo_hygiene.py`, and `unset GIT_DIR GIT_WORK_TREE` at the top
of `scripts/tests/test_validate_release.sh` and `scripts/tests/test_pod_flow.sh`)
were correct, and calling them ineffective was wrong.

## Fix

`.githooks/pre-push` unsets every `GIT_*` variable before running anything.
That closes the class at the boundary, so it no longer depends on each script
that stages a temp repository remembering to do it. The checks below discover
the repository from the working directory, which is what they want anyway.

`scripts/tests/test_hook_disowns_git_env.sh` guards it, negative control first:
a `git init` under a worktree-shaped `GIT_DIR` must set `core.bare = true`, and
the same sequence with the hook's disown block must leave it `false`. Wired
into the pre-push shell lane and `ci.yml`.

The hook also appends `enter`/`exit` lines with the observed `core.bare` and its
pid to `${TMPDIR}/arle-core-bare.log`, outside the repository.

## Rule

Run the probe in the environment that fails, not in the environment that is
convenient. The failing pushes all came from worktrees; the probe ran in the
main checkout, produced a clean result, and that clean result was used to
retract three correct fixes. A negative probe is only evidence when the probe
could have seen the positive.
