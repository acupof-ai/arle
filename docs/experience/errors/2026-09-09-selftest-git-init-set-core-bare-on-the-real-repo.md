# The hygiene selftest reinitialized the real repository — 2026-09-09

> Status: Fixed the same day (`disown_ambient_git`). Four separate breakages of
> the shared checkout, four sessions blocked, root cause found by lane D.

## Context

`scripts/check_repo_hygiene.py --selftest` landed this morning and was wired
into `scripts/pre_push_checks.sh`, CI and `make hygiene`. It builds one world
per check by copying the real artifact into a temporary directory and staging
it:

```python
root = Path(tempfile.mkdtemp(prefix="hygiene-world-"))
subprocess.run(["git", "init", "-q"], cwd=root, check=True)
```

Within the hour, the shared `.git/config` acquired
`core.bare = true` four times (16:58, 17:00, 17:03, 17:08). Every `git status`,
`git add` and `git commit` in the main checkout and in all four lane worktrees
then failed with `fatal: this operation must be run in a work tree`. Four
sessions were blocked; two pushes died mid-flight.

## Root Cause

**Git exports `GIT_DIR` and `GIT_WORK_TREE` to hooks.** `git init` run inside a
hook therefore does not initialize the directory it is standing in — it
reinitializes the repository `GIT_DIR` names. Because the world's temporary
directory is not that repository's work tree, git recorded the only thing
consistent with what it saw: `core.bare = true`.

So the selftest broke the repository it was defending, and only while running
inside pre-push, which is why it looked like an outside process.

The second half is worse and was never observed only because the first half
failed loudly. The checks under test call `git grep` and `git ls-files` with
`cwd=ROOT`, where `ROOT` is the world. With `GIT_DIR` still set, those commands
would have read the **real repository** instead of the world — so the broken
world would have looked clean, and the selftest would have reported that a
check cannot fail when in fact it was never pointed at the broken artifact.

## What made it hard to find

Every lane denied it, correctly: none of them ran a git command that writes
config. `.githooks/pre-push` is 12 lines and does not touch config. Direct
measurement cleared `git worktree add`, `list`, `prune`, `fetch` and
`rev-list`, from both the main checkout and a linked worktree. `chmod a-w` on
`.git/config` did not help either, because git writes `config.lock` and renames
it, which needs write permission on the directory and not on the file.

The tell was in the timing, and it was read wrong twice: the flips landed
during commits and pushes, which was taken as evidence that some other session
was active at those moments rather than as evidence that a hook was running.

## Fix

`disown_ambient_git()` strips every `GIT_*` variable from the environment
before the first world is built. One call at the top of `selftest()` covers
both halves: the world's `git init` initializes the world, and the checks'
`git grep` / `git ls-files` read the world.

Verified by reproducing the trigger:

```
$ git config core.bare false
$ GIT_DIR="$PWD/.git" GIT_WORK_TREE="$PWD" python3 scripts/check_repo_hygiene.py --selftest
$ git config core.bare
false
```

## Rule

A test that builds a scratch repository must first disown the ambient git
environment. Under a hook, `GIT_DIR` and `GIT_WORK_TREE` are exported, and both
halves of the failure follow from that: a write lands on the real repository,
and a read comes back from it.

Corollary: when a shared file changes at times that correlate with commits and
pushes, suspect a hook before suspecting another person.
