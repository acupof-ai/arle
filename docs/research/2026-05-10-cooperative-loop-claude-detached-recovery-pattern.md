---
title: Cooperative loop discipline — Claude commit pattern when codex holds detached HEAD via bisect script
date: 2026-05-10
type: research
status: closed (pattern sedimented, candidate for future SKILL revision)
---

# Cooperative loop discipline — Claude commit when codex holds detached HEAD

> **Purpose**: capture the tick-discipline pattern observed during
> codex's Task #48 matrix bisect (b1a9c1e) where local HEAD was held
> in detached state at candidate SHAs. Claude needed to commit per
> "1 commit per tick" directive but couldn't push to main without
> conflict. The temp-branch-recovery workflow is the resolution.

## §1 The problem

Codex's matrix bisect script does:
```bash
orig=$(git rev-parse --abbrev-ref HEAD)
orig_head=$(git rev-parse --short HEAD)
for cand in <candidate_shas>; do
    git checkout -q "$cand"
    cargo build && cargo test ... > "$log_$cand"
done
git checkout -q "$orig"
echo "RETURNED $(git rev-parse --short HEAD) (was $orig_head)"
```

While the `for` loop is running, local HEAD is in **detached state**
at one of the candidates (e.g. `09ae5a5`). If Claude commits during
this window, the commit lands on detached HEAD, NOT on main.
Standard `git push origin main` then says "Everything up-to-date"
(main is unchanged) but the new commit is orphaned.

## §2 Naive approach (problematic)

Earlier this session I tried committing directly during detached HEAD
state. Result:

```
[分离头指针 5ec37cc] docs(skill+plans): SKILL.md frontmatter ...
... Everything up-to-date  ← push said no-op because main hasn't moved
```

The commit `5ec37cc` was on detached HEAD pointing nowhere reachable
from main. Recovery required:
```bash
git branch backup-5ec37cc 5ec37cc
git checkout main
git cherry-pick 5ec37cc
git push origin main
```

This works but the `5ec37cc` commit on detached HEAD becomes orphaned
(unreachable + GC'able) once `backup-` branch deleted.

## §3 Better pattern — temp branch off detached HEAD

Used during second occurrence:

```bash
# Detected detached HEAD via git status --short --branch (shows "HEAD（非分支）")
# Want to commit safely

# 1. Backup: copy file to /tmp first
cp my-new-file.md /tmp/backup.md
rm my-new-file.md  # avoid interference with codex's checkout flow

# ... wait or proceed ...

# 2. Restore + branch + commit
cp /tmp/backup.md my-new-file.md
git checkout -b claude-detached-tick-recover-$(date +%H%M)
git add my-new-file.md
git commit -m "..."
git push -u origin claude-detached-tick-recover-<timestamp>

# 3. After codex's bisect releases HEAD (script's git checkout -q $orig)
git checkout main
git pull origin main  # in case codex pushed
git cherry-pick <commit-sha-on-temp-branch>
git push origin main
git branch -D claude-detached-tick-recover-<timestamp>
git push origin --delete claude-detached-tick-recover-<timestamp>
```

This pattern:
- Commits go to a NAMED branch, not detached HEAD (no orphan risk)
- Pushed to remote = safe even if local repo state corrupts
- Cherry-pick is a clean way to land the commit on main once HEAD is
  restored
- Cleanup is explicit (delete local + remote temp branch)

## §4 SKILL candidate (single evidence so far — not codifying yet)

**Candidate v1.15.0+ #41 (or similar)**: "When peer agent's script
holds your local HEAD in detached state (via `git checkout` in a
loop), don't commit directly to detached HEAD — create a temp branch
off the detached state, commit there, push to remote for safekeeping,
then cherry-pick onto main once peer's script restores branch state."

Companion to skill #30 (git status before commit, not just before
add): #30 catches the static "what's about to be staged" case; this
pattern catches the dynamic "what's the current branch state during a
peer-agent's bisect" case.

n=1 evidence (this session, Task #48 matrix bisect). Watch for n+1 in
future codex tasks that involve `git checkout` loops (bisect, cross-
commit benches, refactor + verify-old-state cycles).

## §5 Detection rule

At every tick start, BEFORE any commit:

```bash
git status --short --branch | head -1
```

If the first line is `## HEAD（非分支）` (Chinese) or
`## HEAD (no branch)` (English) → detached HEAD detected → use temp-
branch pattern, NOT direct commit to detached.

This adds ~50ms to tick scan but catches a real risk — orphaned
commits that look pushed but aren't on main.

## §6 Cross-references

- `b1a9c1e` codex matrix bisect script that holds HEAD detached
- `5ec37cc` orphan commit example (later recovered as `62e8295`)
- `e5deac8` temp-branch commit example (later recovered as `01bcefa`)
- `2f2e581` cleanup commit + Task #48 BREAKTHROUGH note
- SKILL `kernel-optimization` v1.11.0 #30 (git status discipline)

## §7 Status

Pattern sedimented as research note. Candidate for SKILL revision
when n=2+ evidence accumulates (likely in next codex bisect-style
task).
