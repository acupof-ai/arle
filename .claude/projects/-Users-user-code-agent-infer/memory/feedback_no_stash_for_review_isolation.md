---
name: No stash for review isolation
description: Do not use git stash to isolate a review slice; review can be done together and commits should include only explicitly staged files
type: feedback
---

For this project, do **not** use `git stash` / `git stash pop` to isolate a review or commit slice. The user explicitly corrected this after a staged-review flow temporarily hid local work behind a stash entry.

**Why:** Even when the intent is only to review or commit a subset, stash-based isolation is disruptive and can create restore conflicts against ongoing local edits.

**How to apply:** When only part of the worktree should be committed or reviewed:
1. Stage only the intended files.
2. Review the staged or path-selected diff directly.
3. Commit only the staged files.
4. Leave unrelated local edits in place and call them out explicitly if relevant.
