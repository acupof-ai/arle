---
name: Done means commit and push
description: Do not report work as complete until the relevant changes are committed and pushed to origin/main
type: feedback
---

For this project, "完成" means the work is not only implemented and locally verified, but also **committed and pushed**. A clean local diff without `git commit` / `git push` is not a finished state.

**Why:** The user explicitly corrected the completion bar after a round where the code and verification were done but nothing had been committed or pushed.

**How to apply:** When closing a task, do not say it is complete until:
1. The intended slice is committed on `main`.
2. The commit is pushed to `origin/main`.
3. Any remaining uncommitted changes are called out explicitly if they are outside the committed slice.
