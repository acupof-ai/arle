# Rule

Always commit and push on the current branch. Do not route around a dirty
worktree by creating throwaway push branches or detached-worktree commits.

# Why

The user explicitly corrected that all commits and pushes must happen on the
current branch. Temporary worktrees are fine for inspection or local
reproduction, but the authoritative commit/push path must stay on the current
branch.

# Preventive action

- Before any commit, confirm the current branch is the branch that will be
  pushed.
- If the worktree is dirty with unrelated changes, isolate the intended commit
  with path-limited staging or `git commit --only`, not by diverting the push
  onto another branch.
