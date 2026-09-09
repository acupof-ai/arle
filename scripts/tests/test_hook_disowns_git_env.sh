#!/usr/bin/env bash
# A push from a worktree exports GIT_DIR=<repo>/.git/worktrees/<name> to the
# pre-push hook. That basename is the worktree name, not ".git", and git decides
# bare-vs-not by the basename — so a `git init` in any subprocess below the hook
# reinitializes the SHARED config as bare and breaks every worktree at once.
#
# The negative control runs first and must corrupt the victim. Without it this
# test would still pass if the hook stopped disowning, or if git changed the
# behaviour and there were nothing left to guard.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

( cd "$TMP" && git init -q v && cd v && echo x > f \
  && git -c user.email=t@e -c user.name=t add -A \
  && git -c user.email=t@e -c user.name=t commit -qm init ) >/dev/null 2>&1
victim="$TMP/v"
git -C "$victim" worktree add -q "$TMP/wt" -b probe >/dev/null 2>&1
WT_GIT_DIR="$victim/.git/worktrees/wt"
[ -d "$WT_GIT_DIR" ] || { echo "FAIL: no worktree gitdir at $WT_GIT_DIR" >&2; exit 1; }

bare_of() { git --git-dir="$victim/.git" config core.bare; }
reset_victim() { git --git-dir="$victim/.git" config core.bare false; }

# --- negative control: the worktree GIT_DIR, no disown -> victim goes bare ----
reset_victim
mkdir -p "$TMP/world_a"
( export GIT_DIR="$WT_GIT_DIR"; cd "$TMP/world_a"; git init -q ) >/dev/null 2>&1
[ "$(bare_of)" = "true" ] || {
    echo "FAIL: negative control did not set core.bare; this test proves nothing" >&2
    exit 1
}

# --- treatment: the hook's own disown block, same sequence -------------------
disown_block="$(sed -n '/^while IFS=.=. read -r name _; do$/,/^done < <(env)$/p' "$ROOT/.githooks/pre-push")"
[ -n "$disown_block" ] || { echo "FAIL: .githooks/pre-push has no GIT_* disown block" >&2; exit 1; }

reset_victim
mkdir -p "$TMP/world_b"
( export GIT_DIR="$WT_GIT_DIR"
  eval "$disown_block"
  cd "$TMP/world_b"; git init -q ) >/dev/null 2>&1
[ "$(bare_of)" = "false" ] || {
    echo "FAIL: the disown block did not protect the shared config (core.bare=$(bare_of))" >&2
    exit 1
}

echo "PASS: worktree GIT_DIR corrupts the shared config; the hook's disown prevents it"
