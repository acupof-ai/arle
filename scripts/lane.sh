#!/usr/bin/env bash
# Lane worktrees: one checkout per agent session, one shared build directory.
#
# A lane costs its tracked files (~97 MB) and no build artifacts: every worktree
# compiles into the main checkout's `target/` (8.5 GB, created once). Cargo takes
# one lock per target directory, so lane builds SERIALIZE — that is the price of
# not multiplying the 8.5 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANES="${ARLE_LANES:-$(dirname "$ROOT")/arle-lanes}"

usage() {
    cat <<EOF
usage: scripts/lane.sh <command>

  doctor                repair core.bare in the shared .git/config, then report
  new <name>            worktree at $LANES/<name> on branch lane/<name>
  pr <name> [title]     push the branch and open a PR against main
  list                  worktrees and their branches
  rm <name>             remove the worktree (refuses while it is dirty)

Every lane shares $ROOT/target. Builds serialize on cargo's lock.
EOF
}

# Cargo merges config from the cwd upward. This file sits ABOVE every worktree
# and outside every checkout, so it never dirties the tracked .cargo/config.toml
# — which deliberately sets no target-dir of its own.
share_target() {
    mkdir -p "$LANES/.cargo"
    printf '[build]\ntarget-dir = "%s/target"\n' "$ROOT" > "$LANES/.cargo/config.toml"
}

valid_name() {
    case "$1" in
        ""|*[!A-Za-z0-9_-]*) echo "lane: invalid name '$1' (A-Za-z0-9_-)" >&2; exit 2 ;;
    esac
}

# `core.bare = true` keeps appearing in the SHARED .git/config, which makes every
# worktree fail with "this operation must be run in a work tree". The writer is
# unidentified: all four lanes deny it, .githooks/pre-push does not touch config,
# and `worktree add/list/prune`, `fetch` and `rev-list` were each measured not to
# set it. chmod on the file does not help — git writes config.lock and renames,
# which needs only directory write permission. So repair it instead of guarding.
repair_bare() {
    [ "$(git -C "$ROOT" config core.bare 2>/dev/null)" = "true" ] || return 0
    chmod u+w "$ROOT/.git/config" 2>/dev/null || true
    git -C "$ROOT" config core.bare false
    echo "lane: repaired core.bare=true in the shared .git/config" >&2
}

case "${1:-help}" in
doctor)
    repair_bare
    git -C "$ROOT" rev-parse --is-bare-repository
    git -C "$ROOT" worktree list
    ;;
new)
    name="${2:?lane: new <name>}"; valid_name "$name"
    share_target
    path="$LANES/$name"
    [ -e "$path" ] && { echo "lane: $path already exists" >&2; exit 2; }
    git -C "$ROOT" fetch --quiet origin main 2>/dev/null || true
    git -C "$ROOT" worktree add -b "lane/$name" "$path" main
    echo
    echo "lane $name ready:"
    echo "  cd $path"
    echo "  target-dir -> $ROOT/target (shared; builds serialize)"
    echo
    echo "  pod work from this lane needs its OWN remote tree — pod.sh refuses"
    echo "  one of the pair, so export both:"
    echo "    export POD_TREE=/host/arle-build-$name NODE_TREE=/root/arle-build-$name"
    echo "  A shared tree is a silent-wrong-binary hazard: flock serializes each"
    echo "  sync and each build, not the sync..build sequence, so another lane"
    echo "  syncing between yours and your build leaves you compiling its source."
    echo "  Cost is ~1.5 GB per tree; /host/sccache is shared, so rebuilds are cheap."
    ;;
pr)
    name="${2:?lane: pr <name> [title]}"; valid_name "$name"
    repair_bare
    path="$LANES/$name"
    [ -d "$path" ] || { echo "lane: no worktree at $path" >&2; exit 2; }
    [ -n "$(git -C "$path" status --porcelain)" ] && {
        echo "lane: $name has uncommitted changes; commit them by explicit path first" >&2
        git -C "$path" status --short >&2; exit 2; }
    # Rebase first: a lane branched before main moved would otherwise open a PR
    # that reverts every file main gained in the meantime.
    git -C "$ROOT" fetch --quiet origin main 2>/dev/null || true
    behind=$(git -C "$path" rev-list --count "HEAD..main")
    if [ "$behind" -gt 0 ]; then
        echo "lane: $name is $behind commit(s) behind main; rebasing"
        git -C "$path" rebase main || {
            echo "lane: rebase stopped with conflicts in $path — resolve, then rerun" >&2; exit 2; }
    fi
    git -C "$path" push -u --force-with-lease origin "lane/$name"
    title="${3:-$(git -C "$path" log -1 --pretty=%s)}"
    gh pr create --repo "$(git -C "$ROOT" remote get-url origin | sed 's#.*[:/]\([^/]*/[^/]*\)\.git#\1#')" \
        --base main --head "lane/$name" --title "$title" \
        --body "$(git -C "$path" log main..HEAD --pretty='- %s')"
    ;;
list)
    git -C "$ROOT" worktree list
    ;;
rm)
    name="${2:?lane: rm <name>}"; valid_name "$name"
    path="$LANES/$name"
    git -C "$ROOT" worktree remove "$path"
    git -C "$ROOT" worktree prune
    echo "lane: removed $name (branch lane/$name kept)"
    ;;
*)
    usage
    ;;
esac
