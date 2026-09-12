#!/usr/bin/env bash
# Throwaway-repo tests for scripts/lane.sh `new` rollback. Every case builds a
# fresh git repo, runs the lane-new body via a git wrapper or a forced hook, and
# asserts the on-disk / branch / worktree-registration state afterward. No
# shared state, no GPU, no network.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_LANE="$HERE/../lane.sh"

fail=0
check() {  # $1 description, rest = a test command expected true
    local desc="$1"; shift
    if "$@"; then echo "ok   $desc"; else echo "FAIL $desc" >&2; fail=1; fi
}

# Build a bare throwaway repo and return its path on stdout. $1 is a script to
# run after `git init` (hook setup); the lane.sh under test is copied in and
# its ROOT pointed at the repo root.
make_repo() {
    local d="$1" hook="${2:-}"
    git init -q "$d/r"
    (
        cd "$d/r" || return 1
        git config user.email t@t
        git config user.name t
        git commit -q --allow-empty -m init
        git branch -M main
        mkdir -p scripts
        # Point ROOT at the repo itself rather than its parent.
        sed 's|dirname "${BASH_SOURCE[0]}")/\.\.|dirname "${BASH_SOURCE[0]}")|' \
            "$REAL_LANE" > scripts/lane.sh
        chmod +x scripts/lane.sh
        mkdir -p .git/hooks
        [ -n "$hook" ] && eval "$hook"
    )
}

# A git wrapper that removes the just-created worktree registration after a
# successful `worktree add` (stand-in for the observed concurrent-removal race;
# the real race cannot be forced).
make_race_git() {  # $1 dir; mode is passed to the subshell via GIT_RACE_MODE
    local d="$1"
    mkdir -p "$d/fake-bin"
    cat > "$d/fake-bin/git" <<'EOF'
#!/bin/bash
if [ "$1" = "-C" ]; then
    repo="$2"; shift 2
    if [ "$1" = "worktree" ] && [ "$2" = "add" ]; then
        /usr/bin/git -C "$repo" "$@"; rc=$?
        tgt="${@: -2:1}"
        if [ "${GIT_RACE_MODE:-}" = unlink ]; then
            # Directory stays; its .git link and admin registration are gone.
            rm -f "$tgt/.git"
        else
            # Directory removed entirely (cd in lane.sh then fails).
            rm -rf "$tgt"
        fi
        /usr/bin/git -C "$repo" worktree prune
        exit "$rc"
    fi
    exec /usr/bin/git -C "$repo" "$@"
fi
exec /usr/bin/git "$@"
EOF
    chmod +x "$d/fake-bin/git"
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ── 1. Happy path registers a live worktree ─────────────────────────────────
d="$TMP/happy"; make_repo "$d"
( cd "$d/r" && PATH=/usr/bin:/bin scripts/lane.sh new alpha >/dev/null )
check "happy: branch created" /usr/bin/git -C "$d/r" show-ref --verify --quiet refs/heads/lane/alpha
check "happy: directory is a worktree" /usr/bin/git -C "$d/arle-lanes/alpha" rev-parse --is-inside-work-tree

# ── 2. Nonzero add rolls back (failing post-checkout hook) ──────────────────
d="$TMP/nonzero"; make_repo "$d" 'printf "#!/bin/bash\nexit 1\n" > .git/hooks/post-checkout && chmod +x .git/hooks/post-checkout'
( cd "$d/r" && PATH=/usr/bin:/bin scripts/lane.sh new beta >/dev/null 2>&1 ); rc=$?
check "nonzero: exits 1" test "$rc" -eq 1
check "nonzero: directory removed" test ! -e "$d/arle-lanes/beta"
check "nonzero: branch removed" bash -c '! /usr/bin/git -C "'"$d/r"'" show-ref --verify --quiet refs/heads/lane/beta'
check "nonzero: no worktree entry" bash -c '! /usr/bin/git -C "'"$d/r"'" worktree list | grep -q beta'

# ── 3. Zero add, registration unlinked afterward → rollback, name reusable ──
d="$TMP/unlinked"; make_repo "$d"; make_race_git "$d" unlink
( cd "$d/r" && PATH="$d/fake-bin:$PATH" GIT_RACE_MODE=unlink scripts/lane.sh new gamma >/dev/null 2>&1 ); rc=$?
check "unlinked: exits 1" test "$rc" -eq 1
check "unlinked: directory removed" test ! -e "$d/arle-lanes/gamma"
check "unlinked: branch removed" bash -c '! /usr/bin/git -C "'"$d/r"'" show-ref --verify --quiet refs/heads/lane/gamma'
check "unlinked: no worktree entry" bash -c '! /usr/bin/git -C "'"$d/r"'" worktree list | grep -q gamma'
# The name must be reusable (the trap in the errors entry).
( cd "$d/r" && PATH=/usr/bin:/bin scripts/lane.sh new gamma >/dev/null )
check "unlinked: name reusable after rollback" /usr/bin/git -C "$d/arle-lanes/gamma" rev-parse --is-inside-work-tree

# ── 4. Zero add, directory removed entirely: cd fails but rollback still runs ─
d="$TMP/rmdir"; make_repo "$d"; make_race_git "$d" rmdir
( cd "$d/r" && PATH="$d/fake-bin:$PATH" GIT_RACE_MODE=rmdir scripts/lane.sh new delta >/dev/null 2>&1 ); rc=$?
check "rmdir: exits 1 (cd failure does not abort before rollback)" test "$rc" -eq 1
check "rmdir: branch removed despite the failed cd" bash -c '! /usr/bin/git -C "'"$d/r"'" show-ref --verify --quiet refs/heads/lane/delta'
check "rmdir: no worktree entry" bash -c '! /usr/bin/git -C "'"$d/r"'" worktree list | grep -q delta'
( cd "$d/r" && PATH=/usr/bin:/bin scripts/lane.sh new delta >/dev/null )
check "rmdir: name reusable after rollback" /usr/bin/git -C "$d/arle-lanes/delta" rev-parse --is-inside-work-tree

# ── 5. A pre-existing diverged lane/<name> branch is preserved, dir removed ─
d="$TMP/preexist"; make_repo "$d"
/usr/bin/git -C "$d/r" branch lane/epsilon main >/dev/null
/usr/bin/git -C "$d/r" commit -q --allow-empty -m two
( cd "$d/r" && PATH=/usr/bin:/bin scripts/lane.sh new epsilon >/dev/null 2>&1 ); rc=$?
check "preexist: exits 1" test "$rc" -eq 1
check "preexist: diverged branch kept" /usr/bin/git -C "$d/r" show-ref --verify --quiet refs/heads/lane/epsilon
check "preexist: branch still points at its own commit" bash -c '
  [ "$(/usr/bin/git -C "'"$d/r"'" rev-parse lane/epsilon)" != "$(/usr/bin/git -C "'"$d/r"'" rev-parse main)" ]'
check "preexist: stray directory removed" test ! -e "$d/arle-lanes/epsilon"

if [ "$fail" -eq 0 ]; then
    echo "PASS: lane.sh new rollback (happy, nonzero add, unlinked, directory-gone, preexisting branch)"
    exit 0
fi
echo "FAIL: lane.sh new rollback tests" >&2
exit 1
