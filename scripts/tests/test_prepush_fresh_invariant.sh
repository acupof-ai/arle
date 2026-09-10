#!/usr/bin/env bash
# The pre-push hook no longer scrapes cargo output for "Checking <crate>" to
# decide whether a changed crate rebuilt (the guard false-positived
# examples-only and cfg(test)-only pushes). Correctness now rests on an
# invariant: under the hook's rsync (no -t → updated inputs get mtime=now) and
# the machine lock serializing rsync→cargo, a unit whose dep-info input
# changed MUST rebuild, and an unchanged one MUST be Fresh.
#
# This test proves the invariant with REAL cargo in a scratch target, because
# mocked-cargo worlds can only test the scraper, not cargo's fingerprint.
# It also proves the pre-#302 failure mode (rsync -t restoring old mtimes)
# reports a changed input Fresh — i.e. the mtime=now rsync is what carries the
# guarantee.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
TARGET="$TMP/target"
SNAP="$TMP/snap"
LOCK="$TMP/lock"
STAGE="$TMP/stage"
mkdir -p "$SNAP" "$STAGE/src"

cat > "$STAGE/Cargo.toml" <<'EOF'
[package]
name = "freshinvariant"
version = "0.1.0"
edition = "2021"
EOF

# Read the snapshot rsync flags FROM THE HOOK rather than hardcoding them: the
# mtime=now guarantee is only as good as the hook's actual rsync invocation,
# so a future -a/-t/--times/--archive added there must fail this test.
HOOK="$ROOT/scripts/pre_push_checks.sh"
hook_flags="$(sed -n 's/^SNAPSHOT_RSYNC_FLAGS="\(.*\)"$/\1/p' "$HOOK")"
[ -n "$hook_flags" ] || { echo "FAIL: SNAPSHOT_RSYNC_FLAGS not defined in hook" >&2; exit 1; }
# Reject every mtime-preserving form. --archive implies --times; -t/-a may be
# standalone or bundled into a short cluster (e.g. -rlptD, -a).
for word in $hook_flags; do
    case "$word" in
        -a|-t|--archive|--times)
            echo "FAIL: hook snapshot rsync carries $word; updated files would not get mtime=now" >&2; exit 1 ;;
        -*)
            [[ "$word" == --* ]] && continue
            case "$word" in *a*|*t*) echo "FAIL: hook snapshot rsync cluster $word preserves mtimes" >&2; exit 1 ;; esac
            ;;
    esac
done
# It must still sync recursively with checksums and deletions, or it is a
# different rsync than the hook's snapshot refresh (-r may be bundled, e.g.
# -rlpD).
has_word() {  # exact word match
    local want="$1"; shift
    for word in "$@"; do [ "$word" = "$want" ] && return 0; done
    return 1
}
has_flag_cluster() {  # short flag letter inside a single-dash cluster
    local letter="$1"; shift
    for word in "$@"; do
        case "$word" in
            -[a-zA-Z]*|--*) [[ "$word" == --* ]] && continue; [[ "$word" == *"$letter"* ]] && return 0 ;;
        esac
    done
    return 1
}
has_word --delete $hook_flags || { echo "FAIL: snapshot rsync flags missing --delete" >&2; exit 1; }
has_word --checksum $hook_flags || { echo "FAIL: snapshot rsync flags missing --checksum" >&2; exit 1; }
has_flag_cluster r $hook_flags || { echo "FAIL: snapshot rsync flags not recursive (-r)" >&2; exit 1; }

# Serialized rsync+cargo with the hook's exact flags. $1 = extra flags appended
# (the pre-#302 control appends -t). Returns the count of "Checking" lines.
build() {
    local extra_flags="$1"
    while ! mkdir "$LOCK" 2>/dev/null; do sleep 0.05; done
    # Word-split is intended: the flags come from the hook's quoted scalar.
    rsync $hook_flags $extra_flags "$STAGE/" "$SNAP/" >/dev/null
    local n
    n="$(
        cd "$SNAP"
        CARGO_TERM_COLOR=never CARGO_TARGET_DIR="$TARGET" cargo check 2>&1 \
        | grep -cE 'Checking freshinvariant' || true
    )"
    rmdir "$LOCK"
    echo "$n"
}

# 1) First build compiles; identical re-sync is Fresh.
printf 'pub fn v() -> u32 { 1 }\n' > "$STAGE/src/lib.rs"
[ "$(build '')" = "1" ] || { echo "FAIL: first build did not compile" >&2; exit 1; }
[ "$(build '')" = "0" ] || { echo "FAIL: unchanged re-sync rebuilt (warm cache broken)" >&2; exit 1; }

# 2) Changed input under mtime=now rsync ALWAYS rebuilds, ping-pong between
#    two contents. A 0 here is the false-Fresh the old guard tried to scrape.
printf 'pub fn v() -> u32 { 2 }\n' > "$STAGE/src/lib.rs"
sleep 1
[ "$(build '')" = "1" ] || { echo "FAIL: changed input reported Fresh under mtime=now rsync" >&2; exit 1; }
printf 'pub fn v() -> u32 { 1 }\n' > "$STAGE/src/lib.rs"
sleep 1
[ "$(build '')" = "1" ] || { echo "FAIL: changed-back input reported Fresh under mtime=now rsync" >&2; exit 1; }
[ "$(build '')" = "0" ] || { echo "FAIL: unchanged rebuild after change-back" >&2; exit 1; }

# 3) A cfg(test)-only file is NOT in the non-test unit's dep-info, so changing
#    it must leave the plain `cargo check` lib legitimately Fresh. This is the
#    class of legal Fresh the path-based guard false-positived.
cat > "$STAGE/src/lib.rs" <<'EOF'
pub fn v() -> u32 { 3 }
#[cfg(test)]
mod t { include!("tmod.rs"); }
EOF
printf '#[test] fn x() { assert_eq!(super::v(), 3); }\n' > "$STAGE/src/tmod.rs"
sleep 1
[ "$(build '')" = "1" ] || { echo "FAIL: setup change did not rebuild" >&2; exit 1; }
printf '#[test] fn x() { assert_eq!(super::v(), 99); }\n' > "$STAGE/src/tmod.rs"
sleep 1
[ "$(build '')" = "0" ] || { echo "FAIL: cfg(test)-only change rebuilt the non-test lib (or guard would false-positive)" >&2; exit 1; }

# 4) Control: the pre-#302 rsync (-t preserves the source/COMMIT mtime)
#    reports a genuinely-changed, back-dated input Fresh — reproducing the
#    2026-09-10 false negative that mtime=now fixed. We assert this FAILS to
#    rebuild only when the changed file's mtime is forced into the past, so a
#    future rsync change that silently re-enables mtime preservation flips
#    this expectation and the test breaks loudly.
rm -rf "$TARGET"
printf 'pub fn v() -> u32 { 10 }\n' > "$STAGE/src/lib.rs"
rm -f "$STAGE/src/tmod.rs"
sleep 1
[ "$(build '')" = "1" ] || { echo "FAIL: fresh-target baseline did not compile" >&2; exit 1; }
printf 'pub fn v() -> u32 { 11 }\n' > "$STAGE/src/lib.rs"
touch -t 202001010000 "$STAGE/src/lib.rs"
# -t preserves the 2020 mtime into the snapshot.
old_fresh="$(build '-t')"
if [ "$old_fresh" != "0" ]; then
    echo "FAIL: expected old-mtime changed input to be Fresh under rsync -t (got rebuild=$old_fresh); the -t failure mode no longer reproduces — re-derive the invariant" >&2
    exit 1
fi

echo "PASS: cargo Fresh invariant (changed→rebuild, unchanged→Fresh, cfg(test)-only→Fresh; rsync -t reproduces the old false-Fresh)"
