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

# Serialized rsync+cargo, exactly as the hook runs them. $1 = rsync extra
# flags (the point of the test is -t vs no -t). Returns the number of
# "Checking freshinvariant" lines cargo prints (1 = rebuilt, 0 = Fresh).
build() {
    local rsync_flags="$1"
    while ! mkdir "$LOCK" 2>/dev/null; do sleep 0.05; done
    rsync $rsync_flags -rlpD --delete --checksum "$STAGE/" "$SNAP/" >/dev/null
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
