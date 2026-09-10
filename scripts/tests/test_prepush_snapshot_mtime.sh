#!/usr/bin/env bash
# Regression: the hook refreshes its snapshot with rsync. It used to pass -a
# (=> -t), which restored git-archive's COMMIT-time mtime onto source files. A
# commit older than the last build output then reported a content-CHANGED crate
# Fresh (cargo compares source mtime <= output mtime) — a false negative that
# defeated the freshness gate. The fix rsyncs without -t so transferred files
# take mtime=now, while --checksum leaves content-identical files' mtimes (and
# the warm cache) intact. This test pins both halves with the real cargo+rsync.
set -euo pipefail

command -v cargo >/dev/null || { echo "cargo absent; skipping snapshot-mtime test"; exit 0; }
RSYNC="$(command -v rsync)"

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
REPO="$W/repo"; STAGE="$W/stage"; SNAP="$W/snap"; TGT="$W/target"

mkdir -p "$REPO/src"
cat > "$REPO/Cargo.toml" <<'EOF'
[package]
name = "mtimecrate"
version = "0.1.0"
edition = "2021"
[dependencies]
EOF

git -C "$W" init -q -q "$REPO" 2>/dev/null || { git init -q "$REPO"; }
git -C "$REPO" config user.email t@e
git -C "$REPO" config user.name t

# Two commits with explicit OLD timestamps (both well before "now"), so a
# restored commit mtime is strictly older than any build artifact this test
# produces.
printf 'pub fn v() -> u32 { 1 }\n' > "$REPO/src/lib.rs"
git -C "$REPO" add -A
GIT_AUTHOR_DATE="2020-01-01T00:00:00" GIT_COMMITTER_DATE="2020-01-01T00:00:00" \
  git -C "$REPO" commit -qm A
A="$(git -C "$REPO" rev-parse HEAD)"
printf 'pub fn v() -> u32 { 2 }\n' > "$REPO/src/lib.rs"
git -C "$REPO" add -A
GIT_AUTHOR_DATE="2020-01-02T00:00:00" GIT_COMMITTER_DATE="2020-01-02T00:00:00" \
  git -C "$REPO" commit -qm B
B="$(git -C "$REPO" rev-parse HEAD)"

# Mirror the hook's snapshot refresh exactly (no -t; --checksum; --delete).
refresh() {  # $1 = ref
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  git -C "$REPO" archive "$1" | tar -x -C "$STAGE"
  mkdir -p "$SNAP"
  "$RSYNC" -rlpD --delete --checksum "$STAGE/" "$SNAP/"
}
# Echo REBUILD if cargo compiles mtimecrate, FRESH otherwise.
verdict() {
  ( cd "$SNAP" && CARGO_TARGET_DIR="$TGT" CARGO_TERM_COLOR=never \
      RUSTC_WRAPPER="" cargo build 2>&1 ) \
    | grep -qE 'Compiling mtimecrate( |$)' && echo REBUILD || echo FRESH
}

refresh "$B"
[ "$(verdict)" = REBUILD ] || { echo "FAIL: cold build of B must compile" >&2; exit 1; }

# Content-identical re-refresh: mtime preserved by --checksum -> warm (FRESH).
refresh "$B"
[ "$(verdict)" = FRESH ] || { echo "FAIL: unchanged B must stay Fresh (warm cache)" >&2; exit 1; }

# Switch A->B content differs (old commit mtime): without -t it must REBUILD.
sleep 1
refresh "$A"
[ "$(verdict)" = REBUILD ] || { echo "FAIL: content change A(v1) reported Fresh — mtime-from-commit-time bug" >&2; exit 1; }

# Back to B, content differs again: must REBUILD.
sleep 1
refresh "$B"
[ "$(verdict)" = REBUILD ] || { echo "FAIL: content change B(v2) reported Fresh — mtime-from-commit-time bug" >&2; exit 1; }

echo "PASS: snapshot rsync forces rebuild on content change, keeps Fresh when identical"
