#!/usr/bin/env bash
# Contract: a pod run must only execute a build that its OWN tree produced.
# Two trees can sync+build concurrently; a run resolves $STATE/builds/<label>.
# Before this guard the run checked schema/exit/sha/source but never that the
# build receipt's tree= matched this run's $TREE — at identical heads across
# trees the binary SHAs match, so tree A could silently execute tree B's binary.
# The tree check runs first in the chain (before sha/GPU), so it is testable
# on a Mac with fixture receipts, no GPU.
set -euo pipefail
unset GIT_DIR GIT_WORK_TREE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d -t pod-tree-id-XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

TREE_A="$TMP/host/arle-build-A"; TREE_B="$TMP/host/arle-build-B"; STATE="$TMP/state"
mkdir -p "$TREE_A/scripts" "$TREE_B/scripts" "$STATE"
cp "$ROOT/scripts/pod-remote-run.sh" "$ROOT/scripts/pod-remote-build.sh" "$TREE_A/scripts/"
cp "$ROOT/scripts/pod-remote-run.sh" "$ROOT/scripts/pod-remote-build.sh" "$TREE_B/scripts/"

# A successful build receipt claimed by tree B. (binary path is a nonexistent
# file; the tree ownership check must fire before the sha256 that would need it.)
write_receipt() { # $1 tree
  mkdir -p "$STATE/builds/L"
  cat >"$STATE/builds/L/receipt" <<EOF
schema=arle-build-v1
operation=op-x
profile=release
label=L
tree=$1
source_head=
source_digest=
binary=$TMP/nonexistent-binary
binary_sha=dummy
exit=0
pid=1
finished_at=2026-09-10T00:00:00Z
EOF
}

run_once() { # $1 run-label  -> echoes the run log
  local label="$1"
  local argv="$TMP/argv-$label"
  printf 'x\0' >"$argv"
  POD_TREE="$TREE_A" POD_STATE="$STATE" PROC_ROOT="$TMP/proc" \
    bash "$TREE_A/scripts/pod-remote-run.sh" run L "$label" 0 op-x "$argv" \
    >/dev/null 2>&1 || true
  cat "$STATE/runs/$label/log"
}

# (1) receipt belongs to tree B, run is tree A -> rejected at the tree gate.
write_receipt "$TREE_B"
out="$(run_once cross)"
grep -q "build belongs to another tree" <<<"$out" \
  || fail "tree A executed/accepted a build receipt owned by tree B: $out"
grep -q "run_tree=$TREE_A" <<<"$out" || fail "tree-gate message lacks the run tree: $out"

# (2) same receipt claimed by tree A -> clears the tree gate (then stops at the
# binary-SHA gate, which is expected: no real binary). Must NOT be a tree reject.
rm -rf "$STATE/builds/L"; write_receipt "$TREE_A"
out="$(run_once own)"
grep -q "build belongs to another tree" <<<"$out" \
  && fail "own-tree receipt was wrongly rejected by the tree gate: $out"
grep -q "binary SHA mismatch" <<<"$out" \
  || fail "own-tree receipt should clear the tree gate and stop at sha, got: $out"

echo "pod tree identity contract PASS (cross-reject/own-clears)"
