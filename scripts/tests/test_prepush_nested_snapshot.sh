#!/usr/bin/env bash
# A nested hook run (ARLE_PRE_PUSH_NESTED=1, used by every shell-test fixture)
# skips the machine lock. It therefore must never rsync --delete into the
# SHARED snapshot — a fixture tree would erase the real tree a peer hook is
# compiling (live race 2026-09-11: crates/mlx-sys/vendor array.cpp vanished
# mid-build). The hook forces a private mktemp snapshot for a nested run unless
# the caller explicitly points ARLE_PREPUSH_SNAPSHOT_ROOT at a test-owned dir.
#
# This test drives the real hook with a scratch TMPDIR, pre-seeds the shared
# root with a sentinel, and asserts the sentinel survives a nested cargo run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FIX="$TMP/fix"
BIN="$TMP/bin"
# All ${TMPDIR} paths resolve under this scratch; the real shared snapshot is
# never touched.
SCRATCH_TMPDIR="$TMP/tmp"; mkdir -p "$SCRATCH_TMPDIR"
SHARED="$SCRATCH_TMPDIR/arle-pre-push-snapshot"
mkdir -p "$FIX/scripts/tests"
cp "$ROOT/scripts/pre_push_checks.sh" "$FIX/scripts/pre_push_checks.sh"
for t in $(sed -n '/for test in \\/,/; do/p' "$FIX/scripts/pre_push_checks.sh" | grep -o 'test_[a-z_]*\.sh'); do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$FIX/scripts/tests/$t"
done
printf 'print("fixture hygiene ok")\n' > "$FIX/scripts/check_repo_hygiene.py"
git -C "$FIX" init -q
git -C "$FIX" config user.email t@e; git -C "$FIX" config user.name t
# A .rs change so CARGO_RUNS=1 (the branch that used to take the shared root).
mkdir -p "$FIX/crates/fix/src"
printf '[package]\nname = "fix"\nedition = "2021"\n' > "$FIX/crates/fix/Cargo.toml"
printf 'pub fn f() {}\n' > "$FIX/crates/fix/src/lib.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm base
base="$(git -C "$FIX" rev-parse HEAD)"
printf 'pub fn f() { let _ = 1; }\n' > "$FIX/crates/fix/src/lib.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm rs
tip="$(git -C "$FIX" rev-parse HEAD)"

mkdir -p "$BIN"
cat > "$BIN/cargo" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$BIN/cargo"

# Pre-seed the shared root with a peer's file and a deep nested one.
mkdir -p "$SHARED/crates/mlx-sys/vendor/mlx/mlx"
echo "peer artifact" > "$SHARED/SENTINEL.txt"
echo "peer source" > "$SHARED/crates/mlx-sys/vendor/mlx/mlx/array.cpp"

run_nested() {  # extra env args passed through
  (
    cd "$FIX"
    TMPDIR="$SCRATCH_TMPDIR" ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" "$@" \
      bash "$FIX/scripts/pre_push_checks.sh"
  ) <<<"refs/heads/lane/x $tip refs/heads/lane/x $base"
}

# RED/GREEN world: nested run with NO snapshot override must leave the shared
# root and both sentinels byte-identical (it syncs a private dir instead).
log="$TMP/nested.log"
if ! run_nested >"$log" 2>&1; then
  echo "FAIL: nested hook exited non-zero" >&2; cat "$log" >&2; exit 1
fi
grep -q "using private snapshot .*shared snapshot left untouched" "$log" \
  || { echo "FAIL: nested run did not select a private snapshot" >&2; cat "$log" >&2; exit 1; }
[ "$(cat "$SHARED/SENTINEL.txt")" = "peer artifact" ] \
  || { echo "FAIL: shared-root sentinel was overwritten/deleted" >&2; exit 1; }
[ "$(cat "$SHARED/crates/mlx-sys/vendor/mlx/mlx/array.cpp")" = "peer source" ] \
  || { echo "FAIL: nested run wiped a peer's deep source file (the live race)" >&2; exit 1; }
# The private snapshot is removed by the hook's exit trap, so assert its path
# from the run's log; the sentinels already prove the shared root was untouched.
grep -qE "using private snapshot ${SCRATCH_TMPDIR}/arle-pre-push-nocargo\.[A-Za-z0-9]{6}" "$log" \
  || { echo "FAIL: nested run did not name a private mktemp snapshot" >&2; cat "$log" >&2; exit 1; }

# Control: an EXPLICIT snapshot root is still honored (fixture tests own it and
# intentionally point the hook at a seeded dir).
EXPLICIT="$TMP/explicit-snap"; mkdir -p "$EXPLICIT"
(
  cd "$FIX"
  TMPDIR="$SCRATCH_TMPDIR" ARLE_PRE_PUSH_NESTED=1 ARLE_PREPUSH_SNAPSHOT_ROOT="$EXPLICIT" \
    PATH="$BIN:$PATH" bash "$FIX/scripts/pre_push_checks.sh"
) <<<"refs/heads/lane/x $tip refs/heads/lane/x $base" >"$TMP/explicit.log" 2>&1 \
  || { echo "FAIL: explicit-root nested hook exited non-zero" >&2; cat "$TMP/explicit.log" >&2; exit 1; }
[ -f "$EXPLICIT/crates/fix/src/lib.rs" ] \
  || { echo "FAIL: explicit snapshot root was not synced" >&2; exit 1; }
grep -q "refreshing shared snapshot at $EXPLICIT" "$TMP/explicit.log" \
  || { echo "FAIL: explicit root not recognized as shared/owned" >&2; exit 1; }
# And the real shared root is still untouched by the explicit-root run.
[ "$(cat "$SHARED/SENTINEL.txt")" = "peer artifact" ] \
  || { echo "FAIL: explicit-root run touched the default shared root" >&2; exit 1; }

echo "PASS: nested fixture hook uses a private snapshot and cannot wipe the shared root; explicit override still honored"
