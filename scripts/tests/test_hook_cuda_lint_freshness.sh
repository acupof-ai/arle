#!/usr/bin/env bash
# The CUDA clippy is the only automated gate for the CUDA-Rust surface (no GPU
# CI), and a shared target dir lets cargo report it Fresh across lanes — so the
# gate can pass having checked nothing (a real infer-cuda compile error went
# green on 2026-09-10). The hook now asserts the lint recompiled the crates
# the push changes. This test constructs the stale-artifact world — mock cargo
# reports Fresh — and proves the gate goes red, plus the controls that keep it
# from false-positiving.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
FIX="$TMP/fix"
SNAP_HASH="$(printf '%s' "$FIX" | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-16)"
trap 'rm -rf "$TMP" "${TMPDIR:-/tmp}/arle-pre-push-snapshot-${SNAP_HASH}"' EXIT
BIN="$TMP/bin"; mkdir -p "$BIN"

# Fixture repo: base carries no-op stand-ins for the hook's fast checks, the
# cuda commit adds an infer-cuda crate, the core commit an infer-core one.
mkdir -p "$FIX/scripts/tests" "$FIX/crates/infer-cuda/src"
cp "$ROOT/scripts/pre_push_checks.sh" "$FIX/scripts/pre_push_checks.sh"
# No-op stand-ins for every shell test the hook runs, extracted from the hook
# itself so the fixture tracks the list (a hardcoded list rots when a test is
# added — the green world then fails on the missing stub, not on the assertion).
for t in $(sed -n '/for test in \\/,/; do/p' "$FIX/scripts/pre_push_checks.sh" | grep -o 'test_[a-z_]*\.sh'); do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$FIX/scripts/tests/$t"
done
printf 'print("fixture hygiene ok")\n' > "$FIX/scripts/check_repo_hygiene.py"
git -C "$FIX" init -q
git -C "$FIX" config user.email t@e; git -C "$FIX" config user.name t
git -C "$FIX" add -A && git -C "$FIX" commit -qm base
printf '[package]\nname = "infer-cuda"\nedition = "2021"\n' > "$FIX/crates/infer-cuda/Cargo.toml"
printf 'pub fn fixture() {}\n' > "$FIX/crates/infer-cuda/src/lib.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm cuda
base="$(git -C "$FIX" rev-parse HEAD~1)"; tip="$(git -C "$FIX" rev-parse HEAD)"

# Mock cargo: every invocation exits 0; the CUDA clippy (-v, cuda,no-cuda)
# prints MOCK_LINT_LINES, so the worlds differ only in whether the output shows
# infer-cuda being checked — exactly what a stale shared target suppresses.
cat > "$BIN/cargo" <<'SH'
#!/usr/bin/env bash
verbose=0
for a in "$@"; do [ "$a" = -v ] && verbose=1; done
if [ "$verbose" = 1 ] && [[ " $* " == *" cuda,no-cuda "* ]]; then
  printf '%b' "${MOCK_LINT_LINES:-Checking infer-api }"
fi
exit 0
SH
chmod +x "$BIN/cargo"

ZERO=0000000000000000000000000000000000000000
run_hook() {  # $1 = local sha, $2 = remote sha
  ( cd "$FIX" && PATH="$BIN:$PATH" bash "$FIX/scripts/pre_push_checks.sh" ) <<<"refs/heads/lane/x $1 refs/heads/lane/x $2"
}

# Red world: the push changes infer-cuda, the lint reports it Fresh.
set +e
MOCK_LINT_LINES='Checking infer-api \n' run_hook "$tip" "$base" >"$TMP/red.log" 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] || { echo "FAIL: red world passed — Fresh infer-cuda not caught" >&2; cat "$TMP/red.log" >&2; exit 1; }
grep -q 'infer-cuda Fresh while this push changes it' "$TMP/red.log"

# Green world: the lint checks infer-cuda.
MOCK_LINT_LINES='Checking infer-api \nChecking infer-cuda \nChecking cuda-kernels \n' run_hook "$tip" "$base" >"$TMP/green.log" 2>&1 \
  || { echo "FAIL: green world rejected" >&2; cat "$TMP/green.log" >&2; exit 1; }

# New branch (zero remote sha): the merge-base fallback still sees the CUDA change.
set +e
MOCK_LINT_LINES='Checking infer-api \n' run_hook "$tip" "$ZERO" >"$TMP/new.log" 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] || { echo "FAIL: new-branch red world passed" >&2; cat "$TMP/new.log" >&2; exit 1; }
grep -q 'infer-cuda Fresh while this push changes it' "$TMP/new.log"

# False-positive control: a push that does NOT touch the CUDA crates must not
# assert, however Fresh the lint is — a false-positive gate gets disabled.
mkdir -p "$FIX/crates/infer-core/src"
printf 'pub fn core_fixture() {}\n' > "$FIX/crates/infer-core/src/lib.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm core
corebase="$(git -C "$FIX" rev-parse HEAD~1)"; coretip="$(git -C "$FIX" rev-parse HEAD)"
MOCK_LINT_LINES='Checking infer-api \n' run_hook "$coretip" "$corebase" >"$TMP/ctrl.log" 2>&1 \
  || { echo "FAIL: unchanged-crate push asserted (false positive)" >&2; cat "$TMP/ctrl.log" >&2; exit 1; }

echo "PASS: CUDA lint freshness assertion (red/green/new-branch/control)"
