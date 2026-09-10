#!/usr/bin/env bash
# The CUDA clippy and the cargo test steps are the automated gates for the
# CUDA-Rust surface (no GPU CI), and a shared target dir lets cargo report
# their crates Fresh across lanes — stale artifacts, or artifacts from a lane
# whose trait signatures disagree with this tree (2026-09-10: a concurrent
# lane's 3-param `submit` produced fake E0050/E0063 here). The hook asserts
# every step recompiled the crates the push changes. This test constructs
# that world — mock cargo reports Fresh — and proves the gate goes red, plus
# the controls that keep it from false-positiving.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
FIX="$TMP/fix"
SNAP="$TMP/snapshot"
LOCK="$TMP/snapshot.lock"
trap 'rm -rf "$TMP"' EXIT
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
mkdir -p "$FIX/crates/infer-cuda/examples"
printf 'fn main() { infer_cuda::fixture(); }\n' > "$FIX/crates/infer-cuda/examples/fixture_example.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm cuda
base="$(git -C "$FIX" rev-parse HEAD~1)"; tip="$(git -C "$FIX" rev-parse HEAD)"

# Mock cargo: every invocation exits 0. The CUDA clippy (-v, cuda,no-cuda)
# prints MOCK_LINT_LINES; the examples clippy (-v, --examples) prints
# MOCK_EXAMPLES_LINES; test and non-verbose clippy invocations print a
# Checking line per -p crate, unless MOCK_TEST_SKIP names it — the suppressed
# crate is exactly what a Fresh / cross-contaminated target looks like.
cat > "$BIN/cargo" <<'SH'
#!/usr/bin/env bash
verbose=0
for a in "$@"; do [ "$a" = -v ] && verbose=1; done
if [ "$verbose" = 1 ] && [[ " $* " == *" --examples "* ]]; then
  printf '%b' "${MOCK_EXAMPLES_LINES-Checking fixture_example \n}"
  exit 0
fi
if [ "$verbose" = 1 ] && [[ " $* " == *" cuda,no-cuda "* ]]; then
  printf '%b' "${MOCK_LINT_LINES:-Checking infer-api }"
  exit 0
fi
if [ "$1" = test ] || { [ "$1" = clippy ] && [ "$verbose" = 0 ]; }; then
  take=0
  for a in "$@"; do
    if [ "$take" = 1 ]; then
      if [ "$a" != "${MOCK_TEST_SKIP:-__}" ]; then
        if [ -n "${MOCK_COLOR:-}" ]; then
          printf '\033[1m\033[92m   Compiling\033[0m %s v0.5.8\n' "$a"
        else
          printf 'Checking %s v0.5.8\n' "$a"
        fi
      fi
      take=0
    fi
    [ "$a" = "-p" ] && take=1
  done
fi
exit 0
SH
chmod +x "$BIN/cargo"

ZERO=0000000000000000000000000000000000000000
run_hook() {  # $1 = local sha, $2 = remote sha
  # NESTED bypasses the machine lock (this fixture runs inside a real hook's
  # fast checks, which already hold it) and the cargo is the mock. Override the
  # shared snapshot/lock paths so the fixture never touches the real ones.
  ( cd "$FIX" && ARLE_PRE_PUSH_NESTED=1 ARLE_PREPUSH_SNAPSHOT_ROOT="$SNAP" ARLE_PREPUSH_LOCK_DIR="$LOCK" PATH="$BIN:$PATH" bash "$FIX/scripts/pre_push_checks.sh" ) <<<"refs/heads/lane/x $1 refs/heads/lane/x $2"
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

# Red world for the test-step guard: the same infer-core change, but a test
# step reports it Fresh (mock suppresses its Checking line).
set +e
MOCK_TEST_SKIP=infer-core run_hook "$coretip" "$corebase" >"$TMP/test-red.log" 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] || { echo "FAIL: test-step red world passed — Fresh infer-core not caught" >&2; cat "$TMP/test-red.log" >&2; exit 1; }
grep -q 'reported infer-core Fresh while this push changes it' "$TMP/test-red.log"

# Color world: the hook exports CARGO_TERM_COLOR=always, decorating the test
# steps' output with ANSI codes; the strip must still see the rebuilt crate.
# Uses the core commit so the test-step guard (group 2 lists infer-core) is
# what exercises it — the CUDA lint path sets never and would pass regardless.
MOCK_COLOR=1 run_hook "$coretip" "$corebase" >"$TMP/color.log" 2>&1 \
  || { echo "FAIL: color world rejected rebuilt crates" >&2; cat "$TMP/color.log" >&2; exit 1; }

# Red world for the examples guard: a push changing an example file, the
# examples clippy reports everything Fresh. The lib lint passes (its Checking
# lines are present) so this isolates the examples assertion.
printf 'fn main() { infer_cuda::fixture(); println!("tick"); }\n' > "$FIX/crates/infer-cuda/examples/fixture_example.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm examples
exbase="$(git -C "$FIX" rev-parse HEAD~1)"; extip="$(git -C "$FIX" rev-parse HEAD)"
set +e
MOCK_LINT_LINES='Checking infer-api \nChecking infer-cuda \nChecking cuda-kernels \n' MOCK_EXAMPLES_LINES='' run_hook "$extip" "$exbase" >"$TMP/ex-red.log" 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] || { echo "FAIL: examples red world passed — Fresh examples run not caught" >&2; cat "$TMP/ex-red.log" >&2; exit 1; }
grep -q 'examples lint reported everything Fresh' "$TMP/ex-red.log"

# Green world for the examples guard: the examples clippy checks the example.
MOCK_LINT_LINES='Checking infer-api \nChecking infer-cuda \nChecking cuda-kernels \n' MOCK_EXAMPLES_LINES='Checking fixture_example \n' run_hook "$extip" "$exbase" >"$TMP/ex-green.log" 2>&1 \
  || { echo "FAIL: examples green world rejected" >&2; cat "$TMP/ex-green.log" >&2; exit 1; }


# --- Lock worlds -----------------------------------------------------------
# The hook lock is machine-global at ${TMPDIR}/arle-pre-push-cargo.lock (the
# acquired only for runs that do cargo work. These worlds use a private TMPDIR
# (no collision with a real lock) and no ARLE_PRE_PUSH_NESTED so the lock is
# live. A cargo-free (docs-only, Metal off) run takes a PRIVATE snapshot and
# must NOT wait; a .rs run uses the shared snapshot + lock and must wait.
LOCK_TMP="$TMP/locktmp"; mkdir -p "$LOCK_TMP"
run_hook_locked() {  # $1 = local sha, $2 = remote sha
  ( cd "$FIX" && TMPDIR="$LOCK_TMP" PATH="$BIN:$PATH" bash "$FIX/scripts/pre_push_checks.sh" ) <<<"refs/heads/lane/x $1 refs/heads/lane/x $2"
}
hold_lock() { mkdir "$LOCK_TMP/arle-pre-push-cargo.lock"; printf '%s\n' "$$" > "$LOCK_TMP/arle-pre-push-cargo.lock/pid"; }
release_lock() { rm -rf "$LOCK_TMP/arle-pre-push-cargo.lock"; }

# No-wait world: docs-only, Metal off — no cargo step, so the hook takes a
# private snapshot and must not wait on a peer's held lock. The docs commit has
# coretip as parent, so coretip..docstip is README-only.
export GIT_INDEX_FILE="$TMP/docsidx"
git -C "$FIX" read-tree "$coretip"
dblob="$(printf 'docs\n' | git -C "$FIX" hash-object -w --stdin)"
git -C "$FIX" update-index --add --cacheinfo 100644,"$dblob",README.md
dtree="$(git -C "$FIX" write-tree)"
docstip="$(git -C "$FIX" commit-tree "$dtree" -p "$coretip" -m docs)"
unset GIT_INDEX_FILE
hold_lock
(run_hook_locked "$docstip" "$coretip" >"$TMP/skip.log" 2>&1; echo $? > "$TMP/skip.rc") &
skip_pid=$!
for _ in $(seq 1 60); do kill -0 "$skip_pid" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$skip_pid" 2>/dev/null; then
  kill "$skip_pid" 2>/dev/null || true
  echo "FAIL: docs-only push waited on the snapshot lock" >&2; cat "$TMP/skip.log" >&2; exit 1
fi
[ "$(cat "$TMP/skip.rc")" = "0" ] || { echo "FAIL: docs-only push rejected" >&2; cat "$TMP/skip.log" >&2; exit 1; }
! grep -q 'waiting for peer pre-push hook' "$TMP/skip.log" || { echo "FAIL: no-wait world printed the wait message" >&2; cat "$TMP/skip.log" >&2; exit 1; }
grep -q 'private snapshot' "$TMP/skip.log" || { echo "FAIL: docs-only push did not select a private snapshot" >&2; cat "$TMP/skip.log" >&2; exit 1; }
release_lock

# Wait world: a .rs push runs cargo steps, uses the shared snapshot + lock, and
# waits for the held lock, completing once released.
hold_lock
(run_hook_locked "$coretip" "$corebase" >"$TMP/wait.log" 2>&1; echo $? > "$TMP/wait.rc") &
wait_pid=$!
sleep 2
kill -0 "$wait_pid" 2>/dev/null || { echo "FAIL: .rs push did not wait for the held lock" >&2; cat "$TMP/wait.log" >&2; exit 1; }
grep -q 'waiting for peer pre-push hook' "$TMP/wait.log" || { echo "FAIL: wait message missing" >&2; cat "$TMP/wait.log" >&2; exit 1; }
release_lock
for _ in $(seq 1 120); do kill -0 "$wait_pid" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$wait_pid" 2>/dev/null; then
  kill "$wait_pid" 2>/dev/null || true
  echo "FAIL: hook did not complete after lock release" >&2; cat "$TMP/wait.log" >&2; exit 1
fi
[ "$(cat "$TMP/wait.rc")" = "0" ] || { echo "FAIL: wait world rejected" >&2; cat "$TMP/wait.log" >&2; exit 1; }

echo "PASS: CUDA lint + test-step freshness (red/green/new-branch/control/test-red/color/examples-red/examples-green), snapshot lock (docs no-wait / .rs wait)"
