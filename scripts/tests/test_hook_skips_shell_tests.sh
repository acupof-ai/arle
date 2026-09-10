#!/usr/bin/env bash
# The shell-test batch (lever gate, pod flow, prebuilt export, …) runs for
# minutes. A push that changes nothing under scripts/ / .githooks/ / .github/
# has nothing new for it — CI's Shell Contracts lane runs all of them on every
# PR regardless — so the hook skips the batch and keeps the pre-push SSH
# connection from sitting idle until the remote drops it (exit 141).
# This test pins both directions: a docs-only push skips; a scripts/ push runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
FIX="$TMP/fix"
SNAP_HASH="$(printf '%s' "$FIX" | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-16)"
trap 'rm -rf "$TMP" "${TMPDIR:-/tmp}/arle-pre-push-snapshot-${SNAP_HASH}"' EXIT
BIN="$TMP/bin"; mkdir -p "$BIN"

# Fixture repo: base carries no-op stand-ins for the hook's fast checks.
mkdir -p "$FIX/scripts/tests" "$FIX/docs"
cp "$ROOT/scripts/pre_push_checks.sh" "$FIX/scripts/pre_push_checks.sh"
for t in $(sed -n '/for test in \\/,/; do/p' "$FIX/scripts/pre_push_checks.sh" | grep -o 'test_[a-z_]*\.sh'); do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$FIX/scripts/tests/$t"
done
printf 'print("fixture hygiene ok")\n' > "$FIX/scripts/check_repo_hygiene.py"
git -C "$FIX" init -q
git -C "$FIX" config user.email t@e; git -C "$FIX" config user.name t
git -C "$FIX" add -A && git -C "$FIX" commit -qm base
base="$(git -C "$FIX" rev-parse HEAD)"

# Mock cargo: every invocation is a no-op (this test asserts only which SHELL
# tests the hook launches, not cargo freshness). The verbose CUDA lint prints
# Checking lines for the crates a cuda-kernels push changes, so the freshness
# assertion does not abort before the shell batch under test runs.
cat > "$BIN/cargo" <<'SH'
#!/usr/bin/env bash
verbose=0
for a in "$@"; do [ "$a" = -v ] && verbose=1; done
if [ "$verbose" = 1 ] && [[ " $* " == *" cuda,no-cuda "* ]]; then
  printf 'Checking infer-api \nChecking infer-cuda \nChecking cuda-kernels \n'
fi
exit 0
SH
chmod +x "$BIN/cargo"

run_hook() {  # $1 = local sha, $2 = remote sha
  ( cd "$FIX" && ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" bash "$FIX/scripts/pre_push_checks.sh" ) <<<"refs/heads/lane/x $1 refs/heads/lane/x $2"
}

# World 1 — docs-only push: the shell batch must be skipped (hygiene/fmt stay).
printf 'note\n' > "$FIX/docs/note.txt"
git -C "$FIX" add -A && git -C "$FIX" commit -qm docs
docs_sha="$(git -C "$FIX" rev-parse HEAD)"
set +e
run_hook "$docs_sha" "$base" >"$TMP/docs.log" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] || { echo "FAIL: docs-only push hook exited $rc" >&2; cat "$TMP/docs.log" >&2; exit 1; }
grep -q 'skipping shell test batch' "$TMP/docs.log" \
  || { echo "FAIL: docs-only push did not announce the shell-test skip" >&2; cat "$TMP/docs.log" >&2; exit 1; }
if grep -q 'bash scripts/tests/' "$TMP/docs.log"; then
  echo "FAIL: docs-only push still launched a shell test:" >&2
  grep 'bash scripts/tests/' "$TMP/docs.log" >&2; exit 1
fi

# World 2 — scripts/ push: the shell batch runs (the fixture stubs exit 0).
printf '# stub\n' > "$FIX/scripts/note.sh"
git -C "$FIX" add -A && git -C "$FIX" commit -qm scripts
scr_sha="$(git -C "$FIX" rev-parse HEAD)"
set +e
run_hook "$scr_sha" "$docs_sha" >"$TMP/scr.log" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] || { echo "FAIL: scripts push hook exited $rc" >&2; cat "$TMP/scr.log" >&2; exit 1; }
if grep -q 'skipping shell test batch' "$TMP/scr.log"; then
  echo "FAIL: a scripts/ push must not skip the shell batch" >&2; exit 1
fi
grep -q 'bash scripts/tests/test_pod_flow.sh' "$TMP/scr.log" \
  || { echo "FAIL: scripts/ push did not launch the shell test batch" >&2; cat "$TMP/scr.log" >&2; exit 1; }

# World 3 — a crates/cuda-kernels/ change must still run the batch:
# test_cuda_prebuilt_export / test_kernel_artifact_qualification / test_pod_flow
# read build.rs, kernels.toml, generated/ and the crate package, so a change
# there is exactly what those tests should catch.
mkdir -p "$FIX/crates/cuda-kernels"
printf '// build script touched\n' > "$FIX/crates/cuda-kernels/build.rs"
git -C "$FIX" add -A && git -C "$FIX" commit -qm cuda-kernels
ck_sha="$(git -C "$FIX" rev-parse HEAD)"
set +e
run_hook "$ck_sha" "$scr_sha" >"$TMP/ck.log" 2>&1
rc=$?; set -e
[ "$rc" -eq 0 ] || { echo "FAIL: cuda-kernels push hook exited $rc" >&2; cat "$TMP/ck.log" >&2; exit 1; }
if grep -q 'skipping shell test batch' "$TMP/ck.log"; then
  echo "FAIL: a crates/cuda-kernels/ push must not skip the shell batch" >&2; exit 1
fi
grep -q 'bash scripts/tests/test_kernel_artifact_qualification.sh' "$TMP/ck.log" \
  || { echo "FAIL: cuda-kernels push did not run the kernel artifact test" >&2; cat "$TMP/ck.log" >&2; exit 1; }

echo "PASS: shell-test batch skips for unrelated pushes, runs for scripts and cuda-kernels"
