#!/usr/bin/env bash
# Contracts for scripts/run_shell_tests.sh:
#   - discovered tests run; a per-file TEST-SKIP for this platform is honored
#   - an unannotated failing test fails the runner
#   - an unknown skip token fails the runner (no silent platform typos)
#   - an irrelevant changed-file list skips; a relevant one runs
#   - an undeterminable/empty list runs everything (never runs nothing)
# No git, cargo, network, or platform-specific tools.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

platform="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$platform" in
    linux*)  platform=linux ;;
    darwin*) platform=darwin ;;
esac

# Build a fixture repo: <TMP>/scripts/{run_shell_tests.sh,tests/}.
mkdir -p "$TMP/scripts/tests"
cp "$ROOT/scripts/run_shell_tests.sh" "$TMP/scripts/run_shell_tests.sh"
chmod +x "$TMP/scripts/run_shell_tests.sh"

mk() { printf '%s\n' "$2" > "$TMP/scripts/tests/$1"; chmod +x "$TMP/scripts/tests/$1"; }

fail=0
clean_env=(env -u GITHUB_BASE_REF -u BASE_SHA SHELL_TEST_CHANGED_FILES=)

# 1. all-pass world: one plain pass, one skip annotated for THIS platform.
mk test_pass.sh '#!/usr/bin/env bash
exit 0'
mk test_skipped.sh "#!/usr/bin/env bash
# TEST-SKIP: $platform: fixture skip reason
exit 1"
if "${clean_env[@]}" bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/ok.log" 2>&1; then
    grep -q "SKIP scripts/tests/test_skipped.sh ($platform): fixture skip reason" "$TMP/ok.log" \
        && echo "ok: passing world, own-platform skip printed with reason" \
        || { echo "FAIL: skip line missing" >&2; fail=1; }
    grep -qE '2 tests, 1 ran, 1 skipped, 0 failed' "$TMP/ok.log" \
        || { echo "FAIL: summary count wrong" >&2; cat "$TMP/ok.log" >&2; fail=1; }
else
    echo "FAIL: passing world exited nonzero" >&2; cat "$TMP/ok.log" >&2; fail=1
fi

# 2. unannotated failure is a failure, never a skip.
mk test_red.sh '#!/usr/bin/env bash
echo red-output
exit 7'
if "${clean_env[@]}" bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/red.log" 2>&1; then
    echo "FAIL: unannotated failing test did not fail the runner" >&2; fail=1
else
    grep -q 'FAIL: scripts/tests/test_red.sh exited 7' "$TMP/red.log" \
        && grep -q 'full log: .*test_red.sh.log' "$TMP/red.log" \
        && lf="$(grep -o 'full log: [^ ]*test_red.sh.log' "$TMP/red.log" | awk '{print $3}')" \
        && [ -s "$lf" ] && grep -q red-output "$lf" \
        && echo "ok: unannotated failure fails runner; full output retained on disk" \
        || { echo "FAIL: failure diagnostics or retained log wrong" >&2; cat "$TMP/red.log" >&2; fail=1; }
fi
rm "$TMP/scripts/tests/test_red.sh"

# 3. unknown skip token fails the runner.
mk test_badtok.sh '#!/usr/bin/env bash
# TEST-SKIP: windows: typo platform
exit 0'
if "${clean_env[@]}" bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/tok.log" 2>&1; then
    echo "FAIL: unknown TEST-SKIP token did not fail the runner" >&2; fail=1
else
    grep -q "unknown TEST-SKIP platform 'windows'" "$TMP/tok.log" \
        && echo "ok: unknown skip token fails runner" \
        || { echo "FAIL: token diagnostics wrong" >&2; cat "$TMP/tok.log" >&2; fail=1; }
fi
rm "$TMP/scripts/tests/test_badtok.sh"

# 4. irrelevant changed-file list skips; a relevant one runs.
if env -u GITHUB_BASE_REF -u BASE_SHA SHELL_TEST_CHANGED_FILES=$'docs/note.md\nCHANGELOG.md' \
        bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/irrel.log" 2>&1; then
    grep -q 'skipping shell test batch' "$TMP/irrel.log" \
        && echo "ok: irrelevant list skips" \
        || { echo "FAIL: irrelevant list did not skip" >&2; cat "$TMP/irrel.log" >&2; fail=1; }
else
    echo "FAIL: irrelevant list exited nonzero" >&2; cat "$TMP/irrel.log" >&2; fail=1
fi
if env -u GITHUB_BASE_REF -u BASE_SHA SHELL_TEST_CHANGED_FILES='scripts/lever_gate.sh' \
        bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/rel.log" 2>&1; then
    grep -qE '2 tests, 1 ran, 1 skipped, 0 failed' "$TMP/rel.log" \
        && echo "ok: relevant list runs the batch" \
        || { echo "FAIL: relevant list did not run" >&2; cat "$TMP/rel.log" >&2; fail=1; }
else
    echo "FAIL: relevant list exited nonzero" >&2; cat "$TMP/rel.log" >&2; fail=1
fi

# 5. empty list (manual / undeterminable) runs everything — never zero tests.
if "${clean_env[@]}" bash "$TMP/scripts/run_shell_tests.sh" >"$TMP/empty.log" 2>&1; then
    grep -q 'bash scripts/tests/test_pass.sh' "$TMP/empty.log" \
        && echo "ok: empty list runs the discovered tests" \
        || { echo "FAIL: empty list did not run the batch" >&2; cat "$TMP/empty.log" >&2; fail=1; }
fi

if [ "$fail" -ne 0 ]; then exit 1; fi
echo "test_run_shell_tests: PASS (discovery, skip, red, bad token, relevance, empty-runs-all)"
