#!/usr/bin/env bash
# The pre-push hook runs the body-less content rules over the pushed range, so
# a direct-to-main push that adds what the PR precheck rejects is blocked —
# including the ledger surface (CHANGELOG prose), while the named run-provenance
# ledger (agenda/prereg) stays exempt. One red world per rule: the OLD hook
# accepted the push, the new one must reject with the rule's own line.
#
# Docs-only fixtures: cargo steps are skipped, so no toolchain is needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FIX="$TMP/fix"; BIN="$TMP/bin"; SCRATCH="$TMP/tmp"
mkdir -p "$SCRATCH" "$FIX/scripts/tests" "$BIN"

# Real hook + the real content checker; stub the other fast steps and cargo so
# the test exercises ONLY the content gate (and hygiene/fmt cannot fail it).
cp "$ROOT/scripts/pre_push_checks.sh" "$FIX/scripts/pre_push_checks.sh"
cp "$ROOT/scripts/lane_pr_precheck.py" "$FIX/scripts/lane_pr_precheck.py"
# Hygiene is stubbed (the fixture is not a conforming repo); the real checker
# stays in place because the gate under test calls it via --push-content. Its
# fast-block --selftest invocation also runs, self-contained, on the real file.
printf '#!/usr/bin/env python3\nprint("fixture hygiene ok")\n' > "$FIX/scripts/check_repo_hygiene.py"
# The hook sources the shared shell-test runner unconditionally; copy the real
# one. The batch it runs is skipped for these docs/.rs/.github cases (the
# runner's relevance filter sees no scripts/ change), so no test stubs are
# needed — only the sourceable runner.
cp "$ROOT/scripts/run_shell_tests.sh" "$FIX/scripts/run_shell_tests.sh"
# The .github/ case below matches the runner's relevance filter, so the batch
# actually runs; give it one discovered, passing test (the content gate itself
# is what is under test, not the shell-test batch).
mkdir -p "$FIX/scripts/tests"
printf '#!/usr/bin/env bash\nexit 0\n' > "$FIX/scripts/tests/test_fixture_stub.sh"
printf '#!/usr/bin/env bash\n[[ "$1" == "fmt" ]] && exit 0\nexit 0\n' > "$BIN/cargo"
chmod +x "$BIN/cargo"

git -C "$FIX" init -q
git -C "$FIX" config user.email t@e; git -C "$FIX" config user.name t
git -C "$FIX" branch -M main
git -C "$FIX" remote add origin "$FIX/remote.git"
git -C "$FIX" init -q --bare "$FIX/remote.git" >/dev/null 2>&1
mkdir -p "$FIX/docs"
printf 'base\n' > "$FIX/docs/note.md"
git -C "$FIX" add -A && git -C "$FIX" commit -qm base

run_push() {  # $1=path  $2=content ; exit status of the hook
  local path="$1" content="$2"
  ( cd "$FIX" && mkdir -p "$(dirname "$path")" && printf '%s' "$content" > "$path" \
      && git add -A && git commit -qm change )
  local tip base
  tip="$(git -C "$FIX" rev-parse HEAD)"; base="$(git -C "$FIX" rev-parse HEAD^)"
  ( cd "$FIX"
    TMPDIR="$SCRATCH" ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" \
      ARLE_PRE_PUSH_METAL=0 \
      bash "$FIX/scripts/pre_push_checks.sh"
  ) <<<"refs/heads/main $tip refs/heads/main $base"
}

expect_reject() {  # $1=case  $2=path  $3=content  $4=expected-substring
  local case="$1" path="$2" content="$3" want="$4" log="$TMP/case.log"
  if run_push "$path" "$content" >"$log" 2>&1; then
    echo "FAIL [$case]: hook ACCEPTED a push it must reject" >&2; cat "$log" >&2; exit 1
  fi
  if ! grep -qF "$want" "$log"; then
    echo "FAIL [$case]: rejected but missing expected text '$want'" >&2; cat "$log" >&2; exit 1
  fi
  echo "ok   [$case]: rejected with '${want%%:*}'"
}

expect_accept() {  # $1=case  $2=path  $3=content
  local case="$1" path="$2" content="$3" log="$TMP/case.log"
  if ! run_push "$path" "$content" >"$log" 2>&1; then
    echo "FAIL [$case]: hook rejected an exempt push" >&2; cat "$log" >&2; exit 1
  fi
  echo "ok   [$case]: accepted (exempt)"
}

# Five body-less rules, one rejected push each. The banned machine-path string
# is assembled at runtime (ROOT_SEG + tail), not written literally here: this
# test file is committed content and must itself pass the abs-path rule.
ROOT_SEG="/ho""st"
BADPATH="${ROOT_SEG}/p1-results/results.tsv"
expect_reject abs-path-changelog \
  CHANGELOG.md "$(printf -- '- x\n\nsaw it at %s\n' "$BADPATH")" \
  "abs-path: CHANGELOG.md"

expect_reject comment-ref \
  crates/x/src/lib.rs $'// fixed in #123\npub fn x() {}\n' \
  "comment-ref: crates/x/src/lib.rs"

expect_reject dead-sha \
  docs/note2.md $'see `0123abcd` for the fix\n' \
  "dead-sha: docs/note2.md"

expect_reject gate-registry \
  crates/infer-cuda/examples/foo_parity.rs $'// parity gate\nfn main() {\n  let _ = arg("--negative-control");\n}\n' \
  "gate-registry: crates/infer-cuda/examples/foo_parity.rs"

expect_reject bench-entry \
  crates/y/src/lib.rs $'pub fn y() {}\n' \
  "bench-entry:"

# Exemption pair: the SAME machine path in the named ledger passes.
expect_accept abs-path-agenda-exempt \
  docs/agenda.jsonl "$(printf -- '{"run": "%s"}\n' "$BADPATH")"

# And .github upstream PR references pass comment-ref.
expect_accept comment-ref-github-exempt \
  .github/workflows/x.yml $'# see upstream #123\non: [push]\n'

# Multi-ref: a ref DELETION on the first stdin line must not mask a bad branch
# on the second (`git push origin :old new`). The old `break` stopped after the
# deletion and silently passed the bytes-carrying branch.
( cd "$FIX" && mkdir -p docs && printf 'at %s\n' "$BADPATH" > docs/multi.md \
    && git add -A && git commit -qm multi )
mtip="$(git -C "$FIX" rev-parse HEAD)"; mbase="$(git -C "$FIX" rev-parse HEAD^)"
multi_log="$TMP/multi.log"
if ( cd "$FIX"; TMPDIR="$SCRATCH" ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" \
      ARLE_PRE_PUSH_METAL=0 bash "$FIX/scripts/pre_push_checks.sh" \
    ) <<EOF >"$multi_log" 2>&1
refs/heads/dead $(printf '0%.0s' {1..40}) refs/heads/dead $mbase
refs/heads/main $mtip refs/heads/main $mbase
EOF
then
  echo "FAIL [multi-ref-deletion-first]: hook let a deletion-first push mask the bad branch" >&2
  cat "$multi_log" >&2; exit 1
fi
grep -qF "abs-path: docs/multi.md" "$multi_log" \
  || { echo "FAIL [multi-ref-deletion-first]: rejected but not for the second ref's bad file" >&2; cat "$multi_log" >&2; exit 1; }
echo "ok   [multi-ref-deletion-first]: second ref gated despite first-line deletion"

# Deletion-ONLY stdin and empty stdin must reach the skip decision cleanly, not
# abort on an unbound RELEVANCE_RE under `set -u`: the runner is sourced once
# before the loop, and a loop whose body never runs must still be safe.
del_log="$TMP/deletion-only.log"
ZERO="$(printf '0%.0s' {1..40})"
zero_input="refs/heads/dead ${ZERO} refs/heads/dead $(git -C "$FIX" rev-parse HEAD)"
run_hook_with_input() {  # $1=stdin
  ( cd "$FIX"; TMPDIR="$SCRATCH" ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" \
      ARLE_PRE_PUSH_METAL=0 bash "$FIX/scripts/pre_push_checks.sh" ) <<<"$1"
}
# Deletion-only via stdin.
if ! run_hook_with_input "$zero_input" >"$del_log" 2>&1; then
  echo "FAIL [deletion-only]: hook exited non-zero for a push that deletes only" >&2
  cat "$del_log" >&2; exit 1
fi
# Empty stdin via /dev/null.
if ! ( cd "$FIX"; TMPDIR="$SCRATCH" ARLE_PRE_PUSH_NESTED=1 PATH="$BIN:$PATH" \
        ARLE_PRE_PUSH_METAL=0 bash "$FIX/scripts/pre_push_checks.sh" \
      ) </dev/null >"$del_log" 2>&1; then
  echo "FAIL [empty-stdin]: hook exited non-zero with no stdin" >&2; cat "$del_log" >&2; exit 1
fi
if grep -qiE 'unbound variable|RELEVANCE_RE' "$del_log"; then
  echo "FAIL [deletion-only/empty-stdin]: unbound-variable abort" >&2; cat "$del_log" >&2; exit 1
fi
grep -qE "skipping shell test batch|cargo-free|skipping cargo" "$del_log" \
  || { echo "FAIL [deletion-only/empty-stdin]: did not reach skip decision" >&2; cat "$del_log" >&2; exit 1; }
echo "ok   [deletion-only/empty-stdin]: clean skip, no unbound variable"

echo "PASS: pre-push content gate enforces all five rules, blocks CHANGELOG prose, honors named exemptions, and gates every ref"
