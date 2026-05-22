#!/usr/bin/env bash
# Run `rustfmt --check` only on .rs files that this push/PR actually changed,
# instead of the whole workspace.
#
# Why: when a parallel-track worktree has dirty .rs files (e.g. codex
# mid-tranche), `cargo fmt --all -- --check` trips on those files for every
# unrelated push to main. That made every docs-only commit appear to break
# CI. See docs/experience/errors/2026-05-22-ci-cudarc-leak-and-hygiene-drift.md
# for the full root-cause writeup.
#
# Behavior:
#   - PR  (GITHUB_BASE_REF set): diff against `origin/$GITHUB_BASE_REF`
#   - push (no base ref):         diff against `HEAD~1`
#   - if no .rs files changed:    exit 0 with a notice
#   - if any changed .rs file fails formatting: exit 1
#
# Drift caveat: this is the fast-feedback gate. A periodic full-workspace
# fmt sweep (separate workflow, daily) catches drift on files no commit
# happens to touch.

set -euo pipefail

if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    BASE="origin/${GITHUB_BASE_REF}"
    git fetch --no-tags --depth=200 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1 || true
elif git rev-parse HEAD~1 >/dev/null 2>&1; then
    BASE="HEAD~1"
else
    echo "no HEAD~1 available (likely shallow clone of root commit); skipping fmt scope"
    exit 0
fi

mapfile -t FILES < <(git diff --name-only --diff-filter=ACMRT "${BASE}" HEAD -- '*.rs')

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no .rs files changed between ${BASE} and HEAD; skipping fmt"
    exit 0
fi

echo "checking fmt on ${#FILES[@]} changed .rs files:"
printf '  %s\n' "${FILES[@]}"

# rustfmt itself reads .rustfmt.toml from the workspace root.
rustfmt --check --edition 2024 "${FILES[@]}"
