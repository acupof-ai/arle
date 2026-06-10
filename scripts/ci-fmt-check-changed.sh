#!/usr/bin/env bash
# Run `rustfmt --check` only on .rs files that this push/PR actually changed,
# instead of the whole workspace.
#
# Why: when a parallel track commits an unformatted .rs file, workspace-wide
# `cargo fmt --all -- --check` fails every subsequent unrelated push. Scope to
# the actual delta so CI failure tracks the push, not the tree. See
# docs/experience/errors/2026-05-22-ci-cudarc-leak-and-hygiene-drift.md.
#
# Base resolution (first hit wins):
#   - PR   (GITHUB_BASE_REF set):  diff against origin/$GITHUB_BASE_REF
#   - push (BASE_SHA set):         diff against the pre-push SHA — covers
#                                  multi-commit pushes, unlike HEAD~1
#   - else:                        diff against HEAD~1
#   - no base resolvable:          FULL `cargo fmt --all -- --check`.
#                                  Never skip silently — a depth-1 clone made
#                                  this check a no-op on every push for weeks.
#
# Drift caveat: this is the fast-feedback gate. The nightly full-workspace
# sweep (fmt-nightly.yml) catches drift on files no commit happens to touch.

set -euo pipefail

ZERO_SHA="0000000000000000000000000000000000000000"
BASE=""

if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    git fetch --no-tags --depth=200 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1 || true
    if git rev-parse --verify --quiet "origin/${GITHUB_BASE_REF}" >/dev/null; then
        BASE="origin/${GITHUB_BASE_REF}"
    fi
fi

if [[ -z "$BASE" && -n "${BASE_SHA:-}" && "${BASE_SHA}" != "$ZERO_SHA" ]]; then
    # Pre-push SHA from the workflow (github.event.before). Fetch it if the
    # shallow clone doesn't already contain it.
    if ! git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
        git fetch --no-tags --depth=1 origin "${BASE_SHA}" >/dev/null 2>&1 || true
    fi
    if git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
        BASE="${BASE_SHA}"
    fi
fi

if [[ -z "$BASE" ]] && git rev-parse --verify --quiet HEAD~1 >/dev/null; then
    BASE="HEAD~1"
fi

if [[ -z "$BASE" ]]; then
    echo "no diff base resolvable; falling back to full-workspace fmt check"
    exec cargo fmt --all -- --check
fi

# Portable array fill — macOS ships bash 3.2, which has no `mapfile`.
FILES=()
while IFS= read -r f; do
    FILES+=("$f")
done < <(git diff --name-only --no-renames --diff-filter=ACMT "${BASE}" HEAD -- '*.rs')

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no .rs files changed between ${BASE} and HEAD; skipping fmt"
    exit 0
fi

echo "checking fmt on ${#FILES[@]} changed .rs files (base: ${BASE}):"
printf '  %s\n' "${FILES[@]}"

# rustfmt itself reads .rustfmt.toml from the workspace root.
rustfmt --check --edition 2024 "${FILES[@]}"
