#!/usr/bin/env bash
#
# Runs the scripts/tests/test_*.sh batch for the ci.yml "Shell Contracts" step.
#
# The list is discovered, never enumerated: every scripts/tests/test_*.sh
# runs on both callers, so a new test cannot land in the hook but stay absent
# from CI (a parity-batch test once merged "green" with CI never running it).
#
# A test that cannot run on a platform says so in its own first lines:
#   # TEST-SKIP: linux: <reason>
#   # TEST-SKIP: darwin: <reason>
# The skip is printed with the reason. An unknown platform token is an error;
# an unannotated failure is always a failure. There are no skip lists anywhere
# else.
#
# Whether the batch is worth running follows RELEVANCE_RE below, the same rule
# for both callers. The hook computes changed files itself and exports them as
# SHELL_TEST_CHANGED_FILES; in GitHub Actions the list is derived from the
# PR/push base. With neither, every test runs. A base that cannot be resolved
# runs every test — never a silent skip.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One copy of the relevance rule; the hook sources this file and reuses it.
RELEVANCE_RE='^(scripts|\.github)/|^\.gitignore$|^crates/cuda-kernels/'

main() {
    cd "$REPO_ROOT"
    local ZERO_SHA="0000000000000000000000000000000000000000"
    local changed_files="${SHELL_TEST_CHANGED_FILES:-}"
    local base=""

    if [[ -z "$changed_files" && ( -n "${GITHUB_BASE_REF:-}" || -n "${BASE_SHA:-}" ) ]]; then
        if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
            git fetch --no-tags --depth=200 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1 || true
            git rev-parse --verify --quiet "origin/${GITHUB_BASE_REF}" >/dev/null && base="origin/${GITHUB_BASE_REF}"
        fi
        if [[ -z "$base" && -n "${BASE_SHA:-}" && "${BASE_SHA}" != "$ZERO_SHA" ]]; then
            git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null || \
                git fetch --no-tags --depth=1 origin "${BASE_SHA}" >/dev/null 2>&1 || true
            git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null && base="${BASE_SHA}"
        fi
        if [[ -n "$base" ]]; then
            changed_files="$(git diff --name-only "$base" HEAD 2>/dev/null || true)"
        else
            echo "shell-tests: no diff base resolvable; running the full batch (never skip silently)"
        fi
    fi

    if [[ -n "$changed_files" ]] && ! grep -qE "$RELEVANCE_RE" <<< "$changed_files"; then
        echo "no shell-test inputs (scripts/.github/cuda-kernels/.gitignore) in pushed range; skipping shell test batch"
        return 0
    fi

    local platform
    case "$(uname -s)" in
        Linux*)  platform="linux" ;;
        Darwin*) platform="darwin" ;;
        *)       platform="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
    esac

    local tests=()
    local t
    for t in scripts/tests/test_*.sh; do
        [[ -e "$t" ]] && tests+=("$t")
    done
    if [[ ${#tests[@]} -eq 0 ]]; then
        echo "shell-tests: no scripts/tests/test_*.sh found" >&2
        return 1
    fi

    local failures=() skipped=0 ran=0 total=${#tests[@]}
    local skip_platform skip_reason line rest tok why log rc
    local runner=(bash)
    command -v setsid >/dev/null 2>&1 && runner=(setsid bash)

    # Full per-test output is kept, not only the failure tail: a push hook
    # interleaves this batch with cargo output, so after a failure the test's
    # own diagnostics must still be on disk. A per-run mktemp dir (overlapping
    # hook runs share the box) symlinked from "latest".
    local log_base="${SHELL_TEST_LOG_BASE:-${TMPDIR:-/tmp}/arle-shell-tests}"
    mkdir -p "$log_base"
    # Bound accumulation: drop per-run dirs older than 7 days.
    find "$log_base" -maxdepth 1 -type d -name 'run.*' -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    local log_dir
    log_dir="$(mktemp -d "$log_base/run.XXXXXX")"
    ln -sfn "$log_dir" "$log_base/latest" 2>/dev/null || true
    echo "shell-tests: per-test logs in $log_dir"

    for t in "${tests[@]}"; do
        skip_platform=""; skip_reason=""
        # Directive lives in the file's first 10 lines.
        while IFS= read -r line; do
            case "$line" in
                *"# TEST-SKIP:"*)
                    rest="${line#*# TEST-SKIP: }"
                    tok="${rest%%:*}"
                    why="${rest#*: }"
                    case "$tok" in
                        linux|darwin) ;;
                        *) echo "FAIL: $t: unknown TEST-SKIP platform '$tok' (allowed: linux, darwin)" >&2; return 1 ;;
                    esac
                    [[ "$tok" == "$platform" ]] && { skip_platform="$tok"; skip_reason="$why"; }
                    ;;
            esac
        done < <(head -10 "$t")

        if [[ -n "$skip_platform" ]]; then
            echo "SKIP $t ($skip_platform): $skip_reason"
            skipped=$((skipped + 1))
            continue
        fi

        ran=$((ran + 1))
        log="$log_dir/${t##*/tests/}.log"
        echo "bash $t"
        # setsid gives the test its own process group: a trap that signals its
        # group (`kill "${SRV_PID:-0}"` on early setup failure) cannot take down
        # this runner or the CI step.
        if "${runner[@]}" "$t" >"$log" 2>&1; then
            :
        else
            rc=$?
            echo "FAIL: $t exited $rc — full log: $log" >&2
            tail -40 "$log" >&2
            failures+=("$t")
        fi
    done

    echo "$total tests, $ran ran, $skipped skipped, ${#failures[@]} failed on $platform"
    if [[ ${#failures[@]} -gt 0 ]]; then
        printf 'failed: %s\n' "${failures[@]}" >&2
        return 1
    fi
}

# Sourced by the pre-push hook for RELEVANCE_RE; executed in CI and by hand.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
