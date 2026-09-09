#!/usr/bin/env bash
#
# CI-aligned local validation to run before `git push`.
#
# Usage:
#   scripts/pre_push_checks.sh
#
# Speed: HEAD is exported into a STABLE snapshot dir and refreshed with
# rsync --checksum, so unchanged files keep their mtimes and cargo's
# incremental cache (target/pre-push-quick) stays warm across runs. The
# previous mktemp-per-run snapshot changed every source path on every
# push, invalidating all workspace-crate fingerprints — a full cold
# rebuild (~5-8 min) per push, which is why the .githooks/pre-push hook
# got disabled. Warm runs are now sub-minute.
#
# The snapshot lives OUTSIDE the repo on purpose: inside the repo tree,
# git commands run by the hygiene check would discover the parent repo
# and operate on it instead of the snapshot.
#
# Compile skip: when the pushed range contains no .rs files, all cargo
# steps are skipped (docs/config-only pushes can't break compilation).
# Fast checks (hygiene, fmt, shell tests) run in parallel with the
# cargo steps to hide their latency.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Per-worktree snapshot: a shared dir lets concurrent lane hooks rsync
# different HEADs into one tree, and an interrupted rsync leaves a mix that
# fails content-hash tests (kernel bundle id drift, 2026-09-09).
SNAPSHOT_HASH="$(printf '%s' "$REPO_ROOT" | (sha256sum 2>/dev/null || shasum -a 256) | cut -c1-16)"
SNAPSHOT_ROOT="${TMPDIR:-/tmp}/arle-pre-push-snapshot-${SNAPSHOT_HASH}"
STAGE_ROOT=""

info() { echo "[pre-push] $*"; }
fail() { echo "[pre-push] $*" >&2; }

run() {
    info "$*"
    "$@"
}

cleanup() {
    if [[ -n "${STAGE_ROOT}" && -d "${STAGE_ROOT}" ]]; then
        rm -rf "${STAGE_ROOT}"
    fi
    [[ -n "${FAST_STEP:-}" ]] && rm -f "${FAST_STEP}"
    return 0
}

trap cleanup EXIT

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arle-pre-push-stage.XXXXXX")"
info "refreshing HEAD snapshot at ${SNAPSHOT_ROOT}"
git -C "${REPO_ROOT}" archive HEAD | tar -x -C "${STAGE_ROOT}"
mkdir -p "${SNAPSHOT_ROOT}"
# --checksum keeps mtimes of content-identical files untouched (cargo sees
# them as unchanged); --delete drops files removed from HEAD.
rsync -a --delete --checksum "${STAGE_ROOT}/" "${SNAPSHOT_ROOT}/"
cd "${SNAPSHOT_ROOT}"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
# REPO_ROOT is the *worktree* root, so this used to give every lane its own
# pre-push target tree — 4.8 GB each, 18.6 GB across four lanes, and the env var
# beat the shared `target-dir` in ../arle-lanes/.cargo/config.toml. Anchor it to
# the main checkout (the parent of the common git dir) so all worktrees share
# one, the way ordinary builds already do.
MAIN_ROOT="$(dirname "$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)"
[[ -d "${MAIN_ROOT}" ]] || MAIN_ROOT="${REPO_ROOT}"
export CARGO_TARGET_DIR="${MAIN_ROOT}/target/pre-push-quick"
# cudarc probes the CUDA version at build time; pin it so the cuda,no-cuda
# typecheck works on hosts without nvcc.
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"

# --- Determine whether cargo steps can skip -------------------------------
# Pre-push stdin: <local_ref> <local_sha> <remote_ref> <remote_sha>.
# If the pushed range has zero .rs files, compilation cannot catch anything
# new — skip it. Empty stdin (manual run) defaults to compiling.
# changed_files also drives the CUDA lint freshness assertion below.
SKIP_CARGO=0
CUDA_CRATES_CHANGED=0
changed_files=""
while read -r _local_ref local_sha _remote_ref remote_sha; do
    if [[ "$remote_sha" =~ ^0{40}$ ]]; then
        # New branch: no remote tip to diff. The merge-base with main is the
        # range the push actually adds; HEAD~1 covers a repo with no origin.
        base="$(git -C "${REPO_ROOT}" merge-base HEAD origin/main 2>/dev/null || echo HEAD~1)"
        changed_files="$(git -C "${REPO_ROOT}" diff --name-only "${base}..HEAD" 2>/dev/null || true)"
    else
        changed_files="$(git -C "${REPO_ROOT}" diff --name-only "${remote_sha}..${local_sha}" 2>/dev/null || true)"
    fi
    if ! grep -q '\.rs$' <<< "${changed_files}"; then
        SKIP_CARGO=1
        info "no .rs files in pushed range; skipping cargo steps"
    fi
    if grep -qE '^crates/(infer-cuda|cuda-kernels)/' <<< "${changed_files}"; then
        CUDA_CRATES_CHANGED=1
    fi
    break
done

# --- Fast checks (parallel with cargo) ------------------------------------
# This block runs in the background, so its failure surfaces only as the exit
# status of `wait` and git then prints a bare "failed to push some refs". Record
# the step in flight so the failure can name itself.
# Its own file, not one under STAGE_ROOT: that directory is the HEAD archive
# staging area and does not outlive the snapshot refresh, so the write failed
# and, under `set -e`, took the whole fast block down with it. A diagnostic must
# never be able to fail the thing it is diagnosing — hence `|| true`.
FAST_STEP="$(mktemp "${TMPDIR:-/tmp}/arle-pre-push-faststep.XXXXXX")"
run_fast() { printf '%s\n' "$*" > "${FAST_STEP}" 2>/dev/null || true; run "$@"; }

run_fast_checks() {
    run_fast python3 scripts/check_repo_hygiene.py
    run_fast python3 scripts/check_repo_hygiene.py --selftest
    run_fast cargo fmt --all -- --check
    for test in \
        test_cuda_prebuilt_export.sh \
        test_lever_gate.sh \
        test_kernel_artifact_qualification.sh \
        test_validate_release.sh \
        test_pod_flow.sh \
        test_hook_disowns_git_env.sh \
        test_hook_cuda_lint_freshness.sh; do
        run_fast bash "scripts/tests/${test}"
    done
}
run_fast_checks &
FAST_PID=$!

# --- Cargo steps (serial — cargo locks the target dir) ---------------------
if [[ "${SKIP_CARGO}" == "0" ]]; then
    run cargo check -p arle --no-default-features --features cpu,no-cuda,cli --bin arle
    # CI's test-backend lane runs `-p arle`; the hook did not, so a CLI help
    # rewrite landed on main with cli_smoke red. Same feature set as the check
    # above, so the binary is already built.
    run cargo test -p arle --no-default-features --features cpu,no-cuda,cli --test cli_smoke
    # Clippy (not check) on the cuda lane: catches clippy lints (missing_safety_doc,
    # needless_borrow) that plain check misses — the gap that let quant_linear.rs
    # clippy errors pass the hook and fail CI. Debug profile shares the cache with
    # the arle check above; the old --release forced a second full compilation.
    # The CUDA clippy is the only automated gate for the CUDA-Rust surface
    # (no GPU CI). All lanes share one target dir and cargo fingerprints path
    # deps by mtime, so another lane's newer artifact can make this run report
    # Fresh and check nothing — a real infer-cuda compile error passed the gate
    # green on 2026-09-10. When the pushed range changes the CUDA crates,
    # assert the lint actually recompiled them. The condition is the changed
    # crates only: an unchanged crate is legitimately Fresh, and asserting on
    # it would false-positive the gate into being disabled.
    if [[ "${CUDA_CRATES_CHANGED}" == "1" ]]; then
        cuda_lint_rc=0
        cuda_lint_out="$(CARGO_TERM_COLOR=never cargo clippy -v -p infer-api --no-default-features --features cuda,no-cuda --lib -- -D warnings 2>&1)" || cuda_lint_rc=$?
        printf '%s\n' "${cuda_lint_out}"
        [[ "${cuda_lint_rc}" -eq 0 ]] || exit "${cuda_lint_rc}"
        for crate in infer-cuda cuda-kernels; do
            if grep -qE "^crates/${crate}/" <<< "${changed_files}" \
               && ! grep -qE "(Checking|Compiling) ${crate}( |\$)" <<< "${cuda_lint_out}"; then
                fail "CUDA lint reported ${crate} Fresh while this push changes it — stale shared target; run 'cargo clean -p ${crate}' and retry"
                exit 1
            fi
        done
    else
        run cargo clippy -p infer-api --no-default-features --features cuda,no-cuda --lib -- -D warnings
    fi
    # CI's CPU-only clippy lane (job "cargo clippy (CPU-only surfaces)"): a push
    # that is clean on the cuda lane above still failed CI here (#258). Same
    # feature sets as CI, verbatim — the hook crate list drifts otherwise.
    run cargo clippy -p infer-api --no-default-features --features no-cuda --lib -- -D warnings
    run cargo clippy -p cli --no-default-features --features no-cuda -- -D warnings
    run cargo clippy -p arle --no-default-features --features cpu,no-cuda,cli --bin arle -- -D warnings
    run cargo clippy -p autograd --features no-cuda --lib -- -D warnings
    run cargo clippy -p train --features no-cuda --lib -- -D warnings
    run cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p spec-train -p kv-native-sys -p infer-quant
    run cargo test \
        -p infer-core -p infer-server -p infer-plan -p infer-seam \
        -p infer-moe -p infer-topo -p infer-util -p deepseek-spec -p agent
    run cargo clippy -p kv-native-sys --all-targets -- -D warnings

    # Metal lib check (default-on, Mac only): catches dead-code/unused lints in
    # infer-metal that CI's Metal lane runs with -D warnings. The full binary
    # build + needle gate stays opt-in (ARLE_PRE_PUSH_METAL=1) below.
    # Targets infer-metal directly: infer-api has carried no backend features
    # since runtime backend dispatch (#68), so `-p infer-api --features metal`
    # no longer resolves.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        run cargo check -p infer-metal --no-default-features --features metal
    fi
else
    info "skipping cargo steps (docs/config-only push)"
fi

# --- Wait for fast checks ---------------------------------------------------
if ! wait "${FAST_PID}"; then
    fail "parallel fast-checks FAILED at: $(cat "${FAST_STEP}" 2>/dev/null || echo "unknown step")"
    fail "its output is above, interleaved with the cargo steps"
    exit 1
fi

METAL_CHECKS="${ARLE_PRE_PUSH_METAL:-${AGENT_INFER_PRE_PUSH_METAL:-0}}"

if [[ "${METAL_CHECKS}" == "1" && "$(uname -s)" == "Darwin" ]]; then
    run cargo check -p infer-metal --no-default-features --features metal --profile release-fast
    run cargo build --no-default-features --features metal,no-cuda,cli -p arle --profile release-fast --bin arle
    # Metal correctness gate: needle ladder on the local 0.8B test model.
    GATE_BIN="${CARGO_TARGET_DIR}/release-fast/arle"
    GATE_MODEL="${REPO_ROOT}/models/Qwen3.5-0.8B-MLX-4bit"
    if [[ -x "$GATE_BIN" && -d "$GATE_MODEL" ]]; then
        info "Metal needle gate (Qwen3.5-0.8B-MLX-4bit, lengths 115/300/446)"
        BIN="$GATE_BIN" MODEL="$GATE_MODEL" \
        GATE_PROFILE=metal LENGTHS=115,300,446 RUNS=1 \
        PORT=18189 LEVER_GATE_ALLOW_NO_BASELINE=1 LEVER_GATE_SKIP_TEMP=1 LEVER_GATE_SKIP_CONCURRENT=1 \
        RUST_LOG=warn \
        bash scripts/lever_gate.sh "prepush-$$" || {
            echo "[pre-push] Metal needle gate FAIL" >&2
            exit 1
        }
    else
        info "skipping Metal needle gate (binary or model missing)"
    fi
elif [[ "${METAL_CHECKS}" == "1" ]]; then
    info "skipping Metal-only checks on non-macOS host"
else
    info "skipping Metal checks; set ARLE_PRE_PUSH_METAL=1 (legacy AGENT_INFER_PRE_PUSH_METAL also works) to enable"
fi

info "quick pre-push checks passed"
