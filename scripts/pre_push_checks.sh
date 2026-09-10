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
STAGE_ROOT=""
SNAPSHOT_ROOT=""
LOCK_DIR=""
LOCK_HELD=0

info() { echo "[pre-push] $*"; }
fail() { echo "[pre-push] $*" >&2; }

run() {
    info "$*"
    "$@"
}

cleanup() {
    # A private (cargo-free) snapshot is a mktemp dir only this run uses; the
    # shared snapshot is persistent and must not be removed.
    if [[ "${SNAPSHOT_PRIVATE:-0}" == "1" && -n "${SNAPSHOT_ROOT}" && -d "${SNAPSHOT_ROOT}" ]]; then
        rm -rf "${SNAPSHOT_ROOT}"
    fi
    if [[ -n "${STAGE_ROOT}" && -d "${STAGE_ROOT}" ]]; then
        rm -rf "${STAGE_ROOT}"
    fi
    [[ -n "${FAST_STEP:-}" ]] && rm -f "${FAST_STEP}"
    [[ -n "${SYNCED_FILES:-}" ]] && rm -f "${SYNCED_FILES}"
    [[ "${LOCK_HELD:-0}" -eq 1 ]] && rm -rf "${LOCK_DIR:-}"
    return 0
}

trap cleanup EXIT

# --- Read the push range FIRST to decide what this run must do -------------
# Pre-push stdin: <local_ref> <local_sha> <remote_ref> <remote_sha>.
# Empty stdin (manual run) defaults to compiling.
SKIP_CARGO=0
SKIP_SHELL_TESTS=0
CUDA_CRATES_CHANGED=0
METAL_WANTED="${ARLE_PRE_PUSH_METAL:-${AGENT_INFER_PRE_PUSH_METAL:-0}}"
changed_files=""
while read -r _local_ref local_sha _remote_ref remote_sha; do
    if [[ "$remote_sha" =~ ^0{40}$ ]]; then
        # New branch: no remote tip. The merge-base with main is the range the
        # push actually adds; HEAD~1 covers a repo with no origin.
        base="$(git -C "${REPO_ROOT}" merge-base HEAD origin/main 2>/dev/null || echo HEAD~1)"
        changed_files="$(git -C "${REPO_ROOT}" diff --name-only "${base}..HEAD" 2>/dev/null || true)"
    else
        changed_files="$(git -C "${REPO_ROOT}" diff --name-only "${remote_sha}..${local_sha}" 2>/dev/null || true)"
    fi
    if ! grep -q '\.rs$' <<< "${changed_files}"; then
        SKIP_CARGO=1
    fi
    if grep -qE '^crates/(infer-cuda|cuda-kernels)/' <<< "${changed_files}"; then
        CUDA_CRATES_CHANGED=1
    fi
    # The scripts/tests/*.sh batch takes minutes and holds the pre-push SSH
    # connection idle (push exits 141 after the checks pass). Run it only when
    # the push could change what those tests exercise. Several read real tree
    # files, not only scripts/: cuda_prebuilt_export / kernel_artifact /
    # pod_flow read crates/cuda-kernels/{build.rs,kernels.toml,generated} and
    # the crate package; hook_disowns reads .githooks/pre-push; pod_flow copies
    # the root .gitignore. Trigger on the whole cuda-kernels crate (conservative)
    # plus scripts/ .githooks/ .github/ .gitignore. Hygiene and fmt always run.
    if ! grep -qE '^(scripts|\.githooks|\.github)/|^\.gitignore$|^crates/cuda-kernels/' <<< "${changed_files}"; then
        SKIP_SHELL_TESTS=1
        info "no shell-test inputs (scripts/.githooks/.github/cuda-kernels/.gitignore) in pushed range; skipping shell test batch"
    fi
    break
done
if [[ "${SKIP_CARGO}" == "1" ]]; then
    info "no .rs files in pushed range; skipping cargo steps"
fi

# --- Snapshot + lock selection ---------------------------------------------
# The lock keeps its historical path arle-pre-push-cargo.lock: lanes on the
# old hook and lanes on this one must take the SAME lock during rollout, or two
# hooks compiling into the shared pre-push-quick target would run concurrently.
# It now spans the snapshot refresh through the cargo steps (the old hook held
# it only around cargo), which is exactly the window that must be serialized.
#
# A run WITH cargo steps (any .rs change, or Metal enabled) compiles into the
# SHARED CARGO_TARGET_DIR, so it must build from the ONE shared snapshot and
# hold the machine lock across refresh -> cargo, or artifacts from different
# source roots mix in one target (the snapshot-mtime false-Fresh bug).
#
# A cargo-FREE run (no .rs, Metal off) never touches the shared target; the
# mtime/shared-root problem is moot. It builds from a PRIVATE mktemp snapshot
# and takes NO lock, so a docs-only push never waits behind a peer's minutes-
# long cargo run — waiting held its already-open SSH connection idle until the
# remote dropped it (push exit 141).
CARGO_RUNS=1
if [[ "${SKIP_CARGO}" == "1" && "${METAL_WANTED}" != "1" ]]; then
    CARGO_RUNS=0
fi

if [[ "${CARGO_RUNS}" == "1" && "${ARLE_PRE_PUSH_NESTED:-0}" != "1" ]]; then
    LOCK_DIR="${ARLE_PREPUSH_LOCK_DIR:-${TMPDIR:-/tmp}/arle-pre-push-cargo.lock}"
    waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        holder="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo unknown)"
        if [[ "$holder" != "unknown" ]] && ! kill -0 "$holder" 2>/dev/null; then
            info "removing stale hook lock (dead pid $holder)"
            rm -rf "$LOCK_DIR"
            continue
        fi
        if [[ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +60 2>/dev/null)" ]]; then
            info "removing stale hook lock (pid $holder, older than 60 min)"
            rm -rf "$LOCK_DIR"
            continue
        fi
        [[ "$waited" -eq 0 ]] && info "waiting for peer pre-push hook (pid $holder) to finish"
        sleep 5
        waited=$((waited + 5))
    done
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    LOCK_HELD=1
fi

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arle-pre-push-stage.XXXXXX")"
# A nested run (a shell-test fixture driving the real hook) skips the lock, so
# it must NEVER rsync --delete into the shared snapshot: without the lock its
# tiny fixture tree would erase the real tree a peer hook is compiling (live
# race, 2026-09-11). A nested run may still point at an EXPLICIT test-owned
# root via ARLE_PREPUSH_SNAPSHOT_ROOT (the fixture tests do that); only the
# unset/empty default is forced private. Same private path as cargo-free runs.
nested_unsafe_shared=0
if [[ "${ARLE_PRE_PUSH_NESTED:-0}" == "1" && -z "${ARLE_PREPUSH_SNAPSHOT_ROOT:-}" ]]; then
    nested_unsafe_shared=1
fi
if [[ "${CARGO_RUNS}" == "1" && "${nested_unsafe_shared}" != "1" ]]; then
    # One shared source root for every build in the shared target.
    SNAPSHOT_ROOT="${ARLE_PREPUSH_SNAPSHOT_ROOT:-${TMPDIR:-/tmp}/arle-pre-push-snapshot}"
    mkdir -p "${SNAPSHOT_ROOT}"
    info "refreshing shared snapshot at ${SNAPSHOT_ROOT}"
else
    # Private throwaway snapshot; cleanup removes it on exit.
    SNAPSHOT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arle-pre-push-snapshot.XXXXXX")"
    SNAPSHOT_PRIVATE=1
    if [[ "${CARGO_RUNS}" != "1" ]]; then
        info "cargo-free push; running checks from private snapshot (no lock)"
    else
        info "nested hook without ARLE_PREPUSH_SNAPSHOT_ROOT; using private snapshot ${SNAPSHOT_ROOT} (shared snapshot left untouched)"
    fi
fi
# True iff rsync actually updated a LIB-AFFECTING file of crate $1 in this
# snapshot: crates/<crate>/src/**, crates/<crate>/build.rs, or
# crates/<crate>/Cargo.toml. Examples/tests/benches never change a lib
# fingerprint, so an examples-only push must not demand a lib Checking line.
rsync_touched_lib() {
    local crate="$1"
    grep -qE "^crates/${crate}/(src/|build\.rs$|Cargo\.toml$)" "${SYNCED_FILES}"
}

# True iff rsync updated anything under the crate at all (examples included).
rsync_touched_crate() {
    local crate="$1"
    grep -qE "^crates/${crate}/" "${SYNCED_FILES}"
}

git -C "${REPO_ROOT}" archive HEAD | tar -x -C "${STAGE_ROOT}"
# rsync WITHOUT -t (no mtime preservation): `git archive | tar` stamps files
# with the COMMIT time, and `-a`'s `-t` would restore that older mtime onto a
# source newer build artifacts already exist for — cargo then sees source mtime
# <= output mtime and reports a content-CHANGED crate Fresh (false negative).
# Transferred files take mtime=now so a content change always forces a rebuild;
# --checksum keeps content-IDENTICAL files' mtimes (warm cache hits); -r -l -p
# -D keep recursion/symlinks/perms/specials; --delete drops removed files.
#
# `--out-format='%n'` lists every path rsync UPDATES (content differs by
# checksum, including files the --delete phase drops as `deleting <p>`). These
# are the only paths whose mtime identity changed vs the previous snapshot, so
# they — not the full pushed-range `changed_files` — decide which crates a
# cargo step must have rebuilt. A pushed range can include a crate whose files
# are unchanged IN THIS SNAPSHOT (examples-only changes never touch lib
# fingerprints); keying the Fresh assertion on the push range then failed
# legitimately-Fresh lib builds (2026-09-11 false positive on #312).
SYNCED_FILES="$(mktemp "${TMPDIR:-/tmp}/arle-pre-push-synced.XXXXXX")"
rsync -rlpD --delete --checksum --out-format='%n' "${STAGE_ROOT}/" "${SNAPSHOT_ROOT}/" \
    | sed -E 's/^deleting //' > "${SYNCED_FILES}"
# NOTE: the CUDA_CRATES_CHANGED branch gate stays keyed on the PUSH RANGE (set
# from stdin above): it decides whether this run does the verbose strict lint
# at all. Only the per-crate Fresh ASSERTIONS below consult rsync's actual
# delta — an examples-only range enters the strict branch but its lib
# assertion stays silent because rsync touched no lib fingerprint file.

# Orphan sweeps:
# - Legacy per-worktree snapshot dirs (`snapshot-<hash>`) left by lanes on the
#   old hook (1.9 GB / 20 dirs observed 2026-09-10). The shared dir has no
#   `-<hash>` suffix so the glob never touches it; +7 days also protects
#   not-yet-rebased lanes still refreshing those dirs during rollout.
find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'arle-pre-push-snapshot-*' -mtime +7 -exec rm -rf {} + 2>/dev/null || true
# - Private cargo-free snapshots: normally removed by the EXIT trap, but a
#   SIGKILL skips the trap. Reap any older than two hours (a live one is
#   minutes old).
find "${TMPDIR:-/tmp}" -maxdepth 1 -type d -name 'arle-pre-push-nocargo.*' -mmin +120 -exec rm -rf {} + 2>/dev/null || true
cd "${SNAPSHOT_ROOT}"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
# The shared pre-push target is the compile cache for every worktree (ordinary
# lane builds use the shared target-dir in ../arle-lanes/.cargo/config.toml).
MAIN_ROOT="$(dirname "$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)"
[[ -d "${MAIN_ROOT}" ]] || MAIN_ROOT="${REPO_ROOT}"
export CARGO_TARGET_DIR="${MAIN_ROOT}/target/pre-push-quick"
# cudarc probes the CUDA version at build time; pin it so the cuda,no-cuda
# typecheck works on hosts without nvcc.
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"

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
    if [[ "${SKIP_SHELL_TESTS}" == "1" ]]; then
        return 0
    fi
    for test in \
        test_cuda_prebuilt_export.sh \
        test_lever_gate.sh \
        test_kernel_artifact_qualification.sh \
        test_validate_release.sh \
        test_pod_flow.sh \
        test_pod_tree_identity.sh \
        test_hook_disowns_git_env.sh \
        test_hook_cuda_lint_freshness.sh \
        test_prepush_nested_snapshot.sh \
        test_prepush_snapshot_mtime.sh \
        test_hook_skips_shell_tests.sh \
        test_bench_ab_control_arm.sh; do
        run_fast bash "scripts/tests/${test}"
    done
}
run_fast_checks &
FAST_PID=$!

# Asserts a cargo step rebuilt every crate whose LIB rsync actually updated.
# All lanes share one target dir and cargo fingerprints path deps by mtime, so
# a build from another lane can make this step report a changed crate Fresh —
# stale artifacts, or artifacts from a lane whose trait signatures disagree
# with this tree (2026-09-10: e2's Step 1b lane built a 3-param `submit` into
# the shared target; this tree had 4, and the hook failed with fake E0050/E0063
# that read as real code breakage). cargo prints nothing for a Fresh crate,
# so absence of Checking/Compiling means the step did no work on it.
assert_step_rebuilt() {  # $1 = step label, $2 = step output, rest = the step's -p crates
    local label="$1" out="$2"; shift 2
    # CARGO_TERM_COLOR=always (exported above) inserts ANSI codes between
    # "Compiling" and the crate name; strip them or a rebuilt crate reads Fresh.
    out="$(printf '%s' "${out}" | sed $'s/\x1b\\[[0-9;]*m//g')"
    for crate in "$@"; do
        if rsync_touched_lib "${crate}" \
           && ! grep -qE "(Checking|Compiling) ${crate}( |\$)" <<< "${out}"; then
            fail "${label} reported ${crate} Fresh while rsync updated its lib in the snapshot — shared target cross-contaminated by another lane; run 'CARGO_TARGET_DIR=${CARGO_TARGET_DIR} cargo clean -p ${crate}' and retry"
            exit 1
        fi
    done
}

# --- Cargo steps (serial — cargo locks the target dir) ---------------------
if [[ "${SKIP_CARGO}" == "0" ]]; then
    run cargo check -p arle --no-default-features --features cpu,no-cuda,cli --bin arle
    # CI's test-backend lane runs `-p arle`; the hook did not, so a CLI help
    # rewrite landed on main with cli_smoke red. Same feature set as the check
    # above, so the binary is already built.
    cli_smoke_rc=0
    cli_smoke_out="$(cargo test -p arle --no-default-features --features cpu,no-cuda,cli --test cli_smoke 2>&1)" || cli_smoke_rc=$?
    printf '%s\n' "${cli_smoke_out}"
    [[ "${cli_smoke_rc}" -eq 0 ]] || exit "${cli_smoke_rc}"
    assert_step_rebuilt "cli_smoke" "${cli_smoke_out}" arle
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
        # infer-cuda examples link the CUDA runtime, so clippy --examples (no
        # link) is their only gate; mirrors CI Lint's cuda section. Captured
        # separately: merged into cuda_lint_out, the lib run's "Checking
        # infer-cuda" would mask a Fresh examples run.
        examples_lint_out="$(CARGO_TERM_COLOR=never cargo clippy -v -p infer-cuda --no-default-features --features cuda,no-cuda,nccl --examples -- -D warnings 2>&1)" || cuda_lint_rc=$?
        printf '%s\n' "${cuda_lint_out}"
        printf '%s\n' "${examples_lint_out}"
        [[ "${cuda_lint_rc}" -eq 0 ]] || exit "${cuda_lint_rc}"
        for crate in infer-cuda cuda-kernels; do
            if rsync_touched_lib "${crate}" \
               && ! grep -qE "(Checking|Compiling) ${crate}( |\$)" <<< "${cuda_lint_out}"; then
                fail "CUDA lint reported ${crate} Fresh while rsync updated its lib in the snapshot — shared target cross-contaminated by another lane; run 'cargo clean -p ${crate}' and retry"
                exit 1
            fi
        done
        # rsync updating ANY infer-cuda file (examples included) forces at
        # least one example target to rebuild; a zero-Checking examples run
        # checked nothing. A lib-less (examples-only) sync still arms this,
        # while the lib assertion above stays silent.
        if rsync_touched_crate infer-cuda \
           && compgen -G "crates/infer-cuda/examples/*.rs" > /dev/null \
           && ! grep -qE "(Checking|Compiling) " <<< "${examples_lint_out}"; then
            fail "CUDA examples lint reported everything Fresh while rsync updated crates/infer-cuda — shared target cross-contaminated by another lane; run 'cargo clean -p infer-cuda' and retry"
            exit 1
        fi
    else
        run cargo clippy -p infer-api --no-default-features --features cuda,no-cuda --lib -- -D warnings
        # infer-cuda examples link the CUDA runtime, so clippy --examples (no
        # link) is their only gate; mirrors CI Lint's cuda section.
        run cargo clippy -p infer-cuda --no-default-features --features cuda,no-cuda,nccl --examples -- -D warnings
    fi
    # CI's CPU-only clippy lane (job "cargo clippy (CPU-only surfaces)"): a push
    # that is clean on the cuda lane above still failed CI here (#258). Same
    # feature sets as CI, verbatim — the hook crate list drifts otherwise.
    run cargo clippy -p infer-api --no-default-features --features no-cuda --lib -- -D warnings
    run cargo clippy -p cli --no-default-features --features no-cuda -- -D warnings
    run cargo clippy -p arle --no-default-features --features cpu,no-cuda,cli --bin arle -- -D warnings
    run cargo clippy -p autograd --features no-cuda --all-targets -- -D warnings
    run cargo clippy -p train --features no-cuda --all-targets -- -D warnings
    test_group1_rc=0
    test_group1_out="$(cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p spec-train -p kv-native-sys -p infer-quant 2>&1)" || test_group1_rc=$?
    printf '%s\n' "${test_group1_out}"
    [[ "${test_group1_rc}" -eq 0 ]] || exit "${test_group1_rc}"
    assert_step_rebuilt "cargo test group 1" "${test_group1_out}" \
        chat tools qwen3-spec qwen35-spec spec-train kv-native-sys infer-quant
    test_group2_rc=0
    test_group2_out="$(cargo test \
        -p infer-core -p infer-server -p infer-plan -p infer-kvspace -p infer-seam \
        -p infer-moe -p infer-topo -p infer-util -p deepseek-spec -p agent 2>&1)" || test_group2_rc=$?
    printf '%s\n' "${test_group2_out}"
    [[ "${test_group2_rc}" -eq 0 ]] || exit "${test_group2_rc}"
    assert_step_rebuilt "cargo test group 2" "${test_group2_out}" \
        infer-core infer-server infer-plan infer-kvspace infer-seam \
        infer-moe infer-topo infer-util deepseek-spec agent
    run cargo clippy -p kv-native-sys --all-targets -- -D warnings
    run cargo clippy -p infer-hip -p infer-vulkan -- -D warnings

    # Metal lib check (default-on, Mac only): catches dead-code/unused lints in
    # infer-metal that CI's Metal lane runs with -D warnings. The full binary
    # build + needle gate stays opt-in (ARLE_PRE_PUSH_METAL=1) below.
    # Targets infer-metal directly: infer-api has carried no backend features
    # since runtime backend dispatch (#68), so `-p infer-api --features metal`
    # no longer resolves.
    if [[ "$(uname -s)" == "Darwin" ]]; then
        run cargo check -p infer-metal --no-default-features --features metal
        # Examples have no other compile gate; only the Metal lane carries the
        # feature this one needs.
        run cargo build -p cli --example metal_kv_memory_probe --no-default-features --features metal,no-cuda
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
        LEVER_GATE_ALLOW_NO_BASELINE=1 LEVER_GATE_SKIP_TEMP=1 LEVER_GATE_SKIP_CONCURRENT=1 \
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
