#!/usr/bin/env bash
# One-command Metal GPU capture driver for ARLE.
#
# Subcommands:
#   gputrace    Programmatic MTLCaptureManager capture inside
#               qwen35_compiled_step_session. Requires a Qwen3.5 target.
#               Emits a `.gputrace` bundle. See
#               docs/plans/metal-gdr-kernel-xcode-capture.md §2b.
#   xctrace     Metal System Trace via xctrace --launch. Works with any
#               Metal app (Qwen3 fallback fine). Emits a `.trace` bundle
#               + a GPU-intervals XML dump next to it.
#   help        Show usage.
#
# Env overrides:
#   ARLE_TARGET                gputrace: default Qwen3.5-4B-MLX-4bit local snapshot.
#                              xctrace:  default Qwen3-0.6B-4bit local snapshot.
#                              Legacy `AGENT_INFER_TARGET` also works.
#   ARLE_CAPTURE_STEP          gputrace: default 2 (first post-warmup decode when
#                              --warmup 1 --generation-tokens 2 --use-step-driver).
#                              Counter is process-global: N = W*G for the first
#                              post-warmup decode step.
#                              Legacy `AGENT_INFER_CAPTURE_STEP` also works.
#   ARLE_CAPTURE_PATH          gputrace: default /tmp/qwen35_step_<epoch>.gputrace
#                              Legacy `AGENT_INFER_CAPTURE_PATH` also works.
#   ARLE_XCTRACE_OUT           xctrace:  default /tmp/arle_metal.trace
#                              Legacy `AGENT_INFER_XCTRACE_OUT` also works.
#
# Examples:
#   ./scripts/capture_metal.sh gputrace
#   ARLE_CAPTURE_STEP=12 ./scripts/capture_metal.sh gputrace
#   ./scripts/capture_metal.sh xctrace
#   ARLE_TARGET=/path/to/Qwen3-4B-bf16 ./scripts/capture_metal.sh xctrace

set -euo pipefail

case "$(uname -s)" in
    Darwin) ;;
    *) echo "capture_metal.sh is macOS-only." >&2; exit 1 ;;
esac
case "$(uname -m)" in
    arm64) ;;
    *) echo "capture_metal.sh expects Apple Silicon (arm64)." >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_HUB="${HOME}/.cache/huggingface/hub"
QWEN35_REPO="${HF_HUB}/models--mlx-community--Qwen3.5-4B-MLX-4bit"
QWEN3_REPO="${HF_HUB}/models--mlx-community--Qwen3-0.6B-4bit"

resolve_snapshot() {
    local repo_dir="$1"
    [ -d "${repo_dir}/snapshots" ] || return 1
    local snap
    snap="$(find "${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
    [ -n "${snap}" ] || return 1
    echo "${snap}"
}

CARGO_COMMON=(--release -p infer --no-default-features --features metal,no-cuda)

usage() {
    cat <<EOF
Metal GPU capture driver.

Subcommands:
  gputrace    Programmatic MTLCaptureManager capture (Qwen3.5-only)
  xctrace     xctrace Metal System Trace (any Metal app)
  help        Show this help

Env:
  ARLE_TARGET               path/repo to model (backend-dependent default)
  ARLE_CAPTURE_STEP         gputrace capture step index (default 2)
  ARLE_CAPTURE_PATH         gputrace output (.gputrace bundle)
  ARLE_XCTRACE_OUT          xctrace output (.trace bundle)
  Legacy AGENT_INFER_* names still work.
EOF
}

run_gputrace() {
    cd "${REPO_ROOT}"
    local target
    if [ -n "${ARLE_TARGET:-${AGENT_INFER_TARGET:-}}" ]; then
        target="${ARLE_TARGET:-${AGENT_INFER_TARGET:-}}"
    else
        if ! target="$(resolve_snapshot "${QWEN35_REPO}")"; then
            echo "capture_metal.sh: cannot find Qwen3.5-4B-MLX-4bit snapshot under" >&2
            echo "  ${QWEN35_REPO}/snapshots/" >&2
            echo "Programmatic capture fires only inside qwen35_compiled_step_session;" >&2
            echo "a Qwen3.5 target is required. Download via:" >&2
            echo "  huggingface-cli download mlx-community/Qwen3.5-4B-MLX-4bit" >&2
            exit 1
        fi
    fi

    local step="${ARLE_CAPTURE_STEP:-${AGENT_INFER_CAPTURE_STEP:-2}}"
    local out_path="${ARLE_CAPTURE_PATH:-${AGENT_INFER_CAPTURE_PATH:-/tmp/qwen35_step_$(date +%s).gputrace}}"

    # Fresh destination: MTLCaptureManager refuses to overwrite existing bundles.
    rm -rf "${out_path}"

    echo "=== Metal gputrace capture ==="
    echo "  target:       ${target}"
    echo "  capture step: ${step}"
    echo "  output:       ${out_path}"
    echo ""

    MTL_CAPTURE_ENABLED=1 \
    INFER_CAPTURE_STEP="${step}" \
    INFER_CAPTURE_PATH="${out_path}" \
        cargo run "${CARGO_COMMON[@]}" --bin metal_bench -- \
            --model "${target}" \
            --prompt-tokens 8 \
            --generation-tokens 2 \
            --warmup 1 \
            --runs 1 \
            --use-step-driver "$@"

    if [ -d "${out_path}" ]; then
        echo ""
        echo "=== capture done ==="
        du -sh "${out_path}"
        echo "Open with: open \"${out_path}\""
    else
        echo "capture_metal.sh: expected .gputrace at ${out_path} but it is missing." >&2
        echo "Check stderr above for a '[capture] ...' line; absence means" >&2
        echo "MTL_CAPTURE_ENABLED wasn't propagated or INFER_CAPTURE_STEP missed the counter." >&2
        exit 1
    fi
}

run_xctrace() {
    cd "${REPO_ROOT}"
    command -v xcrun >/dev/null 2>&1 || {
        echo "xcrun not found; install Xcode command-line tools." >&2; exit 1;
    }

    local target
    if [ -n "${ARLE_TARGET:-${AGENT_INFER_TARGET:-}}" ]; then
        target="${ARLE_TARGET:-${AGENT_INFER_TARGET:-}}"
    else
        if ! target="$(resolve_snapshot "${QWEN3_REPO}")"; then
            if ! target="$(resolve_snapshot "${QWEN35_REPO}")"; then
                echo "capture_metal.sh: no default Qwen3 / Qwen3.5 snapshot under ${HF_HUB}." >&2
                echo "Set ARLE_TARGET (legacy AGENT_INFER_TARGET also works) to a model path/repo and retry." >&2
                exit 1
            fi
        fi
    fi

    local out="${ARLE_XCTRACE_OUT:-${AGENT_INFER_XCTRACE_OUT:-/tmp/arle_metal.trace}}"
    rm -rf "${out}"

    # Pre-build so xctrace doesn't time the compile.
    echo "=== metal_bench build ==="
    cargo build "${CARGO_COMMON[@]}" --bin metal_bench

    echo ""
    echo "=== Metal System Trace via xctrace ==="
    echo "  target: ${target}"
    echo "  output: ${out}"
    echo ""
    xcrun xctrace record \
        --template "Metal System Trace" \
        --output "${out}" \
        --launch -- \
        "${REPO_ROOT}/target/release/metal_bench" \
            --model "${target}" \
            --prompt-tokens 8 \
            --generation-tokens 8 \
            --warmup 0 \
            --runs 1 "$@"

    if [ ! -d "${out}" ]; then
        echo "capture_metal.sh: xctrace did not produce ${out}." >&2
        exit 1
    fi

    local dump="${out%.trace}.gpu-intervals.xml"
    echo ""
    echo "=== GPU intervals export ==="
    xcrun xctrace export \
        --input "${out}" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
        > "${dump}" || {
        echo "xctrace export failed; trace is still at ${out}." >&2
        exit 1
    }

    echo ""
    echo "=== capture done ==="
    du -sh "${out}" "${dump}"
    echo "Open with: open \"${out}\""
}

cmd="${1:-help}"
case "${cmd}" in
    gputrace) shift; run_gputrace "$@" ;;
    xctrace)  shift; run_xctrace "$@" ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac
