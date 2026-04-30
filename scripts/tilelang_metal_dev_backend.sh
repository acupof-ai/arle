#!/usr/bin/env bash
# Local ARLE development entrypoint for a TileLang Metal checkout.
#
# Defaults to /tmp/tilelang-metal-pr and validates that this checkout can lower
# ARLE's TileLang attention kernel to Metal before optional serve/bench flows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TILELANG_REPO="${ARLE_TILELANG_REPO:-/tmp/tilelang-metal-pr}"
TILELANG_PYTHON="${ARLE_TILELANG_PYTHON:-}"
if [[ -z "${TILELANG_PYTHON}" ]]; then
    if [[ -x /tmp/arle-tilelang-mac-venv/bin/python ]]; then
        TILELANG_PYTHON="/tmp/arle-tilelang-mac-venv/bin/python"
    else
        TILELANG_PYTHON="python3"
    fi
fi

usage() {
    cat <<EOF
usage: $(basename "$0") <smoke|serve|bench> [args]

Modes:
  smoke [python args...]
      Import TileLang from ARLE_TILELANG_REPO (default: /tmp/tilelang-metal-pr),
      lower ARLE's HD128 attention kernel to Metal, and execute TileLang T.gemm.

  serve [model] [port] [-- metal_serve args...]
      Run smoke first, then launch scripts/start_metal_serve.sh for normal ARLE
      local server debugging.

  bench [model] [port] [-- bench_guidellm args...]
      Run smoke, launch metal_serve in the background, wait for /v1/models, then
      run scripts/bench_guidellm.sh. Defaults to models/Qwen3-0.6B, port 8765,
      and --quick exploration mode.

Environment:
  ARLE_TILELANG_REPO      local TileLang checkout (default: /tmp/tilelang-metal-pr)
  ARLE_TILELANG_PYTHON    Python with the checkout's deps
  ARLE_TILELANG_BENCH_LABEL
EOF
}

run_smoke() {
    if [[ ! -d "${TILELANG_REPO}" ]]; then
        echo "error: TileLang checkout not found: ${TILELANG_REPO}" >&2
        exit 2
    fi
    (
        cd "${TILELANG_REPO}"
        TILELANG_DISABLE_CACHE="${TILELANG_DISABLE_CACHE:-1}" \
            "${TILELANG_PYTHON}" "${REPO_ROOT}/scripts/tilelang_metal_dev_backend.py" \
            --arle-root "${REPO_ROOT}" \
            --tilelang-repo "${TILELANG_REPO}" \
            "$@"
    )
}

wait_for_server() {
    local port="$1"
    local pid="$2"
    for _ in $(seq 1 180); do
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            echo "error: metal_serve exited before readiness" >&2
            return 1
        fi
        if curl -sS -f "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "error: metal_serve did not become ready on port ${port}" >&2
    return 1
}

mode="${1:-smoke}"
case "${mode}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    smoke)
        shift || true
        run_smoke "$@"
        ;;
    serve)
        shift || true
        run_smoke
        exec "${REPO_ROOT}/scripts/start_metal_serve.sh" "$@"
        ;;
    bench)
        shift || true
        model="${ARLE_MODEL:-${REPO_ROOT}/models/Qwen3-0.6B}"
        if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
            model="${1}"
            shift
        fi
        port="${PORT:-8765}"
        if [[ $# -gt 0 && "${1:-}" =~ ^[0-9]+$ ]]; then
            port="${1}"
            shift
        fi
        if [[ "${1:-}" == "--" ]]; then
            shift
        fi
        bench_args=("$@")
        if [[ ${#bench_args[@]} -eq 0 ]]; then
            bench_args=(--quick)
        fi

        run_smoke

        out_dir="$(mktemp -d -t arle-tilelang-metal-dev-XXXXXX)"
        serve_log="${out_dir}/metal_serve.log"
        "${REPO_ROOT}/scripts/start_metal_serve.sh" "${model}" "${port}" -- --warmup 0 \
            >"${serve_log}" 2>&1 &
        serve_pid=$!
        cleanup() {
            kill "${serve_pid}" >/dev/null 2>&1 || true
            wait "${serve_pid}" >/dev/null 2>&1 || true
        }
        trap cleanup EXIT

        wait_for_server "${port}" "${serve_pid}" || {
            echo "metal_serve log: ${serve_log}" >&2
            exit 1
        }

        label="${ARLE_TILELANG_BENCH_LABEL:-tilelang-metal-dev}"
        "${REPO_ROOT}/scripts/bench_guidellm.sh" "${label}" \
            --target "http://127.0.0.1:${port}" \
            --model "${model}" \
            --processor "${model}" \
            "${bench_args[@]}"
        echo "metal_serve log: ${serve_log}"
        ;;
    *)
        echo "error: unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac
