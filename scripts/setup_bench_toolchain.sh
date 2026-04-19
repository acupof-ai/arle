#!/usr/bin/env bash
# Idempotent setup for the bench/dev toolchain.
#
# Pins guidellm under Python 3.11 (uv-managed). guidellm 0.6.0 hangs at
# "Setup complete, starting benchmarks..." on macOS with the default
# multiprocessing start method (fork) under Python 3.11+; the workaround
# is `GUIDELLM__MP_CONTEXT_TYPE=forkserver`, which `scripts/bench_guidellm.sh`
# now exports automatically.
#
# Run this once on a fresh machine, or whenever guidellm needs to be
# reinstalled. Safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GUIDELLM_PIN="0.6.0"
PY_PIN="3.11"

step() { printf '\n>>> %s\n' "$*"; }
ok()   { printf '  ok: %s\n' "$*"; }
fail() { printf '  error: %s\n' "$*" >&2; exit 1; }

# ---- uv ---------------------------------------------------------------------
step "uv (python toolchain manager)"
if ! command -v uv >/dev/null 2>&1; then
    fail "uv not on PATH — install: https://docs.astral.sh/uv/getting-started/installation/"
fi
ok "uv $(uv --version | awk '{print $2}')"

# ---- python 3.11 ------------------------------------------------------------
step "python ${PY_PIN}"
if ! uv python list 2>&1 | grep -q "cpython-${PY_PIN}\..*-macos-aarch64-none.*/"; then
    uv python install "${PY_PIN}"
fi
ok "python ${PY_PIN} available via uv"

# ---- guidellm ---------------------------------------------------------------
step "guidellm ${GUIDELLM_PIN} (under python ${PY_PIN})"
need_install=0
if ! command -v guidellm >/dev/null 2>&1; then
    need_install=1
elif ! guidellm --version 2>&1 | grep -q "guidellm version: ${GUIDELLM_PIN}"; then
    need_install=1
else
    py_shebang="$(head -1 "$(command -v guidellm)")"
    if ! grep -q "/uv/tools/guidellm/" <<<"$py_shebang"; then
        need_install=1
    else
        py_actual="$("$(awk -F'!' '{print $2}' <<<"$py_shebang")" --version 2>&1 | awk '{print $2}')"
        if ! grep -q "^${PY_PIN}\." <<<"$py_actual"; then
            need_install=1
        fi
    fi
fi
if (( need_install )); then
    uv tool uninstall guidellm 2>&1 | tail -1 || true
    uv tool install --python "${PY_PIN}" "guidellm==${GUIDELLM_PIN}" >/tmp/setup_guidellm.log 2>&1
fi
ok "$(guidellm --version | head -1)"
ok "$(head -1 "$(command -v guidellm)")"

# ---- jq + curl preflight (used by scripts/bench_guidellm.sh) ----------------
step "jq, curl"
command -v jq   >/dev/null 2>&1 || fail "jq not on PATH (brew install jq)"
command -v curl >/dev/null 2>&1 || fail "curl not on PATH"
ok "jq $(jq --version)"
ok "curl $(curl --version | head -1 | awk '{print $1, $2}')"

# ---- smoke (offline, no infer server required) ------------------------------
step "smoke: guidellm starts under forkserver"
GUIDELLM__MP_CONTEXT_TYPE=forkserver guidellm --help >/dev/null 2>&1 \
    || fail "guidellm --help failed under forkserver"
ok "guidellm cli reachable"

cat <<'EOF'

Toolchain ready. The bench wrapper at scripts/bench_guidellm.sh exports
GUIDELLM__MP_CONTEXT_TYPE=forkserver automatically. To run a bench:

    ./scripts/bench_guidellm.sh metal-m4max --model Qwen3.5-4B-MLX-4bit \
        --processor /path/to/local/snapshot

EOF
