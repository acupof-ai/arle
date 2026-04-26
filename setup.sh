#!/usr/bin/env bash
# ============================================================================
# ARLE — reproducible dev environment setup
#
# Usage:
#   ./setup.sh              # Full setup: Linux/CUDA toolchain + Zig + venv + build + model
#   ./setup.sh --full       # Alias for the default full setup
#   ./setup.sh --deps-only  # Toolchains + venv only, no build/model
#   ./setup.sh --build-only # Build only (assumes venv exists)
#   ./setup.sh --model-only # Download model only
#   ./setup.sh --web-only   # Bootstrap bun + install web/ frontend deps (cross-platform)
#   ./setup.sh --check      # Verify environment
#   ./setup.sh --clean      # Remove venv and build artifacts
#
# Environment variables:
#   MODEL_ID      — HuggingFace model ID  (default: Qwen/Qwen3-8B)
#   MODEL_DIR     — Local path for model  (default: models/Qwen3-8B)
#   CUDA_HOME     — CUDA toolkit path     (default: /usr/local/cuda)
#   SKIP_MODEL    — Set to 1 to skip model download
#   PYTHON        — Python interpreter     (default: python3)
#   KV_ZIG_VERSION      — Zig version override for kv-native-sys (default: 0.16.0)
#   KV_ZIG_INSTALL_ROOT — Repo-local Zig install root (default: .toolchains/zig)
#
# This script is CUDA/Linux oriented. It now bootstraps the Zig toolchain used
# by `crates/kv-native-sys`; use the Makefile targets for Metal/macOS serving.
# All Python deps are installed into .venv/ — never pollutes system packages.
# Activate manually:  source .venv/bin/activate
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[fail]${NC}  $*"; }
step()  { echo -e "\n${BOLD}${GREEN}▸ $*${NC}"; }

check_cmd() {
    if command -v "$1" &>/dev/null; then
        ok "$1 found: $(command -v "$1")"
        return 0
    else
        fail "$1 not found"
        return 1
    fi
}

ensure_zig() {
    local check_only="${1:-0}"
    local zig_args=(--print-zig)
    if [ "$check_only" = "1" ]; then
        zig_args+=(--check-only)
    fi

    local zig_bin
    zig_bin="$("$SCRIPT_DIR/scripts/setup_zig_toolchain.sh" "${zig_args[@]}")"
    export ZIG="$zig_bin"
    ok "zig: $ZIG"
    info "  zig $("${ZIG}" version)"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-8B}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SKIP_MODEL="${SKIP_MODEL:-0}"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------
MODE="full"
case "${1:-}" in
    --full)        MODE="full" ;;
    --deps-only)   MODE="deps" ;;
    --build-only)  MODE="build" ;;
    --model-only)  MODE="model" ;;
    --web-only)    MODE="web" ;;
    --check)       MODE="check" ;;
    --clean)       MODE="clean" ;;
    --help|-h)
        sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
        exit 0
        ;;
esac

# ---------------------------------------------------------------------------
# Activate venv (if it exists) for all modes except clean
# ---------------------------------------------------------------------------
activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
}

# ============================================================================
# CLEAN
# ============================================================================
do_clean() {
    step "Cleaning build artifacts and venv"
    rm -rf "$VENV_DIR" && ok "Removed .venv/"
    rm -rf target/ && ok "Removed target/"
    rm -rf infer/target/ && ok "Removed infer/target/"
    info "Run ./setup.sh to rebuild from scratch"
}

# ============================================================================
# CHECK — verify everything is ready
# ============================================================================
do_check() {
    step "Checking environment"
    local errors=0

    # Rust
    if check_cmd rustc; then
        info "  rustc $(rustc --version 2>/dev/null | awk '{print $2}')"
    else errors=$((errors + 1)); fi
    check_cmd cargo || errors=$((errors + 1))

    # Zig
    if ensure_zig 1; then
        :
    else
        fail "zig not ready — run ./setup.sh --deps-only"
        errors=$((errors + 1))
    fi

    # CUDA
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        ok "nvcc: $CUDA_HOME/bin/nvcc"
        info "  $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
    else
        fail "nvcc not found at $CUDA_HOME/bin/nvcc"
        errors=$((errors + 1))
    fi

    # GPU
    if check_cmd nvidia-smi; then
        info "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    else errors=$((errors + 1)); fi

    # Venv
    if [ -f "$VENV_DIR/bin/activate" ]; then
        ok "venv: $VENV_DIR"
        activate_venv
    else
        fail "venv not found — run ./setup.sh --deps-only"
        errors=$((errors + 1))
    fi

    # Python (from venv)
    if check_cmd python; then
        info "  python $(python --version 2>/dev/null | awk '{print $2}')"
    else errors=$((errors + 1)); fi

    # Pinned packages
    local pkg_errors=0
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        local pkg ver
        pkg="${line%%==*}"; ver="${line##*==}"
        local actual
        actual=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        actual="${actual:-MISSING}"
        if [ "$actual" = "$ver" ]; then
            ok "  $pkg==$ver"
        else
            fail "  $pkg: want $ver, got $actual"
            pkg_errors=$((pkg_errors + 1))
        fi
    done < <(grep -E '^[a-zA-Z].*==' requirements-build.txt)
    errors=$((errors + pkg_errors))

    # nsjail
    if check_cmd nsjail; then
        info "  sandbox isolation active"
    else
        warn "nsjail not found — tool execution will run without sandbox"
    fi

    # bun + web/ frontend
    local bun_bin=""
    if command -v bun &>/dev/null; then
        bun_bin="$(command -v bun)"
    elif [ -x "$HOME/.bun/bin/bun" ]; then
        bun_bin="$HOME/.bun/bin/bun"
    fi
    if [ -n "$bun_bin" ]; then
        ok "bun $("$bun_bin" --version 2>/dev/null) ($bun_bin)"
        if [ -d web/node_modules/astro ]; then
            ok "  web/node_modules ready"
        else
            warn "  web/node_modules missing — run ./setup.sh --web-only"
        fi
    else
        warn "bun not installed — run ./setup.sh --web-only to bootstrap web/ frontend"
    fi

    # Binaries
    if [ -x target/release/arle ]; then
        ok "target/release/arle built"
    else
        fail "ARLE binary not found — run ./setup.sh --build-only"
        errors=$((errors + 1))
    fi
    if [ -x target/release/infer ]; then
        ok "target/release/infer built"
    else
        fail "infer server binary not found — run ./setup.sh --build-only"
        errors=$((errors + 1))
    fi

    # Model
    if [ -f "$MODEL_DIR/config.json" ]; then
        ok "Model: $MODEL_DIR"
    else
        warn "Model not found at $MODEL_DIR — run ./setup.sh --model-only"
    fi

    echo ""
    if [ "$errors" -eq 0 ]; then
        ok "Environment is ready!"
        echo ""
        info "Activate venv:  ${BOLD}source .venv/bin/activate${NC}"
    else
        fail "$errors issue(s) found"
        return 1
    fi
}

# ============================================================================
# DEPS — toolchain + venv + Python packages
# ============================================================================
do_deps() {
    # --- Rust ---
    step "Rust toolchain"
    if command -v rustc &>/dev/null; then
        ok "Rust $(rustc --version | awk '{print $2}')"
    else
        info "Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        ok "Rust installed: $(rustc --version)"
    fi

    # --- Zig ---
    step "Zig toolchain"
    ensure_zig

    # --- CUDA ---
    step "CUDA toolkit"
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        ok "nvcc: $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
    else
        fail "CUDA toolkit not found at $CUDA_HOME"
        info "Install CUDA toolkit or set CUDA_HOME=/path/to/cuda"
        exit 1
    fi

    # --- nsjail ---
    step "nsjail (sandbox)"
    if command -v nsjail &>/dev/null; then
        ok "nsjail already installed"
    else
        info "Building nsjail from source..."
        apt-get install -y -qq autoconf bison flex gcc g++ git \
            libprotobuf-dev libnl-route-3-dev libtool make pkg-config protobuf-compiler \
            >/dev/null 2>&1
        local nsjail_tmp
        nsjail_tmp="$(mktemp -d)"
        git clone --depth 1 https://github.com/google/nsjail.git "$nsjail_tmp/nsjail" 2>/dev/null
        make -C "$nsjail_tmp/nsjail" -j"$(nproc)" >/dev/null 2>&1
        cp "$nsjail_tmp/nsjail/nsjail" /usr/local/bin/
        rm -rf "$nsjail_tmp"
        ok "nsjail built and installed"
    fi

    # --- Python venv ---
    step "Python virtual environment"
    if [ -f "$VENV_DIR/bin/activate" ]; then
        ok "venv exists: $VENV_DIR"
    else
        info "Creating venv at $VENV_DIR ..."
        # --without-pip: some distros lack ensurepip for newer Python.
        # We bootstrap pip via get-pip.py immediately after.
        "$PYTHON" -m venv --without-pip "$VENV_DIR" 2>/dev/null \
            || "$PYTHON" -m venv "$VENV_DIR"
        ok "venv created"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    info "Python: $(python --version) — $(which python)"

    # Bootstrap pip if missing (happens with --without-pip)
    if ! python -m pip --version &>/dev/null; then
        info "Bootstrapping pip..."
        curl -sSL https://bootstrap.pypa.io/get-pip.py | python -q
    fi
    python -m pip install --upgrade pip -q

    # --- Pinned build deps ---
    step "Python build dependencies (from requirements-build.txt)"
    info "FlashInfer is installed --no-deps (we only need C++ headers)"

    # Install flashinfer separately with --no-deps
    local fi_line
    fi_line=$(grep -E '^flashinfer' requirements-build.txt | head -1)
    pip install "$fi_line" --no-deps -q
    ok "$fi_line (headers only)"

    # Install remaining build deps normally (skip comments, blanks, flashinfer)
    grep -E '^[a-zA-Z]' requirements-build.txt | grep -v 'flashinfer' | \
        pip install -r /dev/stdin -q
    ok "Build deps installed"

    # --- Bench/test deps ---
    step "Bench & test dependencies (from requirements-bench.txt)"
    pip install -r requirements-bench.txt -q
    ok "Bench deps installed"

    # --- Project install ---
    if [ -f pyproject.toml ]; then
        step "Python project (editable install)"
        pip install -e ".[dev]" -q 2>/dev/null || true
        ok "Project installed"
    fi

    # --- Verify pinned versions ---
    step "Verifying pinned versions"
    local ok_count=0
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        local pkg ver
        pkg="${line%%==*}"
        ver="${line##*==}"
        local pkg_name="${pkg//-/_}"
        local actual
        # flashinfer can't be imported without torch — check pip metadata instead
        actual=$(pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        actual="${actual:-MISSING}"
        if [ "$actual" = "$ver" ]; then
            ok "  $pkg==$ver"
            ok_count=$((ok_count + 1))
        else
            fail "  $pkg: want $ver, got $actual"
        fi
    done < <(grep -E '^[a-zA-Z].*==' requirements-build.txt)
    info "$ok_count pinned packages verified"

    echo ""
    ok "All dependencies installed into $VENV_DIR"
    info "Activate: ${BOLD}source .venv/bin/activate${NC}"
}

# ============================================================================
# BUILD — compile Rust + CUDA kernels
# ============================================================================
do_build() {
    step "Building ARLE CLI + infer server (release, CUDA)"
    activate_venv

    # Ensure cargo is on PATH
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    ensure_zig
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
    # Triton AOT needs Python from venv
    export INFER_TRITON_PYTHON="$(which python)"

    info "CUDA_HOME=$CUDA_HOME"
    info "TRITON_PYTHON=$INFER_TRITON_PYTHON"
    info "SM targets: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr '\n' ' ')"

    local start
    start=$(date +%s)

    cargo build --release --features cuda,cli -p agent-infer --bin arle 2>&1 | while IFS= read -r line; do
        case "$line" in
            *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                echo "  $line" ;;
        esac
    done

    cargo build --release --features cuda -p infer 2>&1 | while IFS= read -r line; do
        case "$line" in
            *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                echo "  $line" ;;
        esac
    done

    local elapsed=$(( $(date +%s) - start ))
    ok "Build complete in ${elapsed}s"

    if [ -x target/release/arle ]; then
        info "Binary: target/release/arle ($(du -h target/release/arle | awk '{print $1}'))"
    fi
    if [ -x target/release/infer ]; then
        info "Binary: target/release/infer ($(du -h target/release/infer | awk '{print $1}'))"
    fi
}

# ============================================================================
# MODEL — download model weights
# ============================================================================
do_model() {
    step "Downloading model: $MODEL_ID → $MODEL_DIR"
    activate_venv

    if [ -f "$MODEL_DIR/config.json" ]; then
        ok "Model already exists at $MODEL_DIR"
        info "Delete $MODEL_DIR to re-download"
        return 0
    fi

    mkdir -p "$MODEL_DIR"
    python -c "
from huggingface_hub import snapshot_download
print('Downloading $MODEL_ID ...')
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR',
                  ignore_patterns=['*.bin', '*.pt', 'original/*'])
print('Done')
"
    ok "Model downloaded to $MODEL_DIR"

    if [ -f "$MODEL_DIR/config.json" ]; then
        local params
        params=$(python -c "
import json; c = json.load(open('$MODEL_DIR/config.json'))
print(f\"hidden={c.get('hidden_size','?')}, layers={c.get('num_hidden_layers','?')}\")
" 2>/dev/null || echo "?")
        info "Config: $params"
    fi
}

# ============================================================================
# FULL — run everything
# ============================================================================
do_full() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║           ARLE — environment setup           ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""

    do_deps
    do_build
    if [ "$SKIP_MODEL" != "1" ]; then
        do_model
    else
        warn "Skipping model download (SKIP_MODEL=1)"
    fi

    echo ""
    step "Setup complete!"
    echo ""
    info "Quick start:"
    echo ""
    echo "  # 1. Activate the virtual environment"
    echo "  source .venv/bin/activate"
    echo ""
    echo "  # 2. Set runtime library paths"
    echo "  export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "  # 3. Run agent REPL"
    echo "  ./target/release/arle --model-path $MODEL_DIR"
    echo ""
    echo "  # 4. Run HTTP server"
    echo "  ./target/release/infer --model-path $MODEL_DIR --port 8000"
    echo ""
    echo "  # 5. Run benchmarks"
    echo "  ./scripts/bench_guidellm.sh cuda-local --target http://localhost:8000"
    echo ""
    echo "  # 6. Run tests"
    echo "  cargo test --release"
    echo "  python scripts/test_long_agent.py"
    echo ""
    echo "  # 7. Verify environment"
    echo "  ./setup.sh --check"
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    check) [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"; activate_venv; do_check ;;
    deps)  do_deps ;;
    build) do_build ;;
    model) do_model ;;
    clean) do_clean ;;
    full)  do_full ;;
esac
