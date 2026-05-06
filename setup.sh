#!/usr/bin/env bash
# ============================================================================
# ARLE — reproducible dev environment setup
#
# Usage:
#   ./setup.sh              # Full setup: Linux/CUDA toolchain + venv + build + model
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
#   ARLE_SKIP_WEB — Set to 1 to skip the web/ frontend (bun + Astro) bootstrap
#   PYTHON        — Python interpreter     (default: python3)
#
# Cross-platform: Linux/CUDA + macOS/Metal. Auto-detects host and routes
# CUDA-only steps (nvcc validation, nsjail build, CUDA cargo features) to
# the Linux path; on macOS those are skipped and `cargo build` uses
# `metal,no-cuda`. The KV-tier persistence substrate (`crates/kv-native-sys`)
# is now pure Rust — no external toolchain is required. All Python deps
# install into .venv/.
# Activate manually:  source .venv/bin/activate
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      PLATFORM="unsupported" ;;
esac

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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$PLATFORM" = "unsupported" ]; then
    echo "[fail] unsupported host: $(uname -s) $(uname -m). Linux/x86_64 + Darwin/arm64 only." >&2
    exit 1
fi

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

    if [ "$PLATFORM" = "linux" ]; then
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            ok "nvcc: $CUDA_HOME/bin/nvcc"
            info "  $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
        else
            fail "nvcc not found at $CUDA_HOME/bin/nvcc"
            errors=$((errors + 1))
        fi

        if check_cmd nvidia-smi; then
            info "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        else errors=$((errors + 1)); fi
    else
        if xcrun --find metal &>/dev/null; then
            ok "metal toolchain: $(xcrun --find metal)"
        else
            fail "xcrun metal not found — install Xcode CLT: xcode-select --install"
            errors=$((errors + 1))
        fi
    fi

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

    # Pinned packages from requirements-build.txt.
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

    if [ "$PLATFORM" = "linux" ]; then
        if check_cmd nsjail; then
            info "  sandbox isolation active"
        else
            warn "nsjail not found — tool execution will run without sandbox"
        fi
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
    local expected_server
    if [ "$PLATFORM" = "linux" ]; then expected_server="target/release/infer"
    else expected_server="target/release/metal_serve"; fi
    if [ -x "$expected_server" ]; then
        ok "$expected_server built"
    else
        fail "server binary not found at $expected_server — run ./setup.sh --build-only"
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

    # --- CUDA / Metal ---
    if [ "$PLATFORM" = "linux" ]; then
        step "CUDA toolkit"
        if [ -x "$CUDA_HOME/bin/nvcc" ]; then
            ok "nvcc: $("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep release)"
        else
            fail "CUDA toolkit not found at $CUDA_HOME"
            info "Install CUDA toolkit or set CUDA_HOME=/path/to/cuda"
            exit 1
        fi

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
    else
        step "Apple Silicon / Metal backend"
        if xcrun --find metal &>/dev/null; then
            ok "Metal toolchain (xcrun): $(xcrun --find metal)"
        else
            warn "xcrun metal not found — install Xcode Command Line Tools: xcode-select --install"
        fi
        info "skipping nsjail (Linux-only); Mac will run tools without sandbox"
    fi

    # --- git hooks (inlined from former scripts/install_git_hooks.sh) ---
    step "Git hooks"
    git config core.hooksPath .githooks
    ok "core.hooksPath=.githooks"

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

    # --- Build deps ---
    # TileLang is the only Python AOT dependency for CUDA kernels; the rest is
    # platform-neutral utility support.
    step "Python build dependencies (from requirements-build.txt)"
    if [ "$PLATFORM" = "linux" ]; then
        grep -E '^[a-zA-Z]' requirements-build.txt | pip install -r /dev/stdin -q
        ok "CUDA build deps installed"
    else
        grep -E '^[a-zA-Z]' requirements-build.txt | grep -Ev '^tilelang($|[<=>])' | \
            pip install -r /dev/stdin -q
        ok "Platform-neutral build deps installed (TileLang skipped)"
    fi

    # --- Bench/test deps ---
    step "Bench & test dependencies (from requirements-bench.txt)"
    pip install -r requirements-bench.txt -q
    ok "Bench deps installed"

    # --- TileLang (Linux/CUDA only - AOT codegen for CUDA cubins) ---
    if [ "$PLATFORM" = "linux" ]; then
        step "TileLang (AOT codegen for --features cuda)"
        if python -c "import tilelang" 2>/dev/null; then
            ok "tilelang already installed: $(python -c 'import tilelang; print(tilelang.__version__)')"
        else
            pip install tilelang -q
            ok "tilelang installed: $(python -c 'import tilelang; print(tilelang.__version__)')"
        fi
    fi

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

    if [ "${ARLE_SKIP_WEB:-0}" != "1" ]; then
        do_web || warn "Web frontend setup failed (non-fatal); rerun: ./setup.sh --web-only"
    fi
}

# ============================================================================
# BUILD — compile Rust + CUDA kernels
# ============================================================================
do_build() {
    if [ "$PLATFORM" = "linux" ]; then
        step "Building ARLE CLI + infer server (release, CUDA)"
    else
        step "Building ARLE CLI + infer server (release, Metal)"
    fi
    activate_venv

    # Ensure cargo is on PATH
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

    local arle_features infer_features
    if [ "$PLATFORM" = "linux" ]; then
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
        # TileLang AOT needs Python from venv
        export INFER_TILELANG_PYTHON="$(which python)"

        info "CUDA_HOME=$CUDA_HOME"
        info "TILELANG_PYTHON=$INFER_TILELANG_PYTHON"
        if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
            info "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (override)"
        elif [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
            info "CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES (override)"
        else
            local detected
            detected=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr '\n' ' ')
            if [ -n "$detected" ]; then
                info "SM targets (auto-detect from nvidia-smi): $detected"
            else
                info "SM targets: T1 default {sm_80, sm_86, sm_89, sm_90}"
                info "  set TORCH_CUDA_ARCH_LIST to override; see docs/plans/sm-coverage.md"
            fi
        fi
        arle_features="cuda,cli"
        infer_features="cuda"
    else
        info "Apple Silicon: building with metal,no-cuda"
        arle_features="metal,no-cuda"
        infer_features="metal,no-cuda"
    fi

    local start
    start=$(date +%s)

    cargo build --release --no-default-features --features "$arle_features" -p agent-infer --bin arle 2>&1 | while IFS= read -r line; do
        case "$line" in
            *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                echo "  $line" ;;
        esac
    done

    local server_bin
    if [ "$PLATFORM" = "linux" ]; then
        cargo build --release --no-default-features --features "$infer_features" -p infer --bin infer 2>&1 | while IFS= read -r line; do
            case "$line" in
                *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                    echo "  $line" ;;
            esac
        done
        server_bin="target/release/infer"
    else
        cargo build --release --no-default-features --features "$infer_features" -p infer --bin metal_serve 2>&1 | while IFS= read -r line; do
            case "$line" in
                *warning:*|*error:*|*Compiling*infer*|*Compiling*agent*)
                    echo "  $line" ;;
            esac
        done
        server_bin="target/release/metal_serve"
    fi

    local elapsed=$(( $(date +%s) - start ))
    ok "Build complete in ${elapsed}s"

    if [ -x target/release/arle ]; then
        info "Binary: target/release/arle ($(du -h target/release/arle | awk '{print $1}'))"
    fi
    if [ -x "$server_bin" ]; then
        info "Binary: $server_bin ($(du -h "$server_bin" | awk '{print $1}'))"
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
# WEB — bun + web/ frontend deps (Astro + Vite landing site)
# Cross-platform; safe to run on macOS and Linux without CUDA.
# ============================================================================
do_web() {
    step "Web frontend (web/ — Astro + Vite + bun)"

    if [ ! -d web ]; then
        warn "web/ directory not present; skipping"
        return 0
    fi

    # Make sure bun is on PATH (handle prior install at $HOME/.bun)
    if ! command -v bun &>/dev/null && [ -x "$HOME/.bun/bin/bun" ]; then
        export PATH="$HOME/.bun/bin:$PATH"
    fi

    if ! command -v bun &>/dev/null; then
        info "Installing bun (bun.sh)..."
        curl -fsSL https://bun.sh/install | bash
        export PATH="$HOME/.bun/bin:$PATH"
    fi

    if ! command -v bun &>/dev/null; then
        fail "bun install did not put bun on PATH; check \$HOME/.bun/bin"
        return 1
    fi

    ok "bun $(bun --version)"
    info "Installing web/ dependencies..."
    (cd web && bun install --frozen-lockfile)
    ok "web/ ready — make web-dev / make web-build"
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
    if [ "${ARLE_SKIP_WEB:-0}" != "1" ]; then
        do_web || warn "Web frontend setup failed (non-fatal); rerun: ./setup.sh --web-only"
    fi

    echo ""
    step "Setup complete!"
    echo ""
    info "Quick start:"
    echo ""
    echo "  # 1. Activate the virtual environment"
    echo "  source .venv/bin/activate"
    echo ""
    if [ "$PLATFORM" = "linux" ]; then
        echo "  # 2. Set runtime library paths (Linux/CUDA)"
        echo "  export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
        echo ""
    fi
    echo "  # Run agent REPL"
    echo "  ./target/release/arle --model-path $MODEL_DIR"
    echo ""
    if [ "$PLATFORM" = "linux" ]; then
        echo "  # Run HTTP server (CUDA)"
        echo "  ./target/release/infer --model-path $MODEL_DIR --port 8000"
    else
        echo "  # Run HTTP server (Metal)"
        echo "  ./target/release/metal_serve --model-path $MODEL_DIR --port 8000"
    fi
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
    web)   do_web ;;
    clean) do_clean ;;
    full)  do_full ;;
esac
