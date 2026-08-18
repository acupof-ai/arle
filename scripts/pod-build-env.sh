# Build environment for the H20 (sm_90) box — the single source of truth.
# Sourced by scripts/pod-remote-build.sh on the pod. Edit here, never inline a
# one-off env in an exec command (that drift is how builds became non-deterministic).
export ARLE_RUST_TOOLCHAIN_DIR="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"
if [ -x "$ARLE_RUST_TOOLCHAIN_DIR/bin/cargo" ] && [ -x "$ARLE_RUST_TOOLCHAIN_DIR/bin/rustc" ]; then
  export PATH="$ARLE_RUST_TOOLCHAIN_DIR/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH"
  export RUSTC="${RUSTC:-$ARLE_RUST_TOOLCHAIN_DIR/bin/rustc}"
else
  export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
fi
export CUDA_HOME=/usr/local/cuda
# Shared 180-core box: unbounded cargo parallelism spawns that many nvcc at
# once (~4 GB each) and OOM-kills the build. 32 keeps peak under ~130 GB.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-32}"
# ELKEID (host security agent) SIGKILLs any process that forks while a CUDA
# device is visible — nvcc/ptxas/cicc fork heavily during cuda-kernels build.
# The build does not need a GPU (it cross-compiles for sm_90), so hide devices
# from the build process. The runtime (pod-remote-run.sh) sets its own
# CUDA_VISIBLE_DEVICES from the claimed GPU.
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST=9.0          # H20 == sm_90
export CMAKE_CUDA_ARCHITECTURES=90
# TileLang AOT regen — REQUIRED for full arle / cuda-kernels builds now that the
# generated/ artifacts are gitignored (kernels regenerate on demand; nvcc 12.8+
# cross-compiles every SM incl. Blackwell). Resolve a python that imports tilelang:
# prefer the dedicated venv, else any python3 that has it (the build image ships it
# via requirements-build.txt). If neither, leave unset — a from-source CUDA build
# then fails loudly; install via `scripts/pod.sh setup-tilelang`. (Single-crate
# builds like -p autograd don't touch this.)
if [ -z "${INFER_TILELANG_PYTHON:-}" ]; then
  # Match the CI trusted producer's pin exactly — a loose >= probe let the pod
  # build kernels with a different codegen than the published bundle.
  tilelang_pin="$(grep -oE 'tilelang==[0-9.]+' "${TREE:-/host/arle-build}/requirements-build.txt" 2>/dev/null | cut -d= -f3)"
  tilelang_pin="${tilelang_pin:-0.1.13}"
  for python in \
    "${TREE:-/host/arle-build}/crates/cuda-kernels/tools/tilelang/.venv/bin/python" \
    /host/arle-build/crates/cuda-kernels/tools/tilelang/.venv/bin/python \
    "$(command -v python3 2>/dev/null)"; do
    [ -x "$python" ] || continue
    if "$python" -c "import importlib.metadata as m; assert m.version('tilelang') == '$tilelang_pin'" >/dev/null 2>&1; then
      export INFER_TILELANG_PYTHON="$python"
      break
    fi
  done
fi
# CUTLASS for the W4AFP8 MoE kernel: the FlashMLA vendored CUTLASS 4.x
# (crates/cuda-kernels/vendor/flashmla/csrc/cutlass/) — build.rs points at it
# directly. No standalone download needed.
# Persistent kernel-artifact cache (keyed on kernel source × SM × nvcc). build.rs
# regenerates kernels on demand (generated/ gitignored); this cache restores a prior
# build's identical artifacts instead of re-running TileLang+nvcc. MUST live on /host
# (the container $HOME is ephemeral). Measured: a forced kernel regen 251s -> 53s on a
# warm cache. Disable with ARLE_CUDA_KERNEL_CACHE=0.
export ARLE_CUDA_KERNEL_CACHE_DIR="${ARLE_CUDA_KERNEL_CACHE_DIR:-/host/arle-kernel-cache}"
# Fast network path: use ckl's reverse SOCKS5 proxy only when it is actually up.
# A stale proxy env makes rustup fail before it tries the pod's working direct route.
if timeout 1 bash -lc '</dev/tcp/127.0.0.1/1080' >/dev/null 2>&1; then
  export all_proxy="${all_proxy:-socks5h://127.0.0.1:1080}"
  export https_proxy="${https_proxy:-$all_proxy}"
  export http_proxy="${http_proxy:-$all_proxy}"
else
  unset all_proxy https_proxy http_proxy
fi
# Shared compile cache: sccache caches each rustc compilation keyed by content, so a
# FRESH tree (or a toolchain-switch rebuild) reuses unchanged crates instead of
# recompiling — the cross-POD_TREE / cross-restart reuse the per-tree target/ can't give.
# Cache lives on /host (persistent). .cargo/config.toml sets rustc-wrapper = "sccache";
# if sccache is absent we must explicitly clear RUSTC_WRAPPER to override that config.
# Install: scripts/pod.sh setup-sccache.
if command -v sccache >/dev/null 2>&1; then
  export SCCACHE_DIR="${SCCACHE_DIR:-/host/sccache}"
  export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
  export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
  # Same cache for nvcc: build.rs honors ARLE_NVCC_WRAPPER for the cuda-kernels /
  # deepep .cu compiles (docs/environment.md). Cargo has no nvcc wrapper knob, so
  # this env var is the only place to express it. Arch is already pinned to sm_90
  # above, so a warm build reuses the exact per-SM cubins.
  export ARLE_NVCC_WRAPPER="${ARLE_NVCC_WRAPPER:-sccache}"
else
  export RUSTC_WRAPPER=""
fi
