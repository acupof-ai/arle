# Build environment for the H20 (sm_90) box — the single source of truth.
# Sourced by scripts/pod-remote-build.sh on the pod. Edit here, never inline a
# one-off env in an exec command (that drift is how builds became non-deterministic).
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0          # H20 == sm_90
export CMAKE_CUDA_ARCHITECTURES=90
# tilelang AOT venv (needed only for the full arle binary / cuda-kernels sm_90 AOT,
# not for single-crate builds like -p autograd). Create with scripts/pod.sh setup-tilelang.
export INFER_TILELANG_PYTHON="${INFER_TILELANG_PYTHON:-/host/arle-build/crates/cuda-kernels/tools/tilelang/.venv/bin/python}"
# Fast network path: ckl's reverse SOCKS5 proxy (pod -> local network). The pod's
# direct route to crates.io / static.rust-lang.org stalls; the proxy does not.
export all_proxy="${all_proxy:-socks5h://127.0.0.1:1080}"
export https_proxy="${https_proxy:-$all_proxy}"
export http_proxy="${http_proxy:-$all_proxy}"
# Shared compile cache: sccache caches each rustc compilation keyed by content, so a
# FRESH tree (or a toolchain-switch rebuild) reuses unchanged crates instead of
# recompiling — the cross-POD_TREE / cross-restart reuse the per-tree target/ can't give.
# Cache lives on /host (persistent). GRACEFUL: only wraps rustc if sccache is installed,
# so a missing binary never breaks the build (install: scripts/pod.sh setup-sccache).
if command -v sccache >/dev/null 2>&1; then
  export SCCACHE_DIR="${SCCACHE_DIR:-/host/sccache}"
  export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
  export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
fi
