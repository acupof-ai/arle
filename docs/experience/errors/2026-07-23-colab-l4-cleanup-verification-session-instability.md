# Colab L4 cleanup verification blocked by unstable execution sessions

## Context

Commit `834d6d54e4d1584ee5e153546ac04100800e43bc` needed a cold `sm_89` source build, Qwen3.5-4B inference, needle gate, CUDA MoE test, and matched throughput A/B.

A real Google Colab L4 was allocated and identified as NVIDIA L4 (`sm_89`, 23,034 MiB), driver `580.82.07`, CUDA `12.8.93`.

## Root Cause

The first cold build correctly disabled prebuilt artifacts and reached native CUDA/FlashMLA compilation, then aborted in `cuda-kernels/build.rs` because Colab's system TileLang imported an unpatched `apache-tvm-ffi` and hit duplicate `TypeAttr __ffi_repr__` registration.

The repo's `scripts/pod-tilelang-env.sh` is the correct fix, but Colab's `/usr/bin/python3` lacks `ensurepip`; its venv must first be created with `virtualenv`. The original session disappeared during that repair. Three subsequent L4 allocations failed before the first command with `WebSocketConnectionClosedException: socket is already closed`.

## Fix

No code workaround was added. The exact environment sequence for the next stable session is:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv --system-site-packages \
  crates/cuda-kernels/tools/tilelang/.venv
POD_TREE=/content/arle bash scripts/pod-tilelang-env.sh
```

Require `patched-ok`, then rebuild with:

```bash
CARGO_BUILD_RUSTC_WRAPPER='' \
ARLE_CUDA_KERNEL_CACHE=0 \
ARLE_TILELANG_REGEN=1 \
TORCH_CUDA_ARCH_LIST=8.9 \
CMAKE_CUDA_ARCHITECTURES=89 \
cargo build --release --features cuda -p arle --bin arle
```

`ARLE_CUDA_KERNELS_PREBUILT_DIR` must remain unset.

## Rule

A Colab allocation is not a verification result. The gate remains `pending-remote` until the cold build, decoded inference, needle matrix, CUDA test, and matched benchmark all complete. Never convert environment reachability or partial compilation into a runtime PASS.
