# Triton AOT Integration

`infer` now builds `silu_mul`, `add`, and the embedding lookup kernels through Triton AOT by default.

## What this covers

- Build-time generation of Triton AOT cubins for:
  - `silu_mul`
  - `add`
  - `embedding`
  - `embedding_decode`
  - `embedding_batched`
- Generated C wrappers linked into the normal Rust build
- Default runtime routing of the corresponding ops onto Triton-generated launchers
- `extract_vec` / `write_vec` now using `cudarc` device-to-device memcpy instead of a custom CUDA copy kernel
- A focused `triton_silu_smoke` binary that compares the Triton path against a CPU reference

`build.rs` now skips compiling the replaced legacy CUDA translation units `csrc/activation.cu`, `csrc/elementwise.cu`, and `csrc/embedding.cu`.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Bootstrap a repo-local Triton Python once:

```bash
uv venv tools/triton/.venv
uv pip install -p tools/triton/.venv/bin/python triton
```

Then either point the build to that interpreter explicitly:

```bash
export INFER_TRITON_PYTHON=$PWD/tools/triton/.venv/bin/python
```

or let `build.rs` auto-probe `tools/triton/.venv/bin/python` before trying `python3` / `python`.

If `nvidia-smi` is unavailable where you build, also set the target SM manually.

```bash
export TORCH_CUDA_ARCH_LIST="12.0"        # PyTorch native, RTX 5090 only
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # T1 fat binary
```

`TORCH_CUDA_ARCH_LIST` (PyTorch / vLLM standard) drives the explicit Triton AOT
compile target, so it is the default escape hatch when the build environment
cannot query a live GPU. `CMAKE_CUDA_ARCHITECTURES` works as alias. See
[`docs/plans/sm-coverage.md`](../../../../docs/plans/sm-coverage.md) for the
full tier policy.

### Windows

Official Triton does not ship Windows wheels. Use [`triton-windows`](https://github.com/woct0rdho/triton-windows) instead:

```powershell
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:INFER_TRITON_PYTHON = ".venv\Scripts\python.exe"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```

Requires CUDA 12+, Python 3.9–3.12, and an NVIDIA GPU with compute capability 7.5+ (GTX 16xx or newer).

## Build

```bash
cargo build --release
```

Generated Triton artifacts are written to Cargo `OUT_DIR`, typically under:

```text
target/release/build/infer-*/out/triton_aot/
```

## Validation

Sanity-check the default `silu_mul` path against a host-side reference:

```bash
cargo run --release --bin triton_silu_smoke -- --seq-len 32 --hidden-dim 4096 --iters 20
```

Run the focused GPU tests for the newly replaced paths:

```bash
cargo test --release embedding_variants -- --nocapture
cargo test --release extract_write_vec_roundtrip -- --nocapture
cargo test --release add_and_add_inplace -- --nocapture
```

## Common failures

- `Could not find a Python interpreter with Triton installed`
  - Set `INFER_TRITON_PYTHON`, or bootstrap `tools/triton/.venv` with `uv`.
- `GPU detection failed`
  - Set `TORCH_CUDA_ARCH_LIST` explicitly if `nvidia-smi` is not available during build.
- `Triton AOT generator failed`
  - Re-run the build and inspect the generator stderr printed by `build.rs`; the generator accepts an explicit `cuda:<sm>:32` target derived from `TORCH_CUDA_ARCH_LIST`.
- `CUDA_ERROR_NO_BINARY_FOR_GPU` or similar runtime load errors
  - Rebuild on the target GPU environment; the generated Triton cubin is target-specific.
