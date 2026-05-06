# Elementwise BF16x4 Vectorization — guidellm sweep, cuda, 2026-05-06

## Goal

Record the required benchmark slot for the CUDA elementwise `add` and
`silu_mul` vectorization tranche.

Goal type: optimization / pending-remote.

## Hypothesis

Vectorizing BF16 elementwise loads and stores four elements at a time should
reduce instruction and memory transaction overhead for the MLP activation and
residual-add helper kernels without changing BF16-rounded outputs.

## Command

Pending bench tooling repair:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
cargo build --release -p infer --features cuda --bin infer

scripts/bench_guidellm.sh cuda-elementwise-vectorized
```

## Environment

- **Backend:** cuda.
- **Model:** Qwen/Qwen3-4B target for the required regression sweep.
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB VRAM.
- **Driver / CUDA:** NVIDIA driver 595.71.05, CUDA 13.2 (`/opt/cuda`).
- **Commit:** pending commit for this tranche.
- **Feature set:** `cargo build --release -p infer --features cuda --bin infer`.
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14` required on
  this host because default `gcc` is 16.1.1.

## Results

Pending. The canonical `guidellm` sweep did not run in this workspace because
`guidellm` is not installed on PATH or in the current `.venv`.

TileLang AOT is available in the repo `.venv` and the focused CUDA tests pass.
The temporary `/tmp/arle-tilelang-venv` bootstrap was abandoned before use;
the working interpreter is `$PWD/.venv/bin/python`.

Completed local checks:

| check | result |
|---|---|
| `cargo fmt --check` | pass |
| `git diff --check -- crates/cuda-kernels/csrc/misc/elementwise_basic.cu infer/src/ops/tests.rs` | pass |
| single-file nvcc compile for `elementwise_basic.cu` | pass |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | pass |
| `cargo clippy -p infer --no-default-features --features cuda,no-cuda -- -D warnings` | pass |
| `CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python cargo test --release -p infer --features cuda ops::tests::test_add_batch_tail -- --nocapture` | pass |
| `CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python cargo test --release -p infer --features cuda ops::tests::test_silu_mul_batch_tail_and_in_place -- --nocapture` | pass |
| Qwen3-4B curl smoke, prompt `The cat sat on the mat. The dog`, 24 greedy tokens | pass; coherent continuation |

Blocked checks:

| check | blocker |
|---|---|
| `scripts/bench_guidellm.sh cuda-elementwise-vectorized` | `guidellm` CLI missing |

## Problems

- Canonical bench is blocked by missing `guidellm` tooling, not by the modified
  elementwise kernel.
- This host also needs `NVCC_CCBIN=/usr/bin/g++-14`; using default GCC 16
  failed before reaching the modified kernel.

## Learnings

- Keep elementwise optimization tranches independently build-checkable with
  direct nvcc compilation so local AOT environment failures do not hide syntax
  or ABI issues in the changed `.cu` file.
- Do not claim performance wins for small helper kernels without a canonical
  `guidellm` sweep; this entry is a placeholder until the TileLang build
  environment is repaired.

## Delta Vs Baseline

Pending. Use the most recent CUDA Qwen3-4B `guidellm` baseline when this entry
is closed with real numbers.

## Artefacts

- Pending raw output directory:
  `bench-output/2026-05-06-cuda-elementwise-vectorized/`
