# A2 Autograd NCCL All-Reduce Finite-Diff Gate

## Goal

Unlock Path A A2 by adding a differentiable all-reduce sum primitive to
`autograd`, reusing the existing CUDA/NCCL collective backend instead of adding a
new communication stack.

## Hypothesis

For replicated distributed loss, `all_reduce_sum` is self-adjoint: forward
all-reduces activations and backward must all-reduce upstream gradients. A
2-rank finite-diff check on the distributed total loss should match the analytic
gradient within relative tolerance `1e-2` at `eps=1e-3`.

## Params

- Code path: `ops::all_reduce_sum` -> `Backend::all_reduce_sum_device`.
- CUDA backend: D2D copy into a fresh output handle, then NCCL in-place sum.
- Gate example: `crates/autograd/examples/a2_nccl_all_reduce_autograd.rs`.
- Formula per rank: `y = all_reduce_sum(x_rank)`, `loss_rank = sum(y * y)`.
- Numeric derivative: central difference over summed per-rank losses.

## Env

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPUs: H20 GPU4 and GPU5.
- Build env:
  - `CUDA_HOME=/usr/local/cuda`
  - `CUDARC_CUDA_VERSION=12090`
  - `INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`
  - `LIBRARY_PATH=/tmp/nccl-cu12-227/nvidia/nccl/lib:$LIBRARY_PATH`
  - `LD_LIBRARY_PATH=/tmp/nccl-cu12-227/nvidia/nccl/lib:$LD_LIBRARY_PATH`
- Note: `/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnccl.so` failed on
  this driver with "CUDA driver version is insufficient for CUDA runtime
  version"; the gate used `/tmp/nccl-cu12-227`.

## Results

Command:

```bash
ARLE_A2_WORLD=2 ARLE_A2_CUDA_DEVICES=4,5 \
cargo run -p autograd --release --no-default-features \
  --features cuda,nccl --example a2_nccl_all_reduce_autograd
```

Output:

```text
NCCL version 2.27.7+cuda12.9
a2_nccl_all_reduce_autograd world=2 devices=[4, 5] probe=rank0[2] eps=1.0e-3
loss_minus=1.327489929e2 loss_base=1.327500000e2 loss_plus=1.327510071e2
analytic=1.000000000e0 numeric=1.007080078e0 rel_err=7.030e-3 tol=1.0e-2
PASS
```

Local gates:

```bash
cargo fmt --check
cargo test -p autograd --release --no-default-features --features no-cuda \
  all_reduce_sum_single_rank_forward_backward_is_identity -- --nocapture
cargo clippy -p autograd --release --no-default-features --features no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo clippy -p autograd --release \
  --no-default-features --features cuda,nccl,no-cuda --examples -- -D warnings
```

## Problems

- `cuda-kernels/nccl` currently triggers the full CUDA/TileLang build, even for
  an autograd example that only needs NCCL FFI and `NcclBackend`. This is
  correct but unnecessarily slow; a future `collective-nccl` feature would make
  A2 iteration cheaper.
- The system NCCL under CUDA 12.9 is not usable with the current `.62` driver;
  use the known-compatible `/tmp/nccl-cu12-227` path for this gate.

## Learnings

The primitive is now licensed at the autograd level: forward and backward
collectives produce the correct distributed-loss gradient on a real 2-GPU NCCL
run. The next A2 step is wiring this op at the TP row-parallel boundaries in the
train model, then repeating a model-level multi-rank finite-diff gate.
