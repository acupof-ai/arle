# SiLU Mul fast exp kill

## Context

Goal thread: optimize Qwen3.5 and DSV4 operators for speed and stability.
After the FP8 KV refill win and FP4/W4A8 KV scan kill, the next small
Qwen3.5-relevant operator candidate was `silu_mul_cuda` in
`crates/cuda-kernels/csrc/misc/elementwise_basic.cu`.

Hypothesis: replacing device `expf` with CUDA `__expf` inside
`silu_mul_one` would reduce SiLU gate cost without changing the BF16 output
contract.

Only one variable was changed:

```cpp
float silu = g / (1.0f + expf(-g));
```

to:

```cpp
float silu = g / (1.0f + __expf(-g));
```

The code change was reverted before commit.

## Root Cause

Component A/B on RTX 4070 Ti SUPER, sm_89:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- ops_cuda/silu_mul_batch --quiet
```

| Arm | Criterion time | Throughput |
|---|---:|---:|
| baseline `expf` | 9.7938-9.8049 us, point 9.7987 us | 6.6840-6.6916 Gelem/s |
| treatment `__expf` | 9.7881-9.8006 us, point 9.7942 us | 6.6869-6.6955 Gelem/s |

Correctness gate passed for the treatment:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda \
  ops::tests::test_silu_mul_batch_tail_and_in_place -- --nocapture
```

Result: PASS, including tail and in-place alias cases.

The measured delta is about -0.05% and the intervals overlap. That is not a
license for a hot-path math intrinsic change.

## Fix

Killed the fast-exp SiLU micro-optimization and kept the existing `expf`
implementation. No runtime code changed.

The next SiLU work needs better evidence than a scalar intrinsic swap:

1. Inspect SASS or ncu first to verify whether `expf` is actually a visible
   bottleneck in this kernel.
2. If it is, test a larger structural change such as an approximate sigmoid
   polynomial or fused split-gate SiLU path with a numerical gate.
3. Keep the BF16 output comparison and in-place alias test as the first
   correctness gate.

## Rule

Do not land math-intrinsic changes on tiny overlapping microbench deltas. For
small elementwise kernels, require either a clearly separated component
interval or a downstream request-level regression gate that shows the change is
not noise.
