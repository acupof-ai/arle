# DSv4 MHC hc4 Sinkhorn Win

## Context

Phase 3 P3.4 A7 optimized `dsv4_mhc_params_kernel` in
`crates/cuda-kernels/csrc/misc/dsv4_mhc.cu`.

The local DeepSeek V4 1B config uses `hidden_size = 1024`, `hc_mult = 4`,
`hc_sinkhorn_iters = 20`, and `hc_eps = 1e-6`. The direct microbench added in
`7c7bc17` covers:

- `dsv4_mini_decode_t1_h4096_m24_hc4`
- `dsv4_mini_batch_t64_h4096_m24_hc4`

## What Worked

Specializing the thread0 Sinkhorn normalization for `hc_mult == 4` removes the
dynamic `n` loop bounds from the 4x4 row/column normalization path. The generic
path remains unchanged for other `hc_mult` values.

This is a kernel-only change: no launch shape, residual RMS reduction,
mix scaling, sigmoid, Sinkhorn iteration count, or output layout changed.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_mhc_params --save-baseline p3_4_a7_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_mhc_params --baseline p3_4_a7_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| decode t1/h4096/m24/hc4 | 42.329 us | 23.692 us | -43.980% | 0.00 | LICENSE |
| batch t64/h4096/m24/hc4 | 42.315 us | 23.761 us | -43.821% | 0.00 | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  gen_mhc_params_uses_rms_scaled_mixes -- --nocapture
```

Result:
`test model::deepseek::weights::tests::gen_mhc_params_uses_rms_scaled_mixes ... ok`.

## Rule

For DSv4 MHC parameter generation on the local hc4 substrate, specializing
tiny thread0 normalization loops is licensed when the generic fallback remains
and matched Criterion evidence passes both t1 and t64 shapes. Do not extend
this to other `hc_mult` values without separate component A/B and correctness
gates.
