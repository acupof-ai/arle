# DSv4 Route Top2 Selection Win

## Context

Phase 3 P3.3 A7 optimized `dsv4_route_kernel` in
`crates/cuda-kernels/csrc/moe/dsv4_route.cu`.

The local DeepSeek V4 1B checkpoint uses `num_experts_per_tok = 2`; the route
microbench added in `cf72142` covers learned-bias sqrtsoftplus routing for:

- `dsv4_mini_decode_t1_e16_top2`
- `dsv4_mini_batch_t64_e16_top2`

## What Worked

Special-casing `topk == 2` in the learned-bias route branch avoids the generic
insertion loop over `topk`. The generic path remains for other topk values.

This is a local kernel-only change: no launch-shape, scoring math, or output
normalization changes were made.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_route --save-baseline p3_3_a7_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_route --baseline p3_3_a7_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| decode t1/e16/top2 | 9.3875 us | 8.6982 us | -7.3898% | 0.00 | LICENSE |
| batch t64/e16/top2 | 9.5705 us | 8.7822 us | -8.3935% | 0.00 | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_route_cuda_top2_sqrtsoftplus_bias -- --nocapture
```

Result: `test ops::tests::test_dsv4_route_cuda_top2_sqrtsoftplus_bias ... ok`.

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  moe_forward_routed_computes_gate_routes_and_localizes_ep -- --nocapture
```

Result:
`test model::deepseek::mlp::tests::moe_forward_routed_computes_gate_routes_and_localizes_ep ... ok`.

## Rule

For DSv4 local route selection, topk-specific ranking can be licensed when it
keeps the generic fallback and has matched Criterion evidence across t1 and
t64. Do not broaden the specialization to other topk values without separate
bench and correctness gates.
