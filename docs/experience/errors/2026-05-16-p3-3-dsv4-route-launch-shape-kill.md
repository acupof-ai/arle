# P3.3 DSv4 Route Launch Shape KILL

## Context

Kernel: `dsv4_route_kernel` in `crates/cuda-kernels/csrc/moe/dsv4_route.cu`.

Scope: P3.3 A5 tested whether shrinking the route-select block size helps the
local DeepSeek V4 1B-style learned-bias router. The current kernel performs the
route selection in `threadIdx.x == 0`, so smaller blocks were a plausible way
to reduce wasted CTA resources without changing routing math.

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_route --save-baseline p3_3_a5_before
```

Treatment command used the same filter with `--baseline p3_3_a5_before`.

## Root Cause

Hypothesis: because only one thread does useful work, route-select block sizes
below the default 256 may improve local route latency.

Evidence did not support the hypothesis. Baseline:

| Shape | Baseline |
| --- | ---: |
| decode t1/e16/top2 | 9.4377 us |
| batch t64/e16/top2 | 9.6159 us |

Block=1 treatment:

| Shape | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | --- |
| decode t1/e16/top2 | 9.4304 us | -0.0634% | 0.60 | KILL |
| batch t64/e16/top2 | 9.6175 us | +0.1524% | 0.27 | KILL |

Block=32 treatment:

| Shape | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | --- |
| decode t1/e16/top2 | 9.4610 us | +0.3292% | 0.01 | KILL |
| batch t64/e16/top2 | 9.7026 us | +0.8815% | 0.00 | KILL |

Neither variant reached the >=3% license gate, and block=32 regressed both
local shapes.

## Fix

Treatment reverted. Keep `dsv4_route_kernel` launching with
`DSV4_ROUTE_BLOCK == 256`.

## Rule

Do not retune the DSv4 route-select block size for the local 1B E16/top2 path
without a new shape family and matched evidence. For this path, the measured
latency is dominated by launch/sync framing rather than CTA size.
