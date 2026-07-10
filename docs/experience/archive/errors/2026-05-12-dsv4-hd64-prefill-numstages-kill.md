# DSV4 HD64 prefill NUM_STAGES sweep kill

## Context

User objective: optimize Qwen3.5 and DeepSeek V4 operators for fastest stable
execution. After DSV4-mini HD64 prefill tile, swizzle, and thread-count
alternatives were killed, the next prefill-only single-variable candidate was
TileLang pipeline stage count:

```python
NUM_STAGES = ...
```

The baseline HD64 prefill kernel uses `NUM_STAGES=2`, `NUM_THREADS=128`,
`BLOCK_M=64`, `BLOCK_N=64`, and `T.use_swizzle(panel_size=8)`.

Baseline command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  'ops_cuda/tilelang_(prefill|decode)_hd64_dsv4mini' --quiet
```

Treatment command:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/tilelang_prefill_hd64_dsv4mini --quiet
```

Environment:

| Param | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78) |
| cudarc override | `CUDARC_CUDA_VERSION=13010` |
| Operator | `tilelang_batch_prefill_paged_hd64_q16_kv1_run_cuda` |
| Shape | DSV4-mini HD64, `q16_kv1`, `q_tokens=2048` |

## Root Cause

Measured A/B:

| Arm | Time | Delta vs baseline | Verdict |
|---|---:|---:|---|
| `NUM_STAGES=2` baseline | `170.20-170.32 us`, point `170.27 us` | baseline | keep |
| `NUM_STAGES=1` | `170.58-170.67 us`, point `170.63 us` | `+0.21%` | kill: regression |
| `NUM_STAGES=3` | `170.21-170.27 us`, point `170.24 us` | `-0.02%` | kill: overlap/noise |

`NUM_STAGES=1` is a small regression. `NUM_STAGES=3` overlaps the baseline and
does not justify a TileLang AOT setting change.

## Fix

Killed the HD64 prefill `NUM_STAGES={1,3}` sweep and restored:

```python
NUM_STAGES = 2
```

No runtime code change is kept.

## Rule

Do not revisit HD64 DSV4-mini prefill `NUM_STAGES=1` or `NUM_STAGES=3` without
a new profiler trace showing pipeline staging is the bottleneck. For the
current sm_89 component bench, `NUM_STAGES=2` remains the stable setting.
