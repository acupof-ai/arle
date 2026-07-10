# DSV4 HD64 decode NUM_STAGES sweep kill

## Context

User objective: optimize Qwen3.5 and DeepSeek V4 operators for fastest stable
execution. After DSV4-mini HD64 swizzle and thread-count alternatives were
killed, the next decode-only single-variable candidate was TileLang pipeline
stage count:

```python
NUM_STAGES = ...
```

The baseline HD64 decode kernel uses `NUM_STAGES=2`, `NUM_THREADS=128`,
`BLOCK_M=64`, `BLOCK_N=256`, and `T.use_swizzle(panel_size=8)`.

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
  ops_cuda/tilelang_decode_hd64_dsv4mini --quiet
```

Environment:

| Param | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| Driver / CUDA | 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78) |
| cudarc override | `CUDARC_CUDA_VERSION=13010` |
| Operator | `tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda` |
| Shape | DSV4-mini HD64, `q16_kv1`, `kv_len=4096` |

## Root Cause

Measured A/B:

| Arm | Time | Delta vs baseline | Verdict |
|---|---:|---:|---|
| `NUM_STAGES=2` baseline | `97.282-97.848 us`, point `97.591 us` | baseline | keep |
| `NUM_STAGES=1` | `97.146-97.731 us`, point `97.413 us` | `-0.18%` | kill: overlap/noise |
| `NUM_STAGES=3` | `97.154-97.729 us`, point `97.418 us` | `-0.18%` | kill: overlap/noise |

Both treatment intervals overlap the baseline. The point estimates are below a
meaningful operator threshold and do not justify a TileLang AOT churn.

## Fix

Killed the HD64 decode `NUM_STAGES={1,3}` sweep and restored:

```python
NUM_STAGES = 2
```

No runtime code change is kept.

## Rule

Do not revisit HD64 DSV4-mini decode `NUM_STAGES=1` or `NUM_STAGES=3` without
a new profiler trace showing pipeline staging is the bottleneck. For the
current sm_89 component bench, `NUM_STAGES=2` remains the stable setting.
