# DSV4 HD64 prefill NUM_THREADS sweep kill

## Context

User objective: optimize Qwen3.5 and DeepSeek V4 operators for fastest stable
execution. After HD64 prefill tile-size and swizzle alternatives were killed,
the next single-variable DSV4-mini prefill candidate was changing only the
TileLang thread count:

```python
NUM_THREADS = ...
```

The baseline HD64 prefill kernel uses `NUM_THREADS=128`, `BLOCK_M=64`,
`BLOCK_N=64`, `NUM_STAGES=2`, and `T.use_swizzle(panel_size=8)`.

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

Measured / observed A/B:

| Arm | Result | Verdict |
|---|---|---|
| `NUM_THREADS=128` baseline | `170.20-170.32 us`, point `170.27 us` | keep |
| `NUM_THREADS=64` | bench panicked: `CUDA_ERROR_INVALID_VALUE` from `tilelang_batch_prefill_paged_hd64_q16_kv1_run_cuda` | kill |
| `NUM_THREADS=256` | AOT failed before bench: TileLang `Layout infer conflict between p and p_bf16 in T.Parallel loop` | kill |

The 64-thread arm produced a cubin but failed at runtime. The 256-thread arm
does not pass TileLang layout inference for the current fragment layout.
Neither treatment is a valid runtime optimization candidate.

## Fix

Killed the HD64 prefill `NUM_THREADS={64,256}` sweep and restored:

```python
NUM_THREADS = 128
```

No runtime code change is kept.

## Rule

Do not revisit HD64 DSV4-mini prefill `NUM_THREADS=64` or `NUM_THREADS=256`
without first changing the TileLang fragment layout and proving AOT/runtime
validity. The current stable prefill thread count remains `128`.
