# DSV4 HD64 decode NUM_THREADS sweep kill

## Context

User objective: optimize Qwen3.5 and DeepSeek V4 operators for fastest stable
execution. After the HD64 swizzle sweep failed, the next DSV4-mini decode
candidate was changing only the TileLang decode thread count:

```python
NUM_THREADS = ...
```

The baseline HD64 decode kernel uses `NUM_THREADS=128`, `BLOCK_M=64`,
`BLOCK_N=256`, `NUM_STAGES=2`, and `T.use_swizzle(panel_size=8)`.

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

Measured / observed A/B:

| Arm | Result | Verdict |
|---|---|---|
| `NUM_THREADS=128` baseline | `97.282-97.848 us`, point `97.591 us` | keep |
| `NUM_THREADS=64` | bench panicked: `CUDA_ERROR_INVALID_VALUE` from `tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda` | kill |
| `NUM_THREADS=256` | AOT failed before bench: TileLang `Layout infer conflict between p and p_bf16 in T.Parallel loop` | kill |

The 64-thread arm produced a cubin but failed at runtime. The 256-thread arm
did not pass TileLang AOT layout inference for the existing fragment layout.
Neither is a valid candidate for a runtime optimization.

This does not prove `128` is globally optimal for every future HD64 shape, but
it is the only stable thread count among the tested local variants for the
current DSV4-mini `(q16, kv1, head_dim=64)` decode substrate.

## Fix

Killed the HD64 decode `NUM_THREADS={64,256}` sweep and restored:

```python
NUM_THREADS = 128
```

No runtime code change is kept.

## Rule

Do not revisit HD64 DSV4-mini decode `NUM_THREADS=64` or `NUM_THREADS=256`
without first changing the TileLang fragment layout and proving AOT/runtime
validity. The current stable decode thread count remains `128`.
