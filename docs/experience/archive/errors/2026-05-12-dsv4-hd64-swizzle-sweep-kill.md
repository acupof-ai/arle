# DSV4 HD64 TileLang swizzle sweep kill

## Context

User objective: optimize Qwen3.5 and DeepSeek V4 operators for fastest stable
execution. After the Qwen3.5 FP8/W4A8 activation-quant launch-smem trims, the
next DSV4 candidate was the HD64 TileLang attention substrate used by the
`dsv4mini` component benches.

Prior DSV4 evidence already killed HD64 prefill tile-size changes and decode
page-metadata hoisting. This experiment tested only TileLang block swizzle
panel size:

```python
T.use_swizzle(panel_size=...)
```

No runtime model wiring or ABI changed.

Commands:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  'ops_cuda/tilelang_(prefill|decode)_hd64_dsv4mini' --quiet

CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/tilelang_decode_hd64_dsv4mini --quiet

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
| Decode operator | `tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda` |
| Prefill operator | `tilelang_batch_prefill_paged_hd64_q16_kv1_run_cuda` |
| Decode shape | DSV4-mini HD64, `q16_kv1`, `kv_len=4096` |
| Prefill shape | DSV4-mini HD64, `q16_kv1`, `q_tokens=2048` |

## Root Cause

The hypothesis was that changing TileLang's swizzle panel size might improve
block scheduling for the HD64 `(q16, kv1)` DSV4-mini shape without touching
math, tile sizes, or memory layout.

Measured A/B:

| Operator | Panel size | Time | Delta vs baseline | Verdict |
|---|---:|---:|---:|---|
| decode | 8 baseline | `97.282-97.848 us`, point `97.591 us` | baseline | keep |
| decode | 4 | `97.266-97.792 us`, point `97.505 us` | `-0.09%` | kill: overlap/noise |
| decode | 16 | `97.114-97.642 us`, point `97.353 us` | `-0.24%` | kill: overlap/noise |
| prefill | 8 baseline | `170.20-170.32 us`, point `170.27 us` | baseline | keep |
| prefill | 4 | `190.69-190.89 us`, point `190.80 us` | `+12.06%` | kill: regression |
| prefill | 16 | `212.67-212.78 us`, point `212.72 us` | `+24.93%` | kill: regression |

Decode `panel_size=4` and `panel_size=16` are not licensed because their
intervals overlap the baseline and their point-estimate deltas are below any
meaningful operator threshold. Prefill smaller/larger panel sizes are clear
regressions.

The mechanism is likely scheduling-shape-specific: this HD64 substrate already
uses the inherited `panel_size=8`, and the tested alternatives either do not
move decode scheduling enough to matter or disrupt prefill scheduling. That
mechanism remains a hypothesis; the decision evidence is the controlled
component A/B above.

## Fix

Killed the HD64 swizzle sweep. Restored both HD64 TileLang kernels to:

```python
T.use_swizzle(panel_size=8)
```

No runtime code change is kept.

## Rule

Do not revisit HD64 DSV4-mini TileLang swizzle `panel_size={4,16}` without a
new shape or a profiler trace proving block scheduling is the bottleneck. For
the current sm_89 component benches, `panel_size=8` remains the fastest stable
setting: decode alternatives are noise, and prefill alternatives regress
substantially.
