# DSV4 HD64 decode page metadata hoist kill

## Context

Goal thread: optimize Qwen3.5 and DSV4 operators while keeping every decision
evidence-backed. After the DSV4 HD64 decode `BLOCK_N=256` win, the next
isolated hypothesis was to mirror the HD64 prefill kernel's page-metadata
hoist in the decode kernel.

Baseline command:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/tilelang_decode_hd64_dsv4mini --quiet
```

Shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| batch_size | 4 |
| seq_len | 4096 |
| page_size | 16 |
| num_q_heads | 16 |
| num_kv_heads | 1 |
| head_dim | 64 |
| `BLOCK_M` | 64 |
| `BLOCK_N` | 256 |

## Root Cause

The source hypothesis was plausible but wrong for TileLang 0.1.9 codegen on
this shape.

Candidate change:

- Add `page_idx_j`, `in_page_j`, and `valid_j` 1D fragments.
- Precompute page metadata once per `j`.
- Reuse those fragments in the `(j, d)` K/V load loop instead of recomputing
  `abs_col // PAGE_SIZE`, `abs_col % PAGE_SIZE`, and `KV_indices[...]` for
  every head-dim lane.

Measured result:

| Candidate | Criterion time | Throughput | Verdict |
|---|---:|---:|---|
| baseline current decode | `97.159-97.744 us`, point `97.425 us` | `172.21 Gelem/s` | baseline |
| page metadata hoist | `132.45-132.59 us`, point `132.53 us` | `126.59 Gelem/s` | kill, `+36.04%` latency |

The likely mechanism is extra fragment/register/layout pressure in the decode
kernel. Unlike prefill, decode has only one real Q row and already pays a large
padded `BLOCK_M=64` tensor-core layout cost. The hoisted fragments made the
compiled kernel materially slower even though they remove repeated scalar
index arithmetic in source.

## Fix

Reverted the candidate. Keep
`crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd64.py` at the existing
inline `(j, d)` metadata calculation.

No runtime code changed in the final tree for this experiment.

## Rule

Do not transfer the HD64 prefill page-metadata hoist to HD64 decode. In
TileLang kernels, reducing source-level duplicate index arithmetic can still
lose when the added fragments increase layout/register pressure. The
wall-clock component bench is the license source, not the source diff shape.

