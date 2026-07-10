# DSv4 Compressor Update Streaming Softmax Win

## Context

Phase 3 P3.7 A1 optimized `dsv4_compressor_update_kernel` in
`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`.

The setup bench in `a888c35` covers the local DSv4 1B compressor shapes from
`infer/models/dsv4-mini-1B-init/config.json`:

- r4 CSA compressor, overlap, apply compressor RoPE
- r4 CSA indexer compressor, overlap, no compressor RoPE
- r96 HCA compressor, no overlap, apply compressor RoPE

## What Worked

The treatment removes the per-column `float logits[256]` local cache from the
compressor softmax path. It computes `max_logit` in one pass, then recomputes
the logit in a second pass while accumulating `denom` and weighted KV value.

This keeps the single CTA launch shape, input/output ABI, overlap handling,
RoPE handling, and pending-tail update unchanged.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_compressor_update --save-baseline p3_7_a1_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_compressor_update --baseline p3_7_a1_before
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_csa_first_r4_h64_overlap_rope` | `10.108 us` | `9.7640 us` | `-3.4436%` | `0.00` | LICENSE |
| `dsv4_mini_csa_decode_r4_h64_overlap_rope` | `10.418 us` | `10.289 us` | `-1.1927%` | `0.00` | SUPPORTING |
| `dsv4_mini_indexer_decode_r4_h64_overlap_no_rope` | `9.8593 us` | `9.7576 us` | `-1.0347%` | `0.00` | SUPPORTING |
| `dsv4_mini_hca_decode_r96_h64_rope` | `32.999 us` | `32.520 us` | `-1.4503%` | `0.00` | SUPPORTING |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_compressor_update_cuda_overlap_decode -- --nocapture
```

Result:
`test ops::tests::test_dsv4_compressor_update_cuda_overlap_decode ... ok`.

## Tradeoffs

- License strength: one measured shape crosses the 3% gate; the other three
  are positive but below the license threshold. This is not evidence that
  decode/HCA are materially faster, only that they did not regress in this
  local Criterion run.
- Compute tradeoff: score logits are recomputed in the second pass. That is
  cheaper than keeping the 256-float local cache for the licensed r4 first
  block shape on SM89, but it is not a broad algorithmic conclusion.
- Numerical behavior: accumulation order changes slightly, so the CUDA test
  checks a direct overlap decode case against a CPU-computed softmax/norm
  reference.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.

## Rule

For DSv4 compressor update on the local SM89 substrate, streaming softmax is
licensed for the r4 first-block compressor shape. Treat the smaller positive
decode/HCA shifts as supporting evidence only; do not cite them as standalone
wins without a separate >=3% A/B.
