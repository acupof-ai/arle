# DSv4 Compressor Pending Fast Path Win

## Context

Phase 3 P3.7 A3 optimized the `completed == 0` path in
`dsv4_compressor_update_kernel` at
`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`.

The setup bench in `6419cb7` added pending-only shapes because most DSv4
decode calls do not finish a compressor block. Before this change, even
`completed == 0` copied every existing pending row back to the same pending
buffer before appending the new raw token.

## What Worked

When no compressed block is completed, existing pending rows are already in the
correct slots. The treatment returns early after appending only the new raw
token rows to `pending_kv` and `pending_score`.

Completed-block processing, overlap cache updates, compressed row writes, RoPE,
and launch shape are unchanged.

## Evidence

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_compressor_update --save-baseline p3_7_pending_setup
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --bench ops_bench --features cuda -- \
  ops_cuda/dsv4_compressor_update --baseline p3_7_pending_setup
```

Results:

| Shape | Baseline | Treatment | Change | p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dsv4_mini_csa_first_r4_h64_overlap_rope` | `9.7731 us` | `9.7687 us` | `-0.1696%` | `0.10` | NO CHANGE |
| `dsv4_mini_csa_decode_r4_h64_overlap_rope` | `10.300 us` | `10.272 us` | `-0.1074%` | `0.41` | NO CHANGE |
| `dsv4_mini_indexer_decode_r4_h64_overlap_no_rope` | `9.7681 us` | `9.7711 us` | `-0.0008%` | `0.99` | NO CHANGE |
| `dsv4_mini_csa_pending_r4_h64_overlap_rope` | `6.9420 us` | `6.9000 us` | `-1.2477%` | `0.56` | NO CHANGE |
| `dsv4_mini_indexer_pending_r4_h64_overlap_no_rope` | `7.8091 us` | `7.7985 us` | `-0.0870%` | `0.00` | NOISE |
| `dsv4_mini_hca_decode_r96_h64_rope` | `32.538 us` | `32.517 us` | `-0.0431%` | `0.25` | NO CHANGE |
| `dsv4_mini_hca_pending_r96_h64_rope` | `10.436 us` | `6.7637 us` | `-37.583%` | `0.00` | LICENSE |

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_compressor_update_cuda -- --nocapture
```

Result:
`test_dsv4_compressor_update_cuda_pending_only_appends_raw ... ok` and
`test_dsv4_compressor_update_cuda_overlap_decode ... ok`.

## Tradeoffs

- License strength: strong for r96 HCA pending update; no material win for r4
  pending because launch overhead dominates the smaller copy.
- Semantic dependency: the fast path relies on pending rows being compact at
  the start of `pending_kv`/`pending_score`, which is the existing cache
  invariant exercised by the new test.
- Numerical behavior: appended score still adds APE using `abs_pos % ratio`;
  existing pending scores are preserved instead of recomputed.
- CUDA Graph compatibility: unchanged; ABI and launch shape are stable.

## Rule

For DSv4 compressor update, do not recopy old pending rows when
`completed == 0`. Append only new raw rows; this is licensed for the local r96
HCA pending decode shape and neutral for completed-block paths.
