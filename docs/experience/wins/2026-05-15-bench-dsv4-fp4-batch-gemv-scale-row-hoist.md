# DeepSeek V4 batch FP4 GEMV scale-row hoist - 2026-05-15

## Goal

- Finish the dirty DSv4 FP4 batch GEMV cleanup by applying the same scale-row
  hoist pattern already validated for raw FP4, raw FP8, and batch FP8 GEMV.

## Hypothesis

- `dsv4_fp4_gemv_batch_kernel` still called `dsv4_block_scale` inside the
  inner `k` loop. For a fixed output row, `block_h`, `block_w`, selected scale
  row, and scale-row base offset are invariant across that loop. Hoisting those
  values should remove repeated integer work without changing FP4 packing,
  FP4 decode, E8M0 scale decode, reduction, batch layout, or launch shape.

## Command

Component A/B:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp4_gemv_batch
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --features cuda \
  test_dsv4_fp4_batched_gemv -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `dsv4_fp4_gemv_batch_cuda`
- Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, SM89, 16376 MiB VRAM
- Driver / CUDA: 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- Feature set: `cargo bench -p infer --features cuda --bench ops_bench`
- Non-default flags / env vars: `CUDARC_CUDA_VERSION=13010`,
  `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Params

| Param | hidden shape | MoE shape |
|---|---:|---:|
| batch | 4 | 4 |
| rows | 1024 | 512 |
| cols | 1024 | 1024 |
| scale_rows | 8 | 4 |
| scale_cols | 8 | 8 |
| scale block | 128x128 | 128x128 |
| input | BF16 `[batch, cols]` | BF16 `[batch, cols]` |
| weights | packed FP4 E2M1 bytes | packed FP4 E2M1 bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Pre-rebase dirty-tree component A/B. Only
`dsv4_fp4_gemv_batch_kernel` changed:

```cpp
// before: dsv4_block_scale(scales, row, k, N, K, scale_rows, scale_cols)
// after: hoist block_h/block_w/scale_row_offset outside the k loop
```

| Shape | Saved baseline estimate | Scale-row hoist | Delta |
|---|---:|---:|---:|
| b4 1024x1024 | about `25.280 us` | `24.783-24.810 us`, point `24.793 us` | `-1.93%` |
| b4 512x1024 | about `15.824 us` | `15.730-15.755 us`, point `15.740 us` | `-0.53%` |

Criterion's saved-baseline comparison:

| Shape | Criterion time change | p-value | Criterion note |
|---|---:|---:|---|
| b4 1024x1024 | `-3.6721% .. -0.6140%`, point `-1.9259%` | `0.01 < 0.05` | Change within noise threshold |
| b4 512x1024 | `-0.6401% .. -0.4094%`, point `-0.5314%` | `0.00 < 0.05` | Change within noise threshold |

Throughput:

| Shape | Scale-row hoist | Criterion throughput change |
|---|---:|---:|
| b4 1024x1024 | `169.17 Gelem/s` | `+1.9637%` point |
| b4 512x1024 | `133.24 Gelem/s` | `+0.5342%` point |

## Post-Rebase Dispatch Audit

After rebasing onto `origin/main`, upstream code had added
`dsv4_fp4_gemv_batch_tiled_kernel`, and `dsv4_fp4_gemv_batch_cuda` dispatches
`B > 1` through that tiled kernel. The committed hoist therefore only affects
the legacy fallback path (`B == 1`) on the rebased tree; the mandated Prelude
bench above is still the dirty-tree evidence for the original cleanup, not a
current active B=4 win.

Control run on the rebased tree with the active tiled path unchanged:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  dsv4_fp4_gemv_batch --save-baseline prelude_fp4_batch_tiled_unhoisted
```

| Shape | Rebased active tiled path |
|---|---:|
| b4 1024x1024 | `19.097-19.209 us`, point `19.157 us` |
| b4 512x1024 | `13.788-13.848 us`, point `13.820 us` |

Temporary tiled-kernel scale-row hoist A/B against that rebased baseline
was tested and not committed:

| Shape | Temporary tiled hoist | Criterion time change | Decision |
|---|---:|---:|---|
| b4 1024x1024 | `19.246-19.389 us`, point `19.325 us` | `+0.4163% .. +1.1560%`, point `+0.7870%` | kill |
| b4 512x1024 | `13.792-13.824 us`, point `13.808 us` | `-0.2743% .. +0.0940%`, point `-0.0909%` | no evidence |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp4_batched_gemv ... ok
```

## Problems

- This is a component bench only, not a DSv4 request-level wall-clock result.
- Both tested shapes were positive but Criterion still classed the change as
  within its noise threshold. Treat the root-cause and request-level impact as
  hypothesis-grade until a later DSv4 decode trace includes this kernel in the
  full wall-clock frame.
- The local branch was 131 commits behind `origin/main` while the dirty patch
  was first validated. Rebase preserved the legacy-kernel cleanup, but upstream
  now routes `B > 1` to the tiled kernel, so there is no post-rebase active B=4
  win to claim from this commit.

## Learnings

- The scale-row hoist is much smaller for batch FP4 than for batch FP8 on this
  SM89 component bench. It remains a low-risk cleanup because it removes
  repeated invariant integer work and does not alter memory layout, dtype
  decode, launch geometry, or output shape.
- The same idea does not transfer automatically to the rebased active tiled
  B>1 path: the controlled temporary patch regressed the 1024x1024 shape by
  `+0.7870%` and showed no evidence on the 512x1024 shape. Punt tiled FP4 batch
  scale-row work unless a new trace identifies it as request-level material.
- Do not extrapolate this microbench into end-to-end DSv4 decode impact. The
  2026-05-14 nsys trace still ranks NCCL and allocation/memset/readback effects
  above this component axis.

## Tradeoff

- LOC complexity: +7 CUDA lines in one kernel.
- SM/hardware specificity: none; the transformation is scalar index hoisting.
- CUDA Graph compatibility: unchanged.
- Peak VRAM / warmup cost: unchanged.
- Numerical correctness risk: low; correctness test passed and scale lookup
  selects the same block row/column as `dsv4_block_scale`.

## Delta vs Baseline

Baseline:

- `65e0c3d bench(cuda): add dsv4 raw batch gemv cases`
- Related precedent:
  [`2026-05-12-bench-dsv4-fp8-batch-gemv-scale-row-hoist.md`](2026-05-12-bench-dsv4-fp8-batch-gemv-scale-row-hoist.md)

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP4 batch b4 1024x1024 latency | about `25.280 us` | `24.793 us` | `-1.93%` |
| FP4 batch b4 512x1024 latency | about `15.824 us` | `15.740 us` | `-0.53%` |

Rebased active B>1 tiled path control:

| metric | unhoisted tiled | temporary tiled hoist | delta | decision |
|---|---:|---:|---:|---|
| FP4 batch b4 1024x1024 latency | `19.157 us` | `19.325 us` | `+0.7870%` | not committed |
| FP4 batch b4 512x1024 latency | `13.820 us` | `13.808 us` | `-0.0909%` | not committed |
