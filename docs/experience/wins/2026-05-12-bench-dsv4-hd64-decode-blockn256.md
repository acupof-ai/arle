# DSV4-mini HD64 decode BLOCK_N=256 - 2026-05-12

## Goal

- Add a component benchmark for the existing DSV4-mini HD64 TileLang decode
  substrate and tune its decode-only KV tile without wiring an unsupported
  DeepSeek V4 runtime path.

## Hypothesis

- The HD64 decode kernel serves one Q row per request but inherited
  `BLOCK_N=16` from the page size. Increasing the KV tile should reduce
  per-page loop overhead for long decode contexts while keeping the existing
  `BLOCK_M=64` tensor-core layout that TileLang can compile on sm_89.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/tilelang_decode_hd64_dsv4mini --quiet
```

## Environment

- **Backend:** CUDA
- **Operator:** `tilelang_batch_decode_paged_hd64_q16_kv1_run_cuda`
- **Model shape:** DSV4-mini substrate only, not a wired model path
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB VRAM
- **Driver / CUDA:** 595.71.05 / CUDA 13.2 (`nvcc` 13.2.78)
- **Commit under test:** working tree based on `87932ef`; this entry is
  committed with the code delta
- **Feature set:** `cargo bench -p infer --features cuda --bench ops_bench`
- **Non-default flags / env vars:** `NVCC_CCBIN=/usr/bin/g++-14`,
  `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Params

| Param | Value |
|---|---:|
| batch_size | 4 |
| seq_len | 4096 |
| page_size | 16 |
| total_pages | 1024 |
| num_q_heads | 16 |
| num_kv_heads | 1 |
| head_dim | 64 |
| q_dim | 1024 |
| kv_dim | 64 |
| KV dtype | BF16 |

## Results

All rows use the same bench shape and only change the TileLang HD64 decode
tile parameter under test.

| Candidate | Status | Criterion time | Throughput | Delta vs baseline |
|---|---|---:|---:|---:|
| baseline `BLOCK_M=64`, `BLOCK_N=16` | pass | 156.63-156.85 us, point 156.73 us | 107.04 Gelem/s | baseline |
| `BLOCK_M=16`, `BLOCK_N=16` | AOT kill | `warp_col_tiles` constraint failed (`got 4`) | n/a | n/a |
| `BLOCK_M=32`, `BLOCK_N=16` | AOT kill | TileLang layout conflict between `p` and `p_bf16` | n/a | n/a |
| `BLOCK_M=64`, `BLOCK_N=32` | pass | 118.60-118.71 us, point 118.65 us | 141.40 Gelem/s | -24.30% |
| `BLOCK_M=64`, `BLOCK_N=64` | pass | 105.03-105.08 us, point 105.06 us | 159.70 Gelem/s | -32.97% |
| `BLOCK_M=64`, `BLOCK_N=128` | pass | 100.09-100.67 us, point 100.35 us | 167.18 Gelem/s | -35.98% |
| `BLOCK_M=64`, `BLOCK_N=256` | **kept** | 97.123-97.738 us, point 97.404 us | 172.24 Gelem/s | **-37.85%** |
| `BLOCK_M=64`, `BLOCK_N=512` | launch kill | `CUDA_ERROR_INVALID_VALUE` | n/a | n/a |

Final rerun on the kept worktree:

| metric | value |
|---|---:|
| time interval | 97.179-97.779 us |
| time point | 97.453 us |
| throughput interval | 171.58-172.64 Gelem/s |
| throughput point | 172.16 Gelem/s |

## Problems

- This is not a full DeepSeek V4 serving benchmark. The HD64 kernel is a
  DSV4-mini-class GQA substrate already present in the build, while full V4
  MLA/MoE/block-FP8 serving remains unwired.
- No guidellm run is possible for this operator today because no runtime model
  dispatch consumes the HD64 DSV4-mini TileLang symbol.
- The benchmark is performance-only. It exercises launch and memory/layout
  behavior, but it is not a numerical baseline for a real checkpoint.

## Learnings

- The inherited `BLOCK_N=16` was too small for long single-token HD64 decode.
  Increasing to 256 cuts the component latency by `37.85%`.
- Reducing padded `BLOCK_M` is not currently viable with TileLang 0.1.9 on
  sm_89: `16` fails tensor-core emitter constraints and `32` fails layout
  inference.
- `BLOCK_N=512` crosses a launch/resource boundary on sm_89. For this shape,
  256 is the largest measured stable KV tile.

## Rule

- For DSV4/HD64 substrate work, do not assume inherited HD128 TileLang tile
  parameters are optimal. Add a shape-specific component bench first, then tune
  exactly one tile axis at a time and keep compile/launch failures as kill
  evidence.
