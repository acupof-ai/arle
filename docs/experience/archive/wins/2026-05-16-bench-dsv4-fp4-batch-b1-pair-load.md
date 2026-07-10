# DeepSeek V4 batch FP4 B1 pair load - 2026-05-16

## Goal

- Test Phase 3 P3.2 A2: reduce duplicate FP4 packed-byte loads in
  `dsv4_fp4_gemv_batch_kernel`, the B=1 raw path behind
  `dsv4_fp4_gemv_batch_cuda`.

## Hypothesis

- Baseline maps one thread iteration to one logical K element. For FP4, two
  adjacent K elements share one packed byte, so adjacent lanes load the same
  byte and decode one nibble each.
- Pairing low/high nibble work in one thread should reduce packed weight byte
  loads from 1024 to 512 per output row for `K=1024` and halve loop
  iterations. Input BF16 loads and FMA count stay the same.
- Scale lookup intentionally remains per `k0`/`k1`; this does not mix in the
  P3.2 A1 scale-column hoist axis.

## Command

Baseline:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --save-baseline p3_2_a2_before
```

Treatment:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/dsv4_fp4_gemv_batch_b1 --baseline p3_2_a2_before
```

Correctness:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test --release -p infer --lib --features cuda \
  test_dsv4_fp4_batched_gemv_b1_raw -- --nocapture
```

## Environment

- Backend: CUDA
- Operator: `dsv4_fp4_gemv_batch_cuda`, B=1 raw dispatch
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
| batch | 1 | 1 |
| rows | 1024 | 512 |
| cols | 1024 | 1024 |
| scale_rows | 8 | 4 |
| scale_cols | 8 | 8 |
| scale block | 128x128 | 128x128 |
| input | BF16 `[1, cols]` | BF16 `[1, cols]` |
| weights | FP4 E2M1 packed bytes | FP4 E2M1 packed bytes |
| scales | FP8 E8M0 bytes, all `127` (=1.0) | FP8 E8M0 bytes, all `127` (=1.0) |

## Results - Component A/B

Only `dsv4_fp4_gemv_batch_kernel` changed:

```cpp
// before: one logical K element per thread iteration
// after: one packed byte per iteration, handling low and high nibbles together
```

| Shape | Baseline point | Pair-load point | Criterion change | p-value | Verdict |
|---|---:|---:|---:|---:|---|
| `dsv4_mini_hidden_1024x1024` | `11.348 us` | `9.5436 us` | `-15.860%` | `0.00 < 0.05` | LICENSE |
| `dsv4_mini_moe_512x1024` | `8.9282 us` | `8.0207 us` | `-10.049%` | `0.00 < 0.05` | LICENSE |

Throughput:

| Shape | Baseline | Pair load | Delta |
|---|---:|---:|---:|
| `dsv4_mini_hidden_1024x1024` | `92.402 Gelem/s` | `109.87 Gelem/s` | `+18.9%` |
| `dsv4_mini_moe_512x1024` | `58.723 Gelem/s` | `65.367 Gelem/s` | `+11.3%` |

## Results - Correctness

```text
test ops::tests::test_dsv4_fp4_batched_gemv_b1_raw ... ok
```

## Tradeoffs

- LOC complexity: +8 kernel lines; no new kernel or dispatch API.
- SM89 specificity: measured locally on RTX 4070 Ti SUPER / SM89; algorithm is
  not SM-specific.
- Shared memory budget: unchanged.
- Register budget: higher per thread due two decoded weights, two scale
  columns, and two input values.
- CUDA Graph compatibility: unchanged; launch shape and arguments are stable.
- Generality across batch sizes: B=1 raw path only; B>1 tiled FP4 path remains
  separate P3.8.
- Generality across shape: both hidden and MoE shapes pass the license gate.
- Numerical correctness margin: arithmetic grouping changes, so a B=1 direct
  FFI correctness test was added and passed with `0.01` tolerance.

## Problems

- This is a component bench only. DeepSeek V4 CUDA serving is not yet a
  request-level performance target, so no Guidellm A/B is available.
- `cargo test --release -p infer --features cuda test_dsv4_fp4_batched_gemv_b1_raw`
  without `--lib` currently compiles unrelated integration tests and fails on
  pre-existing API drift in `spec_decode_radix_pollution.rs` and
  `scheduler_kv_pressure_drains.rs`.
- The root-cause mechanism is still hypothesis-grade without an instruction
  profile. The license evidence is the controlled Criterion component A/B plus
  the direct B=1 correctness test.

## Learnings

- FP4 B=1 raw GEMV had a real duplicate packed-byte load pattern that FP8 did
  not have.
- Keeping B=1 raw cases separate from B>1 tiled cases is necessary; the
  original batch=4 bench would not have measured this kernel.

## Delta vs Baseline

Baseline:
`e85bd0d bench(cuda): add dsv4 fp4 batch b1 raw case`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| FP4 batch B1 1024x1024 latency point | `11.348 us` | `9.5436 us` | `-15.86%` |
| FP4 batch B1 512x1024 latency point | `8.9282 us` | `8.0207 us` | `-10.05%` |
