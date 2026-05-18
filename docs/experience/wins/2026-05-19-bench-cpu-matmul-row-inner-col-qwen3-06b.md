# CPU matmul row-inner-col loop order at Qwen3-0.6B shapes - AMD Ryzen 7 3700X, 2026-05-19

## Goal

- **(optimization)** Replace the CPU reference matmul loop order used by
  `autograd::backend::cpu_matmul_forward` from column-strided
  `row -> col -> inner` to contiguous `row -> inner -> col`, then measure the
  same Qwen3-0.6B OPD shape catalog as the naive baseline.

## Hypothesis

- The baseline proved the naive loop is memory-layout bound: each inner FMA
  walks B with a stride of `n` on row-major storage. Reordering to
  `row -> inner -> col` keeps each B row and output row contiguous while
  preserving the per-output inner accumulation order. Expected result:
  >=10x speedup on at least 6 of 8 shapes, with sigma/mean <5%.

## Command

```bash
cargo run -p autograd --example cpu_matmul_microbench --release \
  | tee bench-output/2026-05-19-cpu-matmul-row-inner-col-qwen3-06b/cpu_matmul_row_inner_col.txt
```

Benchmark constants:

```text
backend = cpu_matmul_forward
warmup_runs = 1
measured_runs = 5
target_fmas_per_run = 1000000000
seed_a = 0x00A1_10C5
seed_b = 0x00B7_70C5
```

## Environment

| Item | Value |
|---|---|
| Backend | CPU `cpu_matmul_forward` |
| CPU | AMD Ryzen 7 3700X 8-Core Processor, 8C/16T, Zen 2 |
| OS / arch | Linux x86_64 |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| Runtime substrate commit | `8e8effd` plus this row-inner-col patch |
| Feature set | `cargo run -p autograd --example cpu_matmul_microbench --release` |
| Non-default flags | none |

## Results

### Treatment run

| shape | FMAs | median (s) | mean (s) | sigma/mean | GFLOPs/s | iters/run | x/fwd | fwd cost (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_proj    [4,1024] @ [1024,2048]` | 8.389e6 | 0.000790 | 0.000791 | 0.462% | 21.224 | 119 | 28 | 0.022134 |
| `k_proj    [4,1024] @ [1024,1024]` | 4.194e6 | 0.000409 | 0.000409 | 0.157% | 20.488 | 238 | 28 | 0.011464 |
| `v_proj    [4,1024] @ [1024,1024]` | 4.194e6 | 0.000407 | 0.000408 | 0.302% | 20.587 | 238 | 28 | 0.011409 |
| `o_proj    [4,2048] @ [2048,1024]` | 8.389e6 | 0.000818 | 0.000818 | 0.127% | 20.512 | 119 | 28 | 0.022902 |
| `gate_proj [4,1024] @ [1024,3072]` | 1.258e7 | 0.001514 | 0.001516 | 1.439% | 16.627 | 79 | 28 | 0.042380 |
| `up_proj   [4,1024] @ [1024,3072]` | 1.258e7 | 0.001831 | 0.001845 | 1.778% | 13.745 | 79 | 28 | 0.051265 |
| `down_proj [4,3072] @ [3072,1024]` | 1.258e7 | 0.001479 | 0.001480 | 0.752% | 17.018 | 79 | 28 | 0.041406 |
| `lm_head   [4,1024] @ [1024,151936]` | 6.223e8 | 0.145455 | 0.146767 | 2.162% | 8.557 | 1 | 1 | 0.145455 |

All treatment shapes are inside the sigma <5% bar.

### Delta vs naive baseline

Baseline: [`2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md`](2026-05-19-bench-cpu-matmul-naive-qwen3-06b.md).

| shape | baseline median (s) | treatment median (s) | speedup |
|---|---:|---:|---:|
| `q_proj` | 0.042127 | 0.000790 | **53.3x** |
| `k_proj` | 0.021061 | 0.000409 | **51.5x** |
| `v_proj` | 0.020998 | 0.000407 | **51.6x** |
| `o_proj` | 0.043031 | 0.000818 | **52.6x** |
| `gate_proj` | 0.060108 | 0.001514 | **39.7x** |
| `up_proj` | 0.060163 | 0.001831 | **32.9x** |
| `down_proj` | 0.073041 | 0.001479 | **49.4x** |
| `lm_head` | 0.881362 | 0.145455 | **6.1x** |

| aggregate | baseline | treatment | speedup |
|---|---:|---:|---:|
| matmul cost per single Qwen3-0.6B forward | 9.856 s | 0.348 s | **28.3x** |
| matmul cost per OPD step, approx 3 forwards | 29.569 s | 1.045 s | **28.3x** |

The license criterion from the baseline entry was >=10x on at least 6 of 8
shapes. This passes on 7 of 8 shapes; only `lm_head` is below 10x, but still
improves by 6.1x and the aggregate forward estimate improves by 28.3x.

## Problems

- This is still a scalar CPU kernel, not a BLAS replacement. `lm_head` remains
  only 8.6 GFLOPs/s because the output row is 151,936 floats and repeatedly
  streams a large row; a blocked/SIMD or matrixmultiply-backed kernel remains
  open.
- This bench measures forward matmul only. `cpu_matmul_backward` calls
  `cpu_matmul_forward` twice after host transposes, so it should inherit the
  faster kernel, but backward-specific wall-clock still needs a separate
  B3 follow-up.
- The naive baseline entry did not print per-shape sigma. The treatment
  harness now repeats enough work per measured run to report sigma/mean; the
  observed speedups are large enough that this does not change the license
  decision, but future A/Bs should use the updated harness on both sides.

## Learnings

- For small-M transformer matmuls, loop order alone recovers most of the
  missing CPU locality. We do not need a dependency or threading to get the
  first 28x aggregate forward improvement.
- The OPD CPU path is still matmul-dominated at production shape. The next
  useful CPU performance axis is either backward-specific measurement or a
  true blocked/SIMD kernel for the remaining `lm_head` gap.
- The optimized loop preserves per-output inner accumulation order, and the
  new slow-reference test keeps CPU forward correctness pinned independently
  of the optimized production function.

## Delta vs baseline

- Median per-shape speedups: 6.1x to 53.3x.
- Aggregate projected Qwen3-0.6B forward matmul cost: 9.856 s -> 0.348 s
  (-96.5%).
- Aggregate projected OPD three-forward matmul cost: 29.569 s -> 1.045 s
  (-96.5%).

## Artefacts

- Bench source: `crates/autograd/examples/cpu_matmul_microbench.rs`
- Raw: `bench-output/2026-05-19-cpu-matmul-row-inner-col-qwen3-06b/cpu_matmul_row_inner_col.txt`
- Raw sha256:
  `6519fd74db8cd2d6275472ba6f006dc132043a5ace2c405960a4822f6db0f161`
