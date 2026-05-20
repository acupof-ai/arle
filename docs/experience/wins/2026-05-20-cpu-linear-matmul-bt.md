# CPU linear matmul_bt eliminates per-call weight transpose

## Goal

Remove the physical `transpose(weight)` from train-side Qwen3.5
`linear_forward`. The old path materialized `weight^T` before every matmul;
at Qwen3-0.6B shapes that copied every projection weight, including a
608 MiB-class `lm_head`, on each forward.

## Hypothesis

Computing `A @ weight^T` directly with a strided-B SGEMM should preserve the
same math while avoiding the copy. The win must hold across all Qwen3-0.6B
linear shapes, not only `lm_head`, because `linear_forward` is shared.

## Params

- Backend: CPU
- Shapes: Qwen3-0.6B linear catalogue at `M=4`
- Runs: 5 measured, 1 warmup
- Command:

```bash
cargo run -p autograd --example cpu_linear_transpose_bt_ab --release \
  | tee bench-output/2026-05-20-cpu-linear-transpose-bt-ab/run.txt
```

## Results

```text
shape       m     k       n    current_s     direct_s    speedup    cur_sigma     bt_sigma       diff  cur_GF/s      bt_GF/s
q_proj      4  1024    2048     0.016751     0.000938     17.859        0.278        1.242   1.001e-5     1.002       17.887
k_proj      4  1024    1024     0.008141     0.000463     17.594        0.311        0.384   1.001e-5     1.030       18.128
v_proj      4  1024    1024     0.008121     0.000466     17.441        0.282        0.378   1.001e-5     1.033       18.017
o_proj      4  2048    1024     0.016520     0.000940     17.568        0.400        0.247   2.098e-5     1.016       17.841
gate_proj   4  1024    3072     0.027044     0.001471     18.383        0.546        0.500   1.121e-5     0.931       17.106
up_proj     4  1024    3072     0.026395     0.001491     17.708        0.358        0.735   1.121e-5     0.953       16.883
down_proj   4  3072    1024     0.027082     0.001452     18.655        0.612        0.799   2.670e-5     0.929       17.335
lm_head     4  1024  151936     0.525850     0.084717      6.207        0.496        0.684    0.000e0     2.367       14.692
```

Every measured linear shape is faster with direct `A @ weight^T`. The
projection shapes improve 17-19x because the old path was dominated by
physical transpose copies. `lm_head` improves 6.2x while preserving exact
output for that sampled data.

## Implementation

- Added `autograd::ops::matmul_bt` for rank-2 `A:[M,K]`, `B:[N,K]`.
- Added CPU forward/backward helpers:
  - `cpu_matmul_bt_forward`
  - `cpu_matmul_bt_backward`
- Switched `crates/train/src/qwen35.rs::linear_forward` from
  `transpose(weight) + matmul` to `matmul_bt(flat_x, weight)`.

## Verification

- `cargo run -p autograd --example cpu_linear_transpose_bt_ab --release`
- `cargo fmt --check -p autograd -p train`
- `cargo test -p autograd --test test_backend --release`
- `cargo test -p autograd --test m1_ops --release matmul_bt_grad_matches_numeric -- --nocapture`
- `cargo test -p autograd --release`
- `cargo test -p train --test test_qwen35_forward --release`
- `cargo test -p train --test test_opd_step --release`
- `cargo test -p train --test test_opd_determinism --release`
- `cargo test -p train --test test_opd_grad_check --release -- --nocapture`
- `cargo test -p train --release`
- `cargo check --workspace`
- `cargo clippy -p autograd --all-targets --release -- -D warnings`
- `cargo clippy -p train --all-targets --release -- -D warnings`
- `cargo build --workspace --release`

## Problems

The new op is intentionally rank-2 only because Qwen3.5 `linear_forward`
already flattens rank-3 inputs before projection. Extending `matmul_bt` to
batched rank-3 should be a separate change if another caller needs it.

## Learnings

For train-side CPU OPD, the transpose copy was larger than the GEMM on every
projection shape. Optimizing matmul kernels while repeatedly materializing
`weight^T` leaves most of the linear path on the floor; the correct primitive
is an explicit `A @ B^T` op.
