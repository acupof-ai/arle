# CPU matmul M=1 dispatch routes rollout-style lm_head to saxpy

## Goal

Fix the single-row `lm_head` matmul regime exposed by the killed
`forward_last_logits` experiment. The existing OPD CPU matmul dispatch used
`matrixmultiply` for wide `N`, which is correct for `M=4` full-sequence
`lm_head` but inefficient for `M=1`.

## Hypothesis

For `M=1, K=1024, N=151936`, `matrixmultiply::sgemm` pays packing overhead
that cannot amortize across multiple output rows. The simple saxpy row-major
loop should be faster. For `M=4` at the same `K,N`, matrixmultiply should stay
faster, so the dispatch must key on `M` as well as `N`.

## Params

- Backend: CPU `cpu_matmul_forward`
- CPU: AMD Ryzen 7 3700X
- Shapes:
  - `M=1, K=1024, N=151936`
  - `M=4, K=1024, N=151936`
- Runs: 5 measured, 1 warmup
- Command:

```bash
cargo run -p autograd --example cpu_matmul_m1_dispatch_ab --release \
  | tee bench-output/2026-05-20-cpu-matmul-m1-dispatch-ab/run.txt
```

## Results

```text
shape         m          route   gflops/s     median_s       mean_s  sigma_pct max_abs_diff
lm_head_m1    1        current      8.569     0.036313     0.036692      2.024   0.000000e0
lm_head_m1    1          saxpy      8.567     0.036321     0.036330      0.089   0.000000e0
lm_head_m1    1 matrixmultiply      4.178     0.074484     0.074486      0.074  1.716614e-5
lm_head_m4    4        current     16.529     0.075301     0.075669      0.749  2.193451e-5
lm_head_m4    4          saxpy      8.652     0.143863     0.144805      1.251   0.000000e0
lm_head_m4    4 matrixmultiply     16.513     0.075377     0.075355      0.176  2.193451e-5
```

`M=1` current route now matches saxpy and is **2.05x faster** than
matrixmultiply by median wall-clock (`0.074484 / 0.036313`). `M=4` current
route still matches matrixmultiply and keeps the previous `lm_head` full-row
path intact.

## Verification

- `cargo run -p autograd --example cpu_matmul_m1_dispatch_ab --release`
- `cargo fmt --check -p autograd`
- `cargo test -p autograd --test test_backend --release`
- `cargo test -p autograd --release`
- `cargo test -p train --test test_opd_determinism --release`
- `cargo test -p train --test test_opd_grad_check --release -- --nocapture`
- `cargo test -p train --release`
- `cargo check --workspace`
- `cargo clippy -p autograd --all-targets --release -- -D warnings`
- `cargo clippy -p train --all-targets --release -- -D warnings`
- `cargo build --workspace --release`

## Problems

This is a kernel route win, not an end-to-end OPD step win by itself. The
current OPD rollout path still computes full-sequence logits. Re-licensing a
last-row rollout path must be a separate single-variable A/B on top of this
M-aware dispatch.

## Learnings

Dispatch for OPD CPU matmul must consider both `M` and `N`. Reducing FLOPs by
slicing to one row changes the kernel regime; without the M-aware route, the
lower-FLOP path can still lose wall-clock.
