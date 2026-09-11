# Metal on-device backward matmul step speedup — Metal, 2026-09-12

> Status: Shipped (pending merge)

## Goal

Wall-clock per Metal training micro-step on the forward-matmul + sum +
backward loop (`crates/autograd/examples/bench_step_matmul.rs`).

## Hypothesis

The four backward matmul entry points (`matmul_bt`,
`matmul_backward_device`, `matmul_bt_backward_device`,
`matmul_bt_input_grad_device`) had no Metal override, so they fell through
to the `Backend` trait-default host path: read back A, B, and the upstream
gradient across the Metal boundary, run the SGEMMs on CPU, and re-upload the
gradients. Lazy on-device `mlx_transpose(_axes)` + `mlx_matmul` overrides
that return unevaluated nodes should remove those per-step copies; the saved
bytes scale as d², so the gain should grow with `d` and wash out at tiny d.

## Parameters

```bash
# baseline A: readback-realize fix only (main via #366)
target/release/examples/bench_step_matmul --backend metal --d <d> --batch 4 --iters 100
# treatment B: A + the four on-device overrides
target/release/examples/bench_step_matmul --backend metal --d <d> --batch 4 --iters 100
```

- Baseline: commit `29065b17f` (A, merged as #366).
- Treatment: A plus the four Metal matmul overrides.
- Shapes: d ∈ {128, 1024, 2048}, batch 4; x @ w + sum + backward per step.
- Trials: 2 interleaved A/B repetitions per d (A then B, same session,
  release profile). Numbers are medians of the 2 reps.

## Environment

- Host: Apple Silicon (Darwin 25.3.0), local Metal.
- Binary: `--release` (thin LTO, cu=1), `--no-default-features --features
  metal`.
- No model: deterministic fixture data inside the example.

## Results

| d | A step_ms (host fallback) | B step_ms (on-device) | ratio B/A |
|---:|---:|---:|---:|
| 128 | 0.322 | 0.282 | 1.14× (wash; sub-millisecond, noisy) |
| 1024 | 2.385 | 0.921 | 2.59× |
| 2048 | 10.895 | 3.626 | 3.00× |

| d | A steps/s | B steps/s |
|---:|---:|---:|
| 128 | 3109 | 4412 |
| 1024 | 420 | 1086 |
| 2048 | 93 | 276 |

At d=128 the loop is dominated by per-step Python-free scalar overhead and
the copies are kilobytes; the 14% gap is within run-to-run noise of the
0.2–0.4 ms range (a wash). From d=1024 up the saved readback/upload traffic is
multi-megabytes per step and the speedup is 2.6–3.0×.

## Correctness

The speedup carries no parity risk beyond the new code itself, covered by
`crates/autograd/tests/test_metal_device_matmul.rs` (6 Metal-vs-CPU tests):
all four methods against the CPU reference, the rank-3 batched
transpose-axes path, and the `need_grad_a`/`need_grad_b` short-circuit. All
pass at rel tolerance 1e-3, matching the other Metal parity tests.

## Rule

A GPU backend that overrides forward matmul but leaves the `*_device`
backward methods on the host-fallback default pays the host round-trip per
training step; the cost is invisible on tiny shapes and multiples on
production shapes. Verify on-device dispatch at a production-size shape, not
the unit-test size.
