# OPD grad clip keeps f64 scale for huge finite norms

## Goal

Finish the OPD gradient-clipping numerical fix for the case where the true
finite gradient norm is larger than `f32::MAX`.

## Context

`2026-05-20-opd-grad-clip-f64-norm.md` moved squared-norm accumulation to
`f64`, which fixed finite large gradients such as `[1e20, -1e20]`. A second
edge remained: `compute_global_norm` cast the final norm back to `f32` before
`clip_grad_norm` computed the scale. For gradients `[f32::MAX, -f32::MAX]`,
the true norm is finite in `f64` but larger than `f32::MAX`, so the cast became
`inf` and the scale was still zero.

## What Worked

`clip_grad_norm` now keeps the internal norm and scale calculation in `f64`.
The public `GradClip::clip` logging surface still returns `f32`, preserving the
trait contract, but the actual gradient mutation no longer depends on a lossy
pre-scale cast.

Failed-before evidence:

```text
test global_norm_above_f32_max_still_scales_to_finite_grads ... FAILED
gradients with finite true scale must not be zeroed: [0.0, -0.0]
```

After the fix:

- `[1e20, -1e20]` still clips to a finite nonzero norm near `1e20`.
- `[f32::MAX, -f32::MAX]` clips to finite nonzero gradients with norm near
  `1e38`.

## Performance Cross-Check

This is not a performance win claim. The OPD moderate CPU profile was rerun as
a regression check because `grad_clip` is on the OPD step path.

| metric | prior f64-accum fix | f64 scale fix |
|---|---:|---:|
| median steps/sec | 1.070481 | 1.077134 |
| total_step_seconds | 15.422849 | 15.283996 |
| grad_clip_seconds | 0.831679 | 0.828545 |

The harness remains noisy (`sigma_pct` about 14.5%), so the licensed result is
only the numerical correctness fix.

## Verification

- `cargo fmt --check -p train`
- `cargo test -j 1 -p train global_norm_above_f32_max_still_scales_to_finite_grads --release`
- `cargo test -j 1 -p train global_norm_large_finite_grads_do_not_overflow_to_zero --release`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo test -j 1 -p train --release`
- `cargo check -j 1 --workspace`
- `cargo build -j 1 --workspace --release`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`

## Rule

For safety code like gradient clipping, widening only the accumulation is not
enough if the scale calculation then narrows through `f32::MAX`. Keep the
normalization scalar in the widest available type until the final per-element
multiply.
