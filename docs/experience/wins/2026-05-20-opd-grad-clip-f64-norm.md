# OPD grad clip uses f64 norm accumulation

## Goal

Keep OPD gradient clipping numerically correct when gradients are finite but
large enough that `f32` square accumulation overflows.

## Context

`opd_step` clips student gradients after backward. The previous
`compute_global_norm` accumulated `sum(g * g)` in `f32`. A finite gradient such
as `[1e20, -1e20]` overflowed the squared sum to `inf`, made the clip scale
zero, and silently wrote `[0.0, -0.0]`.

## What Worked

`compute_global_norm` now accumulates the squared norm in `f64` and casts the
final square root back to `f32`. Normal clipping semantics stay the same, but
large finite gradients no longer get zeroed by norm overflow.

Failed-before evidence:

```text
test global_norm_large_finite_grads_do_not_overflow_to_zero ... FAILED
finite large gradients must not be zeroed by norm overflow: [0.0, -0.0]
```

After the fix, the same targeted test passes and the full train release suite
passes.

## Performance Cross-Check

This is not a performance win claim. The OPD moderate CPU profile was run only
as a regression check because `grad_clip` is on the OPD step path.

| metric | before | after |
|---|---:|---:|
| median steps/sec | 1.071313 | 1.070481 |
| total_step_seconds | 15.380135 | 15.422849 |
| grad_clip_seconds | 0.823372 | 0.831679 |

The profile remains noisy (`sigma_pct` about 14.5%), so this cross-check only
rules out an obvious local regression. The licensed result is the numerical
correctness fix.

## Verification

- `cargo fmt --check -p train`
- `cargo test -j 1 -p train global_norm_large_finite_grads_do_not_overflow_to_zero --release`
- `cargo clippy -j 1 -p train --all-targets --release -- -D warnings`
- `cargo test -j 1 -p train --release`
- `cargo check -j 1 --workspace`
- `cargo build -j 1 --workspace --release`
- `cargo run -j 1 -p train --example opd_step_cpu_moderate_profile --release`

## Rule

Gradient clipping must preserve finite large gradients as scaled finite values.
If the norm calculation overflows before the scale is computed, the clipper has
become a numerical bug rather than a safety guard.
