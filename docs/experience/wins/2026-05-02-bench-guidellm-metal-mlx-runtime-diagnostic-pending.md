# Metal MLX Runtime Diagnostic Pending Bench

## Context

P0-2 in `docs/plans/2026-05-02-omlx-inspired-optimizations.md` asked for an
MLX 0.32+ bump to unlock Apple M5 Neural Accelerator prefill speedups. The
current upstream release/tag check showed no 0.32+ line to vendor; M5 support
landed before the vendored 0.31.1 snapshot. The useful code change is therefore
a runtime diagnostic, not a fake version bump.

## What Worked

- Exposed `mlx::core::version()` through the `mlx-sys` bridge.
- Exposed MLX's actual `metal::is_nax_available()` result so startup logs do
  not claim M5 acceleration when the SDK/Metal build omitted NAX kernels.
- Metal startup now logs MLX version, macOS version, Apple chip string, and
  whether the linked MLX runtime reports NAX availability.
- Added parser tests for release/dev MLX versions and the macOS 26.2 gate.

## Benchmark

Status: `pending-remote`

No M5 guidellm run was captured locally. Run
`scripts/bench_guidellm.sh metal-mlx-runtime-diagnostic-m5` on M5 hardware with
macOS 26.2+ and compare against the existing Metal 0.31.1 baseline. On M3/M4,
record a no-regression smoke because this tranche only adds load-time logging.

## Rule

Do not bump vendored MLX to an imaginary version. Version-gated accelerator
claims must be tied to an upstream MLX tag and a startup diagnostic that appears
in bench logs.
