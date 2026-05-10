# P2.5 — W4A8 Marlin Auto-Config Substrate

## Context

P2.5/M'' targets the QQQ `thread_config_t` schedule-selection pattern for
ARLE's W4A8 Marlin GEMM. The source audit found a tradeoff: QQQ has a cleaner
runtime selector, while ARLE already has sm_89-specific L2 cache-hint
`cp.async` and broader compile-time config coverage.

## What Worked

- Added a QQQ-style `thread_config_t` selector to
  `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`.
- Preserved ARLE's `cp_async4_stream` / `cp_async1_stream` cache-hint paths.
- Preserved the historical default by having Rust pass explicit legacy
  `(thread_k, thread_n)` values unless `INFER_MARLIN_W4A8_AUTOCONFIG=1`.
- Added `INFER_MARLIN_W4A8_AUTOCONFIG=1` as an opt-in A/B switch for the
  QQQ-style selector.

## Correctness

Both arms passed the W4A8-vs-BF16 accuracy gate with the qzeros-fixed W4A8
fixture:

| Arm | Env | Result |
|---|---|---|
| Legacy explicit config | unset | 32/32 token match, 0.0% diff |
| Auto-config selector | `INFER_MARLIN_W4A8_AUTOCONFIG=1` | 32/32 token match, 0.0% diff |

## Bench Status

Status: `pending-bench`.

This commit is a substrate + A/B switch. Production default remains the legacy
explicit tile choice, so no performance behavior changes unless the opt-in env
var is set. License still requires a matched-control W4A8 bench:

- baseline: env unset
- treatment: `INFER_MARLIN_W4A8_AUTOCONFIG=1`
- shapes: W4A8 sustained conc=1 and conc=4
- gate: neutral-or-better within +/-5%, soft win at >=2% latency improvement

## Rule

When upstream schedule selection is a tradeoff, land it behind an explicit
A/B switch and keep the local architecture-specific fast path as the default
until bench data proves the new selector.
