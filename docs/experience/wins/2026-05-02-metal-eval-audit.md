# Metal Materialize Boundary Audit

## Context

P1-3 of `docs/plans/2026-05-02-omlx-inspired-optimizations.md` calls for a
Metal lazy-graph audit before wiring deeper SSD KV persistence or MLX upgrades.
The risk is accidental per-token scalarization through `eval()`, `async_eval()`,
or `.item()` in scheduler hot loops.

## What Worked

- `infer/src/backend/metal/runtime.rs`,
  `infer/src/backend/metal/scheduler.rs`, and `infer/src/backend/metal.rs` have
  zero direct non-comment materialize calls.
- Existing production materialize boundaries are concentrated in wrapper,
  load-time, model-step, DFlash, GDR, request-state, and sampling code:

| Path | Count | Classification |
| --- | ---: | --- |
| `crates/mlx-sys/src/lib.rs` | 2 | Green: FFI declarations |
| `crates/mlx-sys/src/mlx_bridge.cpp` | 4 | Green: bridge wrappers |
| `crates/mlx-sys/src/mlx_qwen35_model.cpp` | 10 | Yellow: C++ Qwen3.5 step/session boundaries |
| `infer/src/backend/metal/dflash.rs` | 6 | Yellow: DFlash batched staging and sampling boundaries |
| `infer/src/backend/metal/gdr.rs` | 10 | Yellow: GDR replay/validation boundaries |
| `infer/src/backend/metal/generate.rs` | 2 | Yellow: legacy Qwen3 generation async staging |
| `infer/src/backend/metal/loader.rs` | 2 | Green: load-time tensor materialization |
| `infer/src/backend/metal/mlx.rs` | 16 | Green/yellow: wrapper plus mask/tripwire materialization |
| `infer/src/backend/metal/ops.rs` | 2 | Green: async wrapper |
| `infer/src/backend/metal/qwen35.rs` | 10 | Yellow: Qwen3.5 model-step and load-time boundaries |
| `infer/src/backend/metal/request_state.rs` | 27 | Yellow: request-state prefill/decode/sampling boundaries |
| `infer/src/backend/metal/weights.rs` | 1 | Green: load-time merged-weight materialization |

- Added `infer/tests/metal_eval_audit.rs` so new non-comment materialize
  boundaries fail tests until this classification is updated.

## Rule

Keep direct materialization out of `runtime.rs` and `scheduler.rs`. Any new
`eval()`, `async_eval()`, or `.item()` under Metal production code must be
classified as setup/test, request/chunk boundary, or hot-loop red flag in the
audit entry before it lands.
