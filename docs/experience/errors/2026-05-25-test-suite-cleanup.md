# Test Suite Cleanup

## Context

T9 audited the unrelated test blockers called out by
`docs/experience/wins/2026-05-25-kv-tier-observability-code-patch.md`.
The commands ran with default features on a fresh target directory:

```bash
CARGO_TARGET_DIR=/tmp/arle-t9-target-20260525-codex cargo test -p infer 2>&1 | tee /tmp/infer-test-output.log
CARGO_TARGET_DIR=/tmp/arle-t9-target-20260525-codex cargo test -p train 2>&1 | tee /tmp/train-test-output.log
```

Default `infer` and `train` features do not enable CUDA, so this audit did
not touch the P5 GPU process.

## Classification

| Surface | Initial result | Classification | Fix |
| --- | --- | --- | --- |
| `infer/examples/qwen35_dense_module_dump.rs` | `cargo test -p infer` tried to compile it under default non-CUDA features and failed on `infer::model`. | Real bug: CUDA-only example was auto-discovered without `required-features = ["cuda"]`. | Add an explicit `[[example]]` gate in `infer/Cargo.toml`. |
| `infer/examples/qwen35_linear_attn_parity.rs` | Same default-feature build failure on `infer::model`. | Real bug: CUDA-only example was auto-discovered without a CUDA gate. | Add `required-features = ["cuda"]`. |
| `infer/examples/qwen35_linear_attn_substage_dump.rs` | Same default-feature build failure on `infer::model`. | Real bug: CUDA-only example was auto-discovered without a CUDA gate. | Add `required-features = ["cuda"]`. |
| `infer/examples/qwen35_prefill_path_probe.rs` | Latent default-feature build hazard because the whole file is `#![cfg(feature = "cuda")]`. | Real bug: CUDA-only example should be hidden from default `cargo test -p infer`. | Add `required-features = ["cuda"]`. |
| `infer/tests/metal_eval_audit.rs` | Static audit failed on `infer/src/backend/metal/kv_pool.rs` and stale materialize counts. | Real bug: deterministic classification drift; not Metal-hardware-specific and not flaky. | Update the audit table to current counts: `kv_pool.rs=1`, `mlx_qwen35_model.cpp=11`, `mlx.rs=15`, `ops.rs=6`, `request_state.rs=30`. |
| `cargo test -p train` | Passed. | No blocker. | No change. |

No env-specific failures were found, so no test was ignored. No flaky failure
was observed, so no flaky ignore entry was added.

## Root Cause

The default `infer` feature set intentionally avoids backend link dependencies,
but Cargo still auto-discovers examples unless each backend-specific example is
declared with `required-features`. Three Qwen3.5 diagnostics had no manifest
gate, and one `#![cfg(feature = "cuda")]` probe was a latent no-main hazard.

The Metal audit failure was the opposite problem: the audit test was doing its
job, but earlier Metal changes did not update the expected materialize-boundary
classification when the counts changed.

## Fix

- Gate CUDA-only Qwen3.5 diagnostic examples in `infer/Cargo.toml`.
- Refresh `metal_eval_audit`'s materialize-boundary table instead of ignoring
  the test, because the audit is a CPU-safe static guard.

## Verification

```bash
CARGO_TARGET_DIR=/tmp/arle-t9-target-20260525-codex cargo test -p infer --test metal_eval_audit
CARGO_TARGET_DIR=/tmp/arle-t9-target-20260525-codex cargo test -p infer
CARGO_TARGET_DIR=/tmp/arle-t9-target-20260525-codex cargo test -p train
```

- `cargo test -p infer --test metal_eval_audit`: 2 passed.
- `cargo test -p infer`: passed; lib suite reported 588 passed and 14 ignored,
  integration tests and doctest passed.
- `cargo test -p train`: passed; lib suite reported 85 passed, all integration
  tests passed, doc-tests had 0 tests.

## Rule

Backend-specific examples must have explicit `required-features` entries in the
owning crate manifest. Static audit tests should be refreshed with a
classification entry when they catch drift; do not hide deterministic audit
failures behind `#[ignore]`.
