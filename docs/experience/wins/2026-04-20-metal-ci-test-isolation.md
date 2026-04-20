# Metal CI Test Isolation

## Context

`Metal CI` on GitHub's Apple Silicon runner was still failing after the
`metal_serve` packaging cleanup. The failure moved around within the
`backend::metal::gdr` unit-test block: first the Qwen3.5 state-shape smoke
test, then the following `rms_normalize` case. That pattern pointed at
process-global MLX Metal state leaking across tests rather than one
deterministic bad assertion.

## What Worked

- Upgraded `metal_test_guard()` in `infer/src/lib.rs` from a raw mutex guard
  to a dedicated RAII guard that clears MLX's Metal cache at each test
  boundary while still serializing Metal tests.
- Made `test_gdr_state_shapes_qwen35` in `infer/src/backend/metal/gdr.rs`
  truly host-side. The small-shape test still covers `MetalRecurrentState`
  allocation; the Qwen3.5 case now pins only the dimension contract without
  forcing a large device allocation on the default CI runner.
- Kept `metal-ci.yml` aligned with release truth by watching the Metal-facing
  support/release docs (`CHANGELOG.md`, `docs/compatibility.md`,
  `docs/stability-policy.md`, `docs/support-matrix.md`) in addition to
  `docs/release-checklist.md` and `release.yml`.

## Rule

When Metal unit tests touch MLX state, serialize them and clear process-global
Metal caches at test boundaries. Large-dimension "shape" tests should stay
host-side unless the goal is explicitly to validate a device allocation path.
