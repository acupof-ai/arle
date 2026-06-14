# DiffusionGemma, HIP, And Vulkan Serve Guard Fixes

## Context

Review of the last two days' serve-path changes found several correctness
holes:

- Metal DiffusionGemma serve bypassed the unified resource guard and could keep
  running after Ctrl-C until its synchronous generation call returned.
- Diffusion generation used caller-provided `SamplingParams.max_new_tokens`
  even after Engine admission clamped the request budget.
- The C++ DiffusionGemma adaptive commit rule used a one-step previous-token
  check, while the Rust loop uses a rolling `stability_threshold` history.
- HIP/Vulkan serve paths accepted multi-slot scheduler configs even though the
  current executors are single-row.
- HIP DSv4 residency could keep IQ2/Q2 token embeddings on quantized tiers that
  the embedding kernel does not serve, and fused routed gate/up tensors could
  pass the load gate even though forward expects split gate and up weights.

## What Worked

- `infer-api` now routes Metal DiffusionGemma through a weight-only Metal
  resource plan before model load, applies MLX memory/cache/wired limits, and
  passes the serve shutdown flag into the diffusion executor.
- `infer-plan` and `infer-seam` now support cooperative cancellation in the
  block-diffusion loop and buffered executor.
- `infer-core` always normalizes the scheduler-admitted `max_tokens` back into
  `SamplingParams.max_new_tokens`, so diffusion executors cannot exceed the
  request's admitted total-token budget.
- The MLX C++ DiffusionGemma bridge checks cancellation around prefill, block,
  and denoise work, and uses a rolling stability history that matches the Rust
  host loop.
- HIP and Vulkan serve handles clamp scheduler slots to one; the CLI paged-KV
  capacity guard is skipped only for those non-paged-KV backends.
- HIP DSv4 load planning dequantizes IQ2/Q2 token embeddings to bf16 and rejects
  fused routed gate/up layouts with an explicit split-weight error.
- Vulkan feature clippy exposed a pre-existing checked-division lint in
  `infer-vulkan::config`; the derivation now uses `checked_div(...).unwrap_or(0)`
  with unchanged fallback semantics.

## Verification

```bash
git diff --check
cargo test -p infer-plan --release
cargo test -p infer-seam --release
cargo test -p infer-core --release
cargo test -p infer-hip --release --no-default-features
cargo test -p infer-metal --release --no-default-features --features metal
cargo test -p infer-vulkan --release --no-default-features
cargo test -p mlx-sys --release
cargo test -p infer-api --release --no-default-features --features cpu,no-cuda --lib
cargo test -p cli --release --no-default-features --features cpu,no-cuda
cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib
cargo check -p infer-api --release --no-default-features --features hip,no-cuda --lib
cargo check -p infer-api --release --no-default-features --features vulkan,no-cuda --lib
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
cargo clippy -p infer-plan -p infer-seam -p infer-core -p infer-hip -p mlx-sys --release -- -D warnings
cargo clippy -p cli -p infer-api --release --no-default-features --features cpu,no-cuda -- -D warnings
cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda --lib -- -D warnings
cargo clippy -p infer-api --release --no-default-features --features hip,no-cuda --lib -- -D warnings
cargo clippy -p infer-api --release --no-default-features --features vulkan,no-cuda --lib -- -D warnings
```

HIP checks on this Mac used the repository's stub/typecheck lane because ROCm
and `hipcc` are not installed. The bare CUDA/no-cuda check failed at
`cudarc`'s `nvcc --version` probe; the CI-equivalent
`CUDARC_CUDA_VERSION=12080` check passed.

No guidellm benchmark was run: this tranche fixes correctness and serve
admission/cancellation guardrails, does not flip a performance default, and the
required on-device Metal DiffusionGemma / HIP DSv4 smoke remains pending on
the target hardware.

## Rule

Serve-path special cases must fail closed at admission and load boundaries.
Diffusion models need their own budget, resource, and cancellation contracts;
backend-specific single-row executors must not inherit generic paged-KV
capacity or multi-slot assumptions.
