# M5 — Unified Model Architecture Contract

## Context

M5 used the P0 survey retreat from shared `ModelForward` execution to a
backend-neutral architecture contract. `infer/src/model.rs` is CUDA-gated and
CUDA `ModelForward` is paged-KV shaped, so the shared surface now lives in
always-available `infer/src/model_arch.rs` as `ModelArchInfo` plus
`ModelArchSummary`.

This builds on M1 telemetry, M2 KV-tier adapter, M3 scheduler IR, and M4 op
backend convergence: both CUDA and Metal now publish the same model shape into
`EngineTelemetry` without leaking CUDA or MLX types into cross-backend modules.

## What Worked

- `d6d00be feat(model): add backend-neutral architecture contract`
  introduced `ModelArchInfo` / `ModelArchSummary` and stable JSON
  serialization using `ModelArch::display_name()`.
- `cbd0e19 refactor(model): move CUDA architecture shape to ModelArchInfo`
  made CUDA `ModelForward` inherit the shape contract while keeping existing
  scheduler callsites valid through the super-trait.
- `c9a05f8 feat(metal): expose model architecture summary` implemented Metal
  shape reporting for Qwen3, dense Qwen3.5, and Qwen3.5-MoE, preserving
  full-attention-only KV layer accounting for hybrid models.
- `73e3210 feat(metrics): publish backend-neutral model architecture` wired
  `EngineTelemetry.model_arch: Option<ModelArchSummary>` and
  `/v1/stats?format=json` `engine_model_arch` from both backend load paths.

Code delta for the implementation slice: 12 runtime files, +481 / -94 lines.

## Verification

- `cargo fmt --all --check`
- `cargo check --release -p infer --no-default-features --features no-cuda`
- `cargo check --release -p infer --no-default-features --features cuda,no-cuda`
- `cargo check --release -p infer --no-default-features --features metal,no-cuda`
- `cargo check --release -p infer --no-default-features --features cuda,metal,no-cuda`
- `NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo check --release -p infer --features cuda`
- `cargo clippy --release -p infer --no-default-features --features no-cuda -- -D warnings`
- `cargo clippy --release -p infer --no-default-features --features metal,no-cuda -- -D warnings`
- `cargo clippy --release -p infer --no-default-features --features cuda,no-cuda -- -D warnings`
- `NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo clippy --release -p infer --features cuda -- -D warnings`
- `NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 INFER_TEST_MODEL_PATH=infer/models/Qwen3-4B cargo test --release -p infer --features cuda --test greedy_consistency -- --test-threads=1`
- `NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 INFER_TEST_MODEL_PATH=infer/models/Qwen3-4B cargo test --release -p infer --features cuda --test e2e -- --test-threads=1`

Linux Metal note: `metal,no-cuda` check/clippy passed. Metal `cargo test`
linking remains unavailable on this Linux runner because `mlx-sys` skips the
MLX/Metal bridge off macOS, leaving MLX FFI symbols undefined. Pure config
summary tests were run under `no-cuda`.

## Bench Status

GuideLLM canonical bench is `pending-M4.5`. It remains blocked by the regular
decode KV preemption gap tracked in
`docs/experience/errors/2026-05-07-m4-guidellm-canonical-stuck.md`, so this
milestone records correctness and telemetry verification only.

## Rule

- Put shared architecture facts in always-available pure Rust modules; do not
  hang backend-neutral traits off CUDA-gated `model.rs`.
- Keep execution traits and architecture metadata separate. `ModelForward`
  can inherit `ModelArchInfo`, but Metal should not be forced into CUDA
  forward or paged-KV execution shapes.
- Telemetry additions should use `Option<T>` for legacy/mock tolerance, then
  wire both backend load paths before claiming cross-backend convergence.
