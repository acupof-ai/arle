# OPD Checkpoint Save/Load KILL

## Context

Tranche P1 needed the OPD infer-teacher training harness to save a LoRA-only
student adapter so capability eval can compare the base 0.8B student against a
distilled checkpoint.

Change under test:

- `crates/train/examples/opd_step_cuda_infer_teacher_train.rs`
  - added `--save-student-checkpoint <DIR>`
  - added `--save-every <N_STEPS>`
- `crates/train/src/qwen35_checkpoint.rs`
  - added named checkpoint directory support for `final/`
  - kept existing `step_XXXXXX/` save contract

## Evidence

Save path works:

- 0.8B self-teach smoke wrote:
  - `step_000001/adapter_model.safetensors`
  - `step_000001/adapter_config.json`
  - `final/adapter_model.safetensors`
  - `final/adapter_config.json`
- 4B teacher -> 0.8B LoRA student smoke wrote the same checkpoint shape.
- Adapter safetensors validation saw 24 PEFT LoRA tensors for Qwen3.5-0.8B
  attention q/v adapters.

Verification:

```text
CARGO_BUILD_JOBS=1 cargo test -p train qwen35_checkpoint --release
15 passed

CARGO_BUILD_JOBS=1 cargo test -p train --release
75 lib tests + train integration tests passed

CARGO_BUILD_JOBS=1 cargo clippy -p train --all-targets -- -D warnings
passed
```

CUDA compile check:

```text
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo check -p train --example opd_step_cuda_infer_teacher_train --release --features cuda
Finished release profile
```

Load/eval path is blocked:

- `infer/src/backend/cuda/bootstrap.rs::load_qwen3_components` honors
  `INFER_LORA_PATH` and attaches a PEFT adapter.
- `infer/src/backend/cuda/bootstrap.rs::load_qwen35_components` directly calls
  `Qwen35Model::from_safetensors_with_options(...)` and has no LoRA attach hook.
- The saved adapter targets Qwen3.5 tensor names, so the current Qwen3
  adapter loader is not a valid fallback for Qwen3.5 serve.

## Root Cause

The training side can now produce a valid Qwen3.5 LoRA adapter, but the runtime
side cannot load that adapter for Qwen3.5 serve. Capability eval after
distillation therefore cannot run through the requested ARLE serve surface yet.

This is not a checkpoint serialization bug. It is a missing Qwen3.5 serve
adapter-loading path.

## Fix

Keep the save-path patch, but do not claim the P1 train -> save -> serve-load ->
eval loop is complete.

Next tranche should add Qwen3.5 PEFT adapter loading on the CUDA runtime side,
probably mirroring `infer/src/model/qwen3/lora.rs` while preserving Qwen3.5
hybrid-layer naming. The CLI/API surface can stay environment-based first
(`INFER_LORA_PATH`) to avoid touching HTTP request handling.

## Rule

A checkpoint writer is not an eval pipeline until the serving runtime can load
the artifact it writes. For OPD capability claims, gate on the whole chain:
train -> save -> load -> eval -> compare.
