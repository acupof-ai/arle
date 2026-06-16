# A5 FP8 frozen-base QLoRA finite-diff gate

## Context

Path A is building ARLE autograd-native training capability for Qwen3.6-35B-A3B
FP8 LoRA. A1-A4 covered attention, TP all-reduce, checkpointing, and grouped MoE
backward. The next required substrate is a frozen FP8 base weight path for QLoRA:
the base weights stay non-trainable and resident in FP8, while LoRA adapters and
inputs remain differentiable in the autograd graph.

This tranche is deliberately the autograd substrate gate, not the full
Qwen3.6-35B FP8 safetensors loader. `qwen35_loader.rs` still rejects
`F8_E4M3`; real checkpoint loading is the next tranche.

## What Worked

- Added `DeviceHandle::CudaFp8BlockScaled` with FP8 E4M3 bytes plus block-scale
  metadata.
- Added CPU reference dequantization and CUDA upload/readback for block-scaled
  FP8 handles.
- Added CUDA f32 x FP8-block-scaled kernels for:
  - `matmul_bt`: `A[M,K] @ W[N,K]^T -> [M,N]`
  - input-gradient backward: `dY[M,N] @ W[N,K] -> dX[M,K]`
- Added `LinearWithLora::set_base_weight_to_fp8_block_scaled`, which installs the
  frozen FP8 base handle and forces `requires_grad=false`.
- Added `crates/train/tests/test_qlora_fp8.rs`, a synthetic QLoRA finite-diff
  gate that checks both LoRA-B gradient and input gradient.

The kernels are correctness-first scalar reductions. They license the frozen FP8
base autograd contract; they are not a performance claim.

## Verification

Local no-CUDA gates:

```text
cargo fmt --check
PASS

cargo test -p train --release --no-default-features --features no-cuda --test test_qlora_fp8 -- --nocapture
qlora_fp8_fd backend=cpu eps=1.0e-3 lora_b[3] analytic=9.931801818e-3 numeric=9.916721843e-3 rel_err=1.518e-3 input[5] analytic=-1.103788495e0 numeric=-1.103788614e0 rel_err=1.080e-7
PASS: 1 passed

cargo clippy -p autograd --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --test test_qlora_fp8 -- -D warnings
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --test test_qlora_fp8
PASS
```

Remote CUDA gate:

```text
host: .62 / iv-ye8is8fbi8s6iplibbg7
gpu: CUDA_VISIBLE_DEVICES=0 (GPU3 avoided)
source: /data01/arle-patha-qlora-fp8-2c20b3be
target: /data01/arle-target-patha-qlora-fp8
env:
  CUDA_HOME=/usr/local/cuda
  CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_KERNEL_SET=dsv4_flash
  ARLE_CUDA_DISABLE_FLASHMLA=1
  INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
  ARLE_CUDA_TEST_DEVICE=0

cargo test -p train --release --no-default-features --features cuda --test test_qlora_fp8 -- --nocapture

warning: cuda-kernels@0.2.1: TileLang AOT skipped for ARLE_CUDA_KERNEL_SET=dsv4_flash; linked CUDA_ERROR_NOT_SUPPORTED stubs for non-DSv4 TileLang FFI symbols.
qlora_fp8_fd backend=cpu eps=1.0e-3 lora_b[3] analytic=9.931801818e-3 numeric=9.916721843e-3 rel_err=1.518e-3 input[5] analytic=-1.103788495e0 numeric=-1.103788614e0 rel_err=1.080e-7
qlora_fp8_fd backend=cuda eps=1.0e-3 lora_b[3] analytic=9.931800887e-3 numeric=9.916721843e-3 rel_err=1.518e-3 input[5] analytic=-1.103788495e0 numeric=-1.103773713e0 rel_err=1.339e-5
PASS: 2 passed
EXIT:0
```

## Verdict

A5 is licensed at the autograd substrate level: frozen block-scaled FP8 base
weights can be installed under a LoRA linear layer, participate in CUDA forward
and input-gradient backward, and match central finite differences within the
1e-2 relative tolerance.

The remaining wall for the Path A end state is real Qwen3.6 FP8 checkpoint
loading into this handle, followed by model-level finite-diff and the 35B memory
gate.

## Rule

Separate the frozen-quantized-base autograd contract from checkpoint codec work.
Finite-diff the synthetic operator first, then wire real FP8 safetensors into the
same handle and re-run a model-level gate.
