# ARLE Autograd BF16 D2D Bridge

## Goal

Create the Path B bridge that can import infer-owned BF16 device logits into
autograd as an f32 CUDA `DeviceHandle`, without host materialization.

## Hypothesis

A synchronous device-to-device byte copy into autograd-owned staging, followed
by a tiny BF16-bits-to-f32 kernel, is enough for the first `InferTeacher`
wire-up.

## Params

- Source tensor: 5 BF16 values with fixed raw bit patterns
- Backend: autograd `CudaBackend`
- GPU: RTX 4070 Ti SUPER
- Env: `NVCC_CCBIN=/usr/bin/g++-14`, `CUDARC_CUDA_VERSION=13010`,
  `TORCH_CUDA_ARCH_LIST=8.9`

## Results

- `cargo test -p autograd --lib backend_cuda::tests::bf16_device_import_roundtrip_preserves_d2d_bytes_and_widens --release --features cuda`: pass
- `cargo check -p autograd --release --features cuda`: pass
- `cargo check -p autograd --no-default-features --features cuda,no-cuda`: pass
- D2D staging preserves the source BF16 bytes exactly.
- f32 import readback matches exact BF16-to-f32 widening.

## Problems

- This validates same-process CUDA pointer import. Cross-runtime OPD
  correctness and wall-clock are licensed in the `InferTeacher` commits.

## Learnings

The bridge can stay backend-generic at the autograd trait boundary: callers
only need a source device pointer, length, and shape.
