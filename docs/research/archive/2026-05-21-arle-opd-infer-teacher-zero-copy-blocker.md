# ARLE OPD Infer-Teacher Zero-Copy Blocker

## Context

Phase 2 of the Qwen3.5 large-to-small OPD plan asked for a train-side
`InferTeacher` that wraps the production `infer` runtime and returns
device-resident logits into the OPD KL path without a host copy.

Commit `9cd072d` landed the train-side `TeacherForward` trait and the
existing in-process `Qwen35Model` implementation. The next requested step was
an `InferTeacher` stub that type-checks against the `infer` engine surface.

## Evidence

- `infer` does not currently expose a public `infer::Engine` type. The public
  runtime entry is `infer::server_engine::LoadedInferenceEngine`, which
  implements `InferenceEngine`.
- `InferenceEngine` exposes `complete`, `complete_stream`, `tokenize`, and
  `telemetry`. It does not expose a token-id forward path or raw logits.
- Train OPD tensors live in `autograd::TensorStore`, whose CUDA backend owns a
  private `autograd::backend_cuda::CudaBackend` with a private
  `cudarc::driver::CudaContext`, default stream, cuBLAS handle, and
  `DeviceHandle::Cuda(CudaStorage)`.
- Infer Qwen3.5 runtime tensors live under `cuda_kernels::prelude::DeviceContext`
  and `DeviceVec`, with its own CUDA context, compute/copy/comm streams, and
  BF16 `CudaSlice` buffers.
- The two CUDA residency types are not connected by a public trait or adapter:
  `cuda_kernels::DeviceVec` cannot currently be installed into
  `autograd::TensorStore` as a `DeviceHandle::Cuda` without either copying or
  adding a new shared-handle ABI.

## Root Cause

The requested zero-copy bridge is blocked by two missing public contracts:

1. `infer` needs a raw token-id forward/logits API below text completion.
2. `autograd` and `cuda-kernels` need a shared CUDA allocation/stream contract
   or an explicit D2D import path that can safely move infer logits into a
   train `TensorStore`.

This is not a performance tuning issue; the required type boundary is absent.

## Deferred Fix

Recommended next architecture tranche:

1. Add an `infer` raw teacher API that accepts token ids + positions and
   returns a device logits handle plus shape metadata.
2. Define a shared CUDA buffer ABI between `cuda-kernels::DeviceVec` and
   `autograd::DeviceHandle::Cuda`, or add a deliberate D2D copy bridge with a
   wall-clock budget.
3. Re-run Phase 2 commit 3 only after the bridge exists:
   Qwen3-0.6B self-teach step <= 0.4 s and max relative loss error <= 1e-4
   versus `InProcessTeacher`.

## Rule

Do not claim `infer` teacher integration from the text-completion engine
surface. OPD needs raw logits on the train backend; if the only available
surface is text/token completion, stop and add the missing runtime contract
first.
