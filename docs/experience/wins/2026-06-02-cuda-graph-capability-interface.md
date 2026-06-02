# CUDA Graph decode capability is model-reported

## Context

DSv4 performance investigation showed that CUDA Graph coverage is a first-order
gap versus the SGLang DSv4 path. ARLE already had CUDA Graph decode support for
Qwen3 and partial/piecewise support for Qwen3.5, but the scheduler only saw a
boolean. DSv4 returned `false`, so startup logs could not distinguish "runtime
flag disabled" from "model path is not graph-safe yet".

## What Worked

- Added `CudaGraphDecodeSupport` and `CudaGraphDecodeMode` to the generic
  `ModelForward` contract.
- Kept the existing `supports_cuda_graph_decode()` API as a compatibility
  wrapper over the richer model capability.
- Qwen3 now reports full decode graph support, unless the runtime disables CUDA
  Graph or LoRA forces eager decode.
- Qwen3.5 now reports piecewise decode graph support, matching the existing
  consecutive linear-layer graph cache.
- DSv4 now reports explicit unsupported status with the current blockers:
  host `start_pos` launch parameters, per-step scratch allocation, and
  unvalidated TP/EP NCCL graph capture.
- Scheduler warmup logs the model-reported capability before deciding whether
  to capture graphs or run eager warmup.

## Verification

- `cargo fmt --check`
- `CUDARC_CUDA_VERSION=12080 CARGO_TARGET_DIR=/tmp/arle-cargo-check-cuda cargo check -p infer --no-default-features --features cuda,no-cuda`
- `CARGO_TARGET_DIR=/tmp/arle-cargo-check-nocuda cargo check -p infer --no-default-features --features no-cuda`

The CUDA/no-cuda check passed with pre-existing DSv4 unused/unsafe warnings.
The first isolated CUDA check without `CUDARC_CUDA_VERSION` failed because this
Mac has no `nvcc`; it passed after pinning the cudarc CUDA version.

## Pending Remote

No DSv4 performance win is claimed here. This tranche intentionally does not
enable DSv4 graph replay because replay would capture stale host scalar
metadata today. Remote DSv4 A/B is required only after the next tranche makes
the DSv4 decode body graph-safe.

Required next gates:

| Gate | Required result |
|---|---|
| DSv4 metadata | `start_pos`, seq lengths, slot/block metadata are read from stable device buffers or graph-updated nodes |
| DSv4 scratch | decode context owns all per-step scratch with stable pointers |
| DSv4 collectives | TP/EP collective backend is graph-capture supported or forced eager outside graph |
| Correctness | EOS and 32-token decode return sane text with graph on/off |
| Perf | matched warm TPOT shows real graph-on launch-overhead recovery |

## Rule

CUDA Graph "enabled" must be a model-level capability with a reason, not a
silent boolean. Do not flip DSv4 graph support until stale replay metadata is
licensed by correctness tests and trace evidence.
