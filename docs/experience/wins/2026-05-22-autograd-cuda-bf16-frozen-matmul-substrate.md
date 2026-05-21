# Autograd CUDA BF16 Frozen Matmul Substrate

## Goal

Unblock the 9B GPTQModel teacher + 0.8B LoRA student memory path by adding the
first train-side primitives needed for a frozen BF16 student base:
device-resident BF16 storage, `matmul_bt` with a frozen BF16 RHS, and BF16
embedding lookup for both host token ids and rollout decode's device token-id
buffer.

## Hypothesis

The prior OPD bench failed because the LoRA student base still expands to f32
on the train side. A BF16 device handle and BF16 RHS projection path should let
the loader keep frozen base weights in 2-byte storage in a later tranche.

## Params

- Scope: substrate only, not yet wired into Qwen3.5 loader/model.
- New handle: `DeviceHandle::CudaBf16`.
- New op paths:
  - `matmul_bt(f32 lhs, BF16 rhs) -> f32 output`
  - `embedding(BF16 table, i32 ids) -> f32 output`
  - `embedding_from_f32_ids(BF16 table, f32 ids) -> f32 output`
- CUDA env:
  - `NVCC_CCBIN=/usr/bin/g++-14`
  - `INFER_TILELANG_PYTHON=$PWD/.venv/bin/python`
  - `CUDARC_CUDA_VERSION=13010`
  - `TORCH_CUDA_ARCH_LIST=8.9`
  - `CARGO_BUILD_JOBS=1`

## Results

Correctness gates:

```text
cargo test -p autograd --test test_cuda_bf16_frozen_ops --release --features cuda
running 4 tests
test cuda_bf16_upload_readback_roundtrips_as_f32 ... ok
test cuda_embedding_accepts_frozen_bf16_table ... ok
test cuda_embedding_from_f32_ids_accepts_frozen_bf16_table ... ok
test cuda_matmul_bt_accepts_frozen_bf16_rhs ... ok
test result: ok. 4 passed
```

Type gate:

```text
cargo check -p autograd --release --features cuda
Finished `release` profile [optimized] target(s)
```

## Problems

cuBLAS rejected the direct mixed `A=f32, B=bf16, C=f32` `gemm_ex` combination.
The landed path explicitly rounds the activation to BF16 on-device, runs native
BF16 GEMM with FP32 accumulation, then converts the BF16 result back to f32.
That matches the inference precision regime and keeps the tensor device
resident, but it is a precision tradeoff that must be checked at model level.

## Learnings

This is a licensed substrate step, not a performance claim. The next tranche
still needs Qwen3.5 LoRA loader/model wiring so large frozen base tensors choose
these BF16 handles before re-running the 9B GPTQModel -> 0.8B LoRA OPD memory
gate.
