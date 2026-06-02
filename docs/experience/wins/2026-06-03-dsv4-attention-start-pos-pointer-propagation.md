# DSv4 attention start-pos pointer propagation

## Context

DSv4 full decode CUDA Graph replay must not capture per-step `start_pos` values
by value. The batch decode fallback path already computed per-row GPU
`start_pos` pointers, but `forward_attention_gpu_into` dropped that pointer
before the cached compressor/indexer update.

## What Worked

Propagated `start_pos_ptr_u64` into cached attention and switched compressor /
indexer cache updates to the start-pos-pointer ABI when the pointer is present.
This keeps decode cache offsets derived from GPU metadata during future graph
replay instead of from capture-time host constants.

Verification:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

## Rule

Graph replay inputs must be device-visible metadata, not host constants baked
into the captured launch parameters.
