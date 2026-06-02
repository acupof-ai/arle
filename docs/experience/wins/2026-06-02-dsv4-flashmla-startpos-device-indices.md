# DSv4 FlashMLA decode indices read start_pos from device

## Context

Goal: continue DSv4 CUDA Graph readiness by removing host scalar launch
parameters from the FlashMLA sparse-FP8 decode metadata path. The decode
indices builder took `start_pos` as a host `int`, so graph replay would freeze
the captured decode position unless the launch node was updated every step.

## What Worked

- Kept the existing scalar
  `arle_dsv4_flashmla_decode_build_indices_cuda` ABI for compatibility.
- Refactored the CUDA index calculation into one device helper shared by the
  old scalar kernel and a new device-pointer kernel.
- Added
  `arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda` plus Rust FFI
  and wrapper.
- Added a stable one-i32 `fm_decode_start_pos` arena slot in
  `DeepseekAttentionRuntimeCache`.
- Changed the DSv4 FlashMLA decode path to fill that slot on the CUDA stream
  and pass the device pointer into the indices builder.

This is a replay-safety tranche, not the final CUDA Graph enablement. The
`start_pos` slot still needs a graph-safe producer contract for true replay,
for example graph-external metadata update or graph node parameter update.

## Verification

- Local `cargo fmt --check`
- Local `cargo check -p infer --no-default-features --features no-cuda`
- Local `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Local `git diff --check`
- Remote pod worktree `/tmp/arle-dsv4-startpos-115bb81c`, HEAD
  `115bb81c7fcfcea0c79ca791b9b7c29161cad885`.
- Remote `cargo +stable fmt --check`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`
- Remote targeted CUDA C compile:
  `/usr/local/cuda/bin/nvcc -c csrc/attention/dsv4_flashmla_decode_build_indices.cu -o /tmp/dsv4_flashmla_decode_build_indices_115bb81c.o -O3 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 --compiler-options -fPIC -Icsrc -std c++17 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`
- Remote `nm -g /tmp/dsv4_flashmla_decode_build_indices_115bb81c.o` confirmed:
  `arle_dsv4_flashmla_decode_build_indices_cuda` and
  `arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda`.

## Problems

`CUDARC_CUDA_VERSION=12080 cargo +stable check -p cuda-kernels
--no-default-features --features cuda --offline` was started as a full CUDA
symbol verification, but it compiled the whole native CUDA C + FlashMLA set
and reached `vendor/flashmla/csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu`.
That job was intentionally terminated after confirming it was doing full AOT
work, not a targeted validation of this diff.

No runtime benchmark, decode correctness result, CUDA Graph replay result, or
TPOT claim is made from this buildability tranche.

## Pending Graph Enablement

The DSv4 CUDA Graph gate remains closed. Remaining replay blockers include the
dynamic producer for `fm_decode_start_pos`, compressor counters, FP8 KV pool
high-water marks, FlashMLA sched metadata, CUDA graph coverage for small prep
kernels, and TP/EP collective capture semantics.

## Rule

Moving one launch scalar to a device pointer is necessary but not sufficient
for graph replay. Treat it as ABI groundwork until the dynamic metadata
producer is also replay-safe and verified by an actual graph capture test.
