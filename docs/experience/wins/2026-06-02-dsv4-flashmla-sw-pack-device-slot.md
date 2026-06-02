# DSv4 FlashMLA decode SW pack derives slot metadata on device

## Context

Goal: continue DSv4 CUDA Graph readiness by removing per-step host metadata
copies from the FlashMLA sparse-FP8 decode body. The one-token SW FP8 pack path
computed `ring_idx = start_pos % sliding_window` on the host, then copied
single-element `block_id` and `row` arrays to device every decode step before
calling the existing FP8 pack kernel.

## What Worked

- Added `arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda`, a tiny CUDA
  kernel that reads the stable device `start_pos` scalar and writes the
  `[block_id,row]` scratch pair on stream.
- Kept `arle_dsv4_fp8_kv_pack_strided_cuda` unchanged for the real FP8
  quantize/pack work.
- Moved FlashMLA decode `start_pos` staging before the SW pack so the pack and
  indices builder reuse the same device metadata pointer.
- Removed the two one-token `memcpy_htod(&[block_idx])` /
  `memcpy_htod(&[row])` calls from the per-step SW pack path.

## Verification

- Local `cargo fmt --check`
- Local `cargo check -p infer --no-default-features --features no-cuda`
- Local `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Local `git diff --check`
- Remote pod worktree `/tmp/arle-dsv4-swpack-870e76e1`, HEAD
  `870e76e167aa6802f13de6d77a8ce05e9497889a`.
- Remote `cargo +stable fmt --check`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`
- Remote targeted CUDA C compile:
  `/usr/local/cuda/bin/nvcc -c csrc/attention/dsv4_fp8_kv_pack.cu -o /tmp/dsv4_fp8_kv_pack_870e76e1.o -O3 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 --compiler-options -fPIC -Icsrc -std c++17 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`
- Remote `nm -g /tmp/dsv4_fp8_kv_pack_870e76e1.o` confirmed:
  `arle_dsv4_fp8_kv_pack_cuda`,
  `arle_dsv4_fp8_kv_pack_strided_cuda`, and
  `arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda`.

No runtime benchmark, decode correctness result, CUDA Graph replay result, or
TPOT claim is made from this buildability tranche.

## Pending Graph Enablement

The DSv4 CUDA Graph gate remains closed. This removes a hot per-token H2D site,
but `start_pos` still needs a replay-safe producer contract, `topk_length` is
still staged through a scalar fill launch, compressor FP8 pack still builds
host block/row arrays, and TP/EP collectives remain outside a graph-safe
contract.

## Rule

Prefer a device metadata producer over per-step host copies in the decode body.
Keep the real pack kernel single-sourced unless the copied metadata itself is
the bottleneck.
