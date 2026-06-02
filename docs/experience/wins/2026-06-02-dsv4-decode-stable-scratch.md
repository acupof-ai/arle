# DSv4 batched decode uses stable scratch buffers

## Context

Goal: move DSv4 toward CUDA Graph decode support without flipping an unsafe
graph capability bit. The previous batched decode body allocated token
embedding, residual stream, per-row attention scratch, head scratch, and one
vocab-sized logits buffer per row on every decode step. That alone made CUDA
Graph capture invalid even before handling dynamic attention metadata.

## What Worked

- Added DSv4 batched-decode scratch owned by `DeepseekBatchDecodeBuffers`.
- Token-id staging, embedding output, stream buffers, per-row attention input
  and output, head HC scratch, final norm scratch, and logits scratch now have
  stable device pointers.
- Changed N>=2 DSv4 batched decode to write logits directly into each slot's
  `decode_logits` via device-to-device copy instead of returning freshly
  allocated `Vec<DeviceVec>`.
- Added cache-owned CSA selector scratch for the cached attention path:
  `q_i`, selector weights, and selected block ids now reuse stable layer-cache
  device buffers instead of allocating during every decode step.
- Extended the FlashMLA decode arena with `topk_length` and dummy `lse_out`, so
  the sparse-decode fast path no longer allocates those per step. `topk_length`
  is still updated each step because HCA top-k depends on the current compressed
  row count.
- Kept the DSv4 CUDA Graph capability gate closed. `start_pos` is still passed
  as a host launch parameter to attention kernels, and TP/EP NCCL capture is
  still unvalidated.

## Verification

- `cargo fmt --check`
- `CUDARC_CUDA_VERSION=12080 CARGO_TARGET_DIR=/tmp/arle-cargo-check-cuda cargo check -p infer --no-default-features --features cuda,no-cuda`
- `CARGO_TARGET_DIR=/tmp/arle-cargo-check-nocuda cargo check -p infer --no-default-features --features no-cuda`
- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Remote pod worktree `/tmp/arle-dsv4-scratch-7072a9eb`, HEAD
  `7072a9eb02512057047fbbfdb121197fcd9ed295`
- Remote `git diff --check -- infer/src/model/deepseek/state.rs infer/src/model/deepseek/weights.rs docs/experience/wins/2026-06-02-dsv4-decode-stable-scratch.md`
- Remote `/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 /root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu/bin/cargo check -p infer --no-default-features --features cuda,no-cuda --offline`

The CUDA/no-cuda check passed with pre-existing DSv4 warnings.

## Pending Graph Enablement

No TPOT win is claimed from this local tranche. Remote DSv4 correctness and
perf gates still need a real build and matched graph-off / graph-on A/B after
the attention start-position metadata is made graph-safe.

Remaining blockers before DSv4 can report CUDA Graph support:

- `start_pos` is captured by value in `dsv4_prepare_qk_cuda`,
  `dsv4_compressor_update_cuda`, `dsv4_csa_select_cuda`,
  `dsv4_hybrid_attention_cuda`, FlashMLA decode index build, and
  `arle_dsv4_output_inverse_rope_cuda`.
- Compressor counters (`pending_len`, `compressed_rows`) and FP8 pack high-water
  marks (`fp8_kv_sw_bootstrapped`, `fp8_kv_comp_packed_rows`) are updated on the
  Rust host path. Full graph replay would reuse the capture-time values unless
  these move to device metadata or the graph updates kernel-node parameters per
  step.
- TP/EP capture is still structurally separate: DSv4 TP/EP decode contains
  NCCL/DeepEP collectives and replicated-token transport that have not been
  proven graph-safe under ARLE's communicator contract.

## Rule

Do not enable CUDA Graph by changing capability reporting alone. Make the decode
body allocation-free first, then make dynamic kernel metadata replay-safe, then
run remote correctness and TPOT A/B.
