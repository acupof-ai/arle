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

The CUDA/no-cuda check passed with pre-existing DSv4 warnings.

## Pending Remote

No TPOT win is claimed from this local tranche. Remote DSv4 correctness and
perf gates still need a real build and matched graph-off / graph-on A/B after
the attention start-position metadata is made graph-safe.

## Rule

Do not enable CUDA Graph by changing capability reporting alone. Make the decode
body allocation-free first, then make dynamic kernel metadata replay-safe, then
run remote correctness and TPOT A/B.
