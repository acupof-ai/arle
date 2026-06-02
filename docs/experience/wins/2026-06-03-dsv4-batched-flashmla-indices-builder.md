# DSv4 Batched FlashMLA Decode Indices Builder

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. The high-performance profile still fails
closed until full decode graph capture, DeepEP/NCCL graph safety, MTP/EAGLE
graph replay, and batched FlashMLA attention are implemented.

## What Worked

Added a batched `b = N` DSv4 FlashMLA decode indices builder:

- `arle_dsv4_flashmla_decode_build_indices_batched_cuda`
- `dsv4_flashmla_decode_build_indices_batched_raw`

The kernel reads `start_pos[b]` from device memory and writes
`indices[b, topk_unified]` plus `topk_length[b]` in one launch. Live slot ids are
offset by `row * total_blocks * page_block_size`, matching FlashMLA's absolute
slot addressing over a single contiguous shared FP8 KV arena.

The startup contract message was also corrected: `start_pos` has a device-buffer
path now; the remaining attention blocker is the per-row loop with host-selected
per-slot/per-layer cache planning.

## Verification

Pending remote CUDA build and symbol check.

Local checks planned:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

## Rule

For DSv4 batched FlashMLA work, landing an indices primitive is only a substrate
step. Do not claim a performance win until forward wiring runs a same-binary
correctness + TPOT A/B on the target workload.
