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
`indices[b, topk_unified]` plus `topk_length[b]` in one launch. The ABI was
corrected to also consume `slot_layer_block_offsets[b]`: live slot ids are
offset by the actual scheduler slot/layer block offset, not by row order. This
matches ARLE's shared FP8 KV layout, where active rows are not guaranteed to be
contiguous in slot order.

The DSv4 shared FP8 KV pool allocation also now uses `PagedKVPool::num_slots`
instead of the decode batch capacity. Normal scheduler decode already passes
all state slots, but temporary speculative verifier contexts can be sized to
`requests.len()`; using the real KV pool slot count prevents EAGLE verifier
contexts from binding a high slot id into an undersized shared FP8 arena.

The startup contract message was also corrected: `start_pos` has a device-buffer
path now; the remaining attention blocker is the per-row loop with host-selected
per-slot/per-layer cache planning.

## Verification

Remote verification before the slot-offset ABI correction, on
`/data01/build/arle` at commit `75403573`:

- release-fast CUDA build passed in 6m58s and harvested a fresh DSv4 prebuilt
  archive.
- `libkernels_cuda.a` exported
  `arle_dsv4_flashmla_decode_build_indices_batched_cuda`.
- TP8 + EAGLE SGLang-best-practice startup still failed closed, as expected, on
  remaining full-graph blockers.

Artifacts:

- `/tmp/dsv4_batched_indices_20260603/build.log`
- `/tmp/dsv4_batched_indices_20260603/startup_tp8_eagle.log`

Local checks after the slot-offset ABI correction:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification after the slot-offset ABI correction, on
`/data01/build/arle` at commit `6c9559dc`:

- remote HEAD matched local HEAD
  `6c9559dcb198675290e1120260f193e66207e517`, with clean status.
- release-fast CUDA build passed in 6m59s, rebuilt the CUDA tree because
  `cuda_kernels_tree` changed, and harvested a fresh DSv4 prebuilt archive.
- `libkernels_cuda.a` exported
  `arle_dsv4_flashmla_decode_build_indices_batched_cuda`.
- TP8 + EAGLE SGLang-best-practice startup probe failed closed, as expected,
  with the remaining blockers unchanged: full-decode CUDA graph, DeepEP/NCCL
  capture/replay, frozen-KV EAGLE/MTP graph replay, graph-captured
  FlashMLA/SWA/C4/C128 metadata replay, and batched decode attention still
  looping per row.
- After the probe, no `infer` process remained and `nvidia-smi` reported no
  compute apps.

Artifacts:

- `/tmp/dsv4_slot_offsets_20260603/build.log`
- `/tmp/dsv4_slot_offsets_20260603/startup_tp8_eagle.log`

## Rule

For DSv4 batched FlashMLA work, landing an indices primitive is only a substrate
step. Do not claim a performance win until forward wiring runs a same-binary
correctness + TPOT A/B on the target workload.
