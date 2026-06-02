# DSv4 Batch FlashMLA Decode Arena

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. This tranche does not make the high-performance
`sglang` profile runnable by itself. It removes one structural blocker for
batch FlashMLA decode wiring: the Rust decode context previously had only
per-slot `b=1` FlashMLA scratch in each attention cache.

## What Worked

`DeepseekBatchDecodeBuffers` now owns a batch-shaped FlashMLA sparse-decode
arena for the future `b=N` decode dispatch:

- `indices`: `[B, topk_unified]`
- `topk_length`: `[B]`
- `num_splits`: `[B + 1]`
- `sched_meta`: `[num_sm_parts, 8]`
- `lse_accum`: `[B + num_sm_parts, h_q]`
- `o_accum`: `[B + num_sm_parts, h_q, d_v]`
- `lse_out`: `[B, h_q]`
- `slot_layer_block_offsets`: `[B]`

The shared FP8 KV pool also now exposes explicit helpers for:

- the global pool base pointer, which batch FlashMLA needs as one contiguous
  KV base;
- total block count;
- slot-major `(slot, layer)` block offsets.

The per-row shared-pool binding path now uses the same slot/layer block-offset
helper, so the address formula is no longer duplicated. A small unit test locks
the formula:

`block_offset = (slot_idx * layers + layer_idx) * slot_blocks`.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification on `/data01/build/arle`, commit
`098b135f0975c425eb2e9e03e017d311e77d87e9`:

- remote HEAD matched local HEAD, with clean status before the build.
- release-fast CUDA build passed in 24.01s.
- the build used the DSv4 prebuilt CUDA artifact fast path and skipped nvcc /
  TileLang AOT, so this Rust-only tranche did not pay a CUDA rebuild.

Artifact:

- `/tmp/dsv4_batch_flashmla_arena_20260603/build.log`

## Rule

Batch FlashMLA needs batch-owned metadata and scratch before it can be graphed.
A Rust-side arena is only a graph-enablement substrate; it is not a TPOT win
until the row-looped attention core is replaced and the target workload runs.
