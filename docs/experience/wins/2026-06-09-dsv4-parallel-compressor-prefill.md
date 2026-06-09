# DSv4 prefill −31% — parallelize the single-block compressor kernel

**Date:** 2026-06-09. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** (this). **Scope:** `crates/cuda-kernels/csrc/misc/dsv4_attention.cu`.

## Context

Prefill at 64K ran GPU-100%-util but only ~2000 tok/s (~1-2% MFU) — the GPUs were
busy on small/serial kernels, not dense GEMM (see
`2026-06-09-dsv4-roofline-op-analysis` reasoning in chat). The compressor
(`dsv4_compressor_update_kernel`) launched `<<<1, 256>>>` — a **single CUDA
block** that loops `for block in 0..completed` serially, using 1/78 of the SMs.
Called twice per CSA layer (core + indexer) per chunk; at chunk=4096, CSA ratio=4
→ 1024 compressed blocks produced serially in one block.

## What worked

Split the prefill path into two kernels (decode `start_pos_ptr` path left on the
old single-block kernel — there `completed<=1`, so a grid is pointless and the
risk is zero):

- `dsv4_compressor_block_kernel<<<completed, 256>>>` — **one CUDA block per
  compressed-output block**, fills the SM array. Block `b>0` reads its overlap
  "previous-block" tokens DIRECTLY from `kv_raw`/`score_raw` (first-half
  projection); the serial `prev_overlap` carry was only a re-read optimization —
  the source tokens are addressable. Only block 0's cross-chunk overlap reads the
  (frozen, prior-chunk) `prev_overlap` input.
- `dsv4_compressor_finalize_kernel<<<1, 256>>>` — tiny: writes the cross-chunk
  `prev_overlap` (last block's first-half tokens) + the trailing `pending` partial
  block, with a `__syncthreads` separating the prev_overlap reads of OLD pending
  from the pending writes. No in-flight read/write race (runs after the parallel
  compute on the same stream).

Mutated-buffer enumeration (correctness): `compressed[compressed_base+b]` (block
kernel, disjoint per b), `prev_overlap_kv/score` (finalize, last block only),
`pending_kv/score` (finalize, tail or completed==0 append). All reads of OLD
pending/prev_overlap complete before their writes (separate kernel launches,
same stream + the finalize `__syncthreads`).

## Results (8×H20, default config, same binary A/B)

- **64K prefill: 25.7s → 17.6s / 17.7s = −31%** (consistent ×2; baseline was the
  rope-fix+chunk build, this = + the compressor split only).
- **Correctness gate PASS:** needle n=4 (75 tok) retrieves `738291`; n=8 (115 tok)
  **3/3** exact (the rope-fix build was 2/3 — no regression, MoE non-det floor).
- The single-block compressor was **~30% of 64K prefill**, far above the ~2-3%
  a naive O(N) FLOP estimate suggested — single-block serialization (1/78 SM +
  per-block `__syncthreads`) dominated, not the arithmetic.

## Rule

- `<<<1, …>>>` on any per-chunk prefill kernel is a structural bug regardless of
  its FLOP share — single-block = 1/78 SM + serialized `__syncthreads`, and the
  wall cost is dominated by the serialization, not the math. Grid over the
  independent unit (here: compressed blocks).
- A "serial recurrence" (the compressor's `prev_overlap` carry) is often just a
  re-read optimization — if the source data is addressable, each parallel unit
  re-reads its window and the recurrence dissolves; only the genuine cross-chunk
  boundary state (block 0's overlap, the tail pending) needs sequencing.
- Don't trust a naive FLOP-share estimate to dismiss a kernel — the compressor
  read as "~2-3%, O(N)" but measured −31%. Build the cheap A/B; let it correct
  the roofline guess.
