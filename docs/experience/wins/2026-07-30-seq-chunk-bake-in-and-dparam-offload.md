# Seq-chunk bake-in + d_param host-offload — the 65536 writeback wall

> Status: **PENDING-REMOTE (65536 / 131072 in flight on H20 GPU 5 / GPU 1)**.
> Code shipped and locally gated; the GPU verdict lands in this file's Measured
> table before this entry is called green.

## Context

Two things at once, both on the single-GPU 256K writeback path
(`--synthetic-writeback-seq`, ThinkingCap-Qwen3.6-27B-FP8, 1×H20 97871 MiB):

1. MLP seq-chunking (`2026-07-28-mlp-seq-chunked-recompute-256k.md`) and
   full-attention chunking (`95b305c9e`, no entry of its own) each shipped behind
   a `total_rows ≥ 40961` threshold plus an env override. Two knobs, two code
   paths, and the un-chunked path is the one never taken past 40960.
2. 65536 was the next wall. Forward completed (743.1 s), fused_ce 2.95 s, then
   `cuda alloc_zeros failed (slice_bwd)`.

## What changed

**Bake-in.** `mlp_seq_chunk_for_seq` / `attn_seq_chunk_for_seq` (each a 40961-row
gate + an `ARLE_OPD_{MLP,ATTN}_SEQ_CHUNK` env override) collapse into one const,
`runtime_flags::OPD_SEQ_CHUNK = 4096`. Both call sites in `qwen35.rs` lose their
`if chunk > 0 { … } else { plain }` branch. Justification is that chunking is
*exact*, not a tradeoff: MLP is position-wise, and the attention path is
FlashAttention q-tiling with `q_start` + position-sliced RoPE. Sub-threshold cost
is one input copy. CP is untouched — the attention dispatch still sits behind
`!cp.is_enabled()`.

**d_param host-offload.** `forward_full_attention_chunked` keeps k/v
full-sequence and passes them as saved inputs, so `seq_chunked_recompute_backward`
holds full-seq f32 `d_k`/`d_v` accumulators on device across the *entire* chunk
loop. `[1, 24, seq, 256]` f32 (24 = post-`repeat_kv` heads, head_dim 256) is
**1.5 GiB each at seq=65536**, 3.0 GiB for the pair, on top of `d_input`. The
pre-fix OOM peaked at 96788 / 97871 MiB — 1083 MiB of headroom, less than
`slice_backward`'s full-`input_shape` allocation needs.

Fix: after each chunk's `accumulate_into_device`, `offload_to_host` the
accumulator; the next chunk's `ensure_device` brings it back for the accumulate
only. One `ensure_device` sweep after the loop restores all of them before
`GradPairs` is built. Cost is 2 × 1.5 GiB of PCIe traffic per chunk boundary
against a backward already measured in the 2000-second range.

## Verification

Local gates green: `cargo {check,test,clippy}` on `train` and `autograd`
(`clippy -- -D warnings`), plus the CUDA-lane Mac typecheck
(`CUDARC_CUDA_VERSION=12080 cargo check -p train --release --no-default-features
--features cuda,no-cuda`). CPU parity for chunked-vs-unchunked is the existing
`chunked_matches_unchunked` autograd test — unchanged by this diff and still
passing, which is what says the bake-in didn't alter the math.

## Measured — seq ladder (H20, 1 GPU, untraced)

Prior rungs, for the wall this entry moves:

| total rows | forward | fused_ce | backward | verdict |
|---|---:|---:|---:|---|
| 49152 | 811.2 s | 2.18 s | 2719.8 s | `RUN_EXIT=0` loss 7.2336 |
| 57344 | 575.5 s | 2.64 s | 2460.4 s | `RUN_EXIT=0` loss 6.1978 |
| 65536 (pre-fix) | 743.1 s | 2.95 s | — | **OOM `slice_bwd`**, peak 96788 MiB |
| 65536 (post-fix) | | | | *in flight, GPU 5* |
| 131072 | | | | *in flight, GPU 1* |

131072 does **not** hit the `> 65_535` chunked-SDPA branch
(`attention.rs:171`) — the q chunk is `OPD_SEQ_CHUNK` = 4096, so every SDPA call
stays on the fused fast path regardless of total seq. That branch remains
untested, and CP is still the first thing that would reach it
(`reference_sdpa_65535_grid_boundary_untested_under_cp`).

## Rule

A threshold on an *exact* transform is dead weight — it doubles the code paths
and the un-taken one is the one that rots. Gate a tradeoff; bake in an identity.

And when a chunked backward accumulates gradients for saved inputs that are
themselves full-sequence, the accumulators — not the per-chunk working set — are
the device-memory floor. They are live across the whole loop but touched once per
chunk: exactly the profile that wants host residency between touches.
