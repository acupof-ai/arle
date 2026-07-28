# MLP seq-chunked recompute — the 256K writeback VRAM lever

> Status: Shipped; CPU parity GREEN; H20 seq-ladder measured. **Default-ON,
> seq-adaptive** (`total_rows ≥ 40961` → chunk 4096; `ARLE_OPD_MLP_SEQ_CHUNK` env
> override). Not a 256K unlock on its own — single-card writeback ceiling moves
> 40960→49152; 256K is blocked earlier by i32 kernel-index walls + non-MLP
> backward terms (see Ceiling).

## Context

OPD target is 256K seq. The writeback backward re-runs each layer's forward
under grad-checkpoint (`Qwen35Layer::forward`); that recompute materializes the
whole layer's O(seq) activations at once — MLP intermediates alone are 11.4 GiB
at seq=40960 (`2026-07-27-opd-replay-op-mem-attribution.md`), ~73 GiB at 256K.
Single-card OOM long before 256K (40960 already peaks 85.9 GiB / 96 GiB).

The MLP block is position-wise (row `i` out depends only on row `i` in), so its
recompute can be sliced along seq without changing the result.

## What changed

New autograd primitive `checkpoint_seq_chunked` (`ops/checkpoint.rs`): forward
runs the position-wise `replay` once full-seq tape-disabled (frees its own
transients) for the output; the backward (`Tape::seq_chunked_recompute_backward`)
re-runs `replay` on `chunk` seq rows at a time — slicing input + upstream-grad
rows on a disabled scratch tape (detached leaves, so the sub-backward doesn't
scatter through a recorded slice), collecting `d_x_c` + param grads per chunk,
writing `d_x_c` into the full-seq `d_input`'s disjoint rows, accumulating param
grads, then freeing the chunk's live set + `trim_after_checkpoint_replay`. One
new `BackwardOp::SeqChunkedRecompute`; reuses the `CheckpointFn` registry. Peak
recompute drops from `O(seq · intermediate)` to `O(chunk · intermediate)`.

Wired in `Qwen35Layer::forward`: the MLP segment routes through
`checkpoint_seq_chunked` when `runtime_flags::mlp_seq_chunk_for_seq(batch*seq)`
returns > 0 (seq-adaptive, default-on ≥ 40961 total rows), else the plain call.
`chunk == 0` in the op is an inline passthrough (byte-identical, zero overhead).
No CLI flag — the earlier manual `--mlp-seq-chunk` was deleted; `ARLE_OPD_MLP_SEQ_CHUNK`
env forces a chunk (escape hatch for a sub-threshold OOM + manual A/B lever). The
gate is on `batch*seq` (total rows), matching `writeback_offload_for_seq` — the
recompute peak is `O(batch·chunk·intermediate)`, not per-sequence.

## Verification

**CPU parity GREEN** (`autograd` lib test `chunked_matches_unchunked`): a
position-wise MLP block (`silu(x@Wgᵀ)*(x@Wuᵀ)@Wdᵀ`) chunked (chunk=2 over seq=6)
vs unchunked — loss, `d_input`, `d_weight` all ≤1e-5. 37/37 autograd tests pass;
clippy clean; CUDA-lane Mac typecheck (`autograd`+`train`, `cuda,no-cuda`) green.

## Measured — op-level A/B (H20, seq=40960, layer 63 full-attn, per-op pool_used)

| metric | chunk 0 | chunk 4096 | Δ |
|---|---|---|---|
| mean_loss | 8.685793 | 8.685793 | **bit-identical** |
| post_attention pool_used | 60142 MiB | 60142 MiB | 0 (untouched) |
| post_mlp pool_used | 73422 MiB | 62542 MiB | −10880 MiB |
| **inner-backward peak** | **82.0 GiB** | **68.9 GiB** | **−13.1 GiB** |

Loss bit-identical (add-order-exact); `post_attention` unchanged confirms it
touches only the MLP recompute.

## Measured — seq-ladder ceiling (H20, untraced clean wall)

Total-token `--synthetic-writeback-seq`, chunk from the env override:

| total rows | chunk 0 | chunk 4096 |
|---|---|---|
| 40960 | survives (82/96 GiB) | survives |
| **49152** | **backward OOM** (`mul_backward grad_a`) | **completes** (bwd 2460 s) |
| 57344 | — | **backward OOM** (`add_into_device` 2.82 GB — a non-MLP grad-accumulate) |
| ≥61680 | forward **i32 kernel-index wall** | same (chunk-independent) |

**The lever's real value is a survival license, not a wash:** at 49152, chunk=0
OOMs in the backward and chunk=4096 completes — so the default-on threshold sits
at 40961 (just past the last-safe un-chunked point). It is NOT a 256K unlock:
- single-card writeback ceiling moves **40960 → 49152**;
- **57344** OOMs on non-MLP O(seq) backward terms (attention grad + grad-accumulate)
  this lever does not touch — that's the P2 (full-attention) lever;
- **seq > 61680** hits hard i32 kernel-index limits in the *forward* (fused gate-up
  `[1,seq,34816]` and MLP-intermediate overflow 2³¹), neither VRAM nor addressable
  by recompute — a separate i64-index kernel fix is prerequisite to attempting 256K.

## Rule

Position-wise blocks (MLP, norm, residual) recompute-chunk along seq for free —
the peak decouples from seq length, numerically exact (add-order only). But a
VRAM lever's reach is bounded by the *other* seq-coupled terms and by
kernel-index widths: measure the full ladder to find the NEXT wall, and state the
ceiling honestly (here 49152, not 256K). Attention grad + grad-accumulate + i32
indexing are the walls past this one.
