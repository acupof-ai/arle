# MLP seq-chunked recompute — the 256K writeback VRAM lever

> Status: Shipped; CPU parity GREEN; H20 VRAM A/B measured (seq=40960, GREEN).
> Default OFF (`--mlp-seq-chunk 0`) — no default flip.

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
`checkpoint_seq_chunked` when `--mlp-seq-chunk N` (N>0) and the tape is enabled;
else the plain call. Knob: `runtime_flags::mlp_seq_chunk` (default 0 = off).

## Verification

**CPU parity GREEN** (`autograd` lib test `chunked_matches_unchunked`): a
position-wise MLP block (`silu(x@Wgᵀ)*(x@Wuᵀ)@Wdᵀ`) chunked (chunk=2 over seq=6)
vs unchunked — loss, `d_input`, `d_weight` all ≤1e-5. 37/37 autograd tests pass;
clippy clean; CUDA-lane Mac typecheck (`autograd`+`train`, `cuda,no-cuda`) green.

## Measured A/B (H20 sm_90, agent-OPD synthetic masked writeback, seq=40960)

Same-binary A/B, one variable = `--mlp-seq-chunk`. ThinkingCap-Qwen3.6-27B-FP8
shared-frozen-base, LoRA r16 qv. Backward-recompute peak captured per-op via
`ARLE_OPD_OP_MEM_CHECKPOINT_FN` at layer 63 (full-attn); `pool_used_current`,
not driver-used.

| metric | arm A `chunk 0` | arm B `chunk 4096` | Δ |
|---|---|---|---|
| RUN_EXIT | 0 | 0 | — |
| mean_loss | 8.685793 | 8.685793 | **bit-identical** |
| post_attention pool_used | 60142 MiB | 60142 MiB | 0 (untouched) |
| post_mlp pool_used | 73422 MiB | 62542 MiB | −10880 MiB |
| **inner-backward peak** | **82.0 GiB** | **68.9 GiB** | **−13.1 GiB** |

**GREEN.** Loss bit-identical (0.00%, chunking is add-order-exact); `post_attention`
unchanged confirms it touches only the MLP recompute; −13.1 GiB at seq=40960. The
win decouples from seq (peak ≈ `O(chunk·intermediate)` not `O(seq·intermediate)`),
so it scales into the 256K regime the flag exists for. Full seq-ladder
{131072, 262144} × {8192, 2048} still worth running before a default flip.

## Rule

Position-wise blocks (MLP, norm, residual) recompute-chunk along seq for free —
the peak decouples from seq length, and it's numerically exact (add-order only).
Attention is seq-coupled; it does NOT go through this op (q-chunk or fused-FA
backward is its separate lever). The 256K wall is a `pool_used` length-ladder,
not a kernel-speed question — measure at the seq where the term is large.
