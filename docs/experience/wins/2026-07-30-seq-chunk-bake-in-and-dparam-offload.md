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

**cat_seq → one full-seq buffer.** `checkpoint_seq_chunked`'s forward built its
output by `cat_seq`-ing each chunk onto the accumulated prefix
(`checkpoint.rs:227-230`). Every chunk allocated a `[done + chunk]` buffer and
re-copied all prior chunks into it: **O(seq²/chunk)** device traffic, and the
final concat holds old + new simultaneously. At seq=131072 that is 32 chunks and
a 2.5 GiB output, so the last step wants 5 GiB against the 3.2 GiB left after a
94580 MiB forward peak — `alloc_zeros failed (concat_axis2)`.

Now one `zeros([batch, seq, dim])` up front and `write_slice_device` per chunk,
which is exactly what the backward already does for `d_input`
(`tape.rs:1113`). On CUDA that is `slice_backward_f32` — a real device kernel
doing plain assignment (`grad[input_offset] = upstream[idx]`,
`kernels/layout.cu:154`) over disjoint rows, with only shape metadata crossing
PCIe. Peak drops from 2× the output to 1×; traffic from quadratic to linear.

This also explains the pre-existing 40960 `concat_axis2` OOM that
`trim_after_checkpoint_replay` (`tape.rs:1193-1197`) was added to work around —
same root cause, treated at the symptom.

## Verification

Local gates green: `cargo {check,test,clippy}` on `train` and `autograd`
(`clippy -- -D warnings`), plus the CUDA-lane Mac typecheck
(`CUDARC_CUDA_VERSION=12080 cargo check -p train --release --no-default-features
--features cuda,no-cuda`).

**`chunked_matches_unchunked` earned its keep.** The cat_seq rewrite silently
dropped `d_input` — a fresh `alloc_device_tensor` defaults to
`requires_grad=false` (`tensor.rs:611`), and the old code inherited the flag
through its trailing `reshape`. The tape entry was still recorded, so nothing
errored; backward just skipped the node and `grads.get(&x)` returned `None`. One
`set_requires_grad(output_id, requires_grad)` fixes it. A VRAM-only review would
have shipped a silently wrong gradient.

## Measured — seq ladder (H20, 1 GPU, untraced)

Prior rungs, for the wall this entry moves:

| total rows | forward | fused_ce | backward | verdict |
|---|---:|---:|---:|---|
| 49152 | 811.2 s | 2.18 s | 2719.8 s | `RUN_EXIT=0` loss 7.2336 |
| 57344 | 575.5 s | 2.64 s | 2460.4 s | `RUN_EXIT=0` loss 6.1978 |
| 65536 (pre-fix) | 743.1 s | 2.95 s | — | **OOM `slice_bwd`**, peak 96788 MiB |
| 131072 (d_param fix only) | — | — | — | **forward OOM `concat_axis2`**, peak 94580 MiB |
| 131072 (+ cat_seq fix) | | | | *rebuilding* |

131072 never reached the backward: it OOMed in the **forward**, at
`concat_axis2`, which is the third finding of this session and had nothing to do
with the d_param fix.

**The two in-flight rows' wall-clock is not comparable to the rows above them.**
They run concurrently on one box and this forward is host-bound (GPU util 8-19%
at steady state, CPU pinned at 100% with CPU-time tracking wall-clock 1:1), so
they contend for host cycles the earlier rows had to themselves — 65536's forward
is already past 19 min against the 12.4 min it took alone. Peak VRAM is unaffected
(separate cards); only the seconds are. Re-measure serially before quoting a
backward time.

Peak VRAM *is* comparable, and gives the seq slope: 66987 MiB at 65536 vs
94580 MiB at 131072 → **~27.6 GB per 65536 tokens**, intercept ~39.4 GB. Linear
extrapolation puts 262144 at ~149.8 GB: **out of reach on one 97871 MiB card**
regardless of this fix. What that 27.6 GB is made of is the next question, and
the answer decides whether 256K single-card is reachable at all or CP is
mandatory.

131072 does **not** hit the `> 65_535` chunked-SDPA branch
(`attention.rs:171`) — the q chunk is `OPD_SEQ_CHUNK` = 4096, so every SDPA call
stays on the fused fast path regardless of total seq. That branch remains
untested, and CP is still the first thing that would reach it
(`reference_sdpa_65535_grid_boundary_untested_under_cp`).

## Rule

A threshold on an *exact* transform is dead weight — it doubles the code paths
and the un-taken one is the one that rots. Gate a tradeoff; bake in an identity.

When a chunked loop accumulates a full-sequence result — a gradient or an output
— **allocate the final shape once and write disjoint slices into it.** Growing it
by concatenation is quadratic traffic and a doubled peak, and it fails late: it
works at every length you tested and dies at the one you needed. `cat_seq` in the
forward and host-parked accumulators in the backward are the same lesson from
both ends.

A chunk-loop rewrite is a *gradient* change, not just a memory change. The output
buffer's `requires_grad` is part of the contract; a fresh device allocation
doesn't carry it, and nothing errors when it's missing — backward just quietly
skips the node. Keep an end-to-end parity test on the exact quantity (`d_input`,
not just loss) or this class of bug ships.
