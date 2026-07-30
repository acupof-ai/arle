# Seq-chunk bake-in, d_param offload, and the quadratic-fold walls at 65536/131072

> Status: **PENDING-REMOTE — partial.** Four fixes shipped and locally gated. The
> two forward walls (65536 `slice_bwd`, 131072 `concat_axis2`) are both cleared and
> measured; **no completed step at either length** — the 131072 re-run was stopped
> by hand at 99 min, still in forward. Needs a serial 65536 and 131072 to green,
> plus a 40960 run to license deleting `trim_after_checkpoint_replay`.

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

**It does *not* explain the 40960 OOM** that `trim_after_checkpoint_replay`
(`tape.rs:1193-1197`) was added for — an earlier draft of this entry claimed it
did, wrongly. At exactly 40960 the then-live `≥ 40961` threshold meant
`checkpoint_seq_chunked` took its `chunk == 0` passthrough, so `cat_seq` never
ran. That OOM was the **backward**'s `concat_row_chunks` (`matmul.rs:510`, inside
`matmul_bt_lora_backward_tiled`). Both sites raise the identical string
`alloc_zeros failed (concat_axis2)`, which is what made the misattribution easy.

The same quadratic fold exists in **five** places, and the two unnamed so far are
the worse ones: `concat_row_chunks` (`matmul.rs:510`) and `concat_seq_chunks`
(`attention.rs:713`) each fold `concat_axis2` while their callers hold *every*
chunk live in a `Vec` — peak ≈3× the result, not 2×. That is the 40960 wall, and
the trim stays until a remote 40960 run licenses its removal.

## Where the chunked path actually runs — two corrections

**`checkpoint_seq_chunked` records no tape entry during the writeback forward.**
The enclosing `checkpoint()` disables the tape around the whole layer replay
(`checkpoint.rs:27`), so `if outer_enabled && requires_grad` is always false and
the chunk loop runs purely to produce the output. The entry — and with it the
pinned k/v saved inputs — appears only in the **backward** replay, which runs on
a fresh `Tape::new()` (`enabled: true`, `tape.rs:964`). So the full-attention
k/v are transient in the forward: created inside the replay, absent from the
outer checkpoint's `keep` set, reclaimed at `checkpoint.rs:40`. They are neither
offloaded nor pinned across the forward, and are recomputed from scratch in
backward.

Consequence for the `set_requires_grad` fix above: it repairs the *backward
replay's* path, not a forward one. `chunked_matches_unchunked` caught it because
that test calls the op directly on an enabled tape — the configuration the real
OPD forward never presents, and the one the backward replay always does.

**`ckpt_group_size` is dead arithmetic that always returns 1.** Its
`attn_floor = 12 GiB` (`qwen35.rs:327`) alone exceeds the 8 GiB budget, so the
integer division is 0 and `clamp(1, 8)` yields 1 — at *every* sequence length,
seq=128 included. The `mlp` term it computes is never load-bearing. That is why
exactly one layer's checkpoint group is live at a time, and it is a property of
the constant, not a decision anyone made.

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
| 131072 (+ cat_seq fix) | >99 min, not finished | — | — | **no OOM**, peak 85899 MiB — stopped by hand |

The 131072 re-run cleared the wall that killed the previous attempt — the
`concat_axis2` OOM does not recur and the forward peak drops **94580 → 85899 MiB
(−8.7 GB)** — but it was stopped at 99 minutes before the forward completed, so
there is no `phase=` timing and no backward verdict. What the row licenses: the
`cat_seq` fold was the 131072 forward wall. What it does not: that 131072
completes.

**Wall-clock across these rows is not comparable.** An early pair ran
concurrently on one box, and this forward is host-bound (GPU util 8-19% at steady
state, CPU pinned at 100% with CPU-time tracking wall-clock 1:1), so they
contended for host cycles the 49152/57344 rows had to themselves — 65536's
forward went past 19 min against the 12.4 min it took alone. Peak VRAM is
unaffected (separate cards); only the seconds are. Re-measure serially before
quoting any timing.

Peak VRAM *is* comparable, and gives the seq slope: 66987 MiB at 65536 vs
85899 MiB at 131072 → **~19 GB per 65536 tokens** post-fix (it was 27.6 GB before
the `cat_seq` fix removed the doubled fold peak). Extrapolating the post-fix slope
puts 262144 near 124 GB: still **out of reach on one 97871 MiB card**.

**That slope is the 48 linear-attention layers.** `forward_linear_attention`
(`qwen35.rs:2620`) runs the whole sequence in one `linear_attention_core` call —
neither seq-chunked nor host-offloaded, unlike the MLP and full-attention paths.
Only one layer is live at a time, so the peak is one LA layer's O(seq) state, and
`la_layer_peak_bytes` now models it at 211712 B/token against the measured 231424
(**under-models by 9%**, the dangerous direction). Chunking LA is the next lever
and the precondition for 256K on one card; it is *not* a copy of the MLP fix,
because LA is recurrent (`chunk_state` carries along seq) and needs the GDR
chunked form with explicit boundary state.

131072 does **not** hit the `> 65_535` chunked-SDPA branch
(`attention.rs:171`) — the q chunk is `OPD_SEQ_CHUNK` = 4096, so every SDPA call
stays on the fused fast path regardless of total seq. That branch remains
untested, and CP is still the first thing that would reach it
(`reference_sdpa_65535_grid_boundary_untested_under_cp`).

## The structural fix (`2b4509f05`, `fba949e24`)

Four bugs in one day, all the same shape: something that should have been a type
or an invariant was a convention — *remember* to mark the gradient, *remember* to
preallocate, *remember* to update the memory constant. Conventions hold in code
one person writes once; across 64 call sites and 19 files they leak.

**`requires_grad` is now derived, not maintained.** `TapeEntry::record`
(`tape.rs:372`) computes the OR over `input_ids` and marks the output itself, so
31 manual marks, 51 guards and 29 orphaned locals are gone, and "recorded but not
requiring grad" is unreachable. The mark lands on a *disabled* tape too, which is
what a `checkpoint` inner replay needs — `tape.rs:1449` pins that and fails if the
mark moves inside the `enabled` branch (mutation-verified, not just passing).

**`SeqAccum` (`ops/seq_accum.rs`) has no append.** Five sites folded
`concat_axis2` per chunk; three also kept every chunk in a `Vec` until the fold,
for a ≈3× peak. All five now allocate the final shape once and assign disjoint
rows. The quadratic form is no longer expressible. `concat_row_chunks` and
`concat_seq_chunks` are deleted. Net for the two changes: **−170 lines**.

**`ckpt_group_size` was dead arithmetic returning 1 at every length** — verified
for seq ∈ {128, 1024, 8192, 40960, 65536, 131072}, because `attn_floor = 12 GiB`
alone exceeded the 8 GiB budget. "One layer per checkpoint group" was a property
of a stale constant, not a decision. Deleted, along with the three duplicated
`checkpoint_sequential` blocks, which collapse into one `checkpoint_layers`.
In its place `[ckpt-peak]` prints modeled vs actual peak on each new high-water
(`CU_MEMPOOL_ATTR_USED_MEM_HIGH` through two defaulted `Backend` methods, so no
backend type crosses the seam) — the 9% under-model shows up in a log line
instead of in an OOM 3000 seconds later.

## Rule

A threshold on an *exact* transform is dead weight — it doubles the code paths
and the un-taken one is the one that rots. Gate a tradeoff; bake in an identity.

**A convention that fails silently is a bug with a delay.** Prefer deriving a
value at the one place it's consumed over maintaining it at N places that produce
it; prefer an API that can't express the wrong shape over a comment asking for the
right one. Both fixes here were net deletions — that is the tell that the
abstraction was missing rather than that a new one was wanted.

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
