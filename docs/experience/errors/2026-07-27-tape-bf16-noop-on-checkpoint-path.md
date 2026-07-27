# tape-bf16 is a no-op on the checkpointed writeback path — 2026-07-27

## Context

Chasing a 40960→128K single-H20 OPD-writeback VRAM lever. `--tape-precision bf16`
(store retained activations + emitted grads as bf16) looked like the big one:
the doc math said it would halve the ~35 GiB of per-layer forward intermediates.
Flipped the default (`runtime_flags.rs` + `args.rs`), built, ran a matched A/B:
same binary `3e5d6236`, `--tape-precision fp32` vs `bf16`, seq=40960, one GPU each.

## Root cause

Both arms came back **byte-identical**: loss=8.685793 AND pool_used=35497 MiB —
not "within the bf16-grad bar", literally the same digits. If bf16 storage were
converting anything, the backward would read rounded activations and loss would
move in the low bits. It didn't move at all → the flag touched nothing on this
path.

Why: the writeback runs under gradient checkpointing with host offload
(`checkpoint_sequential`, `offload_checkpoints=true`). Inside a checkpoint group
the forward runs tape-disabled and frees intermediates at the closure exit; the
saved input is offloaded to host. So activations are never *retained on the tape*
long enough for `quantize_frozen_bf16` / the bf16 store path to see them. The
bf16 tape flag only bites a non-checkpointed forward, which the long-seq
writeback never uses (ckpt_group_size=1 is forced at long seq).

## Fix

Reverted the default flip — bf16 stays opt-in (`--tape-precision bf16`), byte-for
short-seq non-checkpointed lanes where it still applies. Did NOT ship a
default-flip with zero measured benefit + changed gradient-precision semantics.

## Rule

A storage-precision flag only pays where the tape actually retains the tensor.
Under gradient checkpointing + offload the retained set is one input per group,
not the forward activations — so a bf16-tape lever is a no-op there. Measure the
A/B (loss + pool_used) before believing a precision knob is a VRAM win; identical
loss to the last digit is the tell that the flag is inert, not that it's
"within tolerance". The real 40960→128K lever is single-layer forward
seq-chunked recompute (attention 23.4 GiB > MLP SwiGLU 11.4 GiB), not storage
precision. See `docs/research/2026-07-27-opd-writeback-wall-decomposition.md`.
