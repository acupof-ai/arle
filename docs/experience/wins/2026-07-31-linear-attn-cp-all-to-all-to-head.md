# Linear-attention CP = all-to-all-to-head, model-agnostic — CPU-gated — 2026-07-31

> Status: Core landed + CPU-gated (head-split parity exact; cp==1 byte-identical).
> Multi-rank all-to-all transport (world>1) is pending-remote NCCL.

## Goal

Make gated-delta linear attention correct under context parallelism. The old CP
path ran the recurrence on each rank's local sequence shard in isolation — the
Markovian state never crossed the shard boundary, so every layer past the first
was silently wrong. Qwen3.6 is hybrid (~48 linear-attn layers), so this made 256K
CP training numerically wrong even with the ring full-attn fix.

## What worked — all-to-all-to-head, in autograd not the model

Calibrated against Megatron's gated-delta-net (`megatron/core/ssm/gated_delta_net`),
our exact arch family: the recurrence is head-independent (state, conv taps,
`a_log[h]`, `dt_bias[h]`, `beta[h]`, per-head rmsnorm never cross value-heads), so
CP all-to-alls the sequence into the head axis — each rank gets the full sequence
for 1/N of the heads, runs the complete recurrence locally (no cross-rank
dependency, no approximation), all-to-alls back.

Landed as **`autograd::ops::linear_attention_core_cp`** — model-agnostic, the
sibling of `cp_causal_sdpa` for full attention. Parameterized by
`LinearAttentionParams`, so any gated-delta / Mamba-hybrid model reuses it; the
next model wires one call, not a rewrite. `qwen35.rs` `forward_linear_attention`
gained a `cp` param and calls it — the CP transport lives in autograd, the model
file only names the op.

Design confirmed by a second Megatron source review (workflow): GDN carries the
**fused qkv through one all-to-all** and keeps the **conv1d weight packed**
(section-sliced per rank), splitting to per-head q/k/v only as a view at the
kernel entry. So `linear_attention_core`'s 8-arg interface is **untouched** — the
CP wrapper does per-region all-to-all + `cat` re-fuse + per-rank weight slice
around it. New general `ops::cat(inputs, axis)` (rank/axis-generic, unlike the
rank-4 `cat_seq`/`cat_heads`) backs both the qkv re-fuse and the packed-conv slice.

## Verification (local, CPU)

```
cargo test -p autograd -p train --no-default-features --features no-cuda
cargo clippy -p autograd -p train --no-default-features --features no-cuda
```

- `linear_attention_cp_head_split_reconstructs_full`: running the recurrence on a
  value-head subset and concatenating over subsets == running on all heads, for
  cp=2, cp=4, and GQA 2:1. This is the exact math the all-to-all-to-head transport
  relies on — GPU-free, fails if the recurrence ever coupled heads. Replaces the
  old carry-chain test (which tested the rejected serial-ring design).
- `cat` round-trip + grad (feature axis, and axis-0 rank-2 for packed conv).
- cp==1 falls through to `linear_attention_core` verbatim (byte-identical).
- Full autograd+train no-cuda suite green (39 test binaries, 0 failures); clippy clean.

## Pending-remote

`all_to_all` world>1 is the NCCL seq↔head shuffle (no single NCCL primitive) —
pod-only. On the pod: CP-vs-single loss parity at a seq long enough that linear
layers carry real state, alongside the ring full-attn >65535 gate.

## Rule

CP transport for a model op belongs in autograd (parameterized, model-agnostic),
not the model forward — mirror `cp_causal_sdpa`. Adopt-official-first pays here:
a source read of Megatron's GDN refuted a core-interface refactor and confirmed
our fused-qkv + packed-conv1d design *is* the canonical shape. Head-independence
is what makes all-to-all-to-head exact — prove it with a head-split parity test,
not a run.
