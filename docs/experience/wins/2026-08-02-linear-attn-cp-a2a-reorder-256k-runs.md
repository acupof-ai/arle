# Linear-attn CP a2a + zigzag reorder: real 27B 256K runs end-to-end — 2026-08-02

> Status: Shipped. Pod-verified (HEAD 1734c69cc, GPUs 2/3): f32-anchored
> reorder gate PASS (ce_cp_vs_cpu 8.5e-5, bf16 floor); real 27B cp=2 seq=131072
> completes a full fwd+bwd+optimizer step, no OOM.

## Context

The full-attention CP ring landed and passed (2026-08-01), but the real 27B is
hybrid: 48 of its layers are **linear-attention** (Mamba-style sequential scan),
whose CP transport is a seq↔head all-to-all, not a ring. Two gaps blocked real
256K CP training, both invisible to `nd_parallel_parity` (which has zero
linear-attn layers):

1. `all_to_all_device` world>1 was a pure `Err(pending-remote)` stub — the 27B
   crashed at `linear_attention.rs:429` on the first `all_to_all(q,1,2)`.
2. Once a2a ran, the scan read the sequence **out of order**: zigzag CP + the
   a2a head-shuffle interleave the 2N sequence blocks as `[c0, c_{2N-1}, c1,
   c_{2N-2}, ...]`, but `linear_attention_core_cp` fed that straight into the
   recurrence, which needs true global order. The full-attention ring tolerates
   any block order (per-row absolute-position mask); the scan does not.

## What worked

**a2a device transport (`fd8e38e5c` + `b41b130e5`):** NCCL send/recv group +
transpose-sandwich assembly, reusing `ring_send_recv_kv` / `cuda_slice_device` /
`cuda_concat_parts` — no new kernel. Self-send excluded from the group.

**Zigzag reorder (`b41b130e5`):** a 2N-block permutation in
`linear_attention_core_cp` un-interleaves to global order before the scan and
re-interleaves before the output shuffle. Derived from `cp_size` alone (CP always
uses 2N-way zigzag); pure `slice`+`cat` so backward reassembles for free.

**The gate that actually proved it (`1734c69cc`):** `cp_hidden_parity` ran
`layer_types=[FullAttention, FullAttention]` — it never touched the reorder.
Switching layer 0 to `LinearAttention` (mixed, mirroring the 27B) put the reorder
under the f32 anchor. Since reorder + scan are f32 and the CP path is f32, the
CPU-f32 ground truth separates a reorder bug from bf16 noise — the 27B liveness
run can't (no single-card anchor fits at seq=131072).

## Verification (pod, HEAD 1734c69cc, GPUs 2/3)

- **f32-anchored reorder gate (seq=16, layer 0 LinearAttention):**
  `cp_vs_cpu_f32=3.84e-2 ≤ single_vs_cpu_f32=4.38e-2`; `ce_cp_vs_cpu=8.5e-5`
  (bf16 floor, marginally below single-card's 8.9e-5). PASS, RUN_EXIT=0. **This
  is the decisive numeric proof the reorder is correct.**
- **Real 27B cp=2 seq=131072 (ThinkingCap-Qwen3.6-27B-FP8, LoRA r16 attn-qv):**
  full fwd+bwd+optimizer step, NO OOM. Ranks `DONE loss=2.821535` / `0.214899`
  over 130816 targets each — the 13× rank asymmetry is a zigzag-shard property
  (rank 0 owns the sequence head = thin context = high loss), NOT a bug: the f32
  gate above confirms the reorder is numerically correct.

**256K CP VRAM wall (real 27B, cp=2) — the #70 answer:**

| stage | MiB/GPU |
|------|------:|
| weight resident floor | 34,283 |
| forward peak | 80,319 |
| **backward peak (grad-ckpt recompute)** | **94,175** (~96.6%, ~3.3 GB headroom) |

**cp=2 fits 131072 on the H20's 97.5 GB — cp=4 is not needed.** The wall is
activations, not weights.

## Problems (not this fix, tracked)

- **Step wall-clock ~4.3h** (fwd 3765s + bwd 11837s): the reorder uses `ops::cat`,
  which host-forces every input (`layout.rs:366` `tensor_host`), round-tripping 2N
  blocks device→host→device per group per layer. `slice` is device-lazy; `cat` is
  not. Fix is device-native concat (#74), gated behind this correctness PASS.
- **Post-DONE NCCL teardown hang** leaves RUN_EXIT unwritten though all compute +
  measurement completed. Separate issue.

## Rule

A collective that's order-agnostic for one consumer (ring attention: per-row
position mask) can be order-WRONG for another sharing the same shard layout
(sequential scan: needs true global order). When one shard feeds both, the
order-sensitive consumer owns the reorder — don't leak the shard scheme into the
generic collective, and don't re-shard per layer. And a parity gate only proves
the path it exercises: a gate with zero linear-attn layers said nothing about the
linear-attn reorder — put the feature under test in the model config, or the
green is about a different code path. See
`wins/2026-07-31-zigzag-ring-device-kernel-per-row-positions.md`.
