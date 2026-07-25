# nonpaged prefill kernel: int64 index widening — CUDA, 2026-07-25

> Status: Shipped

## Goal

Remove a silent int32 index overflow in the nonpaged prefill/decode kernel that
corrupts reads at ~0.5–1M context, without changing throughput.

## Hypothesis

`(kv_head * max_seq_len + row) * head_dim + dim` (nonpaged_prefill_attention.cu:74/122)
is all-`int`, evaluated 32-bit before widening. It wraps to a negative offset at
`2^31 / (num_kv_heads·head_dim)` — 1,048,576 for kv=8/hd=256 → silent wrong-tensor
read, no error. Only the K/V indices overflow (`max_seq_len` is the large factor);
q/out are bounded by `token(<65535)·q_dim(<16384) < 2^31` and stay int32. Fold the
loop-invariant int64 term (`kv_head*max_seq_len*head_dim`) into base pointers
`k_base`/`v_base` once; the hot loop indexes `k_base[row*head_dim]` with int32
`row*head_dim` (< max_seq_len·head_dim < 2^31) — overflow-safe, no live 64-bit reg
pair across the loop.

NOT a perf-wash (initial claim was wrong — see Measured). The first version widened
all four indices AND read q directly instead of via a deleted `q_s` shared stage;
cuobjdump showed registers 32→40, theoretical occupancy 100%→75% (register-limited)
on the nonpaged kernel. The hoisted-base + int32-q/out rework above targets that.

## Parameters

```bash
# The kernel fires ONLY on the training-forward / OPD path (full_attn_paged()==false,
# qwen35.rs:6028). A normal serve builds a paged KV pool and fires the TileLang PAGED
# kernel instead — nonpaged never runs there (nsys-confirmed). So profile via the
# training-forward lane, NOT serve; assert_kernel_fired.sh also can't see attention
# FFI (it reads oplib FP8-linear dispatch counters only).
# Static before/after via cuobjdump (deterministic, no ncu attach needed):
cuobjdump -res-usage <lib.so | grep nonpaged>   # registers / smem / occupancy
```

- Baseline: pre-change binary, all-int32 index (K/V overflow at ~1M ctx).
- Treatment: int64 KV base hoisted out of loop, int32 q/out, `q_s` shared stage deleted.
- Correctness gate: `cargo test -p autograd checkpoint` (host) + needle on any path that
  reaches the training-forward kernel.

## Environment

- Host / GPU: H20 sm_90 (measured) · G4 sm_120 (int64-overflow lane, pending).
- Model / dtype: ThinkingCap-27B-FP8 (hd256, 64L).

## Measured (cuobjdump, GPU 3 H20 sm_90)

Register/occupancy across four versions (hd256, block=256; sm_90 cliff: any
REG≥33 drops 8→6 blocks/SM):

| version | REG/thread | shared (B) | SASS | theo. occupancy |
|-|-|-|-|-|
| int32 baseline | 32 | 3884 | 1413 | 100% (8 blk/SM) |
| all-int64 (rejected) | 40 | 3372 | 1561 | 75% (6 blk/SM) |
| hoist int64 kv_base | 37 | 3372 | 1560 | 75% (6 blk/SM) |
| **base-pointer (shipped)** | **32** | **3372** | **1470** | **100% (8 blk/SM)** |

The shipped version folds the only overflow-capable term
(`kv_head*max_seq_len*head_dim`) into loop-invariant base pointers `k_base`/`v_base`
(one int64 each, hoisted — 3 `IMAD.WIDE` total, none in the tile loop), and keeps
`row*head_dim` int32 in the hot loop (< max_seq_len·head_dim < 2^31 even at 4M ctx).
No live 64-bit register pair across the loop → back under the 33-register cliff.

Net vs int32 baseline: **identical registers (32) and occupancy (100%), −512 B shared**
(deleted `q_s[256]`), smallest SASS of the four, and the >1M-ctx int32 overflow fixed.
Genuine perf-wash-or-better. Change #2 (tape.rs) unaffected: `cargo test -p autograd
checkpoint` 8/8.

## Rule

KV linear indices over `head·max_seq·dim` overflow int32 below realistic long-context
sizes — but fold the loop-invariant int64 term into a BASE POINTER once and keep the
per-iteration `row·head_dim` int32 (proven < 2^31). Blanket int64 raised registers
32→40 / occupancy −25pp on sm_90 (register-limited at block=256, REG≥33 cliff). Measure
register/occupancy with cuobjdump on any kernel change; never assume "address-math ⇒
perf wash".
