# cp=4 training seq ceiling is 229376, and the 131072 step got 17.5× faster — 2026-08-19

## Context

`docs/baselines.md` carried two 27B rows at seq=131072, both tagged "older
commit" (2026-08-02 cp=2, 2026-08-03 cp=4). 166 commits and two default flips
landed after them — FA3 replacing the scalar ring (2026-08-04/05) and FlashQLA
GDN chunkwise (2026-08-05) — plus device-native cat, checkpoint reload-to-device,
the guarded in-place `merge_grad`, and the 2026-08-16 O(n) student forward.
Nothing above 131072 had ever run on this substrate: the only rung that ever
reached 262144 was the 2026-07-30 option-B ladder, and that path was deleted and
replaced by the ring.

Binary `ctxlad` at HEAD `9c2c84675`, `--release --features cuda,nccl`, 4×H20
(97,508 MiB), GPUs 4-7. Model `/data00/ThinkingCap-Qwen3.6-27B-FP8` (64 layers =
16 full-attn + 48 gated-delta), LoRA r16 α32 `attention-qv`, workload
`--synthetic-writeback-seq N` (one masked-CE writeback, no rollout). FA3 engaged
— `ring_fa3_route`'s "real kernel is absent" warning never fired.

## Result — the ladder

| seq | RUN_EXIT | forward | fused CE | backward | writeback | peak/rank | loss |
|---|---|---:|---:|---:|---:|---:|---|
| 131072 | 0 | 56.8 s | 1.69 s | 119.05 s | 177.5 s | 74,095 MiB | 7.631271 |
| 163840 | 0 | 73.5 s | 2.01 s | 155.08 s | 230.7 s | 78,959 MiB | 7.189730 |
| 196608 | 0 | 89.8 s | 2.03 s | 192.09 s | 283.6 s | 86,991 MiB | 6.924870 |
| **229376** | **0** | 107.9 s | 2.76 s | 231.38 s | **342.2 s** | **92,655 MiB** | 6.742337 |
| 245760 | FAIL | — | 2.95 s | — | — | — | `alloc_zeros failed (la dqkv)` |
| 262144 | FAIL | 126.5 s | 3.14 s | — | — | 86,607 then released | `alloc_zeros failed (slice_bwd)` |

All four ranks print a bit-identical loss at every passing rung. 229376 peaks at
95.0% of the card.

**The measured cp=4 ceiling is 229376.** Both failures are linear-attention
backward allocations, and both land in `backward` after a clean forward and CE.

## What worked — 131072 is 17.5× faster than the last measurement

| | 2026-08-03 (cp=4, scalar ring) | 2026-08-19 | |
|---|---:|---:|---:|
| forward | ~683 s | 56.8 s | 12.0× |
| backward | ~2415 s | 119.05 s | 20.3× |
| step | ~3100 s | 177.5 s | **17.5×** |

Larger than the product of the two headline flips (FA3 2.17× × FlashQLA 1.99× =
4.3×). The rest is the device-native cat that removed the zigzag reorder's
device→host→device round trip (the 2026-08-02 entry measured that step at 4.3 h),
checkpoint reload-to-device + pinned offload, and the O(n) student forward.

## The two failures

**`slice_bwd` is still the wall — it just moved.** `cuda_slice_backward_device`
(`backend_cuda/layout.rs:414`) still does `alloc_zeros(full input shape)`. The
2026-07-30 diagnosis put it on the CP full-attention KV gather; the ring deleted
that caller, so the wall reads as fixed. It is not: the linear-attention zigzag
reorder in `linear_attention_core_cp` is pure `slice`+`cat`, and 48 of the 27B's
64 layers are linear-attention, so the reorder is now the main caller.

**A rank that errors deadlocks the group before its message prints.** Rank 3
OOMs, unwinds, and Drop calls `ncclCommDestroy` — gdb on the live process:

```
#4  pncclCommDestroy () from /usr/lib/x86_64-linux-gnu/libnccl.so.2
```

The three peers are still spinning inside a collective on that communicator, so
the destroy blocks and the error text never reaches stderr. From outside, a
rank-local OOM looks like: three GPUs at 100% util burning 100% CPU, one at 0%
util sleeping with ~35 GB released, no output, indefinitely. Killing the three
spinners releases the unwind and the real line appears. This is the same
signature recorded on 2026-07-30 and it is unchanged.

## Also measured

The checkpoint memory model under-predicts badly at this scale:

```
[ckpt-peak] batch=1 seq=65536 floor=39772MiB layer=21704MiB
            modeled=61476MiB actual=81383MiB drift=+19907MiB
```

−24%. The gate still engaged correctly here, but it is sizing off a number that
is 19.9 GB low.

## Rule

A wall is fixed when its allocation is gone, not when one caller is deleted.
`slice_bwd`'s full-size zero buffer was diagnosed on 2026-07-30, the caller that
exposed it was replaced by the ring, and the wall was read as retired — it was
waiting on a different caller in the same model. Grep the allocation, not the
call site that reported it.

## Follow-up

1. Fuse the slice grad into its consumer instead of materializing a full-input
   zero buffer (`cuda_write_slice_device` already writes into a caller-provided
   `dest`; the backward is the only site that allocates one first).
2. Make a rank-local error abort the communicator instead of destroying it —
   `ncclCommAbort` on the error path, so the message prints and the peers fail
   fast instead of spinning.
3. cp=8 is unmeasured: only 4 GPUs were free. The linear-attn activation shards
   on the head axis under the all-to-all, so cp=8 should raise the ceiling, but
   that is a projection until it runs.
