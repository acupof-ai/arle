# CP=2 linear-attn state chain deadlock: P2P send grouped with collective broadcast — CUDA, 2026-08-18

## Context

After the 2026-08-17 admission-path fix (prefix-match collectives under CP
divergence), TP=2 CP=2 still hung at the first prefill. NCCL debug log showed
ranks 0,1 (cp_rank=0) posting Send with `sendbuff (nil)`, ranks 2,3
(cp_rank=1) posting Recv, then all 4 ranks stuck.

## Root Cause

The linear attention (Gated DeltaNet) CP state chain grouped the P2P state
send with the end-of-chunk collective broadcast in one NCCL group:

```
ncclGroupStart()
  ncclSend(gdr, cp_rank+1)    // P2P
  ncclSend(conv, cp_rank+1)   // P2P
  ncclBroadcast(gdr, root=cp_size-1)  // collective
  ncclBroadcast(conv, root=cp_size-1) // collective
ncclGroupEnd()  // blocks until ALL ranks post the broadcast
```

`ncclGroupEnd()` blocks on collective submission. cp_rank=0's send was
trapped inside the blocked group, so cp_rank=1's recv (posted earlier in a
separate group) never completed. cp_rank=1 couldn't reach its broadcast
(compute waits on the recv), so cp_rank=0's `ncclGroupEnd()` never unblocked.
Circular wait.

The `sendbuff (nil)` in the NCCL log was a logging artifact of the blocked
group, not a real NULL pointer — the buffers were properly allocated.

## Fix

Remove all NCCL groups from the linear attention CP state chain. NCCL docs
state `ncclSend`/`ncclRecv` are "blocking for the GPU" — the host call
returns after enqueuing on the stream; the GPU stream waits for the peer's
matching op. No host blocking, no deadlock: the recv is always posted
before the send on the peer rank. The broadcast is a standard collective
outside a group.

Ungrouped P2P is safe here because the pattern is one-directional (rank N
recvs, rank N-1 sends) — not mutual send/recv, which is the only case
NCCL's group-fusion rule applies to ("if multiple ncclSend and ncclRecv
operations need to progress concurrently to complete, they must be fused").

Deletion-style: 1 insertion, 8 deletions. The dense attention ring prefill
keeps its groups (P2P-only, no collective — safe).

## Rule

Never group a P2P send/recv with a collective in the same NCCL group.
`ncclGroupEnd()` blocks until all ops in the group are enqueued — for a
collective, that means waiting for all ranks to post it. If a P2P op in
the same group is the prerequisite for the peer reaching that collective,
the P2P op is trapped inside the blocked `ncclGroupEnd()`, and the result
is a circular wait. P2P and collective get separate groups, or no group
at all — ungrouped P2P is GPU-blocking (stream waits for the peer's
matching op), not host-blocking, and is correct for one-directional
patterns.
