# CP linear-attention state chain passed PRE-advance state — CUDA, 2026-08-18

## Context

After the P2P+collective group deadlock fix (same-day entry), TP=2 CP=2
still failed the needle ladder: len=300/446 exact, len=2000 miss (needle
lost), len=8000+ server hang. CP=1 control: 21/21 pass. Both failures
isolated to the CP path.

## Root Cause

The CP linear-attention state chain recv'd the previous rank's state
**before** the sender's advance, so the receiver got the pre-slice state
(identical to its own, from the previous chunk's broadcast). The sender's
post-advance send was too late — the receiver had already advanced.

```
rank 1: recv rank 0's state (PRE-advance, same as own)
both:   advance own slice
rank 0: send POST-advance state → too late, rank 1 already advanced
all:    broadcast rank 1's state = f(PRE-state, slice_1)  ← slice_0 lost
```

Every linear-attention layer lost the sender's slice contribution. Short
sequences (300/446) survived because dense attention (ring prefill)
compensates; at 2000 tokens the linear attention dominates and the
missing slice-0 state drops the needle at position 0.

## Fix

Sequential state chain — align before communication:

```
rank 0: advance(slice_0) → send POST-advance state
rank 1: recv → advance(slice_1) → send to rank 2 (CP>2)
...
last:   recv → advance
all:    broadcast last rank's post-state
```

Rank 0 advances first (starts from the slot's own state). Each subsequent
rank recvs the previous rank's POST-advance state before advancing. The
NCCL P2P matching ensures the recv completes when the sender posts the
send, which is after the sender's advance.

## Rule

In a recurrent state chain, the recv MUST land after the sender's advance.
Receiving the pre-advance state is a silent correctness bug — the state
agrees with the receiver's own (no NaN, no crash), but the sender's slice
contribution is missing. The bug only surfaces at sequence lengths where
the recurrent path dominates over dense attention.

## Follow-up (2026-08-18, pod verification)

The fix (365ec0c4f + borrow-checker scoping 78b01b9e0) built and ran on
pod, but **did not change len=2000 behavior** — the model still refuses
with "not relevant to the context", identical to the pre-fix run.
CP=1 control: 21/21 pass. The state chain was not the root cause of the
needle loss; NCCL P2P recv blocks on the stream until the sender posts
the send (which is after the sender's advance via `comm_waits_for_compute`),
so the old code already received the post-advance state.

len=8000+ crashes the server: lockstep coordinator stalls at tick #211
(min_acked=207, 4 ticks behind), tears down after 120s. Server log shows
36 `recurrent sidecar serialize: 73.4 MiB` entries (18 requests × 2)
across all 4 ranks, synchronized, immediately before the stall. The
serializations themselves are ~37 ms each and not the direct cause;
the stall is in the decode phase that follows.

Both bugs (len=2000 correctness, len=8000 liveness) are live and
unresolved. Root cause is elsewhere in the CP path — candidates:
ring prefill KV rotation, decode cross-CP all-gather, or the
linear-attention decode state usage.

## Follow-up 2 (2026-08-18, pod first-token isolation)

Pod experiment with `max_tokens=1` at len=2000 isolated the bug to the
**prefill path**: CP=2 first token `'I'` (wrong) vs CP=1 `'7'` (correct,
full output `'738291'`). FA3 route active, no scalar fallback.

**Root cause found**: `prefill_row_snapshotted` (the recurrent-state
snapshot path) splits the prefill at L*=1984 into [0,1984) + [1984,2000).
The 2D ring prefill attends ONLY to the current segment's rotating KV
buffers — it cannot read prior segments' pool KV. The tail segment's
last-token hidden state is blind to the prefix, so the needle at
position 0 is lost.

At len=300/446 the recurrent state's compressed representation is
sufficient to generate the correct first token despite the blind tail.
At len=2000 the recurrent state has washed out the needle, and the dense
attention cannot reach it.

**Fix**: skip `prefill_row_snapshotted` under 2D (`!self.two_d_engaged()`
guard in the prefill dispatch). The single-pass `prefill_row_paged_default`
covers the entire prompt in one ring pass.

**Remaining**: chunked prefill (len > 2048) has the same blind spot —
each chunk's ring prefill cannot see prior chunks' pool KV. Needs a
paged-attention + flash-decoding merge over the pool after the ring pass.
