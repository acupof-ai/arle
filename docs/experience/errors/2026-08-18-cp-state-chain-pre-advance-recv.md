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
