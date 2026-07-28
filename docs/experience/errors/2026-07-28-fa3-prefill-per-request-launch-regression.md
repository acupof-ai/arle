# Widening FA3 to prefill cost 51% of TTFT at c=8

## Context

Fixing the spec-verify gate meant replacing `meta.seq_len == 1` with a
predicate that also admitted 17-row verify forwards. I widened it all the way —
FA3 for **every** query length, prefill chunks included — on the reasoning that
one predicate is lower-entropy than two and prefill might get faster too.

At c=1 it looked free: prefill 4257 vs 4352 tok/s, inside the ±3% drift band.
The c=8 point is where it showed:

| c=8, no-spec | old binary | FA3-everywhere |
|---|---:|---:|
| TTFT p50 warm | 12.07 s | **18.23 s** (+51%) |
| prefill tok/s | 4313 | 3784 (−12%) |
| decode mean ITL | 850.6 ms | 947.7 (+11%) |
| wall | 1788 s | 2374 s (+33%) |

c=16 never finished. Both arms sat at ~105% CPU with the GPU at 0–11% and
`/v1/stats` timing out, an hour past the point where the previous binary had
completed.

## Root Cause

FA3 zeroes the page stride when `seqused_k` is set, so a ragged batch cannot
share one launch — the call site loops one launch per request. That is free at
decode/verify shapes, where the kernel is KV-bandwidth-bound and the launch
disappears into the memory time. A prefill chunk is compute-bound and already
fills the SMs, so splitting a 16-row batch into 16 launches per layer — 256 per
step across the 16 full-attention layers — buys nothing and pays scheduling for
it. The TileLang paged prefill kernel takes the whole ragged batch in one
launch.

I read the c=1 wash as "prefill is unaffected" when it only showed "at batch 1
there is nothing to batch". The per-request loop has no cost to expose until
`meta.batch > 1`.

## Fix

`FA3_MAX_QLEN = 64` (`qwen35.rs`). Decode (1 row) and spec verify (block+1)
take FA3; anything longer keeps the TileLang paged kernel. Split-KV is
unconditional inside the gate — every shape that reaches it is short. The
constant separates two populations two orders of magnitude apart (17 vs 2048),
so its exact value is not tuned and does not need to be.

## Rule

**A wash at c=1 is not evidence about a batched path.** Batch-1 measurement
cannot see per-request overhead, by construction — it is the one point where a
per-request loop and a batched launch do the same amount of work. Any change
that turns one launch into `batch` launches must be measured at the concurrency
it will run at, and the metric that exposes it is the phase it touches (TTFT
here), not the wall clock, which averaged the +51% down to something I first
mistook for box contention.

The tell was there in the split: prefill degraded 4× more than decode.
Contention slows both. Look at which phase moved before reaching for an
external explanation. See [[feedback_shared_box_sigkill_needs_pid_trace_before_root_cause]]
for the same failure mode in the other direction.
