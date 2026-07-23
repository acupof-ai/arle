# Relay hello-read timeout leaked into the steady-state reader → TP=4 c8+ serve teardown

> Status: Fixed (837b89d39), pod-confirmed 2026-07-24

## Context

A review-fix (LOW#18) added a read timeout to `accept_n` in
`infer-server/src/multiproc_relay.rs` to stop a silent peer from wedging the
single-threaded coordinator accept loop at boot. It shipped in commit
`104cef160`. Local `cargo test` + Mac CUDA typecheck + CI were all green; the H20
DSv4-Flash TP=4/EP=4 needle gate passed 15/15 and c1/c4 throughput was
perf-neutral. Then at **c8** the serve tore itself down:

```
[relay-coordinator] worker rank 3 completion reader failed: relay envelope length
1768628859 exceeds 64 MiB sanity cap — likely corrupted stream or version mismatch
[coordinator] tearing down the serve ... relay write header: Connection reset by peer
```

c8/c16 = 0 completions. A 1.77 GB length prefix = the framed-envelope reader lost
stream sync.

## Root Cause

`accept_n` set `stream.set_read_timeout(Some(remaining))` before reading the
worker hello (bounding a silent peer), then wrapped **the same stream, timeout
still set**, into the relay channel used for **all steady-state completion reads**
(`TcpChannel::new(stream)`). The timeout was never cleared.

Steady-state relay reads must block indefinitely — a completion can be far apart
between tokens. With the leftover timeout, under TP=4 concurrency ≥8 a completion
read timed out **mid-envelope** (after consuming a partial frame), so the next
`read_envelope` started mid-body and misread the following bytes as a length
prefix → the 1.77 GB garbage length → teardown. The champion binary had no read
timeout on the worker stream → steady-state reads blocked → no desync, which is
why c8 passed there.

The pod agent's first hypothesis was the sibling `104cef160` change (H2,
"un-serialize relay streaming" in the *local* relay pool). That was **wrong** —
H2 is `infer-server/src/lib.rs` `coordinator_local_router`, not on the multiproc
TP=4 path. Reading the actual failing code (`multiproc_relay.rs:726` timeout →
`:755` channel wrap, no clear in between) located the true cause. Textbook §0
case-as-fact: decode the symptom (framing desync), trace to the exact code, don't
trust the plausible-mechanism hypothesis.

## Fix

Clear the timeout to `None` (blocking) after the hello read, before the stream
enters the steady-state relay (`multiproc_relay.rs`, after the hello validation):

```rust
stream.set_read_timeout(None).context("worker stream clear read timeout after hello")?;
```

Pod-confirmed on `837b89d39` (DSv4-Flash TP=4): c8 48/48, c16 64/64 completions,
`TEARDOWN_RECURRED=no`, reproducible on a quiet box.

## Rule

A socket timeout set for ONE read (a boot handshake) must be cleared before the
socket is reused for reads with different blocking semantics — a leaked timeout
desyncs a framed stream, it doesn't just fail one read. And: a review-fix that
changes I/O timing on the **multiproc TP=N** relay is not covered by the
local/TP=1 regression tests — it needs a **TP=N runtime c-sweep**, not just
`cargo test` + typecheck. This one was invisible to CI and surfaced only on the
pod at c8. Land the timing-sensitive multiproc changes behind a pod c8/c16 gate.
