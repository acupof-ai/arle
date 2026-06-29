# Continuous-batching correctness: decode-first ordering, per-token SSE, relay worker pool

## Context

Three bugs fixed in one session, all in the serving and execution layer. Commits b4204940, fd7d057f, c2611d95.

## What Worked

### Fix 1 — Decode-first ordering in Mixed batch (b4204940)

`ForwardMode::Mixed` ran prefill sub-steps before the decode batch in both
`submit_multi_row()` (`crates/infer-cuda/src/executor.rs`) and
`submit_dflash_mixed_rows()` (`crates/infer-metal/src/executor.rs`). Every
new request admission stalled all active decode requests for the full prefill
duration. Swapping to decode-first — decode batch runs, then prefill sub-steps
per row — eliminated the stall. TTFT with 4 active decode requests became ≈
TTFT at c=1.

### Fix 2 — True per-token SSE streaming (fd7d057f)

`sse_response()` in `crates/infer-server/src/lib.rs` called `submit()` and
blocked until the entire generation completed, then emitted one response. At
~200 tokens × 9.9 ms/token the reported "TTFT" was actually 1847 ms —
total generation time. `serve_handle_relay_driver` now calls
`submit_streaming()` and emits one `RelayCompletionDelta` per token;
`coordinator.rs` drains the relay channel and sends SSE chunks as they
arrive. Measured: streaming TTFT **69 ms** vs non-streaming **480 ms** — 7×
improvement. Per-token interval confirmed at 8–9 ms.

### Fix 3 — Pre-spawned relay worker pool (c2611d95)

`serve_handle_relay_driver` (`crates/infer-server/src/lib.rs`) previously
called `std::thread::spawn` inside the `TickAdmissions` loop — one OS thread
per incoming request. Under c=1024 this burst to 1024 threads and triggered
the ELKEID kernel HIDS SIGKILL at ~150 s. Fixed by pre-spawning a fixed pool
of `max_live_requests` worker threads at driver startup with request dispatch
via `mpsc::channel`. Thread count is bounded and stable from startup.
Measured: 210 s c=128 stress test — server survived, thread count stable at
185–186, zero ELKEID kills. Previously died at ~150 s.

## Rule

In a continuous-batching executor, decode must run before prefill in a mixed
forward pass — prefill stalls decode for every admission otherwise. For
streaming correctness, the server path must use a token-channel API, not a
blocking wait; and thread creation must be bounded at startup, not unbounded
per request.
