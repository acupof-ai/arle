# Dead Ctrl-C in the agent REPL — no per-request cancel token + thread::scope join-wedge

## Context

In the in-process `arle` agent REPL, Ctrl-C during generation did nothing —
the REPL hung until the model produced its full `max_tokens` (with
`--max-tokens` auto = `max_position_embeddings`, effectively forever). User
report: "不能 ctrl c 退出".

## Root Cause (evidence, file:line)

Two compounding defects, both confirmed by reading the path:

1. **No cancel signal reaches the generation loop.** `IncomingRequest`
   (`infer/src/scheduler/types.rs:825`) had **no cancel field**. The Metal
   runtime only stops a request when its `delta_tx` is dropped
   (`runtime.rs:72` `delta_closed()`), checked at tick-top / prefill-chunk /
   decode boundaries. The agent's SIGINT `AtomicBool` (`crates/agent/src/lib.rs`)
   was read **only by the agent's own poll loop** (`lib.rs:1275`), never
   threaded into the request or the runtime.

2. **`thread::scope` join-wedge.** `complete_with_optional_cancel`
   (`crates/agent/src/lib.rs:1246`) runs `engine.complete_stream` on a worker
   inside `std::thread::scope`. On SIGINT the agent drops its receiver and
   breaks, but the worker is parked in `inner_rx.blocking_recv()` /
   `request_handle_engine.rs:141` and its drain loop (`:145`), which only ends
   when the **runtime finishes the entire generation** — and the runtime never
   saw a cancel. So `scope`'s implicit `worker.join()` wedged the REPL thread
   until `max_tokens` were produced.

Net: cancel was a client-side flag the producer never observed; the consumer
couldn't unwedge until the producer naturally stopped.

## Fix

Per-request cooperative cancel `Arc<AtomicBool>`, threaded
repl → agent API (`Arc<AtomicBool>`, not `&AtomicBool`) →
`CompletionRequest.cancel` → `IncomingRequest.cancel` →
runtime per-request state. New `cancel_requested()` =
`delta_closed() || cancel.load(Relaxed)`, used at the runtime reap / prefill /
decode pre-checks (`backend/metal/runtime.rs`). On Ctrl-C the runtime stops
generation → drops `delta_tx` → the worker's `complete_stream` returns →
`join()` unwedges → REPL returns. HTTP path unchanged (`cancel: None`, still
uses socket-close).

Verified: SIGINT mid-generation halts token streaming within ~1 s (0 B growth
over the following 2 s; stopped at 256 of 4000 tokens).

## Rule

A cooperative cancel flag must reach the **generation producer** (scheduler /
runtime loop), not just the consumer's poll loop. A `thread::scope` worker that
drains a channel wedges its `join()` until the producer stops — so the producer
must honor cancel, or the consumer's "I gave up" is invisible. When adding a
cancel: thread it through the request contract to the loop that calls the FFI,
and verify behaviorally (streaming must stop), not just that it compiles.
