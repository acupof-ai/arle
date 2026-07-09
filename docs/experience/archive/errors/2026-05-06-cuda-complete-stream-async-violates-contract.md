# 2026-05-06 · CUDA `complete_stream` returned before deltas arrived (async submit violated blocking contract)

## Context
`cargo test --release -p infer --features cuda --test e2e` failed at Phase 3
(stream/non-stream consistency) with an empty streamed string:

```
left:  "<full non-streamed text>"
right: ""
stream/non-stream text mismatch for: Tell me a story
```

The Phase 3 driver calls `engine.complete_stream(...)` then drains via
`try_recv` immediately afterwards (greedy → deterministic, so the two
runs should match byte-for-byte). The drain saw no deltas because the
CUDA engine returned before the scheduler had even admitted Request N+1
(log timeline: `Request N done` → panic → `Request N+1 → slot 0`).

## Root cause
Two separate `InferenceEngine::complete_stream` impls, two contracts:

- `BackendInferenceEngine<Metal|Cpu>` (`backend_engine.rs`) drives the
  backend's `generate_stream` callback synchronously on the calling
  thread. Returns only after the final delta has been pushed to `tx`.
- `RequestHandleInferenceEngine` (CUDA path, `request_handle_engine.rs`)
  called `self.submit_request(req, tx)` and returned immediately,
  handing `tx` to the scheduler thread. By the time the test drained
  `rx`, the scheduler hadn't run a single decode step yet.

Internal callers (e.g. `agent::complete_with_optional_cancel`) survive
because they spawn a worker and poll `rx` in a spin-loop; the e2e Phase 3
"complete_stream then `try_recv`" pattern doesn't, and that pattern
matches the assumption baked into `complete()` and the
backend tests `backend_complete_stream_emits_all_chunks_and_finish_marker`
/ `backend_complete_stream_short_circuits_when_rx_dropped`.

## Fix
`infer/src/server_engine/request_handle_engine.rs::complete_stream`
mirrors the metal/cpu blocking contract: forward through an internal
channel, `blocking_recv` until the finish-marker delta arrives, and
short-circuit silently when the consumer drops `tx`.

```rust
let (inner_tx, mut inner_rx) = tokio::sync::mpsc::unbounded_channel();
self.submit_request(req, inner_tx)?;
while let Some(delta) = inner_rx.blocking_recv() {
    let finished = delta.finish_reason.is_some();
    if tx.send(delta).is_err() {
        while inner_rx.blocking_recv().is_some() {}
        return Ok(());
    }
    if finished { break; }
}
Ok(())
```

After this fix, e2e Phase 3 receives all deltas; the residual mismatch
between non-stream and stream output is the pre-existing batched-decode
divergence tracked in
[`2026-04-13-batched-decode-high-concurrency.md`](2026-04-13-batched-decode-high-concurrency.md)
(stream-blocking is fixed; greedy reproducibility across two
back-to-back calls on the same engine is the open issue).

## Rule
- Backend `complete_stream` is a **blocking** contract: it returns
  only after `tx` has received the finish-marker delta (or the consumer
  dropped `rx`). Any new backend variant must hold this — async-submit
  paths route through an internal channel + `blocking_recv` forwarder.
- Tests/agents that call `complete_stream` can rely on `try_recv` working
  the moment the call returns; they do not need to spin-wait for the
  first delta.
