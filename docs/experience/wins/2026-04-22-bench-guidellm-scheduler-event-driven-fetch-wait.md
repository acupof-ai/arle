# Scheduler Event-Driven Fetch Wait

## Context

The CUDA scheduler still used a `recv_timeout(2ms)` polling loop when all
active requests were parked in `Phase::WaitingFetch`. That kept one CPU core
warm even when the only useful wakeups were new requests or coordinator fetch
events.

## What Worked

- Added an explicit scheduler wakeup channel alongside `request_rx`.
- `SchedulerHandle::submit()` now pings the wakeup channel after enqueueing.
- The CUDA scheduler blocks on `coordinator_events` or the wakeup channel when
  it is fetch-wait bound instead of polling every 2ms.
- Fully idle shutdown still exits cleanly once all scheduler handles drop.

## Rule

When the CUDA scheduler has no runnable prefill/decode work, it should sleep on
real wake sources rather than poll with short timeouts.

## Status

`pending-remote` — local type/test validation only. CUDA before/after bench
still needs a remote `scripts/bench_guidellm.sh` snapshot.
