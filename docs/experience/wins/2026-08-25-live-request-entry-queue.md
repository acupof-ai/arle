# Entry queue for over-capacity live requests — infer-server, 2026-08-25

> Status: pending-remote

## Goal

Over-capacity submits queue at entry instead of failing "server is busy"
(#208). Pod gate: fault injection (one group sleeps 500ms/step, the other
group's step wall deviates <5% from solo baseline) + both groups pass needle
independently + over-capacity requests queue under a live multi-group serve.

## What landed

`LiveRequestGate` (condvar queue) replaces the atomic CAS hard-error in
`ServeHandle::acquire_live_request`. Over-capacity submits block until a slot
frees or shutdown wakes them. The relay submitter thread blocks safely (the
ack pump runs on a separate thread; relay workers release slots at stream
end). `too_many_requests` 429 mapping deleted with the error string.

Unit tests (cpu lane, `cargo test -p infer-server live_gate`): acquire up to
capacity, over-capacity blocks until release, shutdown wakes waiters.

## Parameters

```bash
# pending-remote: multi-group CUDA serve, 2x TP group
# - over-capacity queue: submit max_live+1 concurrent requests, assert no 429
# - fault injection: group 0 sleeps 500ms/step, group 1 step wall <5% deviation
# - needle x3 per group
```

- Baseline: `959f50a92` (queue absent, "server is busy" 429)
- Treatment: this commit (queue at entry)
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod, 2x TP group (pending-remote)
- Model / dtype: ThinkingCap-27B-FP8 (pending-remote)

## Rule

An entry queue at the serve layer is safe when the blocking thread is
dedicated to submission: the ack path and the slot-releasing workers must not
sit behind the same queue.
