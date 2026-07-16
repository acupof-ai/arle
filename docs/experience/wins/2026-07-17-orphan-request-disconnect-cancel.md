# Orphan-request disconnect-cancel + quiesce hard gate

> Status: pending-remote — P4 baseline gates on the 8×H20 pod.

## Context

P4 baseline pod run (in-process serve + cc harness, K=4): cc child killed at
`--cc-timeout` → its in-flight `/v1/messages` SSE request orphaned server-side
and kept decoding → quiesce waited 240s+ then the round proceeded with
`active_requests=1` → the orphan's next decode step hit released KV state →
engine-thread panic (`qwen35.rs:1348 full_attn_kv present`) → zombie run,
`RUN_EXIT=1`.

## What Worked

Three-layer fix (`e0864760c` + `6ad2ba96d`):

1. **Local relay driver handles `CancelRequest`** — the coordinator already
   broadcast it on `InFlightGuard::drop` (2026-07-05 multiproc fix), but the
   single-process lane dropped it in the "unexpected envelope" arm.
2. **Engine loop cancels on dropped stream receiver** — a failed streamer send
   records the handle; the loop cancels it after the step (non-lockstep lane),
   and completions are delivered after control closures so a cancel on an idle
   engine still resolves tickets.
3. **Quiesce hard gate** — after cc children exit, remaining actives are
   orphans by definition: `cancel_all_requests` up front, then require
   `active_requests == 0`; bail after 60s instead of corrupting the engine.

Test: `infer-server dropped_stream_receiver_cancels_the_request` — drop the
streaming receiver, request finishes `Abort` instead of decoding to max_tokens.

## Rule

A request whose client is dead must be cancelled in the engine, not waited
out; a quiesce that can proceed with `active > 0` is a corruption path, not a
drain. Bench: pending-remote — verify on the next P4 baseline run (no engine
panic, quiesce drains ≤60s, RUN_EXIT=0).
