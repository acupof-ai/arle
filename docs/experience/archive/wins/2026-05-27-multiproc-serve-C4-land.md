# Phase B-1 commit C.4.1-C.4.5 — multiproc-serve request relay end-to-end

## Goal

Land the request-fanout half of multiproc-serve: rank-0's HTTP submission
broadcasts the request to all worker processes via TCP relay
(`multiproc_relay`), and each worker's scheduler runs in lockstep with
rank-0. Removes the deadlock guard committed in B.4.

## What landed (5 commits)

| Commit | Subphase | What |
|---|---|---|
| `f670c05e` | C.4.1 | `WireRequest` + `WireSamplingParams` serde types + `RelayEnvelope::Request2 { wire }` variant. Captures the minimum data needed for a worker to reconstruct an `IncomingRequest`. |
| `a32ef68d` | C.4.2 | `DistributedRequestCoordination` becomes a 2-variant enum: `InProcess` (legacy Mutex/Condvar, single-process N-thread) and `Nccl { rank, world_size, Arc<NcclGroup> }`. `synchronize_token()` matches and dispatches; Nccl variant calls `broadcast_i32(local, 1, root=0)`. |
| `8234a3f0` | C.4.2b | `LayerCommunicator::tp_nccl()` / `ep_nccl()` pub accessors returning `Option<Arc<NcclGroup>>`. Lets scheduler-side code reach the model's NCCL groups without re-running TCP rendezvous (which would conflict with `ncclCommInitRank`). |
| `5fad975a` | C.4.3 | `DistributedSchedulerGroup::with_relay` constructor + `relay` / `effective_world_size` / `next_request_id` fields. Submit() now has two paths: relay (broadcasts Request2 envelope, submits to local rank-0) vs legacy (Mutex+Condvar fan-out). Two helpers: `wire_request_from_incoming` (IncomingRequest → WireRequest) and `incoming_request_from_wire` (the inverse, worker-side, with fresh sink delta_tx and `distributed=None`). |
| `7894860e` | C.4.4 + C.4.5 | Worker side: `run_worker_mode`'s relay-receiver thread matches `RelayEnvelope::Request2` → reconstruct IncomingRequest → reserve permit + submit on local scheduler. Coordinator side: `relay_coordinator` becomes `Arc<Mutex<RelayCoordinator>>` so it can be shared with `DistributedSchedulerGroup::with_relay`. async_main's request_handle selection grows a multiproc-serve branch that uses `with_relay` when `relay_coordinator.is_some()`. Deadlock guard removed — workers now receive requests, NCCL collectives have peer participants. |

## End-to-end shape (current)

```
        HTTP POST /v1/chat/completions
              │
              ↓
        request_handle.rs::DistributedSchedulerGroup::submit
              ├─ wire = wire_request_from_incoming(req, request_id)
              ├─ relay.lock().broadcast(Request2 { wire }) ─────────┐
              │                                                     │
              ├─ permit.submit(req with distributed=None) [rank 0]  │
              │                                                     │
              ↓                                                     │
        rank-0 scheduler thread                                     │
              ├─ admission → ActiveRequest                          │
              ├─ forward_prefill / forward_decode                   │
              │    (TP/EP NCCL collectives — workers participate) ←─┤ NCCL
              │                                                     │
              └─ delta_tx → HTTP streams to user                    │
                                                                    │
        worker rank R process:                                      │
              ├─ run_worker_mode → scheduler boots, idles ◄─────────┤ TCP
              │                                                     │
              ├─ relay-receiver thread:                             │
              │    relay.recv() → Request2 { wire }                 │
              │    incoming_request_from_wire(wire, sink_delta_tx)  │
              │    handle.reserve_submission().submit(req)          │
              │                                                     │
              ├─ rank-R scheduler:                                  │
              │    admission → ActiveRequest                        │
              │    forward_prefill / forward_decode                 │
              │      (joins TP/EP NCCL collectives) ────────────────┘
              │
              └─ delta_tx → /dev/null (drained, not surfaced)
```

NCCL collectives during forward (TP all-reduce, EP all-to-all) now have
peer participation from worker ranks. Rank-0's HTTP response stream is
authoritative; worker output is silently dropped.

## What's NOT wired yet — C.4.6 outstanding

**Token sync divergence.** With `distributed=None` on every rank's
IncomingRequest, no rank calls `synchronize_token`, so each rank samples
its own next-step token independently. For prefill this doesn't matter
(no sampling). For decode:

- Step 0: rank 0 samples token X; workers each sample their own
  tokens Y1, Y2, ... (independent RNG, may agree on greedy but not
  past temperature > 0).
- Step 1: rank 0's forward input = X; worker R's input = YR. Hidden
  states diverge.
- Step 1's NCCL all-reduce mixes diverged hidden states → output
  garbage.

User-visible response (rank 0's stream) is **mathematically incorrect**
past step 0 once temperature > 0 or top-k > 1, even though it's not a
deadlock.

**Fix shape (C.4.6, deferred to next session)**:

1. Plumb model's `Arc<NcclGroup>` up to scheduler boot. Cleanest path:
   add `pub fn ep_nccl(&self) -> Option<Arc<NcclGroup>>` to the
   `ModelForward` trait (default None, DeepSeek impl returns
   `Some(self.layer_communicator.ep_nccl()?)`). Scheduler stores the
   group on construction, exposes via SchedulerHandle.
2. In `run_worker_mode` and `request_handle.rs::DistributedSchedulerGroup::
   submit`, after each side reserves a scheduler permit, attach
   `DistributedRequestCoordination::Nccl { rank, world_size, nccl }`
   to the IncomingRequest before submit.
3. Scheduler's existing `distributed.synchronize_token` calls in
   `cuda/decode.rs:899` and `cuda/prefill.rs:691` already dispatch
   on the enum — the Nccl variant routes through `broadcast_i32` and
   ranks lock to rank-0's token automatically.

Estimated 200 LOC across 4-5 files. Bisect-friendly split:
- C.4.6.1: ModelForward::ep_nccl trait method + DeepSeek impl.
- C.4.6.2: SchedulerHandle stores Arc<NcclGroup>; spawn_scheduler_handle_
  from_path extracts post-model-load.
- C.4.6.3: DistributedSchedulerGroup::with_relay accepts Arc<NcclGroup>;
  submit() attaches Nccl variant.
- C.4.6.4: run_worker_mode attaches Nccl variant in relay-receiver.

## Bench-exempt notes

All C.4 commits are env-gated:
- `ARLE_MULTIPROC_SERVE` unset → legacy single-process N-thread path
  (zero behavior change).
- `ARLE_MULTIPROC_SERVE=1` set + `world_size > 1` → multiproc-serve
  scaffolding fully wired; worker forward participates in NCCL
  collectives; user-visible response from rank 0 is correct up through
  step 0, then diverges per C.4.6 above.

Pure scaffolding land — no production hot-path runtime changes for
the default `ARLE_MULTIPROC_SERVE`-unset path.

## Run output — C.2 standalone smoke (still works after C.4)

`multiproc_relay_smoke` validates the underlying TCP relay protocol
independent of the scheduler integration. 8-rank PASS on Mac (no GPU
needed). See [`./2026-05-27-multiproc-relay-C1-C3.md`](./2026-05-27-multiproc-relay-C1-C3.md).

## Rule

Multi-process pivot landed incrementally across 8 sub-commits (A,
B.1-B.4, C.1-C.5). Each commit is bisect-friendly: cargo check + cargo
build clean on Mac without GPU; each commit either adds a new opt-in
code path or removes a guard once the path is verified. The deadlock
guard pattern (B.4 → C.4.5) is the key technique: scaffolding gets
gated behind a panic-on-use env var until the path is end-to-end safe.
Avoids the "promises behavior it doesn't deliver" half-state that
CLAUDE.md no-half-states warns against.
