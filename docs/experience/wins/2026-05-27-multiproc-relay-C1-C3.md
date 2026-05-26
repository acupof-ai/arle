# Phase B-1 commit C.1-C.3 — multiproc_relay land + production wire-in

## Goal

Build the cross-process control plane the multiproc-serve pivot needs.
SGLang/vLLM use torch.distributed for this; ARLE has no torch, so we
land a JSON-over-TCP relay. Phase B-0
([`./2026-05-27-multiproc-nccl-smoke-phase-B0.md`](./2026-05-27-multiproc-nccl-smoke-phase-B0.md))
already proved NCCL EnvBootstrap is cross-process-safe; this entry adds
the request-broadcast plane.

## What landed

| Commit | Subphase | What |
|---|---|---|
| `a4ec723a` | C.1 | `infer/src/multiproc_relay.rs` — RelayEnvelope tagged enum + RelayCoordinator + RelayWorker with length-prefixed JSON wire format. Unit tests for envelope serde + 2-process round-trip. |
| `681710e9` | C.2 | Two-phase bind/accept API: `RelayCoordinator::bind() -> PendingRelayCoordinator` (publish port to env before spawn) then `pending.accept(world_size, timeout)` (after spawn). New `multiproc_relay_smoke` bin verifying end-to-end with N forked children — PASS at world=2 and world=8 on Mac (no CUDA needed). |
| `6701bc9f` | C.3 | Wire RelayCoordinator into `async_main` (bind before workers, accept after, broadcast boot-ping envelope) and RelayWorker into `run_worker_mode` (connect, spawn relay-receiver thread, log envelopes, EOF on coord drop). |

## Run output — multiproc_relay_smoke (local Mac, no GPU)

```
$ ARLE_RELAY_SMOKE_WORLD_SIZE=8 ./target/release/multiproc_relay_smoke
[coordinator pid=45980] world_size=8 port=52849
[worker pid=45981 rank=1] connecting to 127.0.0.1:52849
[worker pid=45982 rank=2] connecting to 127.0.0.1:52849
...
[worker pid=45988 rank=7] connecting to 127.0.0.1:52849
[coordinator] all 7 workers connected
[coordinator] broadcast 5 cycles, dropping coord to signal EOF
[worker rank=1] coordinator EOF after 5 envelopes
[worker rank=2] coordinator EOF after 5 envelopes
...
[worker rank=7] coordinator EOF after 5 envelopes
[coordinator] rank 1..7 exited 0
[coordinator] multiproc_relay_smoke PASS (5 cycles, world=8)
EXIT=0
```

Validates: free-port pick, bind, child spawn via Command::current_exe()
+ env, all N-1 workers connect, envelope-per-cycle broadcast, EOF on
coord drop, clean worker exit. All without any CUDA / NCCL.

## What the boot path looks like today

```
async_main (coordinator process)
  ├─ build_cuda_worker_bootstrap → N entries
  ├─ if ARLE_MULTIPROC_SERVE=1:
  │     RelayCoordinator::bind() → PendingRelayCoordinator + port
  │     env::set_var(ARLE_COORDINATOR_RELAY_PORT, port)
  │     spawn_cuda_worker_processes(workers[1..]) — N-1 children
  │     pending.accept(N, 30s) → RelayCoordinator
  │     broadcast(boot-ping envelope)
  ├─ spawn_cuda_worker_group(workers[..1], DistributedShape{rank=0, ws=N})
  │     → rank-0 scheduler thread
  └─ tokio HTTP server

run_worker_mode (worker process, ARLE_WORKER_RANK=R>0)
  ├─ read env: WORLD_SIZE, INFER_CUDA_DEVICE, ARLE_WORKER_PARENT_FD,
  │            ARLE_COORDINATOR_RELAY_PORT
  ├─ spawn_cuda_worker_group(workers=[my], DistributedShape{rank=R, ws=N})
  │     → rank-R scheduler thread
  ├─ RelayWorker::connect(addr, 30s)
  ├─ spawn relay-receiver thread:
  │     loop {
  │       match relay.recv()? {
  │         Some(env) => log::debug!(...),  // C.4 plugs into scheduler request_rx
  │         None => return Ok(()),           // coordinator EOF
  │       }
  │     }
  └─ block on read(parent_pipe_fd) for shutdown signal
```

## What's NOT wired yet — C.4 brief

C.3 lands the boot-time integration and proves the relay round-trips
on the production path. **Real per-request fanout is C.4** and needs:

### 1. WireRequest type (~50 LOC, new file `infer/src/multiproc_relay_request.rs`)

Subset of `IncomingRequest` (`infer/src/scheduler/types.rs:755-788`) that
can cross JSON:
- `prompt: String`
- `prompt_tokens: Option<Vec<u32>>`
- `max_tokens: usize`
- `sampling: SamplingParams` (already Serialize-able)
- `stop: Option<Vec<String>>`
- `priority: RequestPriority`
- `session_id: Option<SessionId>`
- `request_id: u64` (new, assigned by coordinator)

NOT in WireRequest: `delta_tx`, `trace_context`, `distributed`,
`ingress_numa_node` — these are reconstructed worker-side with
worker-local primitives.

Add `RelayEnvelope::Request2 { wire: WireRequest }` variant (don't
break existing `Request` variant — append-only).

### 2. NCCL-backed DistributedTokenCoordinator (~150 LOC)

Today's `infer/src/scheduler/types.rs:56-133` uses in-process
Mutex/Condvar; can't cross processes. Replace internals with
`Arc<NcclGroup>`:

```rust
pub struct DistributedTokenCoordinator {
    nccl: Arc<NcclGroup>,
    world_size: usize,
}

fn synchronize_token(&self, rank: usize, _step: usize, local: u32) -> Result<u32> {
    let input = if rank == 0 { vec![local as i32] } else { vec![0i32] };
    let out = self.nccl.broadcast_i32(&input, 1, /*root=*/ 0)?;
    Ok(out[0] as u32)
}
```

(`broadcast_i32` already exists, committed in `98afea39`.)

Plumbing: each rank's `Scheduler::with_config`
(`infer/src/scheduler/cuda/core/construction.rs`) needs the `Arc<NcclGroup>`
from `model.layer_communicator.ep_nccl`. Construct one coordinator per
scheduler boot, hold inside scheduler struct. Drop the per-request
`DistributedTokenCoordinator::new(world_size)` call at
`request_handle.rs:338`.

Single-process N-thread mode also benefits: each thread has its own
NCCL group; broadcast_i32 syncs threads correctly. Eliminates the
parallel Mutex+Condvar code path.

### 3. Coordinator-side request fanout (~100 LOC)

In `request_handle.rs:323-359` `DistributedSchedulerGroup::submit`:

```rust
// Existing: submit to local rank-0 scheduler.
let permit_0 = worker.handle.reserve_submission()?;
permit_0.submit(rank_0_req)?;

// NEW (C.4): if multiproc-serve relay is active, broadcast to workers.
if let Some(relay) = self.relay_coord.as_ref() {
    let wire = WireRequest::from_incoming(&rank_0_req);
    relay.lock().unwrap().broadcast(&RelayEnvelope::Request2 { wire })?;
}
```

Add `relay_coord: Option<Arc<Mutex<RelayCoordinator>>>` to
`DistributedSchedulerGroup`; populated from async_main when
ARLE_MULTIPROC_SERVE=1.

Drop the `submission_lock: Mutex<()>` field
(`request_handle.rs:105`) — NCCL collectives are FIFO-ordered per-comm.

### 4. Worker-side request injection (~100 LOC)

In `run_worker_mode`'s relay-receiver thread, replace the `log::debug!`
with:

```rust
Some(RelayEnvelope::Request2 { wire }) => {
    let (sink_tx, sink_rx) = tokio::sync::mpsc::unbounded_channel();
    spawn_rank_delta_drain(sink_rx);  // discard worker's deltas
    let req = IncomingRequest::from_wire(
        wire,
        sink_tx,
        DistributedRequestCoordination::new(rank, world_size, nccl.clone())?,
    );
    let permit = scheduler_handle.reserve_submission()?;
    permit.submit(req)?;
}
```

Worker's relay-receiver thread needs `Arc<SchedulerHandle>` from the
`StartedCudaWorker` returned by `spawn_cuda_worker_group`, and
`Arc<NcclGroup>` from the scheduler's model. Plumb both at thread spawn.

### 5. Smoke test (~80 LOC, new `infer/tests/multiproc_serve_smoke.rs`)

`#[ignore]` (needs 2 GPUs), behind `cuda + nccl` features:
- Spawn `target/release/infer serve` with `ARLE_MULTIPROC_SERVE=1
  ARLE_MULTIPROC_ALLOW_DEADLOCK=` unset, `INFER_CUDA_DEVICES=0,1`,
  tiny DSv4 model.
- Wait for HTTP `:8000/v1/health`.
- POST one greedy `/v1/chat/completions`, `max_tokens=16`,
  `temperature=0`.
- Capture response.content.
- Re-run with `ARLE_MULTIPROC_SERVE` UNSET (single-process 2-thread).
- Assert byte-identical content.

PASS gate for B-1.

### 6. Drop deadlock guard (~5 LOC)

Once C.4 lands and the multiproc_serve_smoke passes, delete the
`ARLE_MULTIPROC_ALLOW_DEADLOCK` panic from `async_main`.

### Estimated total for C.4

~485 LOC across 5 file edits + 1 new file. 1 commit per step (C.4.1
through C.4.6) for bisect safety.

## Bench-exempt notes

All C.1-C.3 commits are env-gated:
- `ARLE_MULTIPROC_SERVE` unset → no behavior change (no env var read,
  no module touched).
- `ARLE_MULTIPROC_SERVE=1` set → panics at deadlock guard
  (`ARLE_MULTIPROC_ALLOW_DEADLOCK=1` bypasses).

Verified: cargo check on Mac (cuda + no-cuda features) clean across
all 3 commits. `multiproc_relay_smoke` end-to-end PASS at world=2 + 8.

## Rule

Cross-process control-plane shipping doesn't need torch.distributed.
JSON-over-TCP plus length-prefixing is ~300 LOC of self-contained code
that works on any host, validates as a standalone smoke before touching
the production HTTP path, and is debuggable with `nc` / tcpdump.
Phase B-0 proved NCCL EnvBootstrap works cross-process; this entry
proves the same for our higher-level request protocol — the data
plane (NCCL) and control plane (TCP) are independently testable.
