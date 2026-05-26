# Phase B-1 scaffolding land — launcher + worker scheduler boot complete

## Goal

Land the launcher + worker-mode scaffolding for ARLE's multiproc-serve
pivot
([`../../plans/2026-05-27-multiproc-serve-pivot.md`](../../plans/2026-05-27-multiproc-serve-pivot.md))
in the smallest commit boundaries that compose into a working
multi-process serving binary. Phase B-0 already proved NCCL
EnvBootstrap is cross-process-safe at world_size=8
([`./2026-05-27-multiproc-nccl-smoke-phase-B0.md`](./2026-05-27-multiproc-nccl-smoke-phase-B0.md));
this entry covers the actual integration into `arle serve`.

## What landed (5 commits)

| Commit | Subphase | What | Verification |
|---|---|---|---|
| `98afea39` | B-1 A | `NcclGroup::broadcast_i32` + `broadcast_bytes` helpers | cargo check clean, mirrors `broadcast_f32` |
| `2e2686f4` | B-1 B.1 | `ARLE_WORKER_RANK` env detection in `fn main()` → branches to `run_worker_mode()` | cargo check; default unset path unchanged |
| `80d45e3b` | B-1 B.2 | `spawn_cuda_worker_processes` + `WorkerChildren` RAII; coordinator forks N-1 children via `Command::current_exe()` with parent pipe handshake | cargo check; coordinator-side gate `ARLE_MULTIPROC_SERVE=1` |
| `0622ef7c` | B-1 B.3 | `DistributedShape` override threading + `spawn_cuda_worker_group` refactor; worker mode actually boots a scheduler at its rank | cargo check; `started_by_local` indexed by per-process slot, not distributed rank |
| `2493b297` | B-1 B.4 | Deadlock guard: `ARLE_MULTIPROC_SERVE=1` panics until commit C lands | cargo check; `ARLE_MULTIPROC_ALLOW_DEADLOCK=1` bypass for spawn-only smoke |

## Architecture in the tree

```
              arle serve (no ARLE_WORKER_RANK)
              ──────────────────────────────────
              fn main()
                │ ARLE_WORKER_RANK unset → coordinator
                ↓
              tokio runtime + async_main
                │
                ├─ build_cuda_worker_bootstrap (N entries)
                ├─ if ARLE_MULTIPROC_SERVE=1 && N > 1:
                │      spawn_cuda_worker_processes(workers[1..])
                │        for each rank R in 1..N:
                │          pipe(2) → child_read fd + parent_write fd
                │          Command::current_exe()
                │            .env(ARLE_WORKER_RANK=R,
                │                 ARLE_WORKER_PARENT_FD=child_read,
                │                 INFER_CUDA_DEVICE=ordinal_for_R,
                │                 inherits MASTER_*, WORLD_SIZE)
                │            .spawn()
                │      worker_bootstrap_for_coord = workers[..1]
                │      distributed_shape_for_coord = Some(rank=0, ws=N)
                │   else:
                │      bootstrap = full N (existing N-thread path)
                │
                └─ spawn_cuda_worker_group(coord_bootstrap, distributed_shape)
                     → rank-0 scheduler thread (in coordinator process)

              arle serve (ARLE_WORKER_RANK=R, R > 0)
              ──────────────────────────────────────
              fn main()
                │ ARLE_WORKER_RANK = R → run_worker_mode
                ↓
              no tokio, no HTTP, no tokenizer
                │
                ├─ read WORLD_SIZE, INFER_CUDA_DEVICE, ARLE_WORKER_PARENT_FD
                ├─ resolve_model_path + detect_model_type
                ├─ build single-entry CudaWorkerBootstrap
                ├─ spawn_cuda_worker_group(
                │      workers=[my_bootstrap],
                │      distributed_shape=Some(rank=R, world_size=N))
                │   → rank-R scheduler thread (in worker process)
                │
                └─ block on read(parent_pipe_fd):
                     Ok(0)/Err → coordinator died → shutdown_started_workers → exit 0
```

NCCL EnvBootstrap joins all N processes at the rank-aware
`ncclCommInitRank` calls inside each rank's `DeepseekModel`
construction; the TCP rendezvous over `MASTER_ADDR:MASTER_PORT` doesn't
care whether ranks are threads or processes (proved phase B-0).

## What's NOT wired yet — phase B-1 commit C

Workers boot with empty `request_rx` queues. The HTTP handler on rank-0
still goes through `DistributedSchedulerGroup` which only sees its own
in-process worker (length 1). So:

- Requests reach rank-0's scheduler only.
- Rank-0's forward issues TP/EP NCCL collectives.
- Worker ranks 1..N-1 never enter forward (no requests).
- Rank-0's collective blocks waiting for worker ranks → deadlock.

Setting `ARLE_MULTIPROC_SERVE=1` therefore panics today (commit
`2493b297` guard). The pattern is finalized, the launcher is correct,
but the **cross-process request relay needs separate engineering**:

### Brief for phase B-1 commit C

**Goal**: rank-0's HTTP handler must deliver each `IncomingRequest` to
all N rank schedulers (1 in-process + N-1 in worker processes).

**Recommended approach (TCP for control plane, NCCL for token sync)**:

1. **Coordinator-side TCP relay**:
   - New module `infer/src/multiproc_relay/`.
   - `Coordinator { listener: TcpListener, workers: Vec<TcpStream> }`.
   - At boot, opens a TCP listener on a coordinator-allocated port
     (env `ARLE_COORDINATOR_REQUEST_PORT`). Spawns N-1 workers via
     existing `spawn_cuda_worker_processes`, sets the env var so
     workers can connect.
   - Workers connect on boot; coordinator accepts N-1 connections.
   - On each `IncomingRequest` submitted by HTTP handler:
     a. Serialize via bincode into `RelayEnvelope { token_ids,
        sampling_params, request_id, ... }`.
     b. Write to all N-1 worker streams.
   - Wire into `DistributedSchedulerGroup` as an alternative submission
     path when running in multiproc-serve mode.

2. **Worker-side TCP receiver**:
   - In `run_worker_mode`, after `spawn_cuda_worker_group` succeeds,
     spawn a thread that connects to coordinator's TCP port.
   - Loop: read length-prefixed bincode → deserialize → reconstruct
     `IncomingRequest` with a worker-local `DistributedRequestCoordination`
     (NCCL-backed; see step 3) → push to local scheduler's `request_rx`.

3. **NCCL-backed `DistributedTokenCoordinator`**:
   - Today's in-process Mutex/Condvar version
     (`infer/src/scheduler/types.rs:56-133`) is broken across processes.
   - Rewrite with `Arc<NcclGroup>` and `synchronize_token` =
     `nccl.broadcast_i32(&[local], 1, root=0)` (broadcast_i32 helper
     landed in commit A `98afea39`).
   - Plumb through `Scheduler::with_config` → `bootstrap.rs:553-564`
     so each rank's scheduler holds its own NCCL-backed coordinator.
   - Single-process N-thread mode also works with this change (each
     thread already has its own `Arc<NcclGroup>`).

4. **Failure mode**: worker TCP disconnect → rank-0 receives `EPIPE` on
   write → rank-0 panics (matches SGLang's "any worker death kills the
   server" failure mode). No partial-rank recovery — out of scope.

5. **Estimated**: ~400 LOC across new module + 2 file edits.
   Bisect-friendly split:
   - C.1 (~150 LOC): NCCL-backed `DistributedTokenCoordinator` + plumb
     `Arc<NcclGroup>` through `Scheduler::with_config`. Single-process
     mode keeps working.
   - C.2 (~200 LOC): TCP coordinator + worker relay modules.
   - C.3 (~50 LOC): wire into `request_handle.rs` `DistributedScheduler
     Group` + drop the `submission_lock: Mutex<()>` (collectives are
     now FIFO-ordered by NCCL).
   - C.4: drop the deadlock guard, set up a 2-rank `multiproc_serve_
     smoke.rs` test (phase B-1 commit D).

## Smoke-test plan (phase B-1 commit D)

After commit C lands:
- New test `infer/tests/multiproc_serve_smoke.rs` (`#[ignore]` since
  needs 2 GPUs).
- Launches `target/release/infer serve` with `ARLE_MULTIPROC_SERVE=1`,
  posts one greedy non-streaming `/v1/chat/completions`, captures
  response bytes.
- Asserts byte-identical to a baseline run with `ARLE_MULTIPROC_SERVE`
  unset (single-process 2-thread mode) — proves multiproc-serve is
  numerically transparent.
- Phase B-1 PASS gate per the design doc.

## Bench-exempt notes

All B-1 commits are env-gated:
- `ARLE_WORKER_RANK` unset → no behavior change.
- `ARLE_MULTIPROC_SERVE` unset → no behavior change.
- `ARLE_MULTIPROC_SERVE=1` set today → panics at deadlock guard.

The phase B-0 binary (`infer/src/bin/multiproc_nccl_smoke.rs`) provides
the actual exercised cross-process path. Verified 8-rank PASS on the
pod.

## Rule

When a foundational architectural pivot has many file-touching commits
but only the final one delivers user-visible behavior, gate the
incremental commits behind a panicking env-var so users can't
accidentally trip the half-state. Don't trust "default-off" alone —
explicit gate prevents confusion when someone copies an old script.
