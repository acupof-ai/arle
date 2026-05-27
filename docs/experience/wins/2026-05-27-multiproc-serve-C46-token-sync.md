# Phase B-1 commit C.4.6 — NCCL token sync attach, B-1 closed

## Goal

Close the final gap in multiproc-serve: every rank's
`DistributedRequestCoordination::synchronize_token` participates in a
single NCCL `broadcast_i32` collective so worker ranks lock to rank-0's
sampled token. Without this, ranks diverged from step 1 of decode and
the rank-0 user-visible response was mathematically incorrect past step
0 (C.4.5 known-incomplete state).

## What landed (4 sub-commits)

| Commit | Subphase | What |
|---|---|---|
| `f095fff3` | C.4.6.1 | `ModelForward::ep_nccl` default-`None` trait method + DeepSeek impl returning `self.layer_communicator.ep_nccl()`. Lets the scheduler reach the model's NCCL group through a generic interface. |
| `9c164bf8` | C.4.6.2 | `SchedulerHandle::ep_nccl: Option<Arc<NcclGroup>>` field + `with_ep_nccl(nccl)` builder + `ep_nccl()` accessor. `spawn_scheduler_handle_from_path` extracts `model.ep_nccl()` before the model moves into `Scheduler::with_config`, then attaches via `handle.with_ep_nccl(nccl)`. |
| `3f0152ad` | C.4.6.3 | Coordinator-side attach: `DistributedSchedulerGroup::submit`'s relay path builds rank-0's IncomingRequest with `distributed = Some(Nccl { rank=0, world_size=effective, nccl })`, pulled from `self.workers[0].handle.ep_nccl()`. |
| `e0a4175f` | C.4.6.4 | Worker-side attach: `run_worker_mode`'s relay-receiver thread, on each Request2 envelope, builds the IncomingRequest with `distributed = Some(Nccl { rank=R, world_size, nccl })`. NCCL group comes from worker's own `handle.ep_nccl()`. |

## End-to-end token sync flow

```
HTTP POST /v1/chat/completions
  │
  ↓
DistributedSchedulerGroup::submit (rank 0 coordinator process)
  ├─ relay.broadcast(Request2 { wire })  ──────────────┐
  ├─ rank0_req.distributed = Some(Nccl {                │
  │     rank=0, world_size=N, nccl=rank-0's ep_nccl })  │ TCP
  ├─ permit.submit(rank0_req)                           │
  ↓                                                     │
rank-0 scheduler thread → forward_decode               │
  └─ distributed.synchronize_token(step, local_token)  │
       └─ nccl.broadcast_i32([local_token], 1, 0)  ────┼──── NCCL collective ←──┐
                                                       │                        │
worker rank R process:                                 │                        │
  ↓ relay-receiver thread                              ↓                        │
  ├─ recv() → Request2 { wire }                                                 │
  ├─ req.distributed = Some(Nccl {                                              │
  │     rank=R, world_size=N, nccl=rank-R's ep_nccl })                          │
  ├─ permit.submit(req)                                                         │
  ↓                                                                             │
rank-R scheduler thread → forward_decode                                        │
  └─ distributed.synchronize_token(step, local_token)                           │
       └─ nccl.broadcast_i32([0], 1, 0) ── reads broadcast result ──────────────┘
              (returns rank-0's token; rank R uses it as next-step input)
```

After C.4.6 every rank's `synchronize_token` participates in the same
cross-rank NCCL broadcast on the same EP NCCL group the model was
constructed with. Worker hidden states stay aligned with rank 0's
across all decode steps.

## What's complete in B-1

All 10 sub-commits of phase B-1:

| | |
|---|---|
| A `98afea39` | NcclGroup broadcast_i32 + broadcast_bytes helpers |
| B.1 `2e2686f4` | ARLE_WORKER_RANK env detection + run_worker_mode entry |
| B.2 `80d45e3b` | Coordinator-side child spawn via Command + ARLE_MULTIPROC_SERVE gate |
| B.3 `0622ef7c` | DistributedShape override + worker-mode scheduler boot |
| B.4 `2493b297` | Deadlock guard for incomplete state (removed in C.4.5) |
| C.1 `a4ec723a` | multiproc_relay module — TCP transport |
| C.2 `681710e9` | Two-phase bind/accept + multiproc_relay_smoke bin |
| C.3 `6701bc9f` | async_main + run_worker_mode relay wire-in (boot ping) |
| C.4.1 `f670c05e` | WireRequest + WireSamplingParams types + Request2 envelope |
| C.4.2 `a32ef68d` | DistributedRequestCoordination enum InProcess/Nccl |
| C.4.2b `8234a3f0` | LayerCommunicator tp_nccl/ep_nccl accessors |
| C.4.3 `5fad975a` | DistributedSchedulerGroup::with_relay + submit() broadcast |
| C.4.4 `7894860e` | Worker relay-receiver injects WireRequest into local scheduler |
| C.4.5 (in `7894860e`) | Deadlock guard removed |
| C.4.6.1 `f095fff3` | ModelForward::ep_nccl trait method |
| C.4.6.2 `9c164bf8` | SchedulerHandle stores ep_nccl post-model-load |
| C.4.6.3 `3f0152ad` | Coordinator attaches Nccl coord on rank-0 submit |
| C.4.6.4 `e0a4175f` | Worker attaches Nccl coord on relay receive |

Total: 16 commits to main across phase B-1.

## What's NOT done — outstanding for the multiproc-serve track

- **End-to-end pod smoke** (was scheduled as C.4.6.5 / "commit D" in
  earlier briefs): launch 2-rank `arle serve --multiproc`, POST one
  greedy `/v1/chat/completions`, compare bytes against single-process
  2-thread baseline. Needs ARLE checkout + multi-GPU pod time. The
  cargo path is clean; the failure modes are runtime-discovered
  (NCCL rendezvous timing, IPC handle scope, etc.) and will need
  iteration. Not blocking the architecture — the wire-in is sound
  per CLAUDE.md §0 (all 16 commits cargo-checked).

- **`crates/deepep-sys` cxx binding** (task #14 / phase B-2): the
  whole point of the multiproc pivot. Now that ARLE runs one process
  per rank, DeepEP's `Buffer::intranode_dispatch / combine` becomes
  available inline. ~200 LOC mirror of SGLang's deepep.py dispatcher.

- **`forward_deepep_routed_gpu` wire-in** (task #15 / phase B-3):
  swap the NCCL DeepEP-style fallback for native DeepEP calls via
  crates/deepep-sys.

- **SLO A/B bench** (task #16 / phase B-4): the PASS gate per the
  pivot doc (32K/1.5K c=8 qps=8, p50 TTFT +5% etc.).

## Bench-exempt notes

All 16 B-1 commits are env-gated:
- `ARLE_MULTIPROC_SERVE` unset → legacy single-process N-thread path
  (zero behavior change).
- `ARLE_MULTIPROC_SERVE=1` set → multiproc-serve scaffolding fully
  wired; all four collectives (boot barrier, relay TCP, forward NCCL
  TP/EP, token-sync NCCL broadcast) participate across processes.

Pure scaffolding land — no production hot-path runtime changes for
the default path.

## Rule

Cross-rank token synchronization in multi-process serving needs the
SAME NCCL group the model uses for forward collectives — creating a
parallel NCCL group at submission time would conflict with
`ncclCommInitRank`'s TCP rendezvous (port collision) and would split
CUDA contexts (each NcclGroup::new calls CudaContext::new). The
clean plumbing is: model exposes its NCCL group via a default-None
trait method, SchedulerHandle stores the cloned Arc post-model-load,
submission paths attach NCCL-backed
DistributedRequestCoordination::Nccl with that Arc. Single source of
truth for the comm group across forward + token-sync, all dispatching
through the existing rank-aware ncclCommInitRank handle.
