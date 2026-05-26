---
title: ARLE multiproc-serve — multi-process-per-rank pivot
date: 2026-05-27
type: design plan
status: draft — phase B-design, license-or-kill phase B-0 unblocked
owner: ckl
supersedes:
  - docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md (Option A sidecar)
related:
  - docs/experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md
  - docs/experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md
  - /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py (SGLang reference)
  - /sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py (SGLang launcher)
---

# ARLE multiproc-serve — multi-process-per-rank pivot

## Why this doc

The 2026-05-26 sidecar work
([`../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md`](../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md))
landed a working torch-free C++ sidecar for native DeepEP — proves the
kernels work end-to-end with byte-deterministic output. But sidecar
implies a custom IPC dispatch/combine bridge between host (Rust) and
sidecar (C++) processes; ~1000 LOC of original protocol design with no
industry precedent.

Reading SGLang's `deepep.py` token dispatcher
(`/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:300-540`)
revealed the much simpler pattern used by every production stack:
**one process per rank, call DeepEP `Buffer::dispatch` / `Buffer::combine`
inline in the MoE forward path**. Tens of LOC instead of ~1000.

The blocker for ARLE was the supposed cost of multi-process refactor.
Mapping the actual ARLE boot shape (see §Current shape) shows the cost
is much smaller than the previous design doc estimated:

- **NCCL is already cross-process-ready**: ARLE's existing `EnvBootstrap`
  via `MASTER_ADDR/MASTER_PORT/WORLD_SIZE`
  (`infer/src/distributed/nccl.rs:38-49`,
  `infer/src/main.rs:407-415`) is the same TCP-rendezvous NCCL uses for
  multi-process. The current N-thread shape is incidental — NCCL doesn't
  know or care it's all one process.
- **Model weights, KV pool, RadixCache, LayerCommunicator are already
  per-thread** (`bootstrap.rs:404`, `construction.rs:132-220`,
  `layer_communicator.rs:43-50`) — survive the refactor unchanged.
- **HTTP / tokenizer are already in the main tokio loop**, not in
  scheduler worker threads (`main.rs:1296-1321`). Just need to gate them
  on rank-0.
- **The shared-state items are short**: `Arc<NcclGroup>` between
  threads (can't survive — but each child re-inits NCCL via
  `EnvBootstrap`),
  `DistributedSchedulerGroup.submission_lock: Mutex<()>`
  (`infer/src/scheduler/cuda/bootstrap.rs:573` — replace with NCCL
  CPU-side barrier or unix socket), and the
  `Arc<AtomicU32>+Barrier` token coordinator in the direct distributed
  generate path (`main.rs:540-545` — replace with NCCL int32
  broadcast).

So the actual refactor is ~5 surface changes, not "multi-week" as the
original doc estimated.

## Non-goals

- Not designing the full request queue replication strategy yet — that's
  phase B-1 once the launcher + worker mode work.
- Not pursuing the Option A sidecar further. The committed sidecar
  scaffolding
  ([`../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md`](../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md))
  stays in tree under the existing build gate as a phase 1.0a-iv
  reference; it gets ripped in phase B-3 cleanup once native DeepEP is
  live.
- Not changing the Metal backend. multiproc-serve targets CUDA only;
  Metal runs single-process on Apple Silicon.

## Current shape (single-process, N threads)

```
                arle serve  (one process)
                ────────────────────────
        ┌─ main tokio runtime ────────────────────────┐
        │   HTTP server (axum) ⟵ user requests        │
        │   Tokenizer (HF tokenizers)                 │
        │   RadixCache / scheduler decode queue       │
        │                                             │
        │   spawn_cuda_worker_group(N) ─┬─ std::thread 0 → device 0 ──┐
        │   (main.rs:1178-1208)         ├─ std::thread 1 → device 1 ──┤
        │                               ├─ ...                        ├ Arc<NcclGroup>
        │                               └─ std::thread N-1 → dev N-1 ─┘
        └──────────────────────────────────────────────────────────────┘
```

Cite: `main.rs:308` (entry), `main.rs:1113` (`build_cuda_worker_bootstrap`),
`bootstrap.rs:573-616` (one std::thread per rank), `layer_communicator.rs:43-56`
(per-thread NCCL groups), `main.rs:407-415` (NCCL env setup).

DeepEP native cannot work here — proved by codex gate 2026-05-26
([`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md)).

## Target shape (multi-process per rank)

```
                                                  ┌─ arle serve --rank=1 (proc 1)
                                                  │   • skip HTTP / tokenizer
                                                  │   • scheduler rank-1 only
                                                  │   • NCCL EnvBootstrap rendezvous
                                                  │   • DeepEP::Buffer (rank=1, world=N)
                                                  └─ ...
        arle serve  (rank-0 / coordinator)
        ───────────────────────────────────
        ┌─ main tokio runtime ──────────────┐  ── fork via std::process::Command
        │   HTTP server (axum)              │     ::current_exe() + env
        │   Tokenizer                       │     ARLE_WORKER_RANK=R
        │   RadixCache, request queue       │
        │   Spawn N-1 children, wait ready  │
        │   Scheduler rank-0 (one thread)   │
        │   NCCL EnvBootstrap rendezvous    │     ┌─ arle serve --rank=N-1 (proc N-1)
        │   DeepEP::Buffer (rank=0, world=N)│     │   ... same shape as rank 1
        │   Broadcast batches → workers via │     │
        │     NCCL CPU-side group or ZMQ    │     └────────────────────────────────
        └───────────────────────────────────┘
```

Mirror of SGLang's pattern at
`/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py:577-722`:
launcher stays as coordinator running HTTP; one child process per scheduler
rank; control plane is ZMQ; data-plane sync is `torch.distributed`
collectives on a CPU process group. ARLE substitutes ZMQ with a unix
socket or NCCL int32 broadcasts (we don't have torch).

## Concrete file-level change set

### B-0 — license-or-kill smoke (one commit)

New: `infer/tests/multiproc_nccl_smoke.rs` (or a small bin under
`tools/`). 2 child processes spawned via `std::process::Command::
current_exe()` with `ARLE_WORKER_RANK=R`. Each child binds CUDA
device R, initializes NCCL via `EnvBootstrap`, rank-0 broadcasts a
small i32 buffer, rank-1 receives it, both do `ncclAllReduce`, parent
collects exit codes.

**PASS**: both children exit 0, all-reduce result matches expectation,
no deadlock for 10 consecutive cycles.
**KILL**: NCCL hangs or `EnvBootstrap` fails cross-process (would
falsify the agent-1 finding about `MASTER_ADDR/PORT` being
process-agnostic — would force going back to Option A).

### B-1 — coordinator/worker split (one to three commits)

Files touched (estimated, will refine during impl):

1. `infer/src/main.rs` (~308-1330) — branch at startup:
   - If `ARLE_WORKER_RANK` unset or `=0` → coordinator path (current
     code, with `spawn_cuda_worker_group` taking N=1 if any workers
     spawned, plus child-process spawn via `current_exe()`).
   - If `ARLE_WORKER_RANK=R>0` → worker path: skip HTTP, skip
     tokenizer, just boot one scheduler at rank R, listen on broadcast
     channel for batches.

2. `infer/src/scheduler/cuda/bootstrap.rs:573-616` — when in coordinator
   mode, spawn only the rank-0 scheduler thread instead of N.

3. `infer/src/main.rs:540-545` (direct distributed generate token
   coordinator) — replace `Arc<AtomicU32> + Barrier` with NCCL int32
   broadcast.

4. Cross-process batch relay — rank-0 schedules; what gets to ranks
   1..N-1? Two candidate transports:
   - **NCCL CPU-side broadcast** of pickled-equivalent batch metadata.
     Needs CPU collective group on NCCL — we already have NCCL on CUDA
     streams. Cleanest, no new dep.
   - **Unix socket relay** from coordinator to each worker. Adds
     `zmq`-style dep, but gives flexibility for async dispatch.

   Decision deferred to B-1 implementation; B-0 doesn't depend on it.

5. `infer/src/distributed/nccl.rs:38-49` — unchanged in terms of API,
   but workers now reach `NcclGroup::new` from a child process. The
   `EnvBootstrap` rendezvous over TCP already supports this.

**PASS gate**: rank-0 + rank-1 child can serve a single non-streaming
chat request end-to-end on 2×H20. Greedy output byte-identical to the
current single-process-2-thread baseline.
**KILL gate**: latency regression > 50% per request at c=1, or correctness
divergence (sampling-level seed mismatch is acceptable; greedy must
match).

### B-2 — `crates/deepep-sys` cxx binding (one commit)

New workspace crate. Pattern mirrors `crates/mlx-sys` (single C++
bridge, cmake or build.rs nvcc compile). Exposes:

- `Buffer::new(world_size, rank, num_nvl_bytes, num_rdma_bytes)`
- `Buffer::get_local_ipc_handle() -> [u8; 64]`
- `Buffer::sync(peer_handles: Vec<[u8; 64]>, peer_device_ids: Vec<u32>)`
- `Buffer::get_dispatch_layout(topk_idx, num_experts) -> (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank)`
- `Buffer::intranode_dispatch(x, topk_idx, topk_weights, ..., config) ->
  (recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert,
   handle: DispatchHandle)`
- `Buffer::intranode_combine(x, handle, config) -> combined_x`

`DispatchHandle` is opaque Rust newtype wrapping the rank_prefix +
channel_prefix + recv_channel_prefix + send_head tuple.

Build gated on `ARLE_DEEPEP_DIR` env var (same as the Option A sidecar
already uses). Optional at compile time.

**PASS**: `cargo test -p deepep-sys --features cuda` passes a 1-rank
smoke (`Buffer::new` succeeds, `get_local_ipc_handle()` returns a
non-zero blob).

### B-3 — wire DeepEP into `forward_deepep_routed_gpu` (one commit)

`infer/src/model/deepseek/mlp.rs:3094-3982` —
`forward_deepep_routed_gpu` currently calls the NCCL DeepEP-style
fallback. After B-2, replace the dispatch/combine sections with calls
into `deepep_sys::Buffer`. Mirror SGLang's
`_DeepEPDispatcherImplNormal._dispatch_core` / `_combine_core`
(`/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:437-534`)
byte-for-byte where possible.

Gate via `ARLE_DSV4_MOE_BACKEND=native-deepep` (already reserved with
explicit bail today by commit `cd780fc2` — change the bail body to the
real call). Default `deepep` (NCCL fallback) unchanged.

Concurrently: rip Option A sidecar (commits 205317d9, fefaef8c) — its
build gate ensures no production user is affected, but the parallel
code path violates `no half-states`.

**PASS**: greedy output byte-identical to NCCL fallback on a 32-prompt
fixed set.

### B-4 — SLO A/B bench (deferred to ckl, pod time)

`scripts/bench_guidellm.sh dsv4-native-deepep` vs
`dsv4-nccl-deepep-fallback`, same env, same commit, only the
`ARLE_DSV4_MOE_BACKEND` flag changes.

Workload: 32K input / 1.5K output, c=8, qps=8, H20.

**PASS**:
- p50 TTFT delta ≥ +5% (≥ 240 ms off 4800 ms target)
- p50 TPOT delta ≥ +5% (≥ 0.9 ms off 18 ms target)
- p99 TTFT not regressed > 3%
- byte-identical greedy on 32-prompt set

**KILL**: any of the above. File errors entry; re-evaluate (probably
roll back to NCCL default and investigate per-layer overhead with
nsys).

## Stop conditions per CLAUDE.md §0 SOLID

- Wall-clock framing required at every gate. nsys "X% of NVTX window"
  must be cross-checked against "Y ms per request total" — kill on the
  conservative number.
- One variable per A/B run. B-3→B-4 bench must hold every flag constant
  except `ARLE_DSV4_MOE_BACKEND`.
- Every PASS/KILL gets a wins or errors entry on the same commit that
  ran the bench.

## Risks to verify before adopting

- **NCCL EnvBootstrap cross-process** — agent-1 analysis says it works
  because `MASTER_ADDR/PORT` is TCP rendezvous, process-agnostic.
  Confirmed in SGLang's exact use pattern. B-0 will prove or kill this
  in 30 minutes of pod time. If it kills, Option A sidecar is the only
  remaining path.
- **Worker batch relay latency** — broadcasting batch metadata from
  rank-0 to ranks 1..N-1 every forward step. Need ≤ 100 µs total per
  step (we have ~18 ms TPOT budget; relay is ≤ 0.5% of that). SGLang
  uses CPU-side `broadcast_object_list` over torch.distributed; we'll
  benchmark our chosen transport in B-1.
- **Request queue rebalance on rank-0 crash** — current sidecar plan
  monitored child PIDs and restarted. Multi-process means rank-0 IS
  the master; if it dies, the whole server dies. Match SGLang's
  watchdog behavior (any rank crash → SIGQUIT main → exit). Document
  the lack of partial-rank recovery as expected.

## Cross-refs

- Superseded sidecar design:
  [`./2026-05-26-dsv4-deepep-process-per-rank.md`](./2026-05-26-dsv4-deepep-process-per-rank.md)
- Sidecar scaffolding wins (stays in tree until B-3 cleanup):
  [`../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md`](../experience/wins/2026-05-26-dsv4-native-deepep-sidecar-scaffolding.md)
- Same-process gate failures (what triggered the pivot):
  [`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md)
- DSv4 SLO baseline (32K/1.5K, c=8, qps=8, H20, target TTFT 4800 ms,
  target TPOT 18 ms):
  [`../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md) §Serving SLO Baseline
- Industry reference (SGLang's exact multi-process pattern):
  `/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py:577-722`,
  `/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py:476-487`,
  `/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:437-534`.
