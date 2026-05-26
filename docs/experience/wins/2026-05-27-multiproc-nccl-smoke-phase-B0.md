# Phase B-0 PASS — NCCL EnvBootstrap cross-process verified

## Goal

License-or-kill the foundational assumption of the multiproc-serve pivot
([`../../plans/2026-05-27-multiproc-serve-pivot.md`](../../plans/2026-05-27-multiproc-serve-pivot.md)):
ARLE's existing NCCL `EnvBootstrap` rendezvous (TCP via
`MASTER_ADDR/MASTER_PORT/WORLD_SIZE`) survives process boundaries. If it
fails cross-process, the entire Option B pivot is dead and Option A
sidecar is the only remaining native-DeepEP path.

## Hypothesis

NCCL `EnvBootstrap`'s `ncclCommInitRank` uses the TCP rendezvous
protocol — process-agnostic by design. ARLE's current N-thread NCCL
init at `infer/src/distributed/nccl.rs:38-49` works because every rank
calls `MASTER_ADDR/PORT`-driven `ncclCommInitRank` with a unique
`rank` and shared `world_size`. Whether those calls come from N
threads of one process or N processes is invisible to NCCL.

## Params

| Knob | Value |
|---|---|
| Binary | `infer/src/bin/multiproc_nccl_smoke.rs` (new) |
| Build | `cargo build --release -p infer --bin multiproc_nccl_smoke --no-default-features --features cuda,nccl` |
| Build time | 5m 58s release on H20 pod |
| Build artifact | 619 KiB stripped binary |
| Spawn model | `std::process::Command::current_exe()` with `ARLE_WORKER_RANK=R` env |
| NCCL init | `EnvBootstrap`, port reserved by coordinator via `TcpListener::bind(0)` + drop |
| Cycles | 10 per run — broadcast f32 + all_reduce f32 |
| World sizes tested | 2, 8 |

## Env

| Component | Value |
|---|---|
| GPU | 8× NVIDIA H20, sm_90 |
| CUDA | 12.x (toolkit present, exact version not checked) |
| NCCL | 2.21.5+cuda12.4 (per existing `infer/src/distributed/nccl.rs` link line) |
| Driver | 535.161.08 (per prior smoke entry) |
| Build commit | `88b19659` (B-0 binary + LayerCommunicator field rollback) |

## Results

### 2-rank PASS

```
[coordinator pid=922614] world_size=2 master=127.0.0.1:39793
[rank 0 pid=922614] joining NCCL EnvBootstrap world=2
[rank 1 pid=922615] joining NCCL EnvBootstrap world=2
[rank 0] NCCL group ready
[rank 1] NCCL group ready
[rank 0] 10 cycles PASS
[rank 1] 10 cycles PASS
[coordinator] rank 1 exited 0
[coordinator] multiproc_nccl_smoke PASS (10 cycles, world=2)
EXIT=0
```

### 8-rank PASS — production scale

```
[coordinator pid=922649] world_size=8 master=127.0.0.1:...
[rank 0..7] joining NCCL EnvBootstrap world=8
[rank 0..7] NCCL group ready
[rank 0..7] 10 cycles PASS
[coordinator] rank 1..7 exited 0
[coordinator] multiproc_nccl_smoke PASS (10 cycles, world=8)
EXIT=0
```

Per cycle each rank does:
1. `broadcast_f32(&[cycle+1.0], 1, root=0)` — rank 0 sends cycle counter, all others receive and assert match.
2. `all_reduce_f32(&[(rank+1) as f32])` — sum across ranks, expected `N*(N+1)/2`.

All 10 cycles × 8 ranks asserted bit-exact against the expected sentinel
values; any drift would have aborted the smoke.

## Problems

None for the gated test. One workflow papercut:
`cargo build --release --bin multiproc_nccl_smoke` (no `-p`) fails with
"no bin target named `multiproc_nccl_smoke` in default-run packages" —
cargo doesn't auto-resolve bins from non-default workspace members.
`-p infer` is required.

## Learnings

1. **NCCL `EnvBootstrap` is fully process-agnostic.** The 8-rank case
   confirms it scales to the production world size — no rendezvous
   timeouts, no peer-discovery failures, no PCIe/IB ordering issues
   from concurrent child process startup.

2. **The multiproc-serve pivot is unblocked.** The hypothetical KILL
   condition documented in the pivot doc ("NCCL hangs or
   `EnvBootstrap` fails cross-process") did not materialize. Option A
   sidecar can now be deprecated as planned in B-3.

3. **Coordinator startup pattern is simple.** ~70 LOC of `Command`
   spawning + readiness drain is enough to launch N children with
   `ARLE_WORKER_RANK` env. No mpirun, no torch, no Python.

4. **NCCL port allocation via `TcpListener::bind(0)` + drop has a
   tiny race window.** The OS reserves the port between bind and the
   child's connect. In a single-host smoke this never fires; for
   production we'll want a retry loop on the rare collision. Documented
   here for future hardening — not blocking.

## Rule

`EnvBootstrap`-style TCP rendezvous is the canonical multi-process NCCL
pattern. ARLE inherits the same primitive whether it runs as 1-process
N-thread (current) or N-process 1-thread (target). Future pivots that
hinge on "does NCCL work across X" should run a 10-cycle smoke under X
before committing to multi-week refactors.

## Next

Phase B-1 — coordinator/worker split in `main.rs`. The Plan agent's
600-800 LOC / 5-commit breakdown (commits A→E) is the implementation
spec. Commit A first (NCCL `broadcast_i32`/`broadcast_bytes` + token
coordinator NCCL wire — pure refactor, still single-process,
bisect-friendly).
