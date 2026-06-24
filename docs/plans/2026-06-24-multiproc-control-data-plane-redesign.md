# Multiproc TP serving — control/data-plane separation + shm transport

Status: design, awaiting sign-off. Architectural (8 components across `cli` /
`infer-server` / `infer-api`) → approach-first per the agent contract.

## Problem (first principles)

A **control-plane / data-plane role-conflation deadlock**. DSv4/Qwen3.5 multiproc
TP runs rank 0 **in the parent process as BOTH** the relay **coordinator** (control
plane — a host event loop: bind, broadcast ticks, read worker completions) **AND**
TP **rank 0's engine** (data plane — must enter NCCL/NVSHMEM collectives in
lockstep with peers). One process, two roles with incompatible blocking semantics.

The `deepep_ll` boot adds collectives allreduce never runs — `Buffer::sync` + uid
`all_gather` + `nvshmem_init` (`deepep.rs:240-284`). The parent's coordinator setup
(`serve_multiproc.rs:bind_relay_and_spawn_workers`) and its engine-thread boot
(`serve.rs:160-166` → `build_cuda_engine`) race; rank 0's engine never reaches the
GPU (pod evidence: GPU 4 = 3 MiB while ranks 1-3 spin at 100% in the barrier
waiting for it) → **boot deadlock**. allreduce-TP4 has no boot collectives → boots.

This is structural, not an EP bug: any per-boot collective on a process that also
serializes on the control plane can deadlock. The fix must remove the conflation
**by construction**, not patch the ordering.

## Target architecture — SPMD: thin coordinator + N symmetric workers

Separate the planes:

- **Data plane**: spawn **all N** ranks (0..N-1) as identical worker processes
  (today only 1..N-1 are spawned; rank 0 is in-process). Every rank builds its
  engine, joins every collective, runs the lockstep driver. No rank is special.
- **Control plane**: the parent becomes a **thin coordinator owning NO TP rank** —
  HTTP ingress, request-id allocation, per-tick admission broadcast to all N,
  completion collection from the output-owner rank, response routing. No GPU, no
  collective participation → it can never be the missing rank at a barrier.

Deadlock impossible by construction: the coordinator never enters a collective; the
N workers are symmetric and all reach every barrier.

### Transport seam (enables both B and shmipc)

Extract a `RelayTransport` trait — the control-plane wire, today hardcoded to
localhost `TcpStream` in `multiproc_relay.rs`:

```
trait RelayTransport {
    fn broadcast_tick(&self, seq, admissions) -> Result<()>;   // coord → N workers, per step
    fn recv_completion(&self) -> Result<RelayCompletionDelta>;  // owner worker → coord
}
```

- `TcpTransport` — the current localhost-TCP impl, moved behind the trait (no
  behavior change).
- `ShmTransport` — a shared-memory ring impl (below), selected by config.

## shmipc transport (perf, separate axis)

The relay is **localhost TCP on the per-decode-step hot path** (RTT ~10-30µs:
syscall + copy + loopback softirq, paid every step), carrying small control +
`token_ids` (the heavy TP tensor exchange is NCCL/NVLink — untouched). A shared-mem
ring (SPSC per direction, `eventfd`/futex notify) cuts that to ~1µs.

- **Substrate already present**: `kv-native-sys` exposes POSIX shm + mmap + a host
  arena; `memmap2` is a workspace dep. No new heavy dependency — extend
  `kv-native-sys` with a ring, or a thin `ShmRing` over its shm + an `eventfd`.
- **LICENSE-OR-KILL (§0) — current verdict: KILL / defer.** Back-of-envelope from
  the mapping: ingress `TickAdmissions` is per-step at 20-40 Hz; TCP→shm saves
  ~10µs against a **25-50 ms step = 0.02-0.04%** (wash). Egress per-token
  completions (Stage-3-only) ≈ ~0.1% (a request's ~N×10µs over its ~N/ratetok·s).
  Both are dwarfed by step time → **do not build shmipc speculatively.** The only
  scenario that could license it is high concurrency, where localhost TCP under
  many small per-step messages may queue — but that is **unmeasurable until B makes
  EP boot**. So: ship B, then measure the per-step relay RTT share at target
  concurrency; license shm transport ONLY if it's then ≥~2-3% wall-clock. A/B
  TCP-vs-shm, same binary, two-config. Never bundle with B (confounder). The
  transport trait (Phase 1) is still worth it — it makes the swap a 1-file follow-up
  IF the measurement ever licenses it, at near-zero cost now.

## Phasing (decouple correctness from perf — no multi-variable confounding)

1. **Transport trait extraction** — `TcpStream` behind `RelayTransport`, zero
   behavior change. Refactor-only; existing multiproc serve still passes. Low risk.
2. **B — SPMD split** (the actual deadlock fix). The hard part is the **Stage-3
   completion-return path is unimplemented**: today only rank-0's *in-process*
   engine feeds HTTP; once rank 0 is a child, its output must flow over the relay
   to the coordinator, which routes to HTTP clients. Sub-steps:
   - `serve_multiproc.rs:479` spawn `0..world_size`; symmetric config to all
     (rank 0 also reads `ARLE_WORKER_ENGINE_CONFIG`).
   - `serve.rs:160-166` drop the in-process engine when multiproc; parent runs the
     thin coordinator loop instead.
   - Implement Stage 3: owner-rank completion → relay → coordinator sink →
     HTTP (`multiproc_relay.rs:355-377` sink registry, today unused by HTTP).
   - Coordinator scheduler: request-id alloc + per-tick admission + completion
     routing (thin; not a full `engine_loop`). Async(HTTP)/sync(relay) bridge via
     a blocking relay thread → tokio mpsc.
   - **Gate**: EP=4 `deepep_ll` boots to ready deterministically (the deadlock is
     gone) + needle exact + c跑 returns ok>0. Re-test EP=8 (production) too.
3. **shmipc transport** — only if Phase-2's measured RTT share justifies it. A/B
   vs TCP, per-step ITL.

Phase 1 is safe prep. Phase 2 is the fix (multi-day; the Stage-3 + coordinator
loop + HTTP rewrite are the weight). Phase 3 is conditional on measurement.

## Risks / open questions

- **Latency regression from B**: in-process rank 0 avoided one relay hop for the
  output owner; B routes it over the relay. This is exactly why Phase 3 (shm)
  pairs with B — measure the added hop, recover it with shm if material.
- **Stage-3 routing** is net-new code (completion ownership, sink lifetime,
  backpressure) — the bulk of Phase 2.
- **Coordinator HTTP/relay async boundary** — keep the relay sync on a dedicated
  thread; bridge to tokio, don't async-rewrite the relay.
- Verification is **CUDA-only on the shared H20 pod** — needs a quiet window / a
  bundle-fed rebuild per the established flow.
