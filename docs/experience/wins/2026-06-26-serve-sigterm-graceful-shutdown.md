# Serve SIGTERM graceful shutdown + the PID-namespace "ghost GPU" trap

`pending-remote` for the in-binary SIGTERM A/B (needs a rebuild that includes
the handler); the **clean-teardown + no-pod-restart** result below is measured.

## Context

DSv4 TP4 multiproc serves on the 8×H20 pod kept appearing to hang at "starting
cuda multiproc coordinator" — port never binds, GPU 0-3 each pinned at ~74 GB.
Two earlier diagnoses were **both wrong**, corrected here by evidence:

1. **"Ghost GPU contexts → pod restart" — WRONG.** `nvidia-smi
   --query-compute-apps` showed PIDs (2381544-47) holding 74 GB that `ps`/`kill`
   **inside the container** reported as "No such process". That is **not** a dead
   process leaking GPU — it is a **PID-namespace mismatch**: `nvidia-smi` prints
   **host** PIDs; container `ps`/`kill` use the **container** PID namespace. `tn
   exec` (runs on the node/host) showed those PIDs **alive** (`Sl`, `etime
   18:45`) and their etime matched our own just-launched serve — they **were**
   our workers, viewed from the host.
2. **"It's just JIT-compiling, be patient" — WRONG.** The `nvcc`/`ptxas` procs
   I'd read as active compile had `etime` of **3 days** (stale zombies);
   `DG_JIT_CACHE_DIR` was cold (4 K) and TileLang cache untouched. No compile was
   running.

## Root cause of the hang

A **flaky deadlock in the multiproc coordinator's startup**, after NCCL
bootstrap and the 74 GB TP-sharded weight load (so NCCL rendezvous *succeeded*),
that never reaches the HTTP bind on the serve port. Evidence: 18 min frozen, GPU
**0 % util** (→ blocked on CPU socket I/O, not a GPU-collective spin), all worker
mains in `tcp_recvmsg`, coordinator main in `futex_wait`, 4 `arle-relay-rank`
reader threads in `recv()`. `RELAY_TIMEOUT` is 120 s and enforced
(non-blocking listener + deadline), yet the serve was alive at 18 min — so it was
past `accept_symmetric` / the engine-ready barrier, wedged in the post-barrier
serving bring-up. Flaky: the **same binary** served cleanly once (port 8082).
Exact line not localized (release binary is stripped — needs a debug-symbol
build to pin barrier-vs-HTTP-bind).

## The SIGTERM code fix (still valid, independent of the above)

`shutdown_signal` (`crates/infer-api/src/serve.rs`) awaited `ctrl_c()` (SIGINT)
**only**. `kill`, `pkill`, `pod_serve.sh`, systemd, orchestrators all send
**SIGTERM** → default disposition → the coordinator dies instantly,
`drop(WorkerChildren guard)` is skipped, and a worker that is **mid-NCCL-collective
while actively serving** gets reaped at an unsafe point → wedged/leaked GPU
context. Fix: `shutdown_signal` now selects over SIGINT **and** SIGTERM
(`tokio::signal::unix::SignalKind::terminate()`), so SIGTERM drives the same
graceful path (serve future returns → `drop(guard)` → pipe EOF → worker Drop →
`ncclCommDestroy` + cudarc GPU free). Companion tooling: `scripts/pod_serve.sh`
+ `scripts/reap_run.py` (`PR_SET_CHILD_SUBREAPER`) so the pod (PID 1 = `sleep
infinity`, never reaps) doesn't accumulate zombies.

## What worked (measured)

`pod_serve.sh stop confirm` on the **wedged** startup-deadlocked serve:
GPU 0-3 → **0 MiB** each, host PIDs 2381544-47 **gone**, zombie count held at 238
(no new leak), **no pod restart**. The teardown is clean because the wedged
workers sat in **interruptible** socket waits (`S` state, not `D`/mid-collective),
so killing all ranks at once let the driver reclaim. (This particular teardown
is attributable to the reaper killing the whole group; the in-binary SIGTERM
handler's distinct value is the *actively-serving* case, which still wants the
rebuilt-binary A/B.)

## Rule

`nvidia-smi --query-compute-apps` prints **host** PIDs — never conclude "ghost /
dead process holding GPU" from container-side `ps`/`kill` saying "No such
process". Cross-check from the host (`tn exec`); a matching `etime` means it is
**your own live serve** in a different PID namespace, and killing all ranks at
once frees the GPU **with no pod restart**. A multiproc/TP serve must also handle
SIGTERM (not just SIGINT) so an actively-serving coordinator shuts down
gracefully instead of reaping TP workers mid-collective.
See [[reference_h20_pod_pid_namespace_gpu_trap]].
