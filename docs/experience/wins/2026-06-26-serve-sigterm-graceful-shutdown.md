# Serve SIGTERM graceful shutdown — fixes leaked GPU contexts on `kill`

`pending-remote` — code fix landed; the SIGTERM-vs-SIGINT ghost A/B must run on
the 8×H20 box after a pod restart clears the existing ghost contexts.

## Context

DSv4 TP4 multiproc serves on the 8×H20 pod began hanging in `ncclCommInitRank`
("starting coordinator", barrier never reaches "engine ready"). Root cause was
NOT a NCCL transport / `/dev/shm` / cuMem / IB issue (all ruled out by A/B) and
NOT a code regression (an earlier build with the SAME code served once, 8082):
GPUs 0-3 were occupied by **ghost GPU contexts** — dead PIDs
(`nvidia-smi --query-compute-apps` shows them; `ps`/`kill` say "No such
process") still holding ~74 GB each, blocking `gpu-reset` ("in use by another
client" = the ghosts themselves) and colliding with new NCCL P2P init.

The ghosts came from `pkill -9` / `kill` of hung serves: a TP worker SIGKILLed
mid-NCCL-collective wedges and leaks its GPU context, and orphaned workers
zombie under the pod's `sleep infinity` PID 1.

## Root cause (code audit)

`shutdown_signal` (`crates/infer-api/src/serve.rs`) awaited `ctrl_c()` (SIGINT)
**only**. `kill`, `pkill`, `pod_serve.sh`, systemd, and orchestrators all send
**SIGTERM** — which hit the default disposition: the multiproc coordinator died
instantly, `drop(WorkerChildren guard)` was skipped, and TP workers were reaped
by the OS via pipe-EOF at unpredictable points (frequently mid-collective). The
SIGINT path was already complete (serve future returns → `drop(guard)` → pipe
EOF → worker Drop chain → `ncclCommDestroy` on every comm/sub-comm + cudarc GPU
free, no leak); only SIGTERM was unhandled.

## What worked

Made `shutdown_signal` select over SIGINT **and** SIGTERM (`tokio::signal::unix::
SignalKind::terminate()`), so SIGTERM drives the same graceful path. Benefits
every serve (single-proc and multiproc coordinator). Companion tooling:
`scripts/pod_serve.sh` + `scripts/reap_run.py` (PR_SET_CHILD_SUBREAPER) so the
pod (PID 1 = `sleep infinity`, never reaps) doesn't leak zombies on stop.

Secondary gaps left for a follow-up (not needed once SIGTERM is graceful):
`ncclCommAbort` fallback / `ncclCommDestroy` timeout for the
peers-already-dead case; wiring the already-plumbed `RelayEnvelope::Shutdown`
so workers quiesce NCCL before the coordinator's fds close.

## Rule

A multiproc/TP serve MUST handle SIGTERM (not just SIGINT) — SIGTERM is the
default stop signal, and an un-graceful coordinator death reaps TP workers
mid-collective, wedging/leaking GPU contexts that only a pod restart clears.
Existing ghost contexts are un-clearable in-container (no live holder to kill,
`gpu-reset` blocked, persistence toggle no-op): only a pod delete+recreate
(GPU released+reset by the device plugin) or node reboot clears them.
See [[reference_h20_pod_sigkill_leaks_ghost_gpu_context]].
