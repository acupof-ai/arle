# OPD multi-arm A/B died repeatedly — shared-resource launch collision, not CUPTI/OOM/external

## Context
After the gated-delta forward fix (recurrent route) + the device LoRA-merge fix
(510×), the OPD math A/B kept dying: training arms SIGKILLed (`AB_EXIT:137`)
after minutes, and eval serves dropped mid-run (`RemoteDisconnected`). Burned a
very long debug across many cycles chasing the wrong cause.

## Root Cause
**Concurrent arle processes (2+ arms) collided on a shared launch resource at
*load time*** (deaths logged at "loading student" = the DeepGEMM JIT-warm
phase). A *single* arm ran 66+ steps stably; any 2 concurrent died fast — the
single-vs-concurrent isolation was the decisive discriminator.

**Attribution (partial — read this honestly):** the "fix" changed THREE things
at once (unique session + port + JIT dir), which is unattributable on its own
(§0). Narrowing from evidence afterward:
- **Port: RULED OUT.** `--port` exists only in `arle serve`, not `arle train
  opd` (the teacher is in-process via `--teacher-runtime infer`). OPD-train binds
  no port → "unique port" did nothing.
- **DeepGEMM JIT cache: PRIME SUSPECT.** The dying v3/v4 launchers set no
  `DG_JIT_CACHE_DIR` (→ shared *default* cache); the surviving multiseed sets a
  unique per-run dir. Mechanism fits: 2 procs concurrently JIT-warming the same
  cache dir at startup → file race/lock/corruption → crash at load. This is the
  only resource with both an evidence delta AND a load-time mechanism.
- **tmux session name: NOT isolated.** Could not confirm whether the dying run
  used unique session names, so a session collision isn't excluded.
- **Single-variable confirmation still owed:** run 2 concurrent arle identical
  except share ONLY the JIT dir (unique session) → if they die, JIT is THE
  cause; if they survive, it was the session. NOT yet done (queued).

With unique session + JIT, 2 concurrent arms survived 25 min + 6 checkpoints,
zero deaths — so the fix *works*, but the attribution is "JIT-cache (prime) or
session" pending the single-variable test.

Every other hypothesis was a **red herring**, ruled out one by one:
- **Box contention** — the user was not using GPU 4-7 at all.
- **Host OOM** — `free` showed 1.7 TB available; cgroup `oom_kill=0`.
- **CUPTI segfault** (`libcupti.so.12.9` in dmesg) — the live arle procs mapped
  **0 libcupti** (`grep -c cupti /proc/PID/maps` = 0); the CUPTI crashes were a
  different process's profiler.
- **Xid 31 GPU MMU fault** — the dmesg entries were ~8.8 days stale (ts delta
  ~765k s vs the recent events).
- **Container restart / kubelet eviction** — `Restart Count: 0`, no OOMKilled
  event.
- **External watchdog / cron** — user confirmed none; no auditd SIGNAL record,
  no cron killing arle.
- **eval `pkill`** — the eval's `stop_serve` was scoped to `--port 8137`, not broad.

The eval `RemoteDisconnected` was a *separate, second* bug: a short client HTTP
timeout on long-CoT (4096-token, 60 s+) generations, counted as wrong → a fake
0.16 (later 0.842 on a biased 19-valid subset). Fix: long request-timeout +
retry + exclude request-errors from the denominator → clean n=100, 0 error.

## Fix
1. Per-run unique tmux session name, engine port, and `DG_JIT_CACHE_DIR`; never
   stack two arle on one GPU; never `pkill -f arle` (kill only own PID/port).
2. Eval client: `request-timeout ≥ 600 s`, retry on disconnect, exclude
   request-errors from accuracy.

## Rule
A SIGKILL that leaves **no trace** in the victim + a **"single stable /
concurrent dies"** signature ⇒ suspect a **shared-resource collision in your own
launcher** (session/port/JIT-cache), not external-kill/OOM/CUPTI. Diagnose by
**isolating single-vs-concurrent first**, and rule out red herrings with their
own evidence (`/proc/PID/maps` for the lib, dmesg *timestamp deltas* for Xid,
`Restart Count` for container restarts, `free`+cgroup for OOM) — don't theorize a
killer you can't name. A no-trace SIGKILL is caught with a kernel
`signal_generate sig==9` tracepoint, but that only fires if you reproduce the
*concurrent* condition. See [[reference_dsv4_deepgemm_jit_cache_persist_62]]
(shared JIT cache) and the eval-robustness note in
[[feedback_flag_silent_noop_passes_exit0_smoke]].
