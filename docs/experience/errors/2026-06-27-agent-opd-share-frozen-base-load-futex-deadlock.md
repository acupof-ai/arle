# Agent-OPD full-loop blocker moves from OOM to a `--share-frozen-base` student-load futex deadlock

## Context

Re-ran the agent-OPD full loop with held-out eval at HEAD `999f5339` on the 8×H20
pod, after the `mem_fraction_static` min-clamp fix (0.5→0.05, commit `999f5339`)
that was supposed to unblock the co-resident 27B FP8 load. Synced the 9 changed
source files (`65a46817..HEAD`) to `/host/arle-build`, built clean
(`BUILD_EXIT=0`, binary 14:48:13), verified `MEM_FRACTION_STATIC_MIN = 0.05` in
the pod source.

Command (per brief):
```
arle train agent-opd --student-model /host/Qwen3.6-27B-FP8 \
  --dataset /root/agent_opd_task.jsonl --staged-root /root/staged \
  --eval-dataset /root/agent_opd_eval.jsonl --eval-staged-root /root/eval_staged \
  --eval-every 1 --rounds 2 --samples-per-prompt 2 --max-turns 24 --max-tokens 1024 \
  --lora-layer-start 32 --rollout-num-slots 1 --save-lora-adapters /root/agentopd_value \
  --pythonpath lib:test
```

Three launch attempts; the clean one (`agentopd3`, GPU 5, no contention) is the
attributable case.

## Root Cause

**The clamp fix WORKED — the OOM is gone.** The run now gets all the way past
CUDA init: the rollout engine loads, prints `borrowing 400 resident FP8 base
projections (zero-copy)`, the autograd backend spins up (186-thread runtime,
CUDA primary context resident, GPU shows 3 MiB context handle). This is far
beyond the prior failure point (previously OOM'd at ~2.2 GB free during the
student embed/lm_head bf16 copy).

**A NEW, different blocker then surfaces: a deterministic futex deadlock in the
`--share-frozen-base` student load.** After `[loading student from
/host/Qwen3.6-27B-FP8]` (`crates/cli/src/train_cli.rs:2136`), the call
`load_qwen35_lora_from_hf_dir_with_shared_base(... shared_base ...)`
(`train_cli.rs:2139` → `crates/train/src/qwen35_loader.rs` FP8 share path
~line 1232, which calls the autograd backend
`import_fp8_block_scaled_device_ptr` / `replace_device_handle` per borrowed
projection — `crates/autograd/src/backend_cuda.rs:1147`) hangs.

Measured evidence of a hard deadlock (NOT slowness, NOT OOM, NOT governance kill),
over a 60s control window with zero change on every axis:
- main thread `/proc/<pid>/stack` = `futex_wait_queue_me → futex_wait`
  (syscall 202, FUTEX_WAIT); **all 186 threads in state S**, not one R or D.
- log mtime frozen at 15:03:13 (~7 min), byte-identical.
- `/proc/<pid>/io` `rchar` = 6411617 bytes, **identical** across the window — the
  student weights are never read (RSS ~250 MB, the 29 GB never loads to host).
- GPU 5 memory flat at **3 MiB** — the student base never uploads (vs the rollout
  engine which DID upload earlier in agentopd2 before that GPU got contended).
- No `RUN_EXIT`, no panic, no `OUT_OF_MEMORY`, no SIGKILL — the process is alive
  and parked, distinct from the prior governance-SIGKILL hypothesis and the
  prior OOM (RUN_EXIT=1).

Likely mechanism (hypothesis, not yet bisected): the share-frozen-base import
wraps the rollout engine's resident FP8 device pointers into the autograd
backend's stream/primary context; a lock or cross-context stream sync between the
still-live rollout engine and the importing autograd backend never releases. The
zero-byte-read + 3 MiB GPU + 186-thread-all-sleeping signature points at a
lock/sync deadlock at the engine↔autograd boundary, not at file I/O.

## Confounders isolated

- **Attempt 1 (`agentopd`, GPU 0):** the arle process died silently mid-load
  because a *foreign concurrent `dsv4` build* (another agent, same shared
  `/host/arle-build` tree, `pod-remote-build.sh dsv4`) relinked
  `target/release/arle` at 14:54:56 underneath the running binary — classic
  shared-tree SIGBUS. Not the OPD code. (Memory:
  `feedback_no_concurrent_pod_builds_delegate_ops`,
  `feedback_always_sync_latest_delete_stale_pod_trees`.)
- **Attempt 2 (`agentopd2`, GPU 1):** a foreign **4-rank job** (consecutive host
  PIDs 2163570–2163573, 94880 MiB each on GPUs 0–3 — the "foreign serve on 0-3"
  the brief warned about) occupied GPU 1; my masked single-GPU run was starved.
  The 47→94 GB climb I first saw was the foreign job ramping, NOT my run.
- **Attempt 3 (`agentopd3`, GPU 5, clean):** no contention, no concurrent build,
  GPU exclusively mine → the futex deadlock above is the real, attributable
  blocker.

## Fix

Not yet fixed — blocker handed back. Decomposition for the next pass:
1. Bisect the deadlock: `--no-share-frozen-base` takes the OTHER load-order branch
   (`train_cli.rs:2044` — student loaded FIRST, no borrow), which sidesteps the
   engine↔autograd import entirely. A/B that to confirm the deadlock is in the
   share path (cheap one-flag test).
2. If confirmed, audit the lock/stream ordering in
   `import_fp8_block_scaled_device_ptr` (`backend_cuda.rs:1147`,
   `upgrade_device_ptr` + the `CudaFp8BlockScaledStorage::new_borrowed` context
   guard) vs the rollout engine's outstanding stream work — likely a missing
   `stream_wait` / a mutex held across the import loop.

## Rule

- **Clearing one blocker exposes the next — re-decode the NEW failure, never
  reuse the OLD label.** The clamp fix did its job (OOM gone, full CUDA init
  reached); the loop then hit a *different* mechanism (futex deadlock in the
  share-frozen-base import). Don't carry "OOM" or "governance" forward — a futex
  stack + zero-byte-read + flat-GPU is a deadlock, attributed at file:line.
- **A futex-parked, all-threads-S, zero-rchar, flat-GPU process is a deadlock,
  not a slow load.** Prove "slow vs hung" with a control window (mtime + rchar +
  GPU mem, twice, ≥60s apart) before concluding either way.
- **On a shared pod build tree, a foreign concurrent build can SIGBUS your
  running binary mid-load — and a foreign multi-rank serve can reappear on 0-3.**
  Before trusting a run, check `ps … cargo build|pod-remote-build` and the full
  8-GPU `--query-compute-apps` map; run on a GPU you've confirmed exclusively
  yours, and never attribute a foreign job's memory ramp to your run.
