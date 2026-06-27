# Agent-OPD `--share-frozen-base` student-load deadlock fixed — `cuCtxSynchronize` → `cuStreamSynchronize`

## Context

The agent-OPD full loop, after the `mem_fraction_static` clamp fix (`999f5339`)
cleared the co-resident 27B FP8 OOM, hit a deterministic hang at the
`--share-frozen-base` student load on the 8×H20 pod
([prior errors entry](../errors/2026-06-27-agent-opd-share-frozen-base-load-futex-deadlock.md)).
The handed-off decode said "main thread in `futex_wait`, all 186 threads S" and
pointed at the FP8 import (`import_fp8_block_scaled_device_ptr`).

## What Worked

**Re-decoded the case at the kernel level — the original symptom was a mis-sample.**
Reproduced clean on an own tree (`/host/arle-opd-deadlock`, free GPU, no foreign
contention) and read `/proc/<pid>/syscall` + per-thread `wchan` directly (gdb
attach trips the node's ELKEID anti-debug and reaps the target):

- main thread is in `poll(nfds=2, timeout=-1)` (`__x64_sys_poll`), **NOT**
  `futex_wait`; the process has **4 threads**, not 186 (the 180-thread figure was
  the idle 180-core rayon/OpenMP pool, all legitimately parked — a red herring).
- instrumenting the loader (`ARLE_OPD_LOAD_TRACE`) proved the load **completes** —
  all 851 tensors materialize; the FP8 import is a pure pointer-wrap
  (`upgrade_device_ptr`) and never blocks. The hang is in the line AFTER the load.

**Root cause:** the post-load cross-stream handoff fence called
`train_backend.device_synchronize()` = **`cuCtxSynchronize`**, which drains the
ENTIRE device primary context. The co-resident rollout engine shares that primary
context (both backends `cuDevicePrimaryCtxRetain` the same ordinal) but runs its
streams with cudarc event-tracking **disabled** (`DeviceContext::on_device`
→ `ctx.disable_event_tracking()`, `crates/cuda-kernels/src/tensor.rs:341`) and
idle-parks between scheduler steps. `cuCtxSynchronize` blocks forever in `poll()`
on the engine's never-host-progressed streams. (Reconciles "GPU flat at 3 MiB":
the student's async uploads were submitted but the FIRST sync — this fence —
deadlocks before they execute, so VRAM never climbs.)

**Fix:** add `Backend::stream_synchronize` (`cuStreamSynchronize` on the train
backend's OWN default stream only) and use it for the two share-frozen-base
handoff fences (agent-opd + rubric-opd in `train_cli.rs`). The engine's resident
FP8 base is committed by its own load+warmup, so the fence only needs the
student's own upload stream drained. `device_synchronize` (`cuCtxSynchronize`) is
LEFT unchanged — its other callers (`opd.rs` weight offload/reload fences)
intentionally drain the whole context. Gated to `!no_share_frozen_base`; default
serve byte-identical.

**Verified (own tree, GPUs 4-7, no contention):**
- STEP 1 A/B: `--no-share-frozen-base` loads the student first with no co-resident
  engine → never deadlocks (GPU climbs 16→38 GB). `--share-frozen-base` baseline
  deterministically hangs at the fence (GPU flat 3 MiB). Confirms the deadlock is
  share-path-specific.
- With the fix: `[opd-load-trace] post stream-sync OK` prints immediately and GPU
  residency climbs to **35.7 GB** (vs the deadlocked baseline stuck at 3 MiB) —
  load + weight upload complete, InferStudent builds. Reproduced across two runs.

## Still open (separate blocker, NOT this deadlock)

After the fence clears, the process is SIGKILLed (whole process group, no `$?`
capture, no panic, no kernel OOM log, GPU 35/97 GB so not VRAM-bound) at the
rollout transition ("building InferStudent" → first generation). Signature is the
node-governance/ELKEID kill the prior commits (`730fde6f`, `0ca638a7`) already
documented as "not a code bug". The held-out pass-rate is blocked by this external
governance kill, not by any code deadlock. The sandbox `process_group(0)` →
`setsid` fork-safety fix (a multithreaded-CUDA-parent `fork()` hazard) is bundled
prophylactically but does not lift the governance kill.

## Bench

pending-remote (correctness/deadlock fix — the "before" is a hang, the "after" is
a clearing fence + GPU residency climb, both measured on the H20 pod above; the
default serve path is byte-identical so no serve-perf delta). No `bench_guidellm`
delta applies to a co-resident OPD-load fence.

## Rule

- **A `poll()`-parked, GPU-flat process is NOT necessarily a `futex` deadlock —
  read `/proc/<pid>/syscall` + per-thread `wchan` before trusting a handed-off
  "futex / N-thread" decode.** A big idle rayon/OpenMP pool (one thread per core)
  inflates the thread count and one of its workers parks on `futex` (syscall 202),
  which a coarse sampler mis-attributes to the main thread. The main thread here
  was `poll(nfds=2,-1)` = a CUDA blocking sync, not a lock.
- **`cuCtxSynchronize` is the wrong fence when a foreign context shares the device
  primary context.** Both `CudaContext::new(ordinal)` retain the SAME primary
  context; `cuCtxSynchronize` from one drains the OTHER's streams. If the foreign
  engine runs event-tracking-disabled + idle-parked, the context-wide drain
  deadlocks. Use `cuStreamSynchronize` on your OWN stream for a self-fence; reserve
  `cuCtxSynchronize` for the deliberate cross-context ordering (weight reload).
- **On an ELKEID/HIDS pod, gdb attach reaps the target — use `/proc` (`syscall`,
  `wchan`, `stack`, `fd`) for the deadlock evidence**, and instrument the code path
  (env-gated trace) to bisect rather than relying on a userspace backtrace.
