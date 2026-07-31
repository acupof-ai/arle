# CP/DP launcher watchdog: no-progress timer, not a total-budget — 2026-07-31

> Status: Core landed (host-side launcher logic). The `nccl` path is pod-only
> (needs NCCL + nvcc); non-nccl cuda typecheck green. Verification pending-remote.

## Context

`maybe_spawn_ranks_and_wait` (`crates/cli/src/train_multiproc.rs`) spawns one
worker per CP/DP rank and waits for the first exit, killing survivors so one
rank's crash can't wedge NCCL. A rank that *hangs* inside a collective never
exits, so `try_wait` alone can't see it — an opt-in `ARLE_TRAIN_STEP_TIMEOUT_SECS`
deadline was added to tear the group down on a hang.

The bug: `start` was stamped once before the wait loop and the deadline checked
`start.elapsed() > limit` — a **total wall-clock budget from launch**, not the
"no-progress deadline" the comment claimed. It can't tell a hang from a slow-but-
progressing step: a legitimate 740 s forward at local-seq 65536 dies under a
300 s cap. This produced a false-positive "hang" during CP debugging (2026-07-30)
that sent the root-cause chase down the wrong path.

## What worked — measure idle-since-last-log, and get the liveness signal free

A truly wedged collective goes silent on *every* rank: they all block waiting on
the hung one, so no rank emits another log line. That makes "any child stderr
line" a sufficient group-liveness signal — no heartbeat file, no per-phase
instrumentation in the worker.

- Pipe each child's stderr (`Stdio::piped`) instead of inheriting it. A per-child
  forwarder thread reads lines, bumps a shared `Arc<AtomicU64>` (ms since `start`)
  on each line, and re-prints it — the worker's own `[cpN]` prefix is preserved.
- The wait loop computes `idle = start.elapsed() - last_progress_ms` and tears
  down only when `idle > limit`. A slow step logging periodically resets the clock;
  a real wedge (all ranks silent) trips it.
- Forwarders are joined after teardown so buffered final lines flush.

Net: the deadline now means what the comment says. Off by default (a legit long
step is never killed); set it to bound a ladder/OOM probe.

## Verification

- **Non-nccl cuda typecheck (Mac, CI Lint mirror):** `CUDARC_CUDA_VERSION=12080
  cargo check -p cli --release --no-default-features --features cuda,no-cuda` —
  green. The nccl block is `#[cfg(feature = "nccl")]`; its new imports
  (`BufRead`, `Arc`, `AtomicU64`) are all nccl-gated so they don't leak.
- **Pending-remote:** the `nccl` path (`--features cuda,nccl`) needs NCCL libs +
  nvcc — pod-only. (`nccl,no-cuda` is not a real target: it fails pre-existing at
  `backend_cuda.rs:2327`, unrelated to this change.) Pod check: launch a CP run
  with a short `ARLE_TRAIN_STEP_TIMEOUT_SECS`, confirm a progressing step logging
  under the cap survives past it, and a `kill -STOP`'d rank trips teardown.

No unit test: the logic is a saturating subtraction inside a spawn/wait loop that
can't run without NCCL + real child processes; extracting a helper solely to test
`idle = elapsed - last_progress` would be over-engineering (YAGNI).

## Rule

A "no-progress" / "hang" timer must reset on progress, or it's a total-budget
timer wearing the wrong name — and it will kill slow-but-live work while a real
hang looks identical to it. For a multi-process group, the cheapest progress
signal is often already on the wire: a wedged collective silences *all* ranks, so
"any child logged" is liveness — pipe stderr, don't instrument the worker.
