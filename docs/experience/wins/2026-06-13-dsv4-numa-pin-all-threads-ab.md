# DSv4 NUMA pin (all-threads, automatic) — +0.9 tok/s B=1 and 25× tighter variance

## Context

The NUMA pin (`numa_pin.rs`, default ON, `ARLE_NUMA_PIN=0` opt-out) slices each
GPU's NUMA-node cpulist disjointly among its ranks and `sched_setaffinity`s the
worker to its slice. Verifying it on the 8×H20 pod (B=1 decode, France-raw bench,
the FP8-decode 44-class binary) exposed a gap: `sched_setaffinity(0)` pins only
the **calling** thread. Per-rank verification:

- workers (ranks 1-7): single-purpose processes, pinned process-wide ✓
- rank 0: lives in the **coordinator** process (HTTP + router + rank-0 engine).
  Its tokio/serving threads spawned before the engine build, so they never
  inherited the slice — the coordinator's main thread sat on all 180 cores
  (`0-179`) while only the engine thread was on `0-21`.

## What Worked

Make the pin **automatic and complete**: build the mask once, then apply it to
**every** thread of the process (walk `/proc/self/task`), not just the caller —
future threads still inherit from the pinned calling thread. The boot log now
reports `pinned/total`.

**改动生效 (verified, /proc affinity):**

| rank | threads | Cpus_allowed_list |
|---|---|---|
| 0 (coordinator) | 43/43 | `0-21` (was `0-179`) |
| 1-3 (numa0) | 13 each | `22-43` / `44-65` / `66-87` |
| 4-7 (numa1) | 13 each | `90-111` … `156-177` |

Every thread of every rank is now on its disjoint slice. (Box: 2 NUMA nodes ×
90 cores; GPUs 0-3→numa0, 4-7→numa1; 4 ranks/node → 22 cores/rank.)

## Results — serial A/B (same worktree binary, two env flips, ×3 each)

| arm | B=1 tok/s (×3) | mean | σ |
|---|---|---|---|
| **NUMA-ON** (all-threads) | 44.54 / 44.52 / 44.52 | **44.48** | **0.01** |
| NUMA-OFF (`ARLE_NUMA_PIN=0`) | 43.79 / 43.29 / 43.93 | 43.55 | 0.27 |

**+0.9 tok/s (+2.1%) and ~25× tighter variance.** The variance collapse is the
headline: the pin removes the scheduler-placement lottery (the documented ±6%
session drift). The all-threads fix also beat the prior per-thread-only pin
(44.5 vs the earlier 44.0±0.2 with rank 0's coordinator unpinned) — pinning the
coordinator NUMA-local *helps* at B=1, so the "squeeze the HTTP pool to 22 cores"
worry was unfounded.

## Rule

- **`sched_setaffinity(0)` is a per-thread call, not per-process** — a
  multi-threaded process (anything hosting a tokio/HTTP pool alongside compute)
  needs an explicit `/proc/self/task` walk to pin the threads that spawned
  before the call. The boot log must report `pinned/total` or the gap is
  invisible (the coordinator looked "pinned" from the engine thread's view).
- **Pinning value is variance reduction, not just mean** — a NUMA-pin A/B must
  report σ across runs; the +2% mean undersells it, the 25× variance collapse is
  the real product (predictable TPOT for the SLO).

## Verification note

Built and measured in an isolated git worktree (`/data01/build/arle-dsv4` @
`97d113e7`) because ckl held the shared pod tree at `994c8f81` for parallel #88
work — no `git checkout` over dirty files, own cargo target, ckl's tree untouched.
