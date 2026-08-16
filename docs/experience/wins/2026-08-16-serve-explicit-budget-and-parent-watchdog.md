# Serve: the explicit memory budget wins, and an engine outlives no supervisor

## Context

Driving `arle serve --backend metal` from a desktop client (48 GiB M-series,
macOS 26.3) surfaced two lifecycle defects that a CLI operator rarely meets.

## What Worked

### The escape hatch now escapes

The Metal resource guard refuses startup with "pass `--memory-budget-bytes`
after verifying headroom", but `resolve_memory_limit` then clamped that same
explicit budget to `available − 6 GiB`, so the documented remedy could not work.
On this machine `Qwen3.6-35B-A3B-4bit` (19 GiB weights) was refused with ~28 GiB
available: budget 22 GiB against a 23 GiB fixed requirement.

The explicit budget now overrides the available-memory heuristic and logs a
warning naming the anti-swap budget it exceeded; the physical
`total − system_reserve` bound still applies. Measured: 35B loads and serves in
**~20 s** (`--max-running-requests 4 --memory-budget-bytes 26G`), where every
default and flag combination previously refused.

Second finding on the way: `static_state_bytes` is `gdr_state_bytes_per_slot ×
num_slots`, so the default 16 slots demanded 15720 MiB of fixed budget for a
model-side cost. At 4 slots it is 245 MiB and the fixed requirement drops 38 →
23 GiB. The KV pool in the same function is explicitly *not* multiplied by
`num_slots` ("num_slots is a zero-HBM soft cap"), so the two sides disagree —
worth a separate look.

### An engine no longer outlives its supervisor

A supervising app that is SIGKILLed or crashes never runs its own cleanup, so
the engine — holding tens of GiB — was left orphaned, and the next launch could
not start (memory held, and previously a fixed port held too).

`arle serve --parent-pid <PID>` starts a watcher thread that exits when the
parent is gone. Two signals, since neither alone suffices: reparenting
(`getppid()` changing, exact but only valid for a direct child) and
`kill(pid, 0) == ESRCH` (works for any ancestor, but a recycled pid can fool
it). The exit path is `libc::_exit`, not `std::process::exit`: during a
multi-GiB weight load another thread holds the allocator lock, and atexit
handlers would block there — producing exactly the orphan the watchdog exists to
prevent.

Measured: with a dead parent pid, the engine prints
`parent <pid> exited; shutting down` and is gone within one 2 s poll.

## Rule

When a guard's own message names a flag as the remedy, that flag has to be
authoritative — a heuristic that re-clamps it makes the advice unactionable and
the operator's verified headroom meaningless.

Cleanup that only runs on a graceful exit is not cleanup. A child holding
significant resources needs a signal that survives its parent being killed
without warning, and the exit path must not depend on locks the rest of the
process may hold.

## Open

The watchdog is verified in isolation. One observed case — an app-spawned engine
still alive minutes after its supervisor died, with the flag present in its
argv — is **not reproduced**, and the conditions are unknown. Instrument the
poll thread and retry with a large model mid-load before treating orphans as
solved.
