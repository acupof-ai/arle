# Multiproc lockstep coordinator hangs forever when a worker silently stops acking — no timeout, no error

## Context

Found while pod-verifying the DSv4 KV three-layer refactor
([plan](../../plans/2026-07-04-dsv4-dsa-kv-three-layer.md),
[wins entry](../wins/2026-07-05-dsv4-p1-p2-p4-needle-gate.md)): serving
DeepSeek-V4-Flash-FP8 TP=4/EP=4 on 4×H20, a request with `prompt_tokens≈8106`
hangs the entire server indefinitely (`/v1/stats` frozen at `steps=0`), while
`prompt_tokens=7661` completes normally in ~5s. Reproduced identically on two
separate commits (current HEAD and a pre-DSv4-refactor commit), so it is
**not caused by the KV-storage refactor** — this entry documents the
orthogonal bug, root-caused independently.

The first pass at this (bisection + A/B control only) guessed "VRAM squeeze
from TP=4 halving rank count vs the validated TP=8 baseline" — **that guess
was never measured and turned out wrong.** This entry supersedes it.

## Root Cause

`crates/infer-server/src/coordinator.rs:88-109` (`wait_for_ack_window`) is
the rate-limiter for the multiproc lockstep tick-broadcast protocol between
the HTTP coordinator process and the N TP worker processes
(`TICK_WINDOW=4` ticks). Its only exit conditions are `any_dead` (a worker's
relay reader marked it dead — crash/socket-close) or `seq < min_acked +
TICK_WINDOW`. **There is no deadline.** The function's own doc comment
explains the crash case is handled ("a CRASHED worker's reader marks it
dead... the next broadcast hits its dead socket and takes the `fail_all`
path") but has no equivalent for a worker that stops acking **without**
erroring or crashing — it just silently never sends another `TickAck`.

At `prompt_tokens=8106` (not 7661), gdb backtraces on a symbol build during a
live repro show:
- All 4 TP worker main threads parked in `infer_server::multiproc_relay::
  read_envelope` → `TcpChannel::recv` → `serve_multiproc::
  run_lockstep_driver`, waiting for a tick broadcast that never arrives.
- The coordinator's dedicated `arle-coordinator` thread parked in
  `coordinator::lockstep_loop` → `wait_for_ack_window` → `nanosleep`,
  spinning forever because `min_acked` never advances.
- On a second repro attempt (different binary build), the exact freeze point
  differed: rank2 was in `send()` while the other 3 ranks + coordinator relay
  readers were in `recv()` — same subsystem, same failure class (relay
  send/recv desync), different exact line — consistent with a genuine
  cross-rank timing race, not one static deadlock site.

Since every HTTP request on the server routes through this same coordinator
lockstep, once one rank silently stops acking the **entire server** hangs
forever, for every request, with zero error/log signal beyond a rate-limited
`"[coordinator] lockstep stalled"` warn every 10s (which reads, misleadingly,
identically for "a slow rank, wait it out" and "a rank that will never ack
again").

## Ruled out (measured, not inferred)

- **Not OOM/VRAM.** GPU memory across all 4 serving GPUs plateaus at
  96999/97871 MiB and stays flat for the entire hang — no growth, no
  OOM-killer, no Xid in `dmesg` during the test window (the one Xid entry
  present was 6 days stale, unrelated).
- **Not a `libnccl` collective hang.** The stuck frames are inside this
  codebase's own TCP relay (`multiproc_relay.rs`), not inside NCCL.

## Not yet pinned down (explicitly deferred, not guessed)

- **Why** a rank stops acking specifically between 7661 and 8106 prompt
  tokens — which rank, and what it is doing instead of acking (chunked-
  prefill step timing? the KV-tier host-demote path — the 7661 control run
  already showed `host_demoted_pages:13`, so write-through KV is active near
  this boundary). Needs per-rank tick-sequence instrumentation or CUDA-side
  tracing on the stalled rank, not done here.
- **TP=8 at the same prompt length is unverified** (this box only has 4 free
  GPUs; GPU1 is held by a foreign tenant). The lockstep subsystem is
  TP>1-specific by construction (`TP=1 never reaches here`), so it cannot
  reproduce at TP=1, but whether TP=8 also breaks at ~8106 tokens is open.

## Rule

- **A retry/wait loop gating shared server state (here: every HTTP request)
  needs an explicit deadline**, not just a dead-worker check — "the worker
  will error before it silently stops responding" is an assumption, not a
  guarantee, and when it's wrong the failure mode is a permanent, silent,
  whole-server hang instead of a bounded error.
- **A guessed root cause ("VRAM squeeze") from bisection + A/B alone, without
  a measured mechanism, is not licensed** — the A/B correctly ruled out "this
  diff caused it" but said nothing about *why*. Root-cause hypotheses need
  their own verification (here: `nvidia-smi` during the hang + gdb
  backtraces), same as any other claim.
